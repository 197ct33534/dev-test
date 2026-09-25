#!/usr/bin/env python3
"""Data-quality monitor worker — unmapped entities, freshness, PIT integrity.

Runs every ``DATA_MONITOR_INTERVAL_MINUTES`` (default 15). Recovers from
transient DB / network errors and shuts down cleanly on SIGTERM / SIGINT.

Usage
-----
    python -m src.workers.data_monitor
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("src.workers.data_monitor")

DEFAULT_INTERVAL_MINUTES = 15
# Alert when newest raw observation is older than this many hours.
DEFAULT_FRESHNESS_MAX_AGE_HOURS = 6.0
# Lookback window for recent PIT failures.
DEFAULT_PIT_LOOKBACK_HOURS = 48.0


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return int(default)
    try:
        return int(float(raw))
    except ValueError:
        return int(default)


def _ensure_aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


async def run_monitor_cycle() -> dict[str, Any]:
    """One tick: count unmapped teams, freshness lag, recent PIT failures."""
    from sqlalchemy import func, select

    from src.db.schema_v2 import (
        CanonicalCompetition,
        CanonicalTeam,
        CompetitionAlias,
        PitFeatureSnapshot,
        RawDataLake,
        TeamAlias,
    )
    from src.db.session import get_session_factory
    from src.monitoring.data_monitor import DataMonitor

    monitor = DataMonitor(logger_=logger)
    now = datetime.now(timezone.utc)
    freshness_max_h = _env_float(
        "DATA_MONITOR_FRESHNESS_MAX_AGE_HOURS", DEFAULT_FRESHNESS_MAX_AGE_HOURS
    )
    pit_lookback_h = _env_float(
        "DATA_MONITOR_PIT_LOOKBACK_HOURS", DEFAULT_PIT_LOOKBACK_HOURS
    )
    pit_since = now - timedelta(hours=pit_lookback_h)

    summary: dict[str, Any] = {
        "ok": True,
        "as_of": now.isoformat(),
        "unmapped_teams": 0,
        "unmapped_team_aliases": 0,
        "unmapped_competitions": 0,
        "unmapped_competition_aliases": 0,
        "newest_raw_observed_at": None,
        "raw_age_hours": None,
        "freshness_ok": True,
        "pit_failures_recent": 0,
        "pit_checked_recent": 0,
    }

    factory = get_session_factory()
    async with factory() as session:
        # --- Unmapped entities ---
        unmapped_teams = int(
            (
                await session.execute(
                    select(func.count())
                    .select_from(CanonicalTeam)
                    .where(CanonicalTeam.mapping_status == "UNMAPPED")
                )
            ).scalar_one()
            or 0
        )
        unmapped_team_aliases = int(
            (
                await session.execute(
                    select(func.count())
                    .select_from(TeamAlias)
                    .where(TeamAlias.mapping_status == "UNMAPPED")
                )
            ).scalar_one()
            or 0
        )
        unmapped_comps = int(
            (
                await session.execute(
                    select(func.count())
                    .select_from(CanonicalCompetition)
                    .where(CanonicalCompetition.mapping_status == "UNMAPPED")
                )
            ).scalar_one()
            or 0
        )
        unmapped_comp_aliases = int(
            (
                await session.execute(
                    select(func.count())
                    .select_from(CompetitionAlias)
                    .where(CompetitionAlias.mapping_status == "UNMAPPED")
                )
            ).scalar_one()
            or 0
        )

        summary["unmapped_teams"] = unmapped_teams
        summary["unmapped_team_aliases"] = unmapped_team_aliases
        summary["unmapped_competitions"] = unmapped_comps
        summary["unmapped_competition_aliases"] = unmapped_comp_aliases

        total_unmapped = (
            unmapped_teams
            + unmapped_team_aliases
            + unmapped_comps
            + unmapped_comp_aliases
        )
        if total_unmapped:
            monitor.warn(
                "Unmapped entities detected",
                code="UNMAPPED_ENTITIES",
                teams=unmapped_teams,
                team_aliases=unmapped_team_aliases,
                competitions=unmapped_comps,
                competition_aliases=unmapped_comp_aliases,
            )
        else:
            logger.info("Unmapped entities: none")

        # --- Freshness (newest raw observation) ---
        newest_raw = (
            await session.execute(select(func.max(RawDataLake.observed_at)))
        ).scalar_one()
        newest_raw = _ensure_aware(newest_raw)
        if newest_raw is not None:
            age_h = max(0.0, (now - newest_raw).total_seconds() / 3600.0)
            summary["newest_raw_observed_at"] = newest_raw.isoformat()
            summary["raw_age_hours"] = round(age_h, 3)
            if age_h > freshness_max_h:
                summary["freshness_ok"] = False
                monitor.alert(
                    "Raw data lake stale",
                    code="FRESHNESS_STALE",
                    age_hours=round(age_h, 2),
                    max_age_hours=freshness_max_h,
                    newest_observed_at=newest_raw.isoformat(),
                )
            else:
                logger.info(
                    "Freshness OK · newest raw %.2fh ago (max %.1fh)",
                    age_h,
                    freshness_max_h,
                )
        else:
            summary["freshness_ok"] = False
            monitor.warn(
                "No raw observations in lake — cannot score freshness",
                code="FRESHNESS_EMPTY",
            )

        # --- PIT integrity (recent snapshots) ---
        pit_checked = int(
            (
                await session.execute(
                    select(func.count())
                    .select_from(PitFeatureSnapshot)
                    .where(PitFeatureSnapshot.created_at >= pit_since)
                )
            ).scalar_one()
            or 0
        )
        pit_fails = int(
            (
                await session.execute(
                    select(func.count())
                    .select_from(PitFeatureSnapshot)
                    .where(
                        PitFeatureSnapshot.created_at >= pit_since,
                        PitFeatureSnapshot.pit_integrity_passed.is_(False),
                    )
                )
            ).scalar_one()
            or 0
        )
        summary["pit_checked_recent"] = pit_checked
        summary["pit_failures_recent"] = pit_fails

        if pit_fails:
            monitor.hard_gate(
                "PIT integrity failures in lookback window",
                code="PIT_INTEGRITY_FAIL",
                failures=pit_fails,
                checked=pit_checked,
                lookback_hours=pit_lookback_h,
            )
        else:
            logger.info(
                "PIT integrity OK · %s recent snapshot(s), 0 failures (lookback %.0fh)",
                pit_checked,
                pit_lookback_h,
            )

    logger.info(
        "data_monitor cycle done · unmapped=%s freshness_ok=%s pit_fails=%s",
        total_unmapped,
        summary["freshness_ok"],
        pit_fails,
    )
    return summary


async def _worker_loop(stop: asyncio.Event) -> None:
    interval_min = max(1, _env_int("DATA_MONITOR_INTERVAL_MINUTES", DEFAULT_INTERVAL_MINUTES))
    interval_s = interval_min * 60
    logger.info(
        "data_monitor_worker started · interval=%s min · PYTHONPATH ok",
        interval_min,
    )

    while not stop.is_set():
        try:
            await run_monitor_cycle()
        except Exception as exc:  # noqa: BLE001 — recover from network/DB blips
            logger.exception("data_monitor cycle failed (will retry): %s", exc)

        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass

    logger.info("data_monitor_worker shut down cleanly")


def main() -> None:
    stop = asyncio.Event()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _ask_stop(signum: int, _frame: Any) -> None:
        name = signal.Signals(signum).name if signum else "?"
        logger.info("Received %s — graceful shutdown", name)
        loop.call_soon_threadsafe(stop.set)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _ask_stop)
        except (ValueError, OSError):
            # Windows / non-main thread edge cases
            pass

    try:
        loop.run_until_complete(_worker_loop(stop))
    finally:
        loop.close()


if __name__ == "__main__":
    main()
