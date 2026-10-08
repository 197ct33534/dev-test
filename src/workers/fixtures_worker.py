#!/usr/bin/env python3
"""Fixtures worker — keep ``upcoming_fixtures`` (fixtures + odds) fresh.

The API / value-signal scanner read odds from the ``upcoming_fixtures`` cache in
``global_matches.db`` and treat it as stale after ``FIXTURES_MAX_AGE_MINUTES``
(default 60). Nothing else refreshes that cache in Docker, so this worker
re-fetches every ``FIXTURES_REFRESH_INTERVAL_MINUTES`` (default 20) for the
leagues in ``FIXTURES_REFRESH_LEAGUES`` (default ``EPL,UWCL``).

Fetching goes through the existing loaders (``data_loader.load_upcoming_fixtures``
→ Fotmob / ``flashscore_league.fetch_league_upcoming`` + odds) and
``global_db.save_upcoming_to_db`` via
:func:`src.api.services.value_bets.refresh_upcoming_fixtures`. An empty fetch
never overwrites the cache.

Usage
-----
    python -m src.workers.fixtures_worker            # loop
    python -m src.workers.fixtures_worker --once     # single pass (cron / smoke)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")

logger = logging.getLogger("src.workers.fixtures_worker")

DEFAULT_INTERVAL_MINUTES = 20
DEFAULT_LEAGUES = ("EPL", "UWCL")


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return int(default)
    try:
        return int(float(raw))
    except ValueError:
        return int(default)


def refresh_leagues() -> list[str]:
    """Leagues to refresh (``FIXTURES_REFRESH_LEAGUES``, comma-separated)."""
    raw = (os.getenv("FIXTURES_REFRESH_LEAGUES") or "").strip()
    if not raw:
        return list(DEFAULT_LEAGUES)
    out: list[str] = []
    for part in raw.split(","):
        code = part.strip().upper()
        if code and code not in out:
            out.append(code)
    return out or list(DEFAULT_LEAGUES)


def run_refresh_cycle(
    leagues: list[str] | None = None,
    *,
    refresh_fn: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Refresh every league once; one league failing never blocks the others."""
    if refresh_fn is None:
        from src.api.services.value_bets import refresh_upcoming_fixtures

        refresh_fn = refresh_upcoming_fixtures

    summary: dict[str, Any] = {"ok": True, "leagues": {}}
    for code in leagues or refresh_leagues():
        try:
            fx = refresh_fn(code)
            n = 0 if fx is None else int(len(fx))
            summary["leagues"][code] = {"ok": True, "rows": n}
            if n:
                logger.info("fixtures refresh · %s rows=%s", code, n)
            else:
                logger.warning("fixtures refresh · %s returned 0 rows (cache kept)", code)
        except Exception as exc:  # noqa: BLE001
            summary["ok"] = False
            summary["leagues"][code] = {"ok": False, "error": type(exc).__name__}
            logger.exception("fixtures refresh failed for %s: %s", code, exc)
    return summary


async def _worker_loop(stop: asyncio.Event) -> None:
    interval_min = max(5, _env_int("FIXTURES_REFRESH_INTERVAL_MINUTES", DEFAULT_INTERVAL_MINUTES))
    interval_s = interval_min * 60
    logger.info(
        "fixtures_worker started · interval=%s min · leagues=%s",
        interval_min,
        ",".join(refresh_leagues()),
    )
    while not stop.is_set():
        try:
            # Network + parsing are blocking — keep the loop responsive to SIGTERM.
            await asyncio.to_thread(run_refresh_cycle)
        except Exception as exc:  # noqa: BLE001
            logger.exception("fixtures cycle failed (will retry): %s", exc)
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass
    logger.info("fixtures_worker shut down cleanly")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        level=logging.INFO,
    )
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--once", action="store_true", help="Run one refresh pass and exit")
    args = parser.parse_args(argv)
    if args.once:
        result = run_refresh_cycle()
        sys.exit(0 if result.get("ok") else 1)

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
            pass

    try:
        loop.run_until_complete(_worker_loop(stop))
    finally:
        loop.close()


if __name__ == "__main__":
    main()
