"""Background value-signal scanner (APScheduler) for FastAPI lifespan.

Every ``SIGNAL_INTERVAL_MINUTES`` (default 20): scan upcoming fixtures within
``SIGNAL_HORIZON_HOURS`` (default 24), fire when EV ≥ ``SIGNAL_MIN_EV`` **or**
``|line_delta|`` ≥ ``SIGNAL_MIN_LINE_DELTA``, dedupe, push Telegram HTML.

Jobs run off the event loop via ``asyncio.to_thread`` so API routes stay fast.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from src.bot.notifier import broadcast_value_signal
from src.bot.signal_store import (
    DEFAULT_DB_PATH,
    make_signal_key,
    mark_notified,
)

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_MINUTES = 20
DEFAULT_HORIZON_HOURS = 24
DEFAULT_MIN_EV_PCT = 8.0
DEFAULT_MIN_LINE_DELTA = 0.5
# 0 = disabled. Production telegram_bot compose sets SIGNAL_MIN_DATA_SCORE=75.
DEFAULT_MIN_DATA_SCORE = 0.0

# Scan below signal EV so line-disparity-only picks still appear.
_SCAN_MIN_EV_PCT = 1.0
_SCAN_LIMIT = 80

_scheduler = None  # apscheduler AsyncIOScheduler | None


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


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


def scheduler_enabled() -> bool:
    return _env_bool("ENABLE_VALUE_SCHEDULER", default=False)


def signal_min_ev_pct() -> float:
    return _env_float("SIGNAL_MIN_EV", DEFAULT_MIN_EV_PCT)


def signal_min_line_delta() -> float:
    return _env_float("SIGNAL_MIN_LINE_DELTA", DEFAULT_MIN_LINE_DELTA)


def signal_min_data_score() -> float:
    """Minimum aggregate data score (0–100). ``0`` disables the gate."""
    return _env_float("SIGNAL_MIN_DATA_SCORE", DEFAULT_MIN_DATA_SCORE)


def bet_data_score(bet: Mapping[str, Any]) -> float | None:
    """Read ``aggregate_data_score`` / ``data_score`` when present."""
    for key in ("aggregate_data_score", "data_score"):
        raw = bet.get(key)
        if raw is None:
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if val != val:  # NaN
            continue
        return val
    return None


def market_line_delta(bet: Mapping[str, Any]) -> float | None:
    """Abs board-line delta for the bet's market (OU / AH), else best available."""
    mkt = str(bet.get("market") or "").strip().upper()
    ou = bet.get("ou_line_delta")
    ah = bet.get("ah_line_delta")

    def _f(v: Any) -> float | None:
        if v is None:
            return None
        try:
            x = float(v)
        except (TypeError, ValueError):
            return None
        if x != x:  # NaN
            return None
        return x

    ou_d, ah_d = _f(ou), _f(ah)
    if mkt == "OU" and ou_d is not None:
        return abs(ou_d)
    if mkt == "AH" and ah_d is not None:
        return abs(ah_d)
    parts = [abs(d) for d in (ou_d, ah_d) if d is not None]
    if not parts:
        return None
    return max(parts)


def is_hot_signal(
    bet: Mapping[str, Any],
    *,
    min_ev_pct: float = DEFAULT_MIN_EV_PCT,
    min_line_delta: float = DEFAULT_MIN_LINE_DELTA,
    min_data_score: float = DEFAULT_MIN_DATA_SCORE,
) -> bool:
    """True when EV ≥ floor **or** |line_delta| ≥ floor (and data-score gate)."""
    if float(min_data_score) > 0.0:
        score = bet_data_score(bet)
        # Missing score: allow (legacy scanner rows); present-but-low: reject.
        if score is not None and float(score) < float(min_data_score):
            return False

    ev_pct = bet.get("ev_pct")
    if ev_pct is None and bet.get("ev") is not None:
        try:
            ev = float(bet["ev"])
            ev_pct = ev * 100.0 if abs(ev) <= 2.0 else ev
        except (TypeError, ValueError):
            ev_pct = None
    try:
        ev_ok = ev_pct is not None and float(ev_pct) >= float(min_ev_pct)
    except (TypeError, ValueError):
        ev_ok = False

    delta = market_line_delta(bet)
    try:
        line_ok = delta is not None and float(delta) >= float(min_line_delta)
    except (TypeError, ValueError):
        line_ok = False

    return bool(ev_ok or line_ok)


def _parse_kickoff(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    s = str(value).strip()
    if not s:
        return None
    try:
        # Handle trailing Z
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def kickoff_within_hours(
    bet: Mapping[str, Any],
    *,
    hours: float = DEFAULT_HORIZON_HOURS,
    now: datetime | None = None,
) -> bool:
    """True if kickoff is in ``(now, now+hours]`` (UTC). Missing KO → False."""
    # Prefer ISO ``kickoff`` from the API row; skip display-only kickoff_vn.
    ko = _parse_kickoff(bet.get("kickoff") or bet.get("match_date"))
    if ko is None:
        return False
    now_utc = now or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    else:
        now_utc = now_utc.astimezone(timezone.utc)
    # Past kickoffs are not actionable.
    if ko <= now_utc:
        return False
    return ko <= now_utc + timedelta(hours=float(hours))


def filter_hot_signals(
    bets: list[Mapping[str, Any]],
    *,
    min_ev_pct: float = DEFAULT_MIN_EV_PCT,
    min_line_delta: float = DEFAULT_MIN_LINE_DELTA,
    min_data_score: float = DEFAULT_MIN_DATA_SCORE,
    horizon_hours: float = DEFAULT_HORIZON_HOURS,
    now: datetime | None = None,
) -> list[Mapping[str, Any]]:
    """Apply 24h window + EV/line/data-score hot filter (pure; no I/O)."""
    out: list[Mapping[str, Any]] = []
    for bet in bets:
        if not kickoff_within_hours(bet, hours=horizon_hours, now=now):
            continue
        if not is_hot_signal(
            bet,
            min_ev_pct=min_ev_pct,
            min_line_delta=min_line_delta,
            min_data_score=min_data_score,
        ):
            continue
        out.append(bet)
    return out


def _scan_candidates() -> list[dict[str, Any]]:
    """Blocking scan — call via ``asyncio.to_thread``."""
    from src.api.services.value_bets import scan_value_bets_api

    result = scan_value_bets_api(
        min_ev_pct=_SCAN_MIN_EV_PCT,
        markets="1X2,AH,OU",
        limit=_SCAN_LIMIT,
        league=None,
    )
    bets = result.get("bets") or []
    return [dict(b) for b in bets if isinstance(b, Mapping)]


async def run_value_signal_job(
    *,
    db_path: Any = None,
    bot_token: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """One scheduler tick: scan → filter → dedupe → push."""
    min_ev = signal_min_ev_pct()
    min_delta = signal_min_line_delta()
    min_score = signal_min_data_score()
    horizon = _env_float("SIGNAL_HORIZON_HOURS", float(DEFAULT_HORIZON_HOURS))
    store = db_path or DEFAULT_DB_PATH

    try:
        candidates = await asyncio.to_thread(_scan_candidates)
    except Exception as exc:  # noqa: BLE001
        logger.exception("value-signal scan failed: %s", exc)
        return {"ok": False, "error": str(exc), "pushed": 0}

    hot = filter_hot_signals(
        candidates,
        min_ev_pct=min_ev,
        min_line_delta=min_delta,
        min_data_score=min_score,
        horizon_hours=horizon,
    )
    pushed = 0
    skipped_dup = 0
    errors: list[str] = []

    for bet in hot:
        key = make_signal_key(
            match_id=bet.get("match_id"),
            market=bet.get("market"),
            selection=bet.get("selection"),
            home=bet.get("home") or bet.get("home_team"),
            away=bet.get("away") or bet.get("away_team"),
            kickoff=str(bet.get("kickoff") or "") or None,
        )
        if dry_run:
            logger.info("dry-run signal %s", key)
            continue
        if not mark_notified(
            key,
            match_id=bet.get("match_id"),
            market=bet.get("market"),
            selection=bet.get("selection"),
            db_path=store,
        ):
            skipped_dup += 1
            continue
        try:
            res = await broadcast_value_signal(
                bet,
                bot_token=bot_token,
                db_path=store,
            )
            pushed += int(res.get("sent") or 0)
            if res.get("errors"):
                errors.extend(list(res["errors"]))
        except Exception as exc:  # noqa: BLE001
            logger.warning("broadcast failed for %s: %s", key, exc)
            errors.append(str(exc))

    summary = {
        "ok": True,
        "candidates": len(candidates),
        "hot": len(hot),
        "pushed": pushed,
        "skipped_dup": skipped_dup,
        "errors": errors,
    }
    if hot or candidates:
        logger.info(
            "value-signal job: candidates=%s hot=%s pushed=%s dup=%s",
            len(candidates),
            len(hot),
            pushed,
            skipped_dup,
        )
    return summary


def settle_scheduler_enabled() -> bool:
    """Settlement job runs with the value scheduler, or alone when enabled."""
    # Default on when value scheduler is on; opt-in via ENABLE_SETTLE_SCHEDULER.
    if scheduler_enabled():
        return _env_bool("ENABLE_SETTLE_SCHEDULER", default=True)
    return _env_bool("ENABLE_SETTLE_SCHEDULER", default=False)


def start_scheduler() -> Any:
    """Create and start ``AsyncIOScheduler`` if enabled. Idempotent."""
    global _scheduler
    if _scheduler is not None:
        return _scheduler

    want_signals = scheduler_enabled()
    want_settle = settle_scheduler_enabled()
    if not want_signals and not want_settle:
        logger.info(
            "Schedulers disabled "
            "(ENABLE_VALUE_SCHEDULER / ENABLE_SETTLE_SCHEDULER ≠ true)"
        )
        return None

    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
    except ImportError:
        logger.error("apscheduler not installed — pip install apscheduler")
        return None

    interval = _env_int("SIGNAL_INTERVAL_MINUTES", DEFAULT_INTERVAL_MINUTES)
    settle_interval = _env_int("SETTLE_INTERVAL_MINUTES", 45)
    sched = AsyncIOScheduler()

    if want_signals:
        sched.add_job(
            run_value_signal_job,
            trigger="interval",
            minutes=max(1, int(interval)),
            id="value_signal_scan",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        logger.info(
            "Value signal scheduler started "
            "(every %s min, EV≥%s%% or |Δline|≥%s, data_score≥%s)",
            interval,
            signal_min_ev_pct(),
            signal_min_line_delta(),
            signal_min_data_score(),
        )

    if want_settle:
        sched.add_job(
            run_settlement_job,
            trigger="interval",
            minutes=max(5, int(settle_interval)),
            id="settle_completed_matches",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        logger.info("Settlement scheduler started (every %s min)", settle_interval)

    sched.start()
    _scheduler = sched
    return sched


async def run_settlement_job() -> dict[str, Any]:
    """Periodic settle of PENDING journal bets vs ``global_matches.db``."""
    from src.services.settlement_service import settle_completed_matches_async

    try:
        summary = await settle_completed_matches_async()
    except Exception as exc:  # noqa: BLE001
        logger.exception("settlement job failed: %s", exc)
        return {"ok": False, "error": str(exc)}
    if summary.get("settled"):
        logger.info(
            "settlement job: settled=%s pnl=%.2f",
            summary.get("settled"),
            float(summary.get("pnl") or 0.0),
        )
    return {"ok": True, **summary}


def stop_scheduler() -> None:
    """Shut down background scheduler (lifespan exit)."""
    global _scheduler
    if _scheduler is None:
        return
    try:
        _scheduler.shutdown(wait=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("scheduler shutdown: %s", exc)
    _scheduler = None
