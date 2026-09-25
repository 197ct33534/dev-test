#!/usr/bin/env python3
"""Settlement worker — journal PnL + Quant Engine v2 (Pinnacle CLV).

Every ``SETTLE_INTERVAL_MINUTES`` (default 30):

1. Legacy journals via ``settle_completed_matches``
2. ``bet_snapshots`` via ``SettlementEngineV2`` (closing odds / CLV)

Recovers from transient errors; SIGTERM / SIGINT shut down cleanly.

Usage
-----
    python -m src.workers.settlement_worker
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timezone
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
logger = logging.getLogger("src.workers.settlement_worker")

DEFAULT_INTERVAL_MINUTES = 30


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return int(default)
    try:
        return int(float(raw))
    except ValueError:
        return int(default)


async def settle_v2_pending() -> dict[str, Any]:
    """Settle PENDING ``bet_snapshots`` with ``SettlementEngineV2`` (+ CLV)."""
    from src.db.session import get_session_factory
    from src.execution.settlement_v2 import SettlementEngineV2

    now = datetime.now(timezone.utc)
    factory = get_session_factory()
    async with factory() as session:
        engine = SettlementEngineV2(session)
        summary = await engine.settle_pending_bets(now)
        await session.commit()
    return {"ok": True, **summary}


async def run_settlement_cycle() -> dict[str, Any]:
    """One tick: legacy journals + v2 engine."""
    from src.services.settlement_service import settle_completed_matches_async

    out: dict[str, Any] = {"ok": True, "journal": None, "v2": None}

    try:
        journal = await settle_completed_matches_async()
        out["journal"] = journal
        logger.info(
            "journal settle · settled=%s pnl=%.2f skipped=%s",
            journal.get("settled"),
            float(journal.get("pnl") or 0.0),
            journal.get("skipped"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("journal settlement failed: %s", exc)
        out["ok"] = False
        out["journal"] = {"ok": False, "error": str(exc)}

    try:
        v2 = await settle_v2_pending()
        out["v2"] = v2
        logger.info(
            "v2 settle · settled=%s pnl=%.2f skipped=%s",
            v2.get("settled"),
            float(v2.get("pnl") or 0.0),
            v2.get("skipped"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("SettlementEngineV2 failed: %s", exc)
        out["ok"] = False
        out["v2"] = {"ok": False, "error": str(exc)}

    return out


async def _worker_loop(stop: asyncio.Event) -> None:
    interval_min = max(5, _env_int("SETTLE_INTERVAL_MINUTES", DEFAULT_INTERVAL_MINUTES))
    interval_s = interval_min * 60
    logger.info("settlement_worker started · interval=%s min", interval_min)

    while not stop.is_set():
        try:
            await run_settlement_cycle()
        except Exception as exc:  # noqa: BLE001
            logger.exception("settlement cycle failed (will retry): %s", exc)

        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass

    logger.info("settlement_worker shut down cleanly")


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
            pass

    try:
        loop.run_until_complete(_worker_loop(stop))
    finally:
        loop.close()


if __name__ == "__main__":
    main()
