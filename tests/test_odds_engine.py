"""Unit tests for OddsHistoryEngine (sharp/soft + as-of consensus)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.data.odds_engine import OddsHistoryEngine, classify_bookmaker


def test_classify_sharp_sources() -> None:
    assert classify_bookmaker("Pinnacle")["is_sharp"] is True
    assert classify_bookmaker("Betfair Exchange")["is_sharp"] is True
    assert classify_bookmaker("bet365")["is_soft"] is True
    assert classify_bookmaker("Betfair Sportsbook")["is_soft"] is True


def test_process_and_consensus() -> None:
    eng = OddsHistoryEngine()
    t0 = datetime(2026, 9, 25, 10, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 9, 25, 11, 0, tzinfo=timezone.utc)

    summary = eng.process_odds_snapshot(
        {
            "match_id": "m1",
            "timestamp": t0.isoformat(),
            "odds": [
                {
                    "bookmaker": "Pinnacle",
                    "market": "1X2",
                    "home": 2.10,
                    "draw": 3.40,
                    "away": 3.50,
                },
                {
                    "bookmaker": "bet365",
                    "market": "1X2",
                    "home": 2.00,
                    "draw": 3.50,
                    "away": 3.80,
                    "timestamp": t0.isoformat(),
                },
            ],
        }
    )
    assert summary["n_sharp"] == 1
    assert summary["n_soft"] == 1

    eng.process_odds_snapshot(
        {
            "match_id": "m1",
            "bookmakers": [
                {
                    "name": "Betfair Exchange",
                    "timestamp": t1.isoformat(),
                    "markets": {
                        "1X2": {"H": 2.20, "D": 3.30, "A": 3.40},
                    },
                }
            ],
        }
    )

    # As-of before t1 → Pinnacle only.
    c0 = eng.get_sharp_consensus_odds("m1", t0)
    assert c0["n_snapshots"] == 1
    assert "Pinnacle" in c0["sources"]
    assert c0["markets"]["1X2"]["H"] == pytest.approx(2.10)

    # As-of t1 → Betfair Exchange (later sharp tick).
    c1 = eng.get_sharp_consensus_odds("m1", t1)
    assert c1["n_snapshots"] == 1
    assert c1["markets"]["1X2"]["H"] == pytest.approx(2.20)


def test_async_wrappers() -> None:
    import asyncio

    async def _run() -> None:
        eng = OddsHistoryEngine()
        ts = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
        await eng.aprocess_odds_snapshot(
            {
                "match_id": "m2",
                "timestamp": ts.isoformat(),
                "bookmaker": "Pinnacle",
                "home": 1.80,
                "draw": 3.70,
                "away": 4.50,
            }
        )
        out = await eng.aget_sharp_consensus_odds("m2", ts)
        assert out["markets"]["1X2"]["H"] == pytest.approx(1.80)

    asyncio.run(_run())
