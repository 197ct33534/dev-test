"""Smoke tests for walk-forward backtesting master."""

from __future__ import annotations

import math

import pytest

from src.backtest.report import build_quant_report, render_quant_report
from src.backtest.run_backtest import (
    BacktestConfig,
    generate_synthetic_matches,
    run_backtest,
)


def test_generate_synthetic_matches_count_and_order() -> None:
    matches = generate_synthetic_matches(n=12, seed=7)
    assert len(matches) == 12
    kicks = [m.kickoff_utc for m in matches]
    assert kicks == sorted(kicks)
    assert all(m.ft_home_goals is not None for m in matches)
    assert all(m.raw_records for m in matches)


def test_build_quant_report_basic() -> None:
    report = build_quant_report(
        settled_bets=[
            {"status": "WON", "pnl": 50.0, "stake": 100.0, "clv_value": 0.05},
            {"status": "LOST", "pnl": -100.0, "stake": 100.0, "clv_value": -0.02},
        ],
        equity_curve=[10_000.0, 10_050.0, 9_950.0],
        y_true=[1, 0],
        y_prob=[0.6, 0.4],
        tier_counts={"A": 3, "B": 1, "X": 1},
        initial_bankroll=10_000.0,
        final_bankroll=9_950.0,
        n_matches=5,
        n_priced=4,
        mode="dry-run",
    )
    assert report.total_bets == 2
    assert report.net_pnl == pytest.approx(-50.0)
    assert report.tier_counts["A"] == 3
    assert math.isfinite(report.brier)
    text = render_quant_report(report)
    assert "Quant Report" in text
    assert "Brier" in text
    assert "Max Drawdown" in text


@pytest.mark.asyncio
async def test_run_backtest_dry_run_smoke() -> None:
    cfg = BacktestConfig(
        start_date=__import__("datetime").date(2024, 1, 1),
        end_date=__import__("datetime").date(2026, 9, 1),
        initial_bankroll=10_000.0,
        dry_run=True,
        concurrency=4,
        seed=123,
    )
    report = await run_backtest(cfg)
    assert report.n_matches == 50
    assert report.mode == "dry-run"
    assert sum(report.tier_counts.values()) == report.n_matches
    assert report.final_bankroll > 0
    text = render_quant_report(report)
    assert "Financial" in text or "Total Bets" in text
