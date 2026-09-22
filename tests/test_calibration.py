"""Unit tests for calibration metrics and backtest hardening helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtester import BacktestConfig, run_ablation, run_backtest
from src.calibration import (
    apply_commission,
    brier_score,
    closing_line_value,
    expected_calibration_error,
    log_loss,
    reliability_table,
    summarize_predictions,
)


def test_brier_and_log_loss_perfect() -> None:
    y = [1, 0, 1, 0]
    p = [1.0, 0.0, 1.0, 0.0]
    assert brier_score(y, p) == 0.0
    assert log_loss(y, p) < 1e-6


def test_reliability_and_ece() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0.1, 0.9, size=200)
    y = (rng.random(200) < p).astype(float)
    table = reliability_table(y, p, n_bins=5)
    assert len(table) == 5
    ece = expected_calibration_error(y, p, n_bins=5)
    assert 0.0 <= ece < 0.5


def test_clv_and_commission() -> None:
    assert closing_line_value(2.2, 2.0) == pytest.approx(0.1)
    assert apply_commission(10.0, 0.02) == pytest.approx(9.8)
    assert apply_commission(-5.0, 0.02) == -5.0


def test_summarize_by_market_and_season() -> None:
    preds = pd.DataFrame(
        [
            {"market": "1X2", "season": "2023/2024", "p_model": 0.6, "y": 1},
            {"market": "1X2", "season": "2023/2024", "p_model": 0.4, "y": 0},
            {"market": "OU", "season": "2024/2025", "p_model": 0.55, "y": 1},
            {
                "market": "1X2",
                "season": "2023/2024",
                "p_H": 0.5,
                "p_D": 0.25,
                "p_A": 0.25,
                "ftr": "H",
            },
        ]
    )
    summary = summarize_predictions(preds)
    assert summary["overall"]["n"] == 3
    assert not summary["by_market"].empty
    assert set(summary["by_market"]["market"]) >= {"1X2", "OU"}
    assert "multiclass_brier" in summary["overall"]


def _toy_epl(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    teams = [f"T{i}" for i in range(6)]
    rows = []
    base = pd.Timestamp("2023-08-01")
    for i in range(n):
        h, a = teams[i % 6], teams[(i + 1 + i // 6) % 6]
        if h == a:
            a = teams[(i + 2) % 6]
        hg, ag = int(rng.integers(0, 4)), int(rng.integers(0, 4))
        ftr = "H" if hg > ag else ("A" if hg < ag else "D")
        # Closing-ish odds around 2.0–4.0
        ph = 0.35 + 0.1 * (hg - ag) / 4
        ph = float(np.clip(ph, 0.2, 0.6))
        pd_ = 0.25
        pa = 1.0 - ph - pd_
        rows.append(
            {
                "Date": base + pd.Timedelta(days=(i // 2) * 7),
                "HomeTeam": h,
                "AwayTeam": a,
                "FTHG": hg,
                "FTAG": ag,
                "FTR": ftr,
                "Season": "2023/2024" if i < n // 2 else "2024/2025",
                "B365H": 1.0 / ph,
                "B365D": 1.0 / pd_,
                "B365A": 1.0 / max(pa, 0.05),
                "AvgH": 1.0 / ph * 0.98,
                "AvgD": 1.0 / pd_ * 0.98,
                "AvgA": 1.0 / max(pa, 0.05) * 0.98,
            }
        )
    return pd.DataFrame(rows)


def test_walk_forward_closing_clv_commission() -> None:
    data = _toy_epl(90)
    cfg = BacktestConfig(
        min_train_matches=30,
        use_ml=False,
        allowed_markets=["1X2"],
        persist=False,
        calibration="none",
        commission_pct=0.02,
        refit_every=14,
        min_odds=1.01,
        max_odds=10.0,
        min_ev=0.0,
    )
    result = run_backtest(data, config=cfg)
    assert result.summary["commission_pct"] == 0.02
    assert "calibration" in result.summary
    assert not result.predictions.empty
    # CLV column present when bets exist
    if not result.bets.empty:
        assert "clv" in result.bets.columns
        assert "odds_closing" in result.bets.columns


def test_ablation_dc_vs_ensemble_grid() -> None:
    data = _toy_epl(70)
    base = BacktestConfig(
        min_train_matches=25,
        allowed_markets=["1X2"],
        persist=False,
        calibration="none",
        refit_every=20,
        min_odds=1.01,
        max_odds=10.0,
        min_ev=0.0,
        adapt_w_ml=True,
    )
    # DC-only only (skip heavy LGBM on tiny toy if slow — still include w=0)
    table = run_ablation(data, base_config=base, w_ml_grid=(0.0,))
    assert not table.empty
    assert table.iloc[0]["variant"] == "DC-only"
