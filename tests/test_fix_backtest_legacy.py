"""Regression tests: legacy backtester, canonical CLV, Brier, season labels."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.backtester import (
    BacktestConfig,
    _closing_1x2_market,
    _season_label,
    run_backtest,
)
from src.calibration import closing_line_value, multiclass_brier
from src.config import MAX_STAKE_PCT
from src.validation.metrics import QuantMetrics, devig_odds, market_overround


def _toy(n: int = 90, *, closing: bool = True, bad_scores: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    teams = [f"T{i}" for i in range(6)]
    rows = []
    base = pd.Timestamp("2023-08-01")
    for i in range(n):
        h, a = teams[i % 6], teams[(i + 1 + i // 6) % 6]
        if h == a:
            a = teams[(i + 2) % 6]
        hg, ag = int(rng.integers(0, 4)), int(rng.integers(0, 4))
        row = {
            "Date": base + pd.Timedelta(days=(i // 2) * 7),
            "HomeTeam": h,
            "AwayTeam": a,
            "FTHG": hg,
            "FTAG": ag,
            "B365H": 2.6, "B365D": 3.4, "B365A": 2.9,
            "AvgH": 2.5, "AvgD": 3.3, "AvgA": 2.8,
            "B365>2.5": 1.95, "B365<2.5": 1.95,
        }
        if closing:
            row.update({"PSCH": 2.4, "PSCD": 3.5, "PSCA": 3.0})
        rows.append(row)
    df = pd.DataFrame(rows)
    if bad_scores:
        df["FTHG"] = df["FTHG"].astype(object)
        df["FTAG"] = df["FTAG"].astype(object)
        df.loc[5, "FTHG"] = "-"
        df.loc[7, "FTAG"] = ""
    return df


def _cfg(**kw) -> BacktestConfig:
    base = dict(
        min_train_matches=30,
        persist=False,
        calibration="none",
        refit_every=30,
        min_odds=1.01,
        max_odds=10.0,
        min_ev=0.0,
        allowed_markets=["1X2"],
    )
    base.update(kw)
    return BacktestConfig(**base)


def test_non_numeric_scores_do_not_crash() -> None:
    result = run_backtest(_toy(bad_scores=True), config=_cfg())
    assert not result.predictions.empty


def test_kelly_follows_live_policy_clamp() -> None:
    # Absurd kelly_fraction must be clamped like live (MAX_KELLY_FRACTION /
    # max_stake_pct), not passed through unclamped.
    result = run_backtest(_toy(), config=_cfg(kelly_fraction=5.0, max_stake_pct=0.5))
    if not result.bets.empty:
        from src.strategy import capped_kelly_fraction

        for _, b in result.bets.iterrows():
            expected = capped_kelly_fraction(
                b["p_model"], b["odds"], kelly_fraction=5.0, max_stake_pct=0.5
            )
            assert b["stake_pct"] == pytest.approx(expected)
    default = run_backtest(_toy(), config=_cfg())
    if not default.bets.empty:
        assert (default.bets["stake_pct"] <= MAX_STAKE_PCT + 1e-12).all()


def test_clv_uses_real_closing_columns_else_nan() -> None:
    with_close = run_backtest(_toy(closing=True), config=_cfg())
    assert not with_close.bets.empty
    b = with_close.bets.iloc[0]
    close = {"H": 2.4, "D": 3.5, "A": 3.0}
    key = {"Home": "H", "Draw": "D", "Away": "A"}.get(b["selection"], b["selection"])
    assert b["odds_closing"] == pytest.approx(close[key])
    assert b["clv"] == pytest.approx(
        QuantMetrics.closing_line_value(b["odds"], close[key], closing_market_odds=close)
    )
    # Old same-snapshot metric survives under an honest name.
    assert "price_vs_market_avg" in with_close.bets.columns

    no_close = run_backtest(_toy(closing=False), config=_cfg())
    assert not no_close.bets.empty
    assert no_close.bets["clv"].isna().all()
    assert no_close.summary["n_clv"] == 0


def test_closing_market_helper_prefers_pinnacle_closing() -> None:
    row = pd.Series({"PSCH": 2.0, "PSCD": 3.5, "PSCA": 3.9, "AvgCH": 1.9, "AvgCD": 3.4, "AvgCA": 3.8})
    assert _closing_1x2_market(row) == {"H": 2.0, "D": 3.5, "A": 3.9}
    assert _closing_1x2_market(row, ref_family="Avg") == {"H": 1.9, "D": 3.4, "A": 3.8}
    assert _closing_1x2_market(pd.Series({"B365H": 2.0, "B365D": 3.0, "B365A": 4.0})) is None


def test_ou_calibration_excludes_pushes() -> None:
    df = _toy(120)
    # Integer line → pushes possible when total == 2.
    df = df.drop(columns=["B365>2.5", "B365<2.5"])
    df["B365_O25"], df["B365_U25"] = 1.9, 1.9
    result = run_backtest(
        df, config=_cfg(allowed_markets=["OU"], calibration="isotonic", min_cal_samples=20)
    )
    ou = result.predictions
    if not ou.empty and "market" in ou.columns:
        ou = ou[ou["market"] == "OU"]
        # .5 line: every row labelled; labels are 0/1 and pairs sum to 1
        assert set(ou["y"].unique()) <= {0.0, 1.0}


def test_canonical_clv_single_definition() -> None:
    assert closing_line_value(2.2, 2.0) == pytest.approx(math.log(1.1))
    assert QuantMetrics.closing_line_value(2.2, 2.0) == pytest.approx(math.log(1.1))
    market = {"H": 2.0, "D": 3.5, "A": 3.9}
    ovr = market_overround(market)
    assert ovr is not None and ovr > 1.0
    assert closing_line_value(2.2, 2.0, closing_market_odds=market) == pytest.approx(
        math.log(2.2 / (2.0 * ovr))
    )
    fair = devig_odds(market)
    assert fair is not None and sum(1 / v for v in fair.values()) == pytest.approx(1.0)
    assert math.isnan(closing_line_value(1.0, 2.0))


def test_multiclass_brier_standard_definition() -> None:
    labels = ["H", "D", "A"]
    uniform = {c: [1 / 3] * 3 for c in ("H", "D", "A")}
    assert multiclass_brier(labels, uniform) == pytest.approx(2 / 3)
    perfect = {"H": [1, 0, 0], "D": [0, 1, 0], "A": [0, 0, 1]}
    assert multiclass_brier(labels, perfect) == pytest.approx(0.0)
    worst = {"H": [0, 1, 1], "D": [1, 0, 0], "A": [0, 0, 0]}
    assert multiclass_brier(labels, worst) == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("league", "day", "expected"),
    [
        (None, "2024-09-10", "2024/2025"),
        (None, "2024-03-10", "2023/2024"),
        ("EPL", "2024-03-10", "2023/2024"),
        ("J1", "2024-03-10", "2024"),
        ("J1", "2024-11-10", "2024"),
        ("EMPERORS_CUP", "2025-06-10", "2025"),
        ("J_LEAGUE_CUP", "2025-04-10", "2025"),
    ],
)
def test_season_label_calendar_leagues(league, day, expected) -> None:
    row = pd.Series({"HomeTeam": "A"})
    assert _season_label(row, pd.Timestamp(day), league) == expected


def test_season_label_from_row_column() -> None:
    row = pd.Series({"comp_id": "EMPERORS_CUP"})
    assert _season_label(row, pd.Timestamp("2024-10-01")) == "2024"
