"""Unit tests for model fair-line pricing + line disparity."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from src.dixon_coles import score_probability
from src.strategy import (
    COMMON_OU_LINES,
    ModelFairLines,
    ah_home_cover_from_matrix,
    attach_fair_line_fields,
    calculate_model_fair_lines,
    line_disparity_score,
    over_cover_from_matrix,
    select_value_bets,
)


def _dc_matrix(lam: float, mu: float, rho: float = -0.05, g: int = 10) -> np.ndarray:
    mat = np.zeros((g + 1, g + 1), dtype=float)
    for x in range(g + 1):
        for y in range(g + 1):
            mat[x, y] = score_probability(x, y, lam, mu, rho)
    return mat / mat.sum()


def test_fair_ou_near_50_percent() -> None:
    lam, mu = 1.4, 1.1
    mat = _dc_matrix(lam, mu)
    fair = calculate_model_fair_lines(lam, mu, mat)
    assert isinstance(fair, ModelFairLines)
    assert fair.fair_total_goals == pytest.approx(lam + mu)
    assert fair.fair_ou_line in COMMON_OU_LINES
    assert abs(fair.p_over_at_fair - 0.5) < 0.08
    # Chosen line should be the closest-to-50% among candidates.
    gaps = {
        L: abs(over_cover_from_matrix(mat, L) - 0.5) for L in COMMON_OU_LINES
    }
    assert gaps[fair.fair_ou_line] == pytest.approx(min(gaps.values()), abs=1e-9)


def test_fair_odds_are_inverse_prob() -> None:
    lam, mu = 1.6, 1.2
    mat = _dc_matrix(lam, mu)
    fair = calculate_model_fair_lines(lam, mu, mat)
    p_h = float(np.tril(mat, k=-1).sum())
    p_d = float(np.trace(mat))
    p_a = float(np.triu(mat, k=1).sum())
    assert fair.fair_odds["H"] == pytest.approx(1.0 / p_h)
    assert fair.fair_odds["D"] == pytest.approx(1.0 / p_d)
    assert fair.fair_odds["A"] == pytest.approx(1.0 / p_a)
    assert fair.fair_odds["Over"] == pytest.approx(1.0 / fair.p_over_at_fair)
    assert fair.fair_odds["AH_Home"] == pytest.approx(1.0 / fair.p_ah_home_at_fair)


def test_fair_ah_near_50_and_matches_dc_helper() -> None:
    lam, mu = 2.2, 0.8  # strong home favourite → non-positive fair AH
    mat = _dc_matrix(lam, mu)
    fair = calculate_model_fair_lines(lam, mu, mat)
    assert fair.fair_ah_line <= 0
    assert abs(fair.p_ah_home_at_fair - 0.5) < 0.08
    p_vec = ah_home_cover_from_matrix(mat, fair.fair_ah_line)
    assert p_vec == pytest.approx(fair.p_ah_home_at_fair)

    # Away favourite → non-negative fair home handicap.
    fair2 = calculate_model_fair_lines(0.8, 2.2, _dc_matrix(0.8, 2.2))
    assert fair2.fair_ah_line >= 0


def test_disparity_signs() -> None:
    fair = ModelFairLines(
        fair_total_goals=2.75,
        fair_ou_line=2.75,
        fair_ah_line=-0.5,
        fair_odds={"H": 2.0},
        p_over_at_fair=0.5,
        p_ah_home_at_fair=0.5,
    )
    row = attach_fair_line_fields(
        {"market": "OU", "ev": 0.12, "selection": "Over 2.25"},
        fair,
        bookie_ou_line=2.25,
        bookie_ah_line=-0.25,
    )
    assert row["ou_line_delta"] == pytest.approx(0.50)
    assert row["ah_line_delta"] == pytest.approx(-0.25)
    assert row["model_fair_line"] == "Tài Xỉu 2.75"
    assert row["bookie_market_line"] == "Tài Xỉu 2.25"
    assert row["line_edge"] == "+0.50 bàn"

    ah_row = attach_fair_line_fields(
        {"market": "AH", "ev": 0.08},
        fair,
        bookie_ou_line=2.25,
        bookie_ah_line=-0.25,
    )
    assert ah_row["model_fair_line"] == "AH -0.5"
    assert ah_row["bookie_market_line"] == "AH -0.25"
    assert ah_row["line_edge"] == "-0.25 bàn"
    assert ah_row["ah_line_delta"] < 0

    # High |delta| OU should score above equal-EV with zero delta.
    s_hi = line_disparity_score(
        ou_line_delta=0.5, ah_line_delta=None, market="OU", ev=0.10
    )
    s_lo = line_disparity_score(
        ou_line_delta=0.0, ah_line_delta=None, market="OU", ev=0.10
    )
    assert s_hi > s_lo


def test_select_value_bets_secondary_sort_by_disparity() -> None:
    """Equal EV → higher line_disparity_score ranks first."""
    bets = pd.DataFrame(
        [
            {
                "home_team": "A",
                "away_team": "B",
                "market": "OU",
                "selection": "Over 2.5",
                "ev": 0.10,
                "ev_pct": 10.0,
                "match_id": "m1",
                "line_disparity_score": 0.10,
            },
            {
                "home_team": "C",
                "away_team": "D",
                "market": "OU",
                "selection": "Over 2.5",
                "ev": 0.10,
                "ev_pct": 10.0,
                "match_id": "m2",
                "line_disparity_score": 0.40,
            },
        ]
    )
    out = select_value_bets(bets, max_per_day=5, apply_sanity=False)
    assert list(out["match_id"]) == ["m2", "m1"]


def test_fair_lines_timing_under_5ms() -> None:
    lam, mu = 1.35, 1.15
    mat = _dc_matrix(lam, mu)
    # Warm-up
    calculate_model_fair_lines(lam, mu, mat)
    times: list[float] = []
    for _ in range(40):
        t0 = time.perf_counter()
        calculate_model_fair_lines(lam, mu, mat)
        times.append((time.perf_counter() - t0) * 1000.0)
    median_ms = float(np.median(times))
    assert median_ms < 5.0, f"median {median_ms:.2f} ms ≥ 5 ms"


def test_vectorised_over_matches_dc_predict() -> None:
    """Sanity: vectorised cover ≈ DixonColes.predict_over_under on same matrix."""
    # Build a tiny fitted-like model by monkeypatching predict_score_matrix.
    lam, mu, rho = 1.5, 1.2, -0.08
    mat = _dc_matrix(lam, mu, rho=rho)

    class _Stub:
        def predict_score_matrix(self, *a, **k):
            return mat

        def predict_over_under(self, home, away, line=2.5, max_goals=None):
            from src.dixon_coles import DixonColesModel

            return DixonColesModel.predict_over_under(
                self, home, away, line=line, max_goals=max_goals
            )

    stub = _Stub()
    for line in (2.0, 2.25, 2.5, 2.75, 3.0):
        dc = stub.predict_over_under("H", "A", line=line)
        vec = over_cover_from_matrix(mat, line)
        assert vec == pytest.approx(dc["over"], abs=1e-9)
