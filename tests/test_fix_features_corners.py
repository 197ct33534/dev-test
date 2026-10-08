"""Regression tests: corner handicap sign / push and NB totals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import skellam

from src.corner_model import CornerPredictor, estimate_nb_dispersion


def _matches(n: int = 120, *, overdispersed: bool = True, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(6)]
    rows = []
    day0 = pd.Timestamp("2024-08-01")
    for i in range(n):
        home = teams[i % 6]
        away = teams[(i + 1 + i // 6) % 6]
        if home == away:
            away = teams[(i + 2) % 6]
        if overdispersed:
            # NB with mean ~5.5 / ~4.5 and strong dispersion.
            hc = int(rng.negative_binomial(3, 3 / (3 + 5.5)))
            ac = int(rng.negative_binomial(3, 3 / (3 + 4.5)))
        else:
            hc, ac = 5, 4  # zero variance → underdispersed vs Poisson
        rows.append(
            {
                "Date": day0 + pd.Timedelta(days=i),
                "HomeTeam": home,
                "AwayTeam": away,
                "FTHG": 1,
                "FTAG": 1,
                "HS": 10,
                "AS": 8,
                "HST": 4,
                "AST": 3,
                "HC": hc,
                "AC": ac,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def pred() -> CornerPredictor:
    return CornerPredictor(backend="poisson", min_prior_matches=1).fit(_matches())


def _teams(p: CornerPredictor) -> tuple[str, str, float, float]:
    home, away = p.teams[0], p.teams[1]
    r = p.expected_corners(home, away)
    return home, away, r["hc"], r["ac"]


def test_handicap_minus_half_is_home_win_by_one(pred: CornerPredictor) -> None:
    home, away, mu1, mu2 = _teams(pred)
    ah = pred.predict_handicap(home, away, handicap=-0.5)
    # Home -0.5 covers iff HC - AC >= 1.
    assert ah["home"] == pytest.approx(float(skellam.sf(0, mu1, mu2)))
    assert ah["push"] == 0.0
    assert ah["home"] + ah["away"] == pytest.approx(1.0)


def test_handicap_minus_one_has_push(pred: CornerPredictor) -> None:
    home, away, mu1, mu2 = _teams(pred)
    ah = pred.predict_handicap(home, away, handicap=-1.0)
    assert ah["home"] == pytest.approx(float(skellam.sf(1, mu1, mu2)))  # diff >= 2
    assert ah["push"] == pytest.approx(float(skellam.pmf(1, mu1, mu2)))  # diff == 1
    assert ah["away"] == pytest.approx(float(skellam.cdf(0, mu1, mu2)))  # diff <= 0
    assert ah["home"] + ah["away"] + ah["push"] == pytest.approx(1.0)
    assert ah["push"] > 0


def test_handicap_plus_one_and_half(pred: CornerPredictor) -> None:
    home, away, mu1, mu2 = _teams(pred)
    ah = pred.predict_handicap(home, away, handicap=1.5)
    # Home +1.5 covers iff HC - AC >= -1.
    assert ah["home"] == pytest.approx(float(skellam.sf(-2, mu1, mu2)))
    assert ah["push"] == 0.0
    # Sign sanity: giving goals (corners) away is harder than receiving them.
    minus = pred.predict_handicap(home, away, handicap=-1.5)
    assert minus["home"] < ah["home"]
    assert minus["home"] == pytest.approx(float(skellam.sf(1, mu1, mu2)))


def test_handicap_quarter_line_split(pred: CornerPredictor) -> None:
    home, away, _mu1, _mu2 = _teams(pred)
    q = pred.predict_handicap(home, away, handicap=-0.25)
    a = pred.predict_handicap(home, away, handicap=-0.5)
    b = pred.predict_handicap(home, away, handicap=0.0)
    assert q["home"] == pytest.approx(0.5 * (a["home"] + b["home"]))
    assert q["push"] == pytest.approx(0.5 * b["push"])
    assert q["home"] + q["away"] + q["push"] == pytest.approx(1.0)


def test_corner_ev_counts_push_refund(pred: CornerPredictor) -> None:
    home, away, _mu1, _mu2 = _teams(pred)
    ah = pred.predict_handicap(home, away, handicap=-1.0)
    legs = pred.predict_corner_ev(
        home,
        away,
        {"ah_home": 2.0, "ah_away": 2.0, "ah_line": -1.0},
        line=10.5,
        min_ev=-1.0,
    )
    by_sel = {leg["selection"]: leg for leg in legs}
    assert by_sel["AH Home -1"]["ev"] == pytest.approx(ah["home"] * 2.0 + ah["push"] - 1.0)
    assert by_sel["AH Away +1"]["ev"] == pytest.approx(ah["away"] * 2.0 + ah["push"] - 1.0)


def test_nb_dispersion_estimator() -> None:
    rng = np.random.default_rng(0)
    mu = np.full(5000, 10.0)
    y_nb = rng.negative_binomial(5, 5 / (5 + 10.0), size=5000)  # alpha = 0.2
    assert estimate_nb_dispersion(y_nb, mu) == pytest.approx(0.2, abs=0.05)
    y_pois = rng.poisson(10.0, size=5000)
    assert estimate_nb_dispersion(y_pois, mu) < 0.02
    assert estimate_nb_dispersion(np.full(10, 10.0), np.full(10, 10.0)) == 0.0


def test_totals_use_nb_when_overdispersed(pred: CornerPredictor) -> None:
    assert pred.total_alpha_ > 0
    home, away, _mu1, _mu2 = _teams(pred)
    ou = pred.predict_over_under(home, away, line=10.5)
    assert ou["over"] + ou["under"] == pytest.approx(1.0)
    assert ou["push"] == 0.0
    assert ou["dispersion"] == pytest.approx(pred.total_alpha_)

    legacy = CornerPredictor(
        backend="poisson", min_prior_matches=1, total_dist="poisson"
    ).fit(_matches())
    assert legacy.total_alpha_ == 0.0
    ou_p = legacy.predict_over_under(home, away, line=10.5)
    # Same mean, fatter tails under NB → tail mass differs.
    assert ou_p["expected_total"] == pytest.approx(ou["expected_total"])
    assert ou_p["over"] != pytest.approx(ou["over"], abs=1e-4)

    pmf = pred.predict_total_pmf(home, away)
    assert pmf.sum() == pytest.approx(1.0)


@pytest.mark.filterwarnings("ignore::Warning")  # GLM perfect fit on constant data
def test_totals_fall_back_to_poisson_when_underdispersed() -> None:
    p = CornerPredictor(backend="poisson", min_prior_matches=1).fit(
        _matches(overdispersed=False)
    )
    assert p.total_alpha_ == 0.0


def test_totals_whole_line_push(pred: CornerPredictor) -> None:
    home, away, _mu1, _mu2 = _teams(pred)
    ou = pred.predict_over_under(home, away, line=10.0)
    assert ou["push"] > 0
    assert ou["over"] + ou["under"] + ou["push"] == pytest.approx(1.0)


def test_lightgbm_backend_estimates_dispersion() -> None:
    pytest.importorskip("lightgbm")
    p = CornerPredictor(backend="lightgbm", min_prior_matches=1).fit(_matches(200))
    assert p.backend_used_ == "lightgbm"
    assert p.total_alpha_ >= 0.0
    home, away = p.teams[0], p.teams[1]
    ou = p.predict_over_under(home, away, line=9.5)
    assert ou["over"] + ou["under"] == pytest.approx(1.0)
