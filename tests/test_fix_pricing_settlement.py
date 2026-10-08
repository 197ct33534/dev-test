"""Exact settlement EV / fair odds / Kelly for AH & O/U; 1X2 unchanged; pick policy."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.config import (
    DEFAULT_KELLY_FRACTION,
    DEFAULT_MIN_EV,
    MAX_KELLY_FRACTION,
    MIN_EV_THRESHOLD,
    kelly_fraction_label,
)
from src.dixon_coles import (
    DixonColesModel,
    settlement_cover_probability,
    settlement_expected_value,
    settlement_fair_odds,
)
from src.recommender import (
    MatchPrediction,
    ValueBetRecommender,
    format_match_card,
    fractional_kelly,
)
from src.strategy import (
    capped_kelly_fraction,
    full_kelly_fraction,
    price_settlement,
    select_value_bets,
    settlement_full_kelly_fraction,
)

OVER_2 = {"win": 0.437, "half_win": 0.0, "push": 0.263, "half_lose": 0.0, "lose": 0.300}


def _brute_kelly(settlement: dict[str, float], odds: float) -> float:
    b = odds - 1.0
    r = {"win": b, "half_win": b / 2, "push": 0.0, "half_lose": -0.5, "lose": -1.0}
    grid = np.linspace(0.0, 0.999, 99901)
    g = sum(p * np.log1p(grid * r[k]) for k, p in settlement.items() if p > 0)
    return float(grid[int(np.argmax(g))])


def test_over_2_whole_line_ev_counts_push_refund() -> None:
    ev = settlement_expected_value(OVER_2, 2.0)
    assert ev == pytest.approx(0.137, abs=1e-9)
    # The old "effective cover" shortcut got the sign wrong.
    assert 0.437 * 2.0 - 1.0 == pytest.approx(-0.126)
    fair = settlement_fair_odds(OVER_2)
    assert fair == pytest.approx(1.0 + 0.300 / 0.437)
    assert settlement_expected_value(OVER_2, fair) == pytest.approx(0.0, abs=1e-12)
    p_star = settlement_cover_probability(OVER_2)
    assert 1.0 / p_star == pytest.approx(fair)


def test_whole_line_kelly_closed_form() -> None:
    b = 1.0
    pw, pl = 0.437 / 0.737, 0.300 / 0.737
    expected = (b * pw - pl) / b
    assert settlement_full_kelly_fraction(OVER_2, 2.0) == pytest.approx(expected)
    assert settlement_full_kelly_fraction(OVER_2, 2.0) == pytest.approx(
        _brute_kelly(OVER_2, 2.0), abs=2e-5
    )


def test_quarter_line_ev_fair_and_kelly() -> None:
    s = {"win": 0.40, "half_win": 0.12, "push": 0.0, "half_lose": 0.10, "lose": 0.38}
    o = 1.95
    b = o - 1.0
    ev = 0.40 * b + 0.12 * b / 2 - 0.10 * 0.5 - 0.38
    assert settlement_expected_value(s, o) == pytest.approx(ev)
    fair = settlement_fair_odds(s)
    assert fair == pytest.approx(1.0 + (0.38 + 0.05) / (0.40 + 0.06))
    assert settlement_expected_value(s, fair) == pytest.approx(0.0, abs=1e-12)
    k = settlement_full_kelly_fraction(s, 2.30)
    assert k > 0
    assert k == pytest.approx(_brute_kelly(s, 2.30), abs=2e-5)
    assert settlement_full_kelly_fraction(s, 1.80) == 0.0  # negative EV


def test_binary_settlement_matches_1x2_formulas() -> None:
    p, o = 0.55, 2.10
    s = {"win": p, "lose": 1 - p}
    assert settlement_expected_value(s, o) == pytest.approx(p * o - 1.0)
    assert settlement_fair_odds(s) == pytest.approx(1.0 / p)
    assert settlement_full_kelly_fraction(s, o) == pytest.approx(full_kelly_fraction(p, o))
    priced = price_settlement(s, o)
    assert priced["kelly"] == pytest.approx(capped_kelly_fraction(p, o))


def _toy_model() -> DixonColesModel:
    rng = np.random.default_rng(3)
    teams = ["A", "B", "C", "D"]
    rows = []
    day = pd.Timestamp("2024-01-01")
    for _ in range(6):
        for h in teams:
            for a in teams:
                if h != a:
                    rows.append(
                        {
                            "Date": day,
                            "HomeTeam": h,
                            "AwayTeam": a,
                            "FTHG": int(rng.poisson(1.5)),
                            "FTAG": int(rng.poisson(1.1)),
                        }
                    )
                    day += pd.Timedelta(days=1)
    return DixonColesModel(xi=0.0).fit(pd.DataFrame(rows))


def test_recommender_ou_and_ah_use_exact_settlement() -> None:
    model = _toy_model()
    rec = ValueBetRecommender(model, min_ev=0.0)
    bets = rec.evaluate_match(
        "A",
        "B",
        over_under={"line": 2.0, "over": 2.05, "under": 1.85},
        asian_handicap={"handicap": -0.75, "home": 2.0, "away": 1.9},
        only_value=False,
    )
    ou_pred = model.predict_over_under("A", "B", line=2.0)
    ah_pred = model.predict_asian_handicap("A", "B", handicap=-0.75)
    by_sel = {b.selection: b for b in bets}

    over = by_sel["Over 2"]
    s_over = {k: ou_pred[f"over_{k}"] for k in ("win", "half_win", "push", "half_lose", "lose")}
    assert s_over["push"] > 0.05
    assert over.ev == pytest.approx(settlement_expected_value(s_over, 2.05))
    assert over.fair_odds == pytest.approx(settlement_fair_odds(s_over))
    assert over.fair_odds == pytest.approx(1.0 / over.p_model)
    assert over.detail["ev_method"] == "settlement"
    assert ou_pred["over"] + ou_pred["under"] == pytest.approx(1.0)

    ah_home = by_sel["AH Home -0.75"]
    s_ah = {k: ah_pred[f"home_{k}"] for k in ("win", "half_win", "push", "half_lose", "lose")}
    assert s_ah["half_win"] > 0.0
    assert ah_home.ev == pytest.approx(settlement_expected_value(s_ah, 2.0))
    k_ref = min(
        settlement_full_kelly_fraction(s_ah, 2.0) * DEFAULT_KELLY_FRACTION, 0.01
    )
    assert ah_home.kelly_fraction == pytest.approx(k_ref)


def test_recommender_1x2_unchanged() -> None:
    model = _toy_model()
    rec = ValueBetRecommender(model, min_ev=0.0)
    bets = rec.evaluate_match("A", "B", odds_1x2={"H": 2.4, "D": 3.4, "A": 3.6}, only_value=False)
    probs = model.predict_match_probs("A", "B")
    for b in bets:
        key = {"Home": "H", "Draw": "D", "Away": "A"}[b.selection]
        assert b.p_model == pytest.approx(probs[key])
        assert b.ev == pytest.approx(probs[key] * b.bookmaker_odds - 1.0)
        assert b.fair_odds == pytest.approx(1.0 / probs[key])
        assert b.kelly_fraction == pytest.approx(
            capped_kelly_fraction(probs[key], b.bookmaker_odds)
        )


def test_select_value_bets_defaults_to_one_pick_per_match() -> None:
    bets = pd.DataFrame(
        [
            {"match_id": "m1", "market": "1X2", "selection": "Home", "ev": 0.09},
            {"match_id": "m1", "market": "AH", "selection": "AH Home -0.5", "ev": 0.08},
            {"match_id": "m2", "market": "OU", "selection": "Over 2.5", "ev": 0.07},
        ]
    )
    out = select_value_bets(bets, max_per_day=10, apply_sanity=False)
    assert list(out["match_id"]) == ["m1", "m2"]
    assert out.iloc[0]["selection"] == "Home"
    multi = select_value_bets(
        bets, max_per_day=10, apply_sanity=False, allow_multi_picks_per_match=True
    )
    assert len(multi) == 3


def test_stake_policy_constants_and_labels() -> None:
    assert DEFAULT_MIN_EV == pytest.approx(0.05)
    assert MIN_EV_THRESHOLD == DEFAULT_MIN_EV
    assert DEFAULT_KELLY_FRACTION == pytest.approx(0.10)
    assert MAX_KELLY_FRACTION >= DEFAULT_KELLY_FRACTION
    assert kelly_fraction_label() == "Kelly 10%"
    # Labels never advertise more than the enforced ceiling.
    assert kelly_fraction_label(0.25) == f"Kelly {MAX_KELLY_FRACTION:.0%}"
    # Legacy helper keeps its own explicit fraction semantics.
    assert fractional_kelly(0.6, 2.0, fraction=0.1) == pytest.approx(0.1 * 0.2 / 1.0)


def test_match_card_renders_configured_kelly_fraction() -> None:
    model = _toy_model()
    rec = ValueBetRecommender(model, min_ev=-1.0)
    bets = rec.evaluate_match("A", "B", odds_1x2={"H": 5.0, "D": 5.0, "A": 5.0}, only_value=False)
    pred = MatchPrediction(
        home_team="A",
        away_team="B",
        lambda_home=1.5,
        lambda_away=1.1,
        all_bets=bets,
        value_bets=bets,
        comparison_rows=[],
        model_summary={},
        min_ev=0.05,
        kelly_fraction=DEFAULT_KELLY_FRACTION,
    )
    card = format_match_card(pred)
    assert "Kelly (25%)" not in card
    assert f"Kelly ({DEFAULT_KELLY_FRACTION:.0%})" in card
    assert math.isfinite(bets[0].kelly_pct)
