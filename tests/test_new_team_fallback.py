"""New / unseen clubs must never KeyError; Kelly capped at 1% bankroll."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src.data_loader import normalize_team_name
from src.dixon_coles import DEFAULT_NEW_TEAM_ATTACK, DEFAULT_NEW_TEAM_DEFENCE, DixonColesModel
from src.models import NEW_TEAM_KELLY_CAP, apply_new_team_kelly_cap, detect_new_teams
from src.recommender import ValueBetRecommender


def _toy_matches() -> pd.DataFrame:
    """Minimal two-club history so Dixon–Coles can fit."""
    rows = []
    teams = ("Arsenal", "Chelsea")
    rng = np.random.default_rng(0)
    base = pd.Timestamp("2024-01-01")
    for i in range(24):
        h, a = teams[i % 2], teams[(i + 1) % 2]
        rows.append(
            {
                "Date": base + pd.Timedelta(days=i * 7),
                "HomeTeam": h,
                "AwayTeam": a,
                "FTHG": int(rng.integers(0, 4)),
                "FTAG": int(rng.integers(0, 4)),
                "FTR": "H",
            }
        )
    df = pd.DataFrame(rows)
    df["FTR"] = np.where(
        df["FTHG"] > df["FTAG"], "H", np.where(df["FTHG"] < df["FTAG"], "A", "D")
    )
    return df


def test_normalize_inter_alias() -> None:
    assert normalize_team_name("Internazionale") == "Inter"
    assert normalize_team_name("Inter Milan") == "Inter"


def test_leuven_maps_to_known_oud_heverlee() -> None:
    """Flashscore 'Leuven' == historical 'Oud-Heverlee Leuven'."""
    known = ["Ajax", "Oud-Heverlee Leuven", "Barcelona"]
    assert normalize_team_name("Leuven", known_teams=known) == "Oud-Heverlee Leuven"
    assert normalize_team_name("OH Leuven", known_teams=known) == "Oud-Heverlee Leuven"
    # Canonical form when no known list
    assert normalize_team_name("Oud-Heverlee Leuven") == "Leuven"


def test_dixon_coles_new_team_priors(caplog: pytest.LogCaptureFixture) -> None:
    model = DixonColesModel(xi=0.0).fit(_toy_matches())
    assert model.has_new_team("Inter", "Arsenal")
    with caplog.at_level(logging.WARNING):
        lam, mu = model.expected_goals("Inter", "Chelsea")
    assert lam > 0 and mu > 0
    assert "Detecting new team: Inter" in caplog.text
    assert model.attack["Inter"] == pytest.approx(DEFAULT_NEW_TEAM_ATTACK)
    assert model.defence["Inter"] == pytest.approx(DEFAULT_NEW_TEAM_DEFENCE)
    # Still flagged after prior injection (fitted_teams_ is stable).
    assert model.has_new_team("Inter", "Chelsea")
    probs = model.predict_match_probs("Inter", "Chelsea")
    assert abs(sum(probs.values()) - 1.0) < 1e-6


def test_kelly_cap_on_new_team_fixture() -> None:
    model = DixonColesModel(xi=0.0).fit(_toy_matches())
    rec = ValueBetRecommender(model, min_ev=0.0)
    bets = rec.evaluate_match(
        "Inter",
        "Arsenal",
        odds_1x2={"H": 3.5, "D": 3.5, "A": 2.0},
        only_value=False,
    )
    assert bets
    assert all(b.kelly_fraction <= NEW_TEAM_KELLY_CAP + 1e-12 for b in bets)
    assert all(b.detail.get("new_team_fallback") for b in bets)
    # Cap helper unit check
    assert apply_new_team_kelly_cap(0.05, has_new_team=True) == NEW_TEAM_KELLY_CAP
    assert apply_new_team_kelly_cap(0.005, has_new_team=True) == 0.005


def test_detect_new_teams_helper() -> None:
    model = DixonColesModel(xi=0.0).fit(_toy_matches())
    unknown = detect_new_teams(model, "FC Internazionale", "Arsenal")
    assert unknown == ["Inter"]
