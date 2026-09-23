"""Sanity filters: drop inflated EV and overconfident 1X2 longshots."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import DEFAULT_MIN_EV, LONGSHOT_ODDS_MIN, LONGSHOT_P_MAX, MAX_EV
from src.strategy import (
    apply_sanity_filters,
    is_sane_value_bet,
    select_value_bets,
)


def test_config_sanity_defaults() -> None:
    assert DEFAULT_MIN_EV == pytest.approx(0.05)
    assert MAX_EV == pytest.approx(0.50)
    assert LONGSHOT_ODDS_MIN == pytest.approx(15.0)
    assert LONGSHOT_P_MAX == pytest.approx(0.10)


def test_drops_ev_above_50_pct() -> None:
    row = {
        "market": "1X2",
        "selection": "Away",
        "ev": 20.0,  # +2000%
        "ev_pct": 2000.0,
        "bookmaker_odds": 51.0,
        "p_model": 0.41,
    }
    assert is_sane_value_bet(row) is False


def test_keeps_realistic_ev_band() -> None:
    row = {
        "market": "1X2",
        "selection": "Home",
        "ev": 0.12,
        "ev_pct": 12.0,
        "bookmaker_odds": 2.10,
        "p_model": 0.53,
    }
    assert is_sane_value_bet(row) is True
    # Still allow up to the hard cap (50%)
    edge = {
        "market": "OU",
        "selection": "Over 2.5",
        "ev": 0.50,
        "bookmaker_odds": 1.90,
        "p_model": 0.79,
    }
    assert is_sane_value_bet(edge) is True
    over = dict(edge)
    over["ev"] = 0.501
    assert is_sane_value_bet(over) is False


def test_drops_1x2_longshot_with_overconfident_p() -> None:
    """odds > 15 and model p > 10% → invalid (Servette @ 51.00 style)."""
    row = {
        "market": "1X2",
        "selection": "Away",
        "ev": 0.25,  # would pass EV cap alone
        "bookmaker_odds": 51.0,
        "p_model": 0.15,
    }
    assert is_sane_value_bet(row) is False
    # Same odds but humble p → keep (EV may still be filtered elsewhere)
    humble = dict(row)
    humble["p_model"] = 0.05
    humble["ev"] = 0.05 * 51.0 - 1.0  # 1.55 → will fail EV>50%
    # Use modest EV so only longshot rule is under test
    humble["ev"] = 0.20
    assert is_sane_value_bet(humble) is True


def test_longshot_rule_only_applies_to_1x2() -> None:
    row = {
        "market": "OU",
        "selection": "Over 2.5",
        "ev": 0.20,
        "bookmaker_odds": 20.0,
        "p_model": 0.20,
    }
    assert is_sane_value_bet(row) is True


def test_select_value_bets_applies_sanity_before_ranking() -> None:
    bets = pd.DataFrame(
        [
            {
                "match_id": "srv_avl",
                "home_team": "Servette",
                "away_team": "Austria Vienna",
                "market": "1X2",
                "selection": "Away",
                "ev": 20.5,
                "ev_pct": 2050.0,
                "bookmaker_odds": 51.0,
                "p_model": 0.42,
            },
            {
                "match_id": "ars_che",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "market": "1X2",
                "selection": "Home",
                "ev": 0.08,
                "ev_pct": 8.0,
                "bookmaker_odds": 1.85,
                "p_model": 0.58,
            },
            {
                "match_id": "liv_eve",
                "home_team": "Liverpool",
                "away_team": "Everton",
                "market": "1X2",
                "selection": "Away",
                "ev": 0.22,
                "ev_pct": 22.0,
                "bookmaker_odds": 51.0,
                "p_model": 0.12,  # longshot + overconfident p
            },
        ]
    )
    filtered = select_value_bets(bets, max_per_day=20, already_today=0)
    assert len(filtered) == 1
    assert filtered.iloc[0]["home_team"] == "Arsenal"
    # Opt-out still available for tests / diagnostics
    raw = select_value_bets(bets, max_per_day=20, apply_sanity=False)
    assert len(raw) == 3


def test_apply_sanity_filters_frame() -> None:
    bets = pd.DataFrame(
        [
            {"market": "1X2", "ev": 0.60, "bookmaker_odds": 2.0, "p_model": 0.8},
            {"market": "1X2", "ev": 0.10, "bookmaker_odds": 2.0, "p_model": 0.55},
        ]
    )
    out = apply_sanity_filters(bets)
    assert len(out) == 1
    assert out.iloc[0]["ev"] == pytest.approx(0.10)
