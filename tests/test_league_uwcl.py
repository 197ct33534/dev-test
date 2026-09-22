"""Tests for multi-league support (EPL / UWCL) and ensemble weight guard."""

from __future__ import annotations

import pandas as pd

from src.data_loader import (
    LEAGUE_CONFIG,
    league_db_path,
    normalize_league,
    _parse_fotmob_score,
    _normalize_uwcl_team,
)
from src.models import ensemble_weight_for_sample


def test_normalize_league_aliases() -> None:
    assert normalize_league("epl") == "EPL"
    assert normalize_league("UWCL") == "UWCL"
    assert normalize_league("Women's Champions League") == "UWCL"


def test_league_db_paths_differ() -> None:
    epl = league_db_path("EPL")
    uwcl = league_db_path("UWCL")
    assert epl.name == "epl_matches.db"
    assert uwcl.name == "uwcl_matches.db"
    assert epl != uwcl
    assert "UWCL" in LEAGUE_CONFIG


def test_parse_fotmob_score() -> None:
    assert _parse_fotmob_score("3 - 0") == (3, 0)
    assert _parse_fotmob_score("1:2") == (1, 2)
    assert _parse_fotmob_score(None) is None


def test_normalize_uwcl_team() -> None:
    assert _normalize_uwcl_team("OL Lyonnes (W)") == "OL Lyonnes"
    assert _normalize_uwcl_team("BK Häcken") == "BK Hacken"
    assert _normalize_uwcl_team("Hacken") == "BK Hacken"
    assert _normalize_uwcl_team("Bayern München (W)") == "Bayern Munich"


def test_fold_team_key_hacken() -> None:
    from src.data_loader import _fold_team_key

    assert _fold_team_key("BK Häcken") == _fold_team_key("Hacken") == "hacken"
    # Alias layer equalises München / Munich before fold in the pipeline
    assert _normalize_uwcl_team("Bayern München") == _normalize_uwcl_team("Bayern Munich")


def test_ensemble_weight_shrinks_on_small_sample() -> None:
    w_full, warn_full = ensemble_weight_for_sample(500, 0.4)
    assert w_full == 0.4
    assert warn_full is None

    w_small, warn_small = ensemble_weight_for_sample(50, 0.4)
    assert w_small < 0.4
    assert w_small <= 0.15
    assert warn_small is not None
    assert "50" in warn_small
