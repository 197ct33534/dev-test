"""League-aware model fitting helpers + vig removal.

Historically this package was the single module ``src/models.py``. Content now
lives in :mod:`src.models.league_fit` so ``vig_removal`` can share the package
namespace without breaking ``from src.models import ...`` call sites.
"""

from __future__ import annotations

from src.models.league_fit import (
    NEW_TEAM_KELLY_CAP,
    MIN_MATCHES_FOR_FULL_ML,
    SMALL_SAMPLE_W_ML_CAP,
    LeagueModels,
    LeagueWeightDixonColesProxy,
    apply_new_team_kelly_cap,
    detect_new_teams,
    ensemble_weight_for_sample,
    expected_goals_multi_comp,
    fit_league_models,
    maybe_wrap_with_league_weights,
    resolve_fixture_teams,
    safe_match_probs,
    teams_needing_data_badge,
)
from src.models.vig_removal import VigRemovalEngine

__all__ = [
    "NEW_TEAM_KELLY_CAP",
    "MIN_MATCHES_FOR_FULL_ML",
    "SMALL_SAMPLE_W_ML_CAP",
    "LeagueModels",
    "LeagueWeightDixonColesProxy",
    "VigRemovalEngine",
    "apply_new_team_kelly_cap",
    "detect_new_teams",
    "ensemble_weight_for_sample",
    "expected_goals_multi_comp",
    "fit_league_models",
    "maybe_wrap_with_league_weights",
    "resolve_fixture_teams",
    "safe_match_probs",
    "teams_needing_data_badge",
]
