r"""League-aware model fitting helpers (Dixon–Coles · LightGBM · Corners).

Centralises ensemble weight policy so small-sample leagues (e.g. UWCL with
``< 100`` historical matches) automatically prefer Dixon–Coles over LightGBM
to reduce overfitting risk.

Also exposes safe prediction helpers for **new / unseen clubs** (league
priors + Kelly stake cap).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import pandas as pd

from src.config import DEFAULT_W_ML, MAX_STAKE_PCT
from src.corner_model import CornerPredictor
from src.data_loader import normalize_league, normalize_team_name
from src.dixon_coles import DixonColesModel

logger = logging.getLogger(__name__)

# Below this match count, LightGBM weight is capped / scaled down.
MIN_MATCHES_FOR_FULL_ML: int = 100
# Absolute cap on ensemble w_ML when sample is thin (≤ default blend).
SMALL_SAMPLE_W_ML_CAP: float = min(0.15, float(DEFAULT_W_ML))
# Max Kelly fraction of bankroll when either side is a new team.
NEW_TEAM_KELLY_CAP: float = float(MAX_STAKE_PCT)


def ensemble_weight_for_sample(
    n_matches: int,
    base_w_ml: float,
    *,
    min_matches: int = MIN_MATCHES_FOR_FULL_ML,
    small_cap: float = SMALL_SAMPLE_W_ML_CAP,
) -> tuple[float, str | None]:
    """Return effective ``w_ML`` and an optional user-facing warning.

    If ``n_matches < min_matches``, scale ``base_w_ml`` by ``n/min`` and cap at
    ``small_cap`` so Dixon–Coles dominates the Ensemble.
    """
    n = max(0, int(n_matches))
    w = float(max(0.0, min(1.0, base_w_ml)))
    if n >= int(min_matches):
        return w, None
    scaled = w * (n / float(min_matches)) if min_matches > 0 else 0.0
    effective = min(scaled, float(small_cap), w)
    warn = (
        f"Lịch sử chỉ có {n} trận (< {min_matches}) → giảm w_ML "
        f"{w:.0%} → {effective:.0%} (ưu tiên Dixon–Coles, tránh overfitting)."
    )
    return effective, warn


def resolve_fixture_teams(
    home_team: str,
    away_team: str,
    known_teams: Sequence[str] | None = None,
) -> tuple[str, str]:
    """Apply :func:`normalize_team_name` to both sides of a fixture."""
    return (
        normalize_team_name(home_team, known_teams),
        normalize_team_name(away_team, known_teams),
    )


def detect_new_teams(
    model: DixonColesModel,
    home_team: str,
    away_team: str,
    *,
    known_teams: Sequence[str] | None = None,
) -> list[str]:
    """Normalize names then return clubs absent from the fitted DC model."""
    home, away = resolve_fixture_teams(home_team, away_team, known_teams)
    unknown = model.unknown_teams(home, away)
    for name in unknown:
        logger.warning(
            "⚠️ Detecting new team: %s. Using default league priors.",
            name,
        )
    return unknown


def apply_new_team_kelly_cap(
    kelly_fraction: float,
    *,
    has_new_team: bool,
    cap: float = NEW_TEAM_KELLY_CAP,
) -> float:
    """Clamp Kelly stake to ``cap`` (default 1% bankroll) for new-team fixtures."""
    k = float(max(0.0, kelly_fraction))
    if has_new_team:
        return float(min(k, float(cap)))
    return k


def safe_match_probs(
    model: DixonColesModel,
    home_team: str,
    away_team: str,
    *,
    known_teams: Sequence[str] | None = None,
) -> tuple[dict[str, float], list[str]]:
    """1X2 probs that never raise ``KeyError`` for unseen clubs.

    Returns
    -------
    probs, new_teams
    """
    home, away = resolve_fixture_teams(home_team, away_team, known_teams)
    new_teams = model.unknown_teams(home, away)
    for name in new_teams:
        logger.warning(
            "⚠️ Detecting new team: %s. Using default league priors.",
            name,
        )
    probs = model.predict_match_probs(home, away)
    return probs, new_teams


@dataclass
class LeagueModels:
    """Fitted models + effective ensemble weight for one league."""

    league: str
    dixon_coles: DixonColesModel
    corner: CornerPredictor | None
    ml: Any | None = None
    w_ml: float = DEFAULT_W_ML
    n_train: int = 0
    warnings: list[str] = field(default_factory=list)
    ml_error: str | None = None

    @property
    def use_ensemble(self) -> bool:
        return self.ml is not None and self.w_ml > 0


def fit_league_models(
    matches: pd.DataFrame,
    *,
    league: str = "EPL",
    xi: float = 0.0018,
    use_ml: bool = True,
    w_ml: float = DEFAULT_W_ML,
    fit_corners: bool = True,
) -> LeagueModels:
    """Fit Dixon–Coles (+ optional LightGBM / Corners) on ``matches``.

    Automatically shrinks Ensemble ``w_ML`` when the training sample is thin
    (especially relevant for ``UWCL``).
    """
    code = normalize_league(league)
    n = int(len(matches))
    w_eff, warn = ensemble_weight_for_sample(n, w_ml)
    warnings: list[str] = []
    if warn:
        warnings.append(warn)
    if code == "UWCL" and n < MIN_MATCHES_FOR_FULL_ML:
        if not warn:
            warnings.append(
                f"UWCL sample size={n} < {MIN_MATCHES_FOR_FULL_ML}: "
                f"using w_ML={w_eff:.0%}."
            )

    dc = DixonColesModel(xi=float(xi)).fit(matches)

    ml_model = None
    ml_error = None
    if use_ml and w_eff > 0:
        try:
            from src.ml_model import EPLMachineLearningModel

            ml_model = EPLMachineLearningModel().fit(matches)
        except Exception as exc:  # noqa: BLE001
            ml_error = str(exc)
            warnings.append(f"LightGBM skipped: {exc}")
            w_eff = 0.0

    corner: CornerPredictor | None = None
    if fit_corners:
        try:
            backend = "poisson" if code == "UWCL" else "auto"
            corner = CornerPredictor(backend=backend).fit(matches)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"CornerPredictor skipped: {exc}")
            corner = None

    return LeagueModels(
        league=code,
        dixon_coles=dc,
        corner=corner,
        ml=ml_model,
        w_ml=float(w_eff),
        n_train=n,
        warnings=warnings,
        ml_error=ml_error,
    )
