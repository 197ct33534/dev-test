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
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from src.config import DEFAULT_W_ML, MAX_STAKE_PCT, MIN_TEAM_MATCHES
from src.corner_model import CornerPredictor
from src.data_loader import normalize_league, normalize_team_name
from src.dixon_coles import (
    DixonColesModel,
    score_probability,
    WEAK_TIER_NEW_TEAM_ATTACK,
    WEAK_TIER_NEW_TEAM_DEFENCE,
)

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
    """Normalize names then return clubs absent / thin in the fitted DC model.

    Thin history (< ``min_team_matches``, default 5) is treated like new so
    callers can apply Kelly caps and UI badges consistently.
    """
    home, away = resolve_fixture_teams(home_team, away_team, known_teams)
    unknown = model.unknown_teams(home, away)
    for name in unknown:
        n = model.team_match_count(name)
        logger.warning(
            "⚠️ Detecting new/thin team: %s (n=%d). Using weak-tier priors.",
            name,
            n,
        )
    return unknown


def teams_needing_data_badge(
    model: DixonColesModel | None,
    home_team: str,
    away_team: str,
    *,
    known_teams: Sequence[str] | None = None,
    min_matches: int | None = None,
    upcoming_date: Any = None,
    db_path: Path | str | None = None,
    comp_id: str | None = None,
) -> list[str]:
    """Clubs lacking finished history in global DB (UI ``[⚠️ Thiếu Data Đội]``).

    Criterion: :func:`src.global_db.has_team_history` — **not** Dixon–Coles thin
    priors or missing ``flashscore_hash``. ``min_matches`` / ``upcoming_date``
    are accepted for call-site compat but ignored (any history → hide badge).
    """
    from src.global_db import has_team_history

    known = known_teams
    if known is None and model is not None:
        known = getattr(model, "teams", None)
    home, away = resolve_fixture_teams(home_team, away_team, known)
    _ = min_matches, upcoming_date  # compat; badge uses any-history only
    out: list[str] = []
    for name in (home, away):
        if not name:
            continue
        if not has_team_history(name, db_path=db_path, comp_id=comp_id):
            if name not in out:
                out.append(name)
    return out


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
            "⚠️ Detecting new/thin team: %s (n=%d). Using weak-tier priors.",
            name,
            model.team_match_count(name),
        )
    probs = model.predict_match_probs(home, away)
    return probs, new_teams


def expected_goals_multi_comp(
    model: DixonColesModel,
    home_team: str,
    away_team: str,
    *,
    home_league_weight: float = 1.0,
    away_league_weight: float = 1.0,
    w_ref: float = 1.0,
    known_teams: Sequence[str] | None = None,
) -> tuple[float, float]:
    """Optional multi-comp λ/μ with ``league_weight`` scaling.

    Thin adapter around
    :meth:`DixonColesModel.expected_goals_with_league_weights`.
    Default weights (1.0) preserve the single-league Streamlit path.
    """
    home, away = resolve_fixture_teams(home_team, away_team, known_teams)
    return model.expected_goals_with_league_weights(
        home,
        away,
        home_weight=float(home_league_weight),
        away_weight=float(away_league_weight),
        w_ref=float(w_ref),
    )


class LeagueWeightDixonColesProxy:
    """Non-mutating wrapper: scale λ/μ by per-team competition weights.

    Does **not** alter the cached inner :class:`DixonColesModel`. Prediction
    methods that depend on ``expected_goals`` are reimplemented so Ensemble /
    recommender paths pick up the offset. Other attributes delegate to
    ``inner``.
    """

    def __init__(
        self,
        inner: DixonColesModel,
        weight_fn: Callable[[str], float],
        *,
        w_ref: float = 1.0,
    ) -> None:
        self._inner = inner
        self._weight_fn = weight_fn
        self._w_ref = float(w_ref)

    @property
    def inner(self) -> DixonColesModel:
        return self._inner

    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        from src.global_db import apply_league_weight_to_rates

        lam, mu = self._inner.expected_goals(home_team, away_team)
        return apply_league_weight_to_rates(
            lam,
            mu,
            float(self._weight_fn(home_team)),
            float(self._weight_fn(away_team)),
            w_ref=self._w_ref,
        )

    def expected_goals_with_league_weights(
        self,
        home_team: str,
        away_team: str,
        *,
        home_weight: float = 1.0,
        away_weight: float = 1.0,
        w_ref: float = 1.0,
    ) -> tuple[float, float]:
        from src.global_db import apply_league_weight_to_rates

        lam, mu = self._inner.expected_goals(home_team, away_team)
        return apply_league_weight_to_rates(
            lam, mu, home_weight, away_weight, w_ref=w_ref
        )

    def predict_score_matrix(
        self,
        home_team: str,
        away_team: str,
        max_goals: int | None = None,
    ) -> np.ndarray:
        lam, mu = self.expected_goals(home_team, away_team)
        g = self._inner.max_goals if max_goals is None else max_goals
        mat = np.zeros((g + 1, g + 1), dtype=float)
        rho = float(self._inner.rho)
        for x in range(g + 1):
            for y in range(g + 1):
                mat[x, y] = score_probability(x, y, lam, mu, rho)
        total = mat.sum()
        if total <= 0:
            raise RuntimeError("Score matrix has non-positive mass; check parameters.")
        return mat / total

    def predict_match_probs(
        self,
        home_team: str,
        away_team: str,
        max_goals: int | None = None,
    ) -> dict[str, float]:
        mat = self.predict_score_matrix(home_team, away_team, max_goals=max_goals)
        return {
            "H": float(np.tril(mat, k=-1).sum()),
            "D": float(np.trace(mat)),
            "A": float(np.triu(mat, k=1).sum()),
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def maybe_wrap_with_league_weights(
    model: DixonColesModel,
    league: str,
    *,
    db_path: Path | str | None = None,
    w_ref: float = 1.0,
) -> DixonColesModel | LeagueWeightDixonColesProxy:
    """Wrap ``model`` when ``global_matches.db`` can supply team weights.

    EPL baseline (all weights 1.0) skips wrapping. Missing / empty global DB
    returns ``model`` unchanged so Streamlit never breaks.
    """
    code = normalize_league(league)
    try:
        from src.global_db import (
            GLOBAL_DB_PATH,
            connect_global_db,
            get_competition_weight,
            global_db_has_matches,
            resolve_team_id,
            team_weight_for_match,
        )
    except Exception:  # noqa: BLE001
        return model

    path = Path(db_path) if db_path is not None else Path(GLOBAL_DB_PATH)
    if not global_db_has_matches(path):
        return model
    # Domestic EPL baseline weights are 1.0 — skip proxy overhead.
    if code == "EPL":
        return model

    cache: dict[str, float] = {}

    def _weight(team: str) -> float:
        name = str(team or "").strip()
        if not name:
            return 1.0
        if name in cache:
            return cache[name]
        try:
            with connect_global_db(path, init=False) as conn:
                tid = resolve_team_id(
                    conn, name, create=False, comp_id=code
                )
                if tid is None:
                    w = get_competition_weight(conn, code, default=1.0)
                else:
                    w = team_weight_for_match(conn, tid, code, default=1.0)
        except Exception:  # noqa: BLE001
            w = 1.0
        cache[name] = float(w)
        return float(w)

    return LeagueWeightDixonColesProxy(model, _weight, w_ref=float(w_ref))


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
    # UWCL / small-sample leagues: force weak-tier priors for unknown clubs.
    use_weak = code == "UWCL" or n < MIN_MATCHES_FOR_FULL_ML
    if code == "UWCL" and n < MIN_MATCHES_FOR_FULL_ML:
        if not warn:
            warnings.append(
                f"UWCL sample size={n} < {MIN_MATCHES_FOR_FULL_ML}: "
                f"using w_ML={w_eff:.0%}."
            )
    if use_weak:
        warnings.append(
            f"Weak-tier DC priors (α={WEAK_TIER_NEW_TEAM_ATTACK}, "
            f"δ={WEAK_TIER_NEW_TEAM_DEFENCE}) for new/thin clubs "
            f"(< {MIN_TEAM_MATCHES} matches)."
        )

    dc = DixonColesModel(
        xi=float(xi),
        use_weak_tier_priors=use_weak,
        new_team_attack=WEAK_TIER_NEW_TEAM_ATTACK if use_weak else None,
        new_team_defence=WEAK_TIER_NEW_TEAM_DEFENCE if use_weak else None,
        min_team_matches=MIN_TEAM_MATCHES,
    ).fit(matches)

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
