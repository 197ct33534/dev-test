r"""Dixon-Coles Poisson model for Premier League match outcomes.

Models home goals X and away goals Y as (dependent) Poisson random
variables with means lambda and mu:

    lambda_ij = exp(alpha_i - delta_j + gamma)
    mu_ij     = exp(alpha_j - delta_i)

where alpha is attack strength, delta is defence strength (higher = better),
and gamma is home advantage.

Low-score dependence is captured by the Dixon-Coles correction
tau(x, y; lambda, mu, rho):

    (0,0) -> 1 - lambda * mu * rho
    (0,1) -> 1 + lambda * rho
    (1,0) -> 1 + mu * rho
    (1,1) -> 1 - rho
    else  -> 1

Optional exponential time weights w = exp(-xi * t) down-weight older
matches (t = days before the latest kick-off in the training set).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import logging

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize
from scipy.stats import poisson

# Default time-decay rate (per day). ~0.0018 ≈ half-life of ~1 year.
DEFAULT_XI = 0.0018
DEFAULT_MAX_GOALS = 10
# Legacy "average" priors (overconfident for newcomers — prefer weak tier).
DEFAULT_NEW_TEAM_ATTACK = 1.0
DEFAULT_NEW_TEAM_DEFENCE = 1.0
# Clubs with fewer historical matches use weak-tier priors (keep in sync with src.config).
MIN_TEAM_MATCHES = 5
# Weak-tier priors for new / thin-sample clubs (< MIN_TEAM_MATCHES matches).
# Sign convention: higher δ = *better* defence (λ = exp(α − δ_opp + γ)).
# Underdog newcomers need weak attack + *poor* defence → lower δ (not 1.6).
WEAK_TIER_NEW_TEAM_ATTACK = 0.4
WEAK_TIER_NEW_TEAM_DEFENCE = 0.4

logger = logging.getLogger(__name__)


def tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles low-score adjustment tau(x, y; lambda, mu, rho).

    Parameters
    ----------
    x, y:
        Home and away goals.
    lam, mu:
        Expected goals lambda (home) and mu (away).
    rho:
        Correlation parameter rho (typically slightly negative).
    """
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    if x == 0 and y == 1:
        return 1.0 + lam * rho
    if x == 1 and y == 0:
        return 1.0 + mu * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def score_probability(
    x: int,
    y: int,
    lam: float,
    mu: float,
    rho: float,
) -> float:
    """P(X=x, Y=y) = tau * Poisson(x; lambda) * Poisson(y; mu)."""
    return float(tau(x, y, lam, mu, rho) * poisson.pmf(x, lam) * poisson.pmf(y, mu))


# ---------------------------------------------------------------------------
# Asian / totals settlement helpers (whole, half, quarter lines)
# ---------------------------------------------------------------------------

Settlement = str  # "win" | "half_win" | "push" | "half_lose" | "lose"


def _is_quarter_line(line: float) -> bool:
    """True when ``line`` sits on a .25 / .75 grid point (split stake)."""
    frac = abs(line) % 1.0
    return abs(frac - 0.25) < 1e-9 or abs(frac - 0.75) < 1e-9


def split_handicap_line(line: float) -> tuple[float, float]:
    """Split a quarter line into its two neighbouring half/whole lines.

    Examples: 2.25 → (2.0, 2.5), -0.75 → (-1.0, -0.5). Non-quarter lines
    return ``(line, line)``.
    """
    if not _is_quarter_line(line):
        return (float(line), float(line))
    lower = np.floor(line * 2.0) / 2.0
    upper = np.ceil(line * 2.0) / 2.0
    return (float(lower), float(upper))


def _settle_atomic_over(total: int, line: float) -> Settlement:
    """Settle Over on a single whole/half line (no quarter split)."""
    if total > line:
        return "win"
    if total < line:
        return "lose"
    return "push"


def _settle_atomic_under(total: int, line: float) -> Settlement:
    if total < line:
        return "win"
    if total > line:
        return "lose"
    return "push"


def _settle_atomic_ah_home(home_goals: int, away_goals: int, handicap: float) -> Settlement:
    """Settle home Asian Handicap: margin = X - Y + handicap."""
    margin = home_goals - away_goals + handicap
    if margin > 0:
        return "win"
    if margin < 0:
        return "lose"
    return "push"


_SETTLEMENT_RANK = {
    "win": 2,
    "half_win": 1,
    "push": 0,
    "half_lose": -1,
    "lose": -2,
}
_RANK_TO_SETTLEMENT = {v: k for k, v in _SETTLEMENT_RANK.items()}


def _combine_settlements(parts: list[Settlement]) -> Settlement:
    """Combine one or two atomic settlements into a single outcome label."""
    if len(parts) == 1:
        return parts[0]
    a, b = parts[0], parts[1]
    avg = (_SETTLEMENT_RANK[a] + _SETTLEMENT_RANK[b]) / 2.0
    nearest = min(_RANK_TO_SETTLEMENT.keys(), key=lambda r: abs(r - avg))
    return _RANK_TO_SETTLEMENT[nearest]


def _settle_over(total: int, line: float) -> list[Settlement]:
    lo, hi = split_handicap_line(line)
    if lo == hi:
        return [_settle_atomic_over(total, lo)]
    return [_settle_atomic_over(total, lo), _settle_atomic_over(total, hi)]


def _settle_under(total: int, line: float) -> list[Settlement]:
    lo, hi = split_handicap_line(line)
    if lo == hi:
        return [_settle_atomic_under(total, lo)]
    return [_settle_atomic_under(total, lo), _settle_atomic_under(total, hi)]


def _settle_ah_home(home_goals: int, away_goals: int, handicap: float) -> list[Settlement]:
    lo, hi = split_handicap_line(handicap)
    if lo == hi:
        return [_settle_atomic_ah_home(home_goals, away_goals, lo)]
    return [
        _settle_atomic_ah_home(home_goals, away_goals, lo),
        _settle_atomic_ah_home(home_goals, away_goals, hi),
    ]


def _mirror_settlement(outcome: Settlement) -> Settlement:
    """Map home-side AH settlement to the away side."""
    return {
        "win": "lose",
        "half_win": "half_lose",
        "push": "push",
        "half_lose": "half_win",
        "lose": "win",
    }[outcome]


def _empty_settlement() -> dict[str, float]:
    return {"win": 0.0, "half_win": 0.0, "push": 0.0, "half_lose": 0.0, "lose": 0.0}


def _effective_cover_prob(settlement: dict[str, float]) -> float:
    """Effective win probability for EV = P × odds − 1.

    Counts a full win as 1 and a half-win as 0.5 (half the stake struck at
    the offered odds). Push / half-lose / lose do not add cover mass; the
    recommender applies the exact settlement EV when sizing bets.
    """
    return float(settlement.get("win", 0.0) + 0.5 * settlement.get("half_win", 0.0))


def _tau_vectorized(
    x: np.ndarray,
    y: np.ndarray,
    lam: np.ndarray,
    mu: np.ndarray,
    rho: float,
) -> np.ndarray:
    """Vectorised tau(x, y; lambda, mu, rho) for match arrays."""
    t = np.ones(len(x), dtype=float)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    t[m00] = 1.0 - lam[m00] * mu[m00] * rho
    t[m01] = 1.0 + lam[m01] * rho
    t[m10] = 1.0 + mu[m10] * rho
    t[m11] = 1.0 - rho
    return t


def _dc_loglike_matches(
    x: np.ndarray,
    y: np.ndarray,
    lam: np.ndarray,
    mu: np.ndarray,
    rho: float,
    weights: np.ndarray,
) -> float:
    """Weighted sum of per-match log-likelihoods (vectorised)."""
    if np.any(lam <= 0) or np.any(mu <= 0):
        return -1e6 * len(x)

    t = _tau_vectorized(x, y, lam, mu, rho)
    if np.any(t <= 0):
        return -1e6 * len(x)

    ll = np.log(t) + poisson.logpmf(x, lam) + poisson.logpmf(y, mu)
    return float(np.dot(weights, ll))


@dataclass
class DixonColesModel:
    """Maximum-likelihood Dixon-Coles model for 1X2 / scoreline prediction.

    Parameters
    ----------
    xi:
        Time-decay rate xi in w = exp(-xi * t). Set 0 to disable.
    max_goals:
        Truncation for the predicted score matrix (0 .. max_goals inclusive).
    use_weak_tier_priors:
        When True (UWCL / small-sample leagues), unknown clubs use weak-tier
        α/δ. Thin clubs (< ``min_team_matches`` history) always use weak tier.
    new_team_attack / new_team_defence:
        Explicit prior overrides; when None, derived from ``use_weak_tier_priors``.
    min_team_matches:
        Clubs with fewer training appearances are treated as thin / new and
        get weak-tier priors instead of noisy MLE strengths.
    """

    xi: float = DEFAULT_XI
    max_goals: int = DEFAULT_MAX_GOALS
    use_weak_tier_priors: bool = False
    new_team_attack: float | None = None
    new_team_defence: float | None = None
    min_team_matches: int = MIN_TEAM_MATCHES

    teams: list[str] = field(default_factory=list, init=False)
    # Clubs with enough history at fit time (excludes thin / prior-only).
    fitted_teams_: set[str] = field(default_factory=set, init=False, repr=False)
    team_match_counts_: dict[str, int] = field(
        default_factory=dict, init=False, repr=False
    )
    thin_teams_: set[str] = field(default_factory=set, init=False, repr=False)
    attack: dict[str, float] = field(default_factory=dict, init=False)
    defence: dict[str, float] = field(default_factory=dict, init=False)
    home_advantage: float = field(default=0.0, init=False)  # γ
    rho: float = field(default=0.0, init=False)  # ρ
    fitted_: bool = field(default=False, init=False)
    optimize_result_: OptimizeResult | None = field(default=None, init=False)

    # ------------------------------------------------------------------
    # Priors / thin-sample
    # ------------------------------------------------------------------

    def _prior_attack(self) -> float:
        if self.new_team_attack is not None:
            return float(self.new_team_attack)
        if self.use_weak_tier_priors:
            return float(WEAK_TIER_NEW_TEAM_ATTACK)
        # Unknown = 0 matches < min → prefer weak tier to avoid +2000% EV.
        return float(WEAK_TIER_NEW_TEAM_ATTACK)

    def _prior_defence(self) -> float:
        """Defence prior. Higher δ = better defence; weak newcomers get low δ."""
        if self.new_team_defence is not None:
            return float(self.new_team_defence)
        if self.use_weak_tier_priors:
            return float(WEAK_TIER_NEW_TEAM_DEFENCE)
        return float(WEAK_TIER_NEW_TEAM_DEFENCE)

    def team_match_count(self, team: str) -> int:
        """Historical match count from the last ``fit`` (0 if unseen)."""
        return int(self.team_match_counts_.get(str(team).strip(), 0))

    def is_thin_team(self, team: str) -> bool:
        """True when the club has fewer than ``min_team_matches`` history rows."""
        return self.team_match_count(team) < int(self.min_team_matches)

    # ------------------------------------------------------------------
    # Expected goals
    # ------------------------------------------------------------------

    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Return (lambda, mu) for ``home_team`` vs ``away_team``.

        lambda = exp(alpha_home - delta_away + gamma)
        mu     = exp(alpha_away - delta_home)

        Unknown / thin clubs use weak-tier attack/defence priors instead of raising.
        """
        self._require_fitted()
        a_home, d_home = self._strengths_for(home_team)
        a_away, d_away = self._strengths_for(away_team)
        lam = float(np.exp(a_home - d_away + self.home_advantage))
        mu = float(np.exp(a_away - d_home))
        return lam, mu

    def expected_goals_with_league_weights(
        self,
        home_team: str,
        away_team: str,
        *,
        home_weight: float = 1.0,
        away_weight: float = 1.0,
        w_ref: float = 1.0,
    ) -> tuple[float, float]:
        """Expected goals scaled by competition ``league_weight``.

        Optional multi-comp path (European cups). Single-league Streamlit
        fits leave weights at 1.0 → identical to :meth:`expected_goals`.

        Math
        ----
        ``λ' = λ · (w_home / w_ref)``, ``μ' = μ · (w_away / w_ref)``
        with ``w_ref = 1.0`` (EPL baseline). See ``src.global_db``.
        """
        from src.global_db import apply_league_weight_to_rates

        lam, mu = self.expected_goals(home_team, away_team)
        return apply_league_weight_to_rates(
            lam, mu, home_weight, away_weight, w_ref=w_ref
        )

    def unknown_teams(self, *teams: str) -> list[str]:
        """Return club names absent from the well-supported training set.

        Includes thin-history clubs (< ``min_team_matches``) that were demoted
        from ``fitted_teams_`` after fit.
        """
        self._require_fitted()
        known = self.fitted_teams_ or set()
        out: list[str] = []
        for t in teams:
            name = str(t).strip()
            if name and name not in known and name not in out:
                out.append(name)
        return out

    def has_new_team(self, home_team: str, away_team: str) -> bool:
        """True when either side was absent / thin in the training set."""
        return bool(self.unknown_teams(home_team, away_team))

    def _strengths_for(self, team: str) -> tuple[float, float]:
        """Return ``(attack, defence)``, injecting weak-tier priors for newcomers."""
        name = str(team).strip()
        known = self.fitted_teams_ or set()
        if name in known and name in self.attack and name in self.defence:
            return float(self.attack[name]), float(self.defence[name])
        # Already injected priors for this process — reuse without re-logging.
        if name in self.attack and name in self.defence and name not in known:
            return float(self.attack[name]), float(self.defence[name])
        atk = self._prior_attack()
        deff = self._prior_defence()
        n_hist = self.team_match_count(name)
        logger.warning(
            "⚠️ Detecting new/thin team: %s (n=%d). Using weak-tier priors "
            "(α=%.2f, δ=%.2f).",
            name,
            n_hist,
            atk,
            deff,
        )
        # Cache so subsequent calls in the same process stay consistent.
        # Do NOT add to fitted_teams_ — Kelly / unknown checks stay valid.
        self.attack[name] = float(atk)
        self.defence[name] = float(deff)
        return float(atk), float(deff)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_score_matrix(
        self,
        home_team: str,
        away_team: str,
        max_goals: int | None = None,
    ) -> np.ndarray:
        """Return P(X=x, Y=y) matrix of shape (max_goals+1, max_goals+1).

        Rows are home goals x, columns are away goals y. The matrix is
        renormalised to sum to 1 after truncating the Poisson support.
        """
        lam, mu = self.expected_goals(home_team, away_team)
        g = self.max_goals if max_goals is None else max_goals
        mat = np.zeros((g + 1, g + 1), dtype=float)
        for x in range(g + 1):
            for y in range(g + 1):
                mat[x, y] = score_probability(x, y, lam, mu, self.rho)
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
        """Return 1X2 probabilities ``{\"H\", \"D\", \"A\"}`` from the score matrix."""
        mat = self.predict_score_matrix(home_team, away_team, max_goals=max_goals)
        home_win = float(np.tril(mat, k=-1).sum())  # x > y
        draw = float(np.trace(mat))
        away_win = float(np.triu(mat, k=1).sum())  # x < y
        return {"H": home_win, "D": draw, "A": away_win}

    def predict_over_under(
        self,
        home_team: str,
        away_team: str,
        line: float = 2.5,
        max_goals: int | None = None,
    ) -> dict[str, float]:
        """Tài/Xỉu (Over/Under) probabilities for any goals line.

        Supports whole, half, and quarter lines (e.g. 1.5, 2.25, 2.5, 2.75, 3.5).
        Quarter lines are settled as a 50/50 split of the two neighbouring
        half/whole lines (win / half-win / push / half-lose / lose).

        Parameters
        ----------
        line:
            Total-goals line ``L``. Over wins when total goals settle above ``L``.
        max_goals:
            Optional truncation override for the score matrix.

        Returns
        -------
        dict
            ``over`` / ``under`` = effective cover probabilities
            ``p_win + 0.5 * p_half_win`` (suitable for EV = P × odds − 1),
            plus detailed settlement masses ``over_*`` / ``under_*``, and
            ``line``.
        """
        mat = self.predict_score_matrix(home_team, away_team, max_goals=max_goals)
        over_s = _empty_settlement()
        under_s = _empty_settlement()
        g = mat.shape[0] - 1
        for x in range(g + 1):
            for y in range(g + 1):
                p = float(mat[x, y])
                if p <= 0.0:
                    continue
                total = x + y
                over_s[_combine_settlements(_settle_over(total, line))] += p
                under_s[_combine_settlements(_settle_under(total, line))] += p

        return {
            "line": float(line),
            "over": _effective_cover_prob(over_s),
            "under": _effective_cover_prob(under_s),
            **{f"over_{k}": float(v) for k, v in over_s.items()},
            **{f"under_{k}": float(v) for k, v in under_s.items()},
        }

    def predict_asian_handicap(
        self,
        home_team: str,
        away_team: str,
        handicap: float = -0.5,
        max_goals: int | None = None,
    ) -> dict[str, float]:
        """Asian Handicap probabilities for any home handicap line.

        ``handicap`` is the line applied to the **home** team (football-data
        ``AHh`` convention), e.g. ``-0.25``, ``-0.5``, ``-0.75``, ``-1.0``,
        ``0``, ``+0.5``.

        Settlement uses margin ``X - Y + handicap`` (and a 50/50 split for
        quarter lines). Away AH is the opposite side of the same line.

        Returns
        -------
        dict
            ``home`` / ``away`` effective cover probabilities, detailed
            ``home_*`` / ``away_*`` settlement masses, and ``handicap``.
        """
        mat = self.predict_score_matrix(home_team, away_team, max_goals=max_goals)
        home_s = _empty_settlement()
        away_s = _empty_settlement()
        g = mat.shape[0] - 1
        for x in range(g + 1):
            for y in range(g + 1):
                p = float(mat[x, y])
                if p <= 0.0:
                    continue
                home_outcome = _combine_settlements(_settle_ah_home(x, y, handicap))
                home_s[home_outcome] += p
                # Away side is the mirror settlement of the home side.
                away_s[_mirror_settlement(home_outcome)] += p

        return {
            "handicap": float(handicap),
            "home": _effective_cover_prob(home_s),
            "away": _effective_cover_prob(away_s),
            **{f"home_{k}": float(v) for k, v in home_s.items()},
            **{f"away_{k}": float(v) for k, v in away_s.items()},
        }

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, matches: pd.DataFrame) -> DixonColesModel:
        """Estimate alpha, delta, gamma, rho by weighted MLE.

        Parameters
        ----------
        matches:
            DataFrame with columns ``HomeTeam``, ``AwayTeam``, ``FTHG``,
            ``FTAG``, and optionally ``Date`` (required when ``xi > 0``).

        Returns
        -------
        self
        """
        required = {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}
        missing = required - set(matches.columns)
        if missing:
            raise ValueError(f"matches missing columns: {sorted(missing)}")
        if matches.empty:
            raise ValueError("matches is empty")

        df = matches.dropna(subset=list(required)).copy()
        df["FTHG"] = df["FTHG"].astype(int)
        df["FTAG"] = df["FTAG"].astype(int)

        self.teams = sorted(
            set(df["HomeTeam"].astype(str)) | set(df["AwayTeam"].astype(str))
        )
        n = len(self.teams)
        if n < 2:
            raise ValueError("Need at least two teams to fit Dixon–Coles")

        team_index = {t: i for i, t in enumerate(self.teams)}
        home_idx = df["HomeTeam"].map(team_index).to_numpy(dtype=int)
        away_idx = df["AwayTeam"].map(team_index).to_numpy(dtype=int)
        x_goals = df["FTHG"].to_numpy(dtype=int)
        y_goals = df["FTAG"].to_numpy(dtype=int)

        weights = self._time_weights(df)

        # Parameter vector:
        #   attack[0..n-2]  (attack[n-1] = -sum(attack[0..n-2])  →  mean α = 0)
        #   defence[0..n-1]
        #   gamma, rho
        n_attack_free = n - 1
        n_params = n_attack_free + n + 2

        def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
            attack = np.zeros(n, dtype=float)
            attack[:n_attack_free] = theta[:n_attack_free]
            attack[-1] = -attack[:n_attack_free].sum()
            defence = theta[n_attack_free : n_attack_free + n]
            gamma = float(theta[-2])
            rho = float(theta[-1])
            return attack, defence, gamma, rho

        def neg_loglike(theta: np.ndarray) -> float:
            attack, defence, gamma, rho = unpack(theta)
            lam = np.exp(attack[home_idx] - defence[away_idx] + gamma)
            mu = np.exp(attack[away_idx] - defence[home_idx])
            return -_dc_loglike_matches(x_goals, y_goals, lam, mu, rho, weights)

        x0 = np.zeros(n_params, dtype=float)
        # Mild home advantage prior start; rho slightly negative (Dixon–Coles).
        x0[-2] = 0.25
        x0[-1] = -0.05

        bounds = (
            [(-3.0, 3.0)] * n_attack_free
            + [(-3.0, 3.0)] * n
            + [(0.0, 1.5), (-0.2, 0.1)]  # γ ≥ 0; ρ kept small so τ > 0
        )

        result = minimize(
            neg_loglike,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-8},
        )
        self.optimize_result_ = result

        attack_arr, defence_arr, gamma, rho = unpack(result.x)
        self.attack = {t: float(attack_arr[i]) for i, t in enumerate(self.teams)}
        self.defence = {t: float(defence_arr[i]) for i, t in enumerate(self.teams)}
        # Per-club appearance counts (home + away). Thin history → weak priors.
        home_c = df["HomeTeam"].astype(str).value_counts()
        away_c = df["AwayTeam"].astype(str).value_counts()
        counts: dict[str, int] = {}
        for t in self.teams:
            counts[t] = int(home_c.get(t, 0)) + int(away_c.get(t, 0))
        self.team_match_counts_ = counts
        min_n = int(self.min_team_matches)
        self.thin_teams_ = {t for t, n in counts.items() if n < min_n}
        # Well-supported clubs only; thin clubs demoted to prior injection path.
        self.fitted_teams_ = {t for t in self.teams if t not in self.thin_teams_}
        atk_prior = self._prior_attack()
        def_prior = self._prior_defence()
        for t in self.thin_teams_:
            self.attack[t] = float(atk_prior)
            self.defence[t] = float(def_prior)
            logger.warning(
                "⚠️ Thin-sample team: %s (n=%d < %d). Using weak-tier priors "
                "(α=%.2f, δ=%.2f).",
                t,
                counts[t],
                min_n,
                atk_prior,
                def_prior,
            )
        self.home_advantage = gamma
        self.rho = rho
        self.fitted_ = True
        return self

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def team_strengths(self) -> pd.DataFrame:
        """Return attack alpha, defence delta, and net = alpha + delta."""
        self._require_fitted()
        rows = [
            {
                "Team": t,
                "Attack": self.attack[t],
                "Defence": self.defence[t],
                "Net": self.attack[t] + self.defence[t],
            }
            for t in self.teams
        ]
        return pd.DataFrame(rows).sort_values("Net", ascending=False).reset_index(drop=True)

    def summary(self) -> dict[str, Any]:
        """Compact fit summary for logging / UI."""
        self._require_fitted()
        return {
            "n_teams": len(self.teams),
            "home_advantage_gamma": self.home_advantage,
            "rho": self.rho,
            "xi": self.xi,
            "converged": bool(self.optimize_result_.success)
            if self.optimize_result_ is not None
            else False,
            "nll": float(self.optimize_result_.fun)
            if self.optimize_result_ is not None
            else None,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _time_weights(self, df: pd.DataFrame) -> np.ndarray:
        """Compute w_i = exp(-xi * t_i) with t_i = days before latest match.

        Dates are normalised to timezone-naive midnights so string / tz-aware
        ``Date`` columns cannot inflate ``t`` via UTC offsets or mixed dtypes.
        """
        n = len(df)
        if self.xi <= 0 or "Date" not in df.columns:
            return np.ones(n, dtype=float)

        raw = pd.to_datetime(df["Date"], errors="coerce", utc=False)
        if raw.isna().all():
            return np.ones(n, dtype=float)

        # Force naive calendar days (strip tz without shifting the civil date).
        def _naive_day(ts: object) -> pd.Timestamp:
            if ts is None or (isinstance(ts, float) and ts != ts) or pd.isna(ts):
                return pd.NaT
            t = pd.Timestamp(ts)
            if t.tzinfo is not None:
                # Keep wall-clock civil day; do not convert via UTC.
                t = t.replace(tzinfo=None)
            return t.normalize()

        dates = pd.to_datetime(raw.map(_naive_day), errors="coerce")
        if dates.isna().all():
            return np.ones(n, dtype=float)

        latest = dates.max()
        days = (latest - dates).dt.total_seconds() / 86400.0
        days = days.fillna(days.median()).to_numpy(dtype=float)
        days = np.clip(days, 0.0, None)
        return np.exp(-self.xi * days)

    def _require_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError("Model is not fitted. Call fit() first.")

    def _require_teams(self, home_team: str, away_team: str) -> None:
        """Deprecated: unknown teams now use priors via ``_strengths_for``."""
        # Keep for callers; no longer raises KeyError.
        _ = self.unknown_teams(home_team, away_team)


if __name__ == "__main__":
    from src.data_loader import load_epl_data

    data = load_epl_data(n_seasons=2)
    model = DixonColesModel(xi=DEFAULT_XI).fit(data)
    print(model.summary())
    print(model.team_strengths().head(10).to_string(index=False))

    home, away = "Arsenal", "Chelsea"
    if home in model.teams and away in model.teams:
        lam, mu = model.expected_goals(home, away)
        probs = model.predict_match_probs(home, away)
        print(f"\n{home} vs {away}: λ={lam:.2f}, μ={mu:.2f}")
        print(f"1X2: H={probs['H']:.1%} D={probs['D']:.1%} A={probs['A']:.1%}")
