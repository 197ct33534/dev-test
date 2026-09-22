r"""Corner count models — Poisson GLM + LightGBM ``CornerPredictor``.

Works for any league whose historical table has ``HC`` / ``AC`` (EPL) or at
least finished scores (UWCL falls back to GLM / league averages when shot
stats are missing). Callers should pass league-specific match DataFrames from
:func:`src.data_loader.load_league_data`.

Legacy :class:`CornerModel` fits independent Poisson GLMs on HC / AC team
effects. New :class:`CornerPredictor` adds rolling shot/corner features
(:mod:`src.features`) and LightGBM Poisson regressors, then exposes:

* Over/Under 9.5 / 10.5 / 11.5 total corners
* Corner handicap (Skellam on independent Poisson rates)
* :meth:`CornerPredictor.predict_corner_ev` for bookmaker EV %
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import poisson, skellam

from src.features import (
    DEFAULT_ROLL_WINDOWS,
    engineer_rolling_features,
    feature_matrix_for_corners,
    team_corner_trend,
)

DEFAULT_MAX_CORNERS = 20
DEFAULT_OU_LINES: tuple[float, ...] = (9.5, 10.5, 11.5)

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb

    _HAS_LGB = True
except ImportError:  # pragma: no cover
    lgb = None  # type: ignore[assignment]
    _HAS_LGB = False


# ---------------------------------------------------------------------------
# Legacy Poisson GLM (kept for Streamlit / scripts that import CornerModel)
# ---------------------------------------------------------------------------


@dataclass
class CornerModel:
    """Poisson-regression corner model for home (HC) and away (AC) counts.

    Parameters
    ----------
    max_corners:
        Truncation for discrete corner distributions (0 … max inclusive).
    """

    max_corners: int = DEFAULT_MAX_CORNERS

    teams: list[str] = field(default_factory=list, init=False)
    fitted_: bool = field(default=False, init=False)
    model_hc_: Any = field(default=None, init=False, repr=False)
    model_ac_: Any = field(default=None, init=False, repr=False)
    league_avg_hc_: float = field(default=0.0, init=False)
    league_avg_ac_: float = field(default=0.0, init=False)

    def fit(self, matches: pd.DataFrame) -> CornerModel:
        """Estimate HC / AC Poisson GLMs on ``HomeTeam`` × ``AwayTeam``."""
        required = {"HomeTeam", "AwayTeam", "HC", "AC"}
        missing = required - set(matches.columns)
        if missing:
            raise ValueError(f"matches missing columns: {sorted(missing)}")

        df = matches.dropna(subset=list(required)).copy()
        df["HC"] = pd.to_numeric(df["HC"], errors="coerce")
        df["AC"] = pd.to_numeric(df["AC"], errors="coerce")
        df = df.dropna(subset=["HC", "AC"])
        df["HC"] = df["HC"].astype(int)
        df["AC"] = df["AC"].astype(int)
        df["HomeTeam"] = df["HomeTeam"].astype(str)
        df["AwayTeam"] = df["AwayTeam"].astype(str)

        if df.empty:
            raise ValueError("No rows with valid HC/AC to fit corner model")

        self.teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
        self.league_avg_hc_ = float(df["HC"].mean())
        self.league_avg_ac_ = float(df["AC"].mean())

        self.model_hc_ = smf.poisson("HC ~ C(HomeTeam) + C(AwayTeam)", data=df).fit(
            disp=False, maxiter=200
        )
        self.model_ac_ = smf.poisson("AC ~ C(HomeTeam) + C(AwayTeam)", data=df).fit(
            disp=False, maxiter=200
        )
        self.fitted_ = True
        return self

    def expected_corners(self, home_team: str, away_team: str) -> dict[str, float]:
        """Return ``{\"hc\", \"ac\", \"total\"}`` Poisson rates.

        Unknown clubs fall back to league-average HC/AC (no ``KeyError``).
        """
        self._require_fitted()
        unknown = [t for t in (home_team, away_team) if t not in self.teams]
        if unknown:
            for name in unknown:
                logger.warning(
                    "⚠️ Detecting new team: %s. Using default league priors.",
                    name,
                )
            return {
                "hc": float(self.league_avg_hc_),
                "ac": float(self.league_avg_ac_),
                "total": float(self.league_avg_hc_ + self.league_avg_ac_),
            }
        newdata = pd.DataFrame({"HomeTeam": [home_team], "AwayTeam": [away_team]})
        lam_hc = float(self.model_hc_.predict(newdata).iloc[0])
        lam_ac = float(self.model_ac_.predict(newdata).iloc[0])
        return {"hc": lam_hc, "ac": lam_ac, "total": lam_hc + lam_ac}

    def predict_total_pmf(
        self,
        home_team: str,
        away_team: str,
        max_corners: int | None = None,
    ) -> np.ndarray:
        """PMF of total corners under independent Poissons."""
        rates = self.expected_corners(home_team, away_team)
        g = self.max_corners if max_corners is None else max_corners
        pmf = poisson.pmf(np.arange(g + 1), rates["total"])
        total = pmf.sum()
        if total <= 0:
            raise RuntimeError("Corner PMF has non-positive mass")
        return pmf / total

    def predict_over_under(
        self,
        home_team: str,
        away_team: str,
        line: float = 9.5,
    ) -> dict[str, float]:
        """Over / under probabilities for total corners vs ``line``."""
        rates = self.expected_corners(home_team, away_team)
        lam = rates["total"]
        over = float(poisson.sf(np.floor(line), lam))
        under = float(poisson.cdf(np.floor(line), lam))
        return {
            "over": over,
            "under": under,
            "line": line,
            "expected_total": lam,
            **{k: rates[k] for k in ("hc", "ac")},
        }

    def team_corner_rates(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Empirical per-team home/away corner for/against averages."""
        df = matches.dropna(subset=["HomeTeam", "AwayTeam", "HC", "AC"]).copy()
        df["HC"] = pd.to_numeric(df["HC"], errors="coerce")
        df["AC"] = pd.to_numeric(df["AC"], errors="coerce")
        df = df.dropna(subset=["HC", "AC"])

        home = (
            df.groupby("HomeTeam")
            .agg(home_for=("HC", "mean"), home_against=("AC", "mean"), home_n=("HC", "size"))
            .rename_axis("Team")
        )
        away = (
            df.groupby("AwayTeam")
            .agg(away_for=("AC", "mean"), away_against=("HC", "mean"), away_n=("AC", "size"))
            .rename_axis("Team")
        )
        out = home.join(away, how="outer").fillna(0.0)
        out["corners_for_avg"] = (
            out["home_for"] * out["home_n"] + out["away_for"] * out["away_n"]
        ) / (out["home_n"] + out["away_n"]).replace(0, np.nan)
        out["corners_against_avg"] = (
            out["home_against"] * out["home_n"] + out["away_against"] * out["away_n"]
        ) / (out["home_n"] + out["away_n"]).replace(0, np.nan)
        return out.sort_values("corners_for_avg", ascending=False).reset_index()

    def summary(self) -> dict[str, Any]:
        self._require_fitted()
        return {
            "n_teams": len(self.teams),
            "league_avg_hc": self.league_avg_hc_,
            "league_avg_ac": self.league_avg_ac_,
            "league_avg_total": self.league_avg_hc_ + self.league_avg_ac_,
            "hc_converged": bool(
                getattr(self.model_hc_, "mle_retvals", {}).get("converged", True)
            ),
            "ac_converged": bool(
                getattr(self.model_ac_, "mle_retvals", {}).get("converged", True)
            ),
        }

    def _require_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError("Corner model is not fitted. Call fit() first.")

    def _require_teams(self, home_team: str, away_team: str) -> None:
        """Deprecated: unknown clubs use league-average rates (no KeyError)."""
        _ = (home_team, away_team)


# ---------------------------------------------------------------------------
# CornerPredictor — LightGBM / Poisson + EV
# ---------------------------------------------------------------------------


def _expected_value(p: float, odds: float) -> float:
    """EV = P · Odds − 1."""
    if odds <= 1.0 or p <= 0:
        return -1.0
    return float(p) * float(odds) - 1.0


@dataclass
class CornerPredictor:
    """Predict home/away corner counts and corner-market probabilities.

    Parameters
    ----------
    backend:
        ``\"auto\"`` (LightGBM if available else Poisson GLM),
        ``\"lightgbm\"``, or ``\"poisson\"``.
    max_corners:
        Support for total-corner PMF truncation.
    roll_windows:
        Rolling feature windows (default 5 & 10).
    min_prior_matches:
        Drop training rows with thinner history than this.
    """

    backend: str = "auto"
    max_corners: int = DEFAULT_MAX_CORNERS
    roll_windows: tuple[int, ...] = DEFAULT_ROLL_WINDOWS
    min_prior_matches: int = 3

    teams: list[str] = field(default_factory=list, init=False)
    fitted_: bool = field(default=False, init=False)
    backend_used_: str = field(default="", init=False)
    feature_names_: list[str] = field(default_factory=list, init=False)
    glm_: CornerModel | None = field(default=None, init=False, repr=False)
    model_hc_: Any = field(default=None, init=False, repr=False)
    model_ac_: Any = field(default=None, init=False, repr=False)
    train_features_: pd.DataFrame | None = field(default=None, init=False, repr=False)
    history_matches_: pd.DataFrame | None = field(default=None, init=False, repr=False)
    league_avg_hc_: float = field(default=5.0, init=False)
    league_avg_ac_: float = field(default=4.5, init=False)

    def fit(self, matches: pd.DataFrame) -> CornerPredictor:
        """Fit corner models on historical matches (needs HC/AC; shots optional)."""
        required = {"Date", "HomeTeam", "AwayTeam", "HC", "AC"}
        missing = required - set(matches.columns)
        if missing:
            raise ValueError(f"matches missing columns: {sorted(missing)}")

        raw = matches.copy()
        raw["HC"] = pd.to_numeric(raw["HC"], errors="coerce")
        raw["AC"] = pd.to_numeric(raw["AC"], errors="coerce")
        raw = raw.dropna(subset=["HC", "AC", "HomeTeam", "AwayTeam", "Date"])
        if raw.empty:
            raise ValueError("No rows with valid HC/AC to fit CornerPredictor")

        self.history_matches_ = raw.sort_values("Date").reset_index(drop=True)
        self.teams = sorted(
            set(self.history_matches_["HomeTeam"].astype(str))
            | set(self.history_matches_["AwayTeam"].astype(str))
        )
        self.league_avg_hc_ = float(self.history_matches_["HC"].mean())
        self.league_avg_ac_ = float(self.history_matches_["AC"].mean())

        # Always keep a GLM fallback for unseen feature rows / new teams.
        self.glm_ = CornerModel(max_corners=self.max_corners).fit(self.history_matches_)

        prefer = self.backend.lower().strip()
        use_lgb = prefer == "lightgbm" or (prefer == "auto" and _HAS_LGB)
        if prefer == "poisson":
            use_lgb = False

        if use_lgb and _HAS_LGB:
            eng = engineer_rolling_features(
                self.history_matches_,
                windows=self.roll_windows,
                min_prior_matches=self.min_prior_matches,
            )
            eng = eng.dropna(subset=["HC", "AC"])
            X, names = feature_matrix_for_corners(eng, windows=self.roll_windows)
            mask = X.notna().all(axis=1)
            X = X.loc[mask]
            y_hc = eng.loc[mask, "HC"].astype(float)
            y_ac = eng.loc[mask, "AC"].astype(float)

            if len(X) < 40:
                use_lgb = False
            else:
                params = {
                    "objective": "poisson",
                    "metric": "poisson",
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "min_data_in_leaf": 20,
                    "verbosity": -1,
                    "n_estimators": 200,
                }
                self.model_hc_ = lgb.LGBMRegressor(**params)
                self.model_ac_ = lgb.LGBMRegressor(**params)
                self.model_hc_.fit(X, y_hc)
                self.model_ac_.fit(X, y_ac)
                self.feature_names_ = names
                self.train_features_ = eng
                self.backend_used_ = "lightgbm"

        if not use_lgb:
            self.backend_used_ = "poisson"
            self.model_hc_ = None
            self.model_ac_ = None
            self.feature_names_ = []

        self.fitted_ = True
        return self

    def _feature_row(self, home_team: str, away_team: str) -> pd.DataFrame | None:
        """Build one inference row from rolling history (as-of last known match)."""
        if self.history_matches_ is None:
            return None
        # Append a synthetic upcoming row then take its engineered features.
        stub = self.history_matches_.iloc[[-1]].copy()
        stub["Date"] = pd.Timestamp(self.history_matches_["Date"].max()) + pd.Timedelta(
            days=1
        )
        stub["HomeTeam"] = home_team
        stub["AwayTeam"] = away_team
        stub["HC"] = np.nan
        stub["AC"] = np.nan
        for col in ("HS", "AS", "HST", "AST"):
            if col in stub.columns:
                stub[col] = np.nan

        combined = pd.concat([self.history_matches_, stub], ignore_index=True)
        eng = engineer_rolling_features(
            combined,
            windows=self.roll_windows,
            min_prior_matches=0,
        )
        last = eng.iloc[[-1]]
        if not self.feature_names_:
            return last
        X, _ = feature_matrix_for_corners(last, windows=self.roll_windows)
        if X.isna().any(axis=None):
            return None
        return X

    def expected_corners(self, home_team: str, away_team: str) -> dict[str, float]:
        """Predict ``C_home``, ``C_away`` and total.

        Uses LightGBM when fitted; falls back to Poisson GLM rates.
        """
        self._require_fitted()
        if self.backend_used_ == "lightgbm" and self.model_hc_ is not None:
            X = self._feature_row(home_team, away_team)
            if X is not None and len(X.columns) == len(self.feature_names_):
                # Align columns
                X = X.reindex(columns=self.feature_names_)
                if not X.isna().any(axis=None):
                    lam_hc = max(0.05, float(self.model_hc_.predict(X)[0]))
                    lam_ac = max(0.05, float(self.model_ac_.predict(X)[0]))
                    return {"hc": lam_hc, "ac": lam_ac, "total": lam_hc + lam_ac}

        assert self.glm_ is not None
        try:
            return self.glm_.expected_corners(home_team, away_team)
        except KeyError:
            # Unknown team → league averages
            return {
                "hc": self.league_avg_hc_,
                "ac": self.league_avg_ac_,
                "total": self.league_avg_hc_ + self.league_avg_ac_,
            }

    def predict_total_pmf(
        self,
        home_team: str,
        away_team: str,
        max_corners: int | None = None,
    ) -> np.ndarray:
        rates = self.expected_corners(home_team, away_team)
        g = self.max_corners if max_corners is None else max_corners
        pmf = poisson.pmf(np.arange(g + 1), rates["total"])
        s = pmf.sum()
        return pmf / s if s > 0 else pmf

    def predict_over_under(
        self,
        home_team: str,
        away_team: str,
        line: float = 10.5,
    ) -> dict[str, float]:
        """P(Over / Under) for total corners vs ``line`` (typically X.5)."""
        rates = self.expected_corners(home_team, away_team)
        lam = rates["total"]
        over = float(poisson.sf(np.floor(line), lam))
        under = float(poisson.cdf(np.floor(line), lam))
        return {
            "over": over,
            "under": under,
            "line": float(line),
            "expected_total": lam,
            "hc": rates["hc"],
            "ac": rates["ac"],
        }

    def predict_lines(
        self,
        home_team: str,
        away_team: str,
        lines: Sequence[float] = DEFAULT_OU_LINES,
    ) -> pd.DataFrame:
        """Over/Under table for several total-corner lines."""
        rows = [
            self.predict_over_under(home_team, away_team, line=float(line))
            for line in lines
        ]
        return pd.DataFrame(rows)

    def predict_handicap(
        self,
        home_team: str,
        away_team: str,
        handicap: float = 0.0,
    ) -> dict[str, float]:
        """Corner handicap probabilities via Skellam (HC − AC).

        Parameters
        ----------
        handicap:
            Asian-style line on home corners (e.g. ``-1.5`` means home −1.5).
            ``home`` wins when ``HC - AC > handicap`` (for .5 lines).
        """
        rates = self.expected_corners(home_team, away_team)
        mu1, mu2 = rates["hc"], rates["ac"]
        # For line L=.5k: home covers if HC-AC >= ceil(L+eps) i.e. > floor(L)
        # P(HC-AC > L) = P(diff >= floor(L)+1) = skellam.sf(floor(L), mu1, mu2)
        thr = int(np.floor(handicap))
        p_home = float(skellam.sf(thr, mu1, mu2))  # P(diff >= thr+1) = P(diff > thr)
        p_away = float(skellam.cdf(thr, mu1, mu2))  # P(diff <= thr)
        # renormalise tiny mass on exact integer lines (push ignored for .5)
        mass = p_home + p_away
        if mass > 0:
            p_home /= mass
            p_away /= mass
        return {
            "home": p_home,
            "away": p_away,
            "handicap": float(handicap),
            "expected_diff": mu1 - mu2,
            "hc": mu1,
            "ac": mu2,
        }

    def predict_corner_ev(
        self,
        home_team: str,
        away_team: str,
        odds_dict: Mapping[str, float],
        line: float = 10.5,
        *,
        min_ev: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Compute EV % for corner Over/Under (and optional AH) vs book odds.

        Parameters
        ----------
        odds_dict:
            Keys among ``over``, ``under``, and optionally ``ah_home``,
            ``ah_away``, ``ah_line`` (default 0.0 / -0.5).
        line:
            Total-corners O/U line (default 10.5).
        min_ev:
            Only return legs with EV ≥ this threshold (fraction).

        Returns
        -------
        list[dict]
            Each item has ``market``, ``selection``, ``line``, ``p_model``,
            ``odds``, ``ev``, ``ev_pct``.
        """
        self._require_fitted()
        out: list[dict[str, Any]] = []
        ou = self.predict_over_under(home_team, away_team, line=float(line))

        for sel, key in (("Over", "over"), ("Under", "under")):
            odds = odds_dict.get(key) or odds_dict.get(sel.lower())
            if odds is None:
                continue
            odds_f = float(odds)
            p = float(ou[key])
            ev = _expected_value(p, odds_f)
            if ev >= float(min_ev):
                out.append(
                    {
                        "market": "Corners",
                        "selection": f"{sel} {float(line):g}",
                        "line": float(line),
                        "p_model": p,
                        "odds": odds_f,
                        "ev": ev,
                        "ev_pct": ev * 100.0,
                        "hc": ou["hc"],
                        "ac": ou["ac"],
                    }
                )

        ah_home = odds_dict.get("ah_home")
        ah_away = odds_dict.get("ah_away")
        if ah_home is not None or ah_away is not None:
            ah_line = float(odds_dict.get("ah_line", -0.5))
            ah = self.predict_handicap(home_team, away_team, handicap=ah_line)
            for sel, key, odds_v in (
                ("AH Home", "home", ah_home),
                ("AH Away", "away", ah_away),
            ):
                if odds_v is None:
                    continue
                odds_f = float(odds_v)
                p = float(ah[key])
                ev = _expected_value(p, odds_f)
                if ev >= float(min_ev):
                    out.append(
                        {
                            "market": "Corners",
                            "selection": f"{sel} {ah_line:+g}",
                            "line": ah_line,
                            "p_model": p,
                            "odds": odds_f,
                            "ev": ev,
                            "ev_pct": ev * 100.0,
                            "hc": ah["hc"],
                            "ac": ah["ac"],
                        }
                    )

        return sorted(out, key=lambda r: r["ev"], reverse=True)

    def trend(self, team: str, *, last_n: int = 10) -> pd.DataFrame:
        """Recent corner-for/against series for charts."""
        self._require_fitted()
        assert self.history_matches_ is not None
        return team_corner_trend(self.history_matches_, team, last_n=last_n)

    def summary(self) -> dict[str, Any]:
        self._require_fitted()
        return {
            "backend": self.backend_used_,
            "n_teams": len(self.teams),
            "league_avg_hc": self.league_avg_hc_,
            "league_avg_ac": self.league_avg_ac_,
            "feature_names": list(self.feature_names_),
            "lightgbm_available": _HAS_LGB,
        }

    def _require_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError("CornerPredictor is not fitted. Call fit() first.")


if __name__ == "__main__":
    from src.data_loader import load_epl_data

    data = load_epl_data(n_seasons=2)
    pred = CornerPredictor(backend="auto").fit(data)
    print(pred.summary())
    home, away = "Arsenal", "Chelsea"
    if home in pred.teams and away in pred.teams:
        print(pred.expected_corners(home, away))
        print(pred.predict_lines(home, away))
        print(
            pred.predict_corner_ev(
                home,
                away,
                {"over": 1.90, "under": 1.90},
                line=10.5,
            )
        )
