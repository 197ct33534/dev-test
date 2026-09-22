r"""LightGBM 1X2 classifier — parallel path to Dixon–Coles.

Builds leakage-safe rolling features from historical EPL matches, trains a
multiclass LightGBM model on labels {H, D, A}, and exposes ``predict_proba``
in the same ``{\"H\", \"D\", \"A\"}`` shape as ``DixonColesModel.predict_match_probs``
so both models can feed a future Ensemble layer.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable, Sequence
import logging

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "lightgbm is required for EPLMachineLearningModel. "
        "Install with: pip install lightgbm"
    ) from exc

from src.config import ML_CALIBRATION_METHOD, ML_CALIBRATION_MIN_ROWS

# Rolling windows requested for goals / corners.
ROLL_WINDOWS: tuple[int, ...] = (3, 5, 10)
FORM_WINDOW: int = 5
CLASS_LABELS: tuple[str, ...] = ("H", "D", "A")
LABEL_TO_INT: dict[str, int] = {"H": 0, "D": 1, "A": 2}

# Per-team event keys stored in chronological history.
_STAT_KEYS: tuple[str, ...] = ("gf", "ga", "cf", "ca", "points")

logger = logging.getLogger(__name__)


def _as_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError(f"Invalid match date: {value!r}")
    # Normalise to naive midnight for day-diff arithmetic.
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()


def _mean_last(values: Sequence[float], n: int) -> float:
    """Mean of the last ``n`` values, or NaN if history is empty."""
    if not values:
        return float("nan")
    window = values[-n:]
    return float(np.mean(window))


def _sum_last(values: Sequence[float], n: int) -> float:
    """Sum of the last ``n`` values (form points), or NaN if empty."""
    if not values:
        return float("nan")
    return float(np.sum(values[-n:]))


def _rest_days(history: list[dict[str, Any]], match_date: pd.Timestamp) -> float:
    """Days since this team's previous match (NaN on season debut)."""
    if not history:
        return float("nan")
    last_date = history[-1]["date"]
    return float((match_date - last_date).days)


def feature_column_names(windows: Sequence[int] = ROLL_WINDOWS) -> list[str]:
    """Ordered feature names used by the classifier."""
    cols: list[str] = []
    for side in ("home", "away"):
        for n in windows:
            cols.extend(
                [
                    f"{side}_gf_r{n}",
                    f"{side}_ga_r{n}",
                    f"{side}_cf_r{n}",
                    f"{side}_ca_r{n}",
                ]
            )
        cols.append(f"{side}_rest_days")
        cols.append(f"{side}_form_{FORM_WINDOW}")
    return cols


def _side_features(
    prefix: str,
    history: list[dict[str, Any]],
    match_date: pd.Timestamp,
    windows: Sequence[int],
) -> dict[str, float]:
    """Rolling / rest / form features for one side of a fixture."""
    series = {k: [row[k] for row in history] for k in _STAT_KEYS}
    out: dict[str, float] = {}
    for n in windows:
        out[f"{prefix}_gf_r{n}"] = _mean_last(series["gf"], n)
        out[f"{prefix}_ga_r{n}"] = _mean_last(series["ga"], n)
        out[f"{prefix}_cf_r{n}"] = _mean_last(series["cf"], n)
        out[f"{prefix}_ca_r{n}"] = _mean_last(series["ca"], n)
    out[f"{prefix}_rest_days"] = _rest_days(history, match_date)
    out[f"{prefix}_form_{FORM_WINDOW}"] = _sum_last(series["points"], FORM_WINDOW)
    return out


def _points_from_result(gf: int, ga: int) -> int:
    if gf > ga:
        return 3
    if gf == ga:
        return 1
    return 0


def _append_team_result(
    history: dict[str, list[dict[str, Any]]],
    team: str,
    match_date: pd.Timestamp,
    gf: int,
    ga: int,
    corners_for: float,
    corners_against: float,
) -> None:
    history[team].append(
        {
            "date": match_date,
            "gf": float(gf),
            "ga": float(ga),
            "cf": float(corners_for),
            "ca": float(corners_against),
            "points": float(_points_from_result(gf, ga)),
        }
    )


def engineer_match_features(
    matches: pd.DataFrame,
    *,
    windows: Sequence[int] = ROLL_WINDOWS,
    min_prior_matches: int = 3,
    include_label: bool = True,
) -> pd.DataFrame:
    """Build leakage-safe rolling features for every historical match.

    Walks matches in chronological order. For each row, features use **only**
    prior games of each team; the current result is appended to history
    afterwards (no look-ahead).

    Parameters
    ----------
    matches:
        DataFrame with ``Date``, ``HomeTeam``, ``AwayTeam``, ``FTHG``, ``FTAG``,
        optionally ``HC`` / ``AC`` / ``FTR``.
    windows:
        Rolling mean lengths (default 3, 5, 10).
    min_prior_matches:
        Drop rows where either side has fewer than this many prior games.
        Set ``0`` to keep debuts (features will be NaN).
    include_label:
        If True, add integer ``label`` (0=H, 1=D, 2=A) from ``FTR`` or goals.

    Returns
    -------
    pd.DataFrame
        One row per eligible match with feature columns (+ metadata / label).
    """
    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"matches missing columns: {sorted(missing)}")

    df = matches.dropna(subset=list(required)).copy()
    df["Date"] = df["Date"].map(_as_timestamp)
    df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
    df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
    df = df.dropna(subset=["FTHG", "FTAG"])
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)

    has_corners = "HC" in df.columns and "AC" in df.columns
    if has_corners:
        df["HC"] = pd.to_numeric(df["HC"], errors="coerce")
        df["AC"] = pd.to_numeric(df["AC"], errors="coerce")
    else:
        df["HC"] = np.nan
        df["AC"] = np.nan

    if "FTR" not in df.columns:
        df["FTR"] = np.where(
            df["FTHG"] > df["FTAG"],
            "H",
            np.where(df["FTHG"] < df["FTAG"], "A", "D"),
        )
    else:
        df["FTR"] = df["FTR"].astype(str).str.upper().str.strip()

    df = df.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

    history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    feat_names = feature_column_names(windows)

    for idx, row in df.iterrows():
        home = str(row["HomeTeam"])
        away = str(row["AwayTeam"])
        match_date = row["Date"]
        home_hist = history[home]
        away_hist = history[away]

        if (
            len(home_hist) >= min_prior_matches
            and len(away_hist) >= min_prior_matches
        ):
            feats = {}
            feats.update(_side_features("home", home_hist, match_date, windows))
            feats.update(_side_features("away", away_hist, match_date, windows))
            record: dict[str, Any] = {
                "Date": match_date,
                "HomeTeam": home,
                "AwayTeam": away,
                "row_index": int(idx) if isinstance(idx, (int, np.integer)) else idx,
                **{c: feats.get(c, float("nan")) for c in feat_names},
            }
            if include_label:
                ftr = str(row["FTR"])
                if ftr not in LABEL_TO_INT:
                    ftr = (
                        "H"
                        if row["FTHG"] > row["FTAG"]
                        else ("A" if row["FTHG"] < row["FTAG"] else "D")
                    )
                record["FTR"] = ftr
                record["label"] = LABEL_TO_INT[ftr]
            rows.append(record)

        # Update histories AFTER emitting features (anti-leakage).
        hc = row["HC"] if pd.notna(row["HC"]) else float("nan")
        ac = row["AC"] if pd.notna(row["AC"]) else float("nan")
        _append_team_result(
            history, home, match_date, int(row["FTHG"]), int(row["FTAG"]), hc, ac
        )
        _append_team_result(
            history, away, match_date, int(row["FTAG"]), int(row["FTHG"]), ac, hc
        )

    if not rows:
        return pd.DataFrame(columns=["Date", "HomeTeam", "AwayTeam", *feat_names])

    out = pd.DataFrame(rows)
    # Stable column order.
    meta = ["Date", "HomeTeam", "AwayTeam", "row_index"]
    extra = [c for c in ("FTR", "label") if c in out.columns]
    return out[meta + feat_names + extra]


def features_for_fixture(
    history: dict[str, list[dict[str, Any]]],
    home_team: str,
    away_team: str,
    match_date: pd.Timestamp | datetime | str,
    *,
    windows: Sequence[int] = ROLL_WINDOWS,
) -> dict[str, float]:
    """Compute the feature vector for an upcoming (or held-out) fixture."""
    match_ts = _as_timestamp(match_date)
    home = str(home_team).strip()
    away = str(away_team).strip()
    feats: dict[str, float] = {}
    feats.update(_side_features("home", history.get(home, []), match_ts, windows))
    feats.update(_side_features("away", history.get(away, []), match_ts, windows))
    return feats


def build_team_history(
    matches: pd.DataFrame,
) -> dict[str, list[dict[str, Any]]]:
    """Replay all matches into per-team chronological event lists."""
    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"matches missing columns: {sorted(missing)}")

    df = matches.dropna(subset=list(required)).copy()
    df["Date"] = df["Date"].map(_as_timestamp)
    df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
    df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce").astype("Int64")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["FTHG", "FTAG"])
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)

    if "HC" in df.columns and "AC" in df.columns:
        df["HC"] = pd.to_numeric(df["HC"], errors="coerce")
        df["AC"] = pd.to_numeric(df["AC"], errors="coerce")
    else:
        df["HC"] = np.nan
        df["AC"] = np.nan

    df = df.sort_values(["Date", "HomeTeam", "AwayTeam"])
    history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for _, row in df.iterrows():
        hc = row["HC"] if pd.notna(row["HC"]) else float("nan")
        ac = row["AC"] if pd.notna(row["AC"]) else float("nan")
        match_date = row["Date"]
        _append_team_result(
            history,
            str(row["HomeTeam"]),
            match_date,
            int(row["FTHG"]),
            int(row["FTAG"]),
            hc,
            ac,
        )
        _append_team_result(
            history,
            str(row["AwayTeam"]),
            match_date,
            int(row["FTAG"]),
            int(row["FTHG"]),
            ac,
            hc,
        )
    return history


@dataclass
class EPLMachineLearningModel:
    """LightGBM multiclass model for Premier League 1X2 outcomes.

    Parameters
    ----------
    windows:
        Rolling mean windows for goals / corners (default 3, 5, 10).
    min_prior_matches:
        Minimum prior games per side required to enter the training set.
    lgbm_params:
        Extra keyword args forwarded to ``lightgbm.LGBMClassifier``.
    random_state:
        RNG seed for reproducibility.
    """

    windows: tuple[int, ...] = ROLL_WINDOWS
    min_prior_matches: int = 3
    lgbm_params: dict[str, Any] = field(default_factory=dict)
    random_state: int = 42
    # Wrap LightGBM with sklearn probability calibration (lowers ECE).
    calibrate: bool = True
    calibration_method: str = ML_CALIBRATION_METHOD  # "sigmoid" | "isotonic"

    feature_names_: list[str] = field(default_factory=list, init=False)
    classes_: list[str] = field(default_factory=lambda: list(CLASS_LABELS), init=False)
    teams: list[str] = field(default_factory=list, init=False)
    fitted_: bool = field(default=False, init=False)
    model_: Any = field(default=None, init=False, repr=False)
    base_model_: Any = field(default=None, init=False, repr=False)
    calibrated_: bool = field(default=False, init=False)
    history_: dict[str, list[dict[str, Any]]] = field(
        default_factory=dict, init=False, repr=False
    )
    train_rows_: int = field(default=0, init=False)
    last_match_date_: pd.Timestamp | None = field(default=None, init=False)
    feature_means_: dict[str, float] = field(default_factory=dict, init=False)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, matches: pd.DataFrame) -> EPLMachineLearningModel:
        """Engineer features and train the LightGBM 1X2 classifier.

        Parameters
        ----------
        matches:
            Historical EPL frame (``Date``, ``HomeTeam``, ``AwayTeam``,
            ``FTHG``, ``FTAG``, optional ``HC``/``AC``/``FTR``).

        Returns
        -------
        self
        """
        featured = engineer_match_features(
            matches,
            windows=self.windows,
            min_prior_matches=self.min_prior_matches,
            include_label=True,
        )
        if featured.empty or "label" not in featured.columns:
            raise ValueError(
                "No training rows after feature engineering — need more history "
                f"(min_prior_matches={self.min_prior_matches})."
            )

        self.feature_names_ = feature_column_names(self.windows)
        x = featured[self.feature_names_].astype(float)
        # League-mean impute (debuts / sparse corners)
        means = x.mean(numeric_only=True)
        self.feature_means_ = {c: float(means.get(c, 0.0) or 0.0) for c in self.feature_names_}
        x = x.fillna(means).fillna(0.0)
        y = featured["label"].astype(int).to_numpy()

        # Ensure all three classes appear; otherwise multiclass predict_proba
        # column order becomes ambiguous.
        present = set(np.unique(y).tolist())
        if present != {0, 1, 2}:
            raise ValueError(
                f"Training labels must include H/D/A (0/1/2); got {sorted(present)}"
            )

        n_rows = int(len(x))
        # Shrink leaf / child constraints on small samples (e.g. early season).
        adaptive_child = max(5, min(20, n_rows // 10)) if n_rows < 200 else 20
        adaptive_leaves = 15 if n_rows < 200 else 31
        adaptive_estimators = 150 if n_rows < 200 else 300

        params = {
            "objective": "multiclass",
            "num_class": 3,
            "n_estimators": adaptive_estimators,
            "learning_rate": 0.05,
            "num_leaves": adaptive_leaves,
            "max_depth": -1,
            "min_child_samples": adaptive_child,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 1.0,
            "random_state": self.random_state,
            "verbosity": -1,
            "n_jobs": -1,
        }
        params.update(self.lgbm_params)

        base = lgb.LGBMClassifier(**params)
        method = str(self.calibration_method or "sigmoid").lower()
        if method not in {"sigmoid", "isotonic"}:
            method = "sigmoid"

        self.calibrated_ = False
        self.base_model_ = base
        # CalibratedClassifierCV needs enough rows for stratified CV folds.
        n_cv = 3 if n_rows >= 90 else (2 if n_rows >= ML_CALIBRATION_MIN_ROWS else 0)
        if self.calibrate and n_cv >= 2:
            try:
                clf: Any = CalibratedClassifierCV(
                    estimator=base,
                    method=method,
                    cv=n_cv,
                )
                clf.fit(x, y)
                self.model_ = clf
                self.calibrated_ = True
                logger.info(
                    "LightGBM wrapped with CalibratedClassifierCV(method=%s, cv=%d)",
                    method,
                    n_cv,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Probability calibration failed (%s); using raw LightGBM.",
                    exc,
                )
                base.fit(x, y)
                self.model_ = base
        else:
            base.fit(x, y)
            self.model_ = base

        self.history_ = build_team_history(matches)
        self.teams = sorted(self.history_.keys())
        self.train_rows_ = int(len(featured))
        self.last_match_date_ = pd.Timestamp(featured["Date"].max())
        self.fitted_ = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        home_team: str,
        away_team: str,
        match_date: pd.Timestamp | datetime | str | None = None,
    ) -> dict[str, float]:
        """Return 1X2 probabilities ``{\"H\", \"D\", \"A\"}``.

        Compatible with ``DixonColesModel.predict_match_probs`` for Ensemble
        blending (same keys, probabilities sum ≈ 1).

        Parameters
        ----------
        home_team, away_team:
            Club names as in the training data.
        match_date:
            Kick-off used for rest-day features. Defaults to one day after
            the latest training match (or today if unknown).
        """
        self._require_fitted()
        home = str(home_team).strip()
        away = str(away_team).strip()
        # New / unseen clubs: empty history + league-mean feature impute (no crash).
        if home not in self.history_:
            logger.warning(
                "⚠️ Detecting new team: %s. Using default league priors.",
                home,
            )
            self.history_[home] = []
        if away not in self.history_:
            logger.warning(
                "⚠️ Detecting new team: %s. Using default league priors.",
                away,
            )
            self.history_[away] = []

        if match_date is None:
            if self.last_match_date_ is not None:
                match_date = self.last_match_date_ + timedelta(days=1)
            else:
                match_date = pd.Timestamp(datetime.utcnow()).normalize()

        feats = features_for_fixture(
            self.history_,
            home,
            away,
            match_date,
            windows=self.windows,
        )
        row = pd.DataFrame([{c: feats.get(c, float("nan")) for c in self.feature_names_}])
        if self.feature_means_:
            row = row.fillna(self.feature_means_)
        row = row.fillna(0.0)
        proba = np.asarray(self.model_.predict_proba(row)[0], dtype=float)

        # Map classifier class order → H/D/A (LightGBM stores classes_ as ints).
        class_ids = list(getattr(self.model_, "classes_", [0, 1, 2]))
        out = {label: 0.0 for label in CLASS_LABELS}
        for cls_id, p in zip(class_ids, proba):
            label = CLASS_LABELS[int(cls_id)]
            out[label] = float(p)

        total = sum(out.values())
        if total <= 0:
            raise RuntimeError("ML predict_proba returned non-positive mass")
        return {k: v / total for k, v in out.items()}

    def predict_match_probs(
        self,
        home_team: str,
        away_team: str,
        match_date: pd.Timestamp | datetime | str | None = None,
    ) -> dict[str, float]:
        """Alias of :meth:`predict_proba` (Dixon–Coles naming parity)."""
        return self.predict_proba(home_team, away_team, match_date=match_date)

    def feature_vector(
        self,
        home_team: str,
        away_team: str,
        match_date: pd.Timestamp | datetime | str | None = None,
    ) -> pd.Series:
        """Return the engineered feature Series for inspection / Ensemble."""
        self._require_fitted()
        if match_date is None:
            match_date = (
                self.last_match_date_ + timedelta(days=1)
                if self.last_match_date_ is not None
                else pd.Timestamp(datetime.utcnow()).normalize()
            )
        feats = features_for_fixture(
            self.history_,
            home_team,
            away_team,
            match_date,
            windows=self.windows,
        )
        return pd.Series(
            {c: feats.get(c, float("nan")) for c in self.feature_names_},
            dtype=float,
        )

    def feature_importance(self) -> pd.DataFrame:
        """Gain-based feature importance from the fitted booster."""
        self._require_fitted()
        booster = None
        for cand in (self.base_model_, self.model_):
            if cand is None:
                continue
            if hasattr(cand, "booster_"):
                booster = cand.booster_
                break
            # CalibratedClassifierCV → first fold estimator
            cals = getattr(cand, "calibrated_classifiers_", None)
            if cals:
                est = getattr(cals[0], "estimator", None)
                if est is not None and hasattr(est, "booster_"):
                    booster = est.booster_
                    break
        if booster is None:
            raise RuntimeError("No LightGBM booster available for feature importance")
        gains = booster.feature_importance(importance_type="gain")
        return (
            pd.DataFrame(
                {"feature": self.feature_names_, "importance": gains.astype(float)}
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_fitted(self) -> None:
        if not self.fitted_ or self.model_ is None:
            raise RuntimeError("EPLMachineLearningModel is not fitted; call fit() first.")
