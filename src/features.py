r"""Feature engineering for AH / OU / Corners markets.

Builds leakage-safe rolling shot & corner stats plus a simple ``xG_proxy``
used by :class:`src.corner_model.CornerPredictor` and downstream scanners.

Formulas
--------
``xG_proxy = 0.1 · Shots + 0.3 · ShotsOnTarget``

Rolling windows default to the last **5** and **10** matches per team
(chronological, no look-ahead).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import pandas as pd

DEFAULT_ROLL_WINDOWS: tuple[int, ...] = (5, 10)

# Per-team history keys accumulated while walking matches chronologically.
_HIST_KEYS: tuple[str, ...] = ("sot", "shots", "corners", "xg")


def xg_proxy(shots: float, shots_on_target: float) -> float:
    """Approximate expected goals from shot volume.

    Parameters
    ----------
    shots:
        Total shots (HS or AS).
    shots_on_target:
        Shots on target (HST or AST).

    Returns
    -------
    float
        ``0.1 * shots + 0.3 * shots_on_target`` (NaN-safe → 0 contribution).
    """
    s = 0.0 if shots != shots or shots is None else float(shots)  # NaN check
    sot = (
        0.0
        if shots_on_target != shots_on_target or shots_on_target is None
        else float(shots_on_target)
    )
    return 0.1 * s + 0.3 * sot


def _as_ts(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError(f"Invalid date: {value!r}")
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()


def _mean_last(values: Sequence[float], n: int) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values[-n:]))


def add_match_xg_proxy(matches: pd.DataFrame) -> pd.DataFrame:
    """Add ``home_xg_proxy`` / ``away_xg_proxy`` from HS/HST & AS/AST.

    Missing shot columns yield NaN proxies (not zeros) so callers can dropna.
    """
    out = matches.copy()
    has_home = "HS" in out.columns and "HST" in out.columns
    has_away = "AS" in out.columns and "AST" in out.columns

    if has_home:
        hs = pd.to_numeric(out["HS"], errors="coerce")
        hst = pd.to_numeric(out["HST"], errors="coerce")
        out["home_xg_proxy"] = 0.1 * hs + 0.3 * hst
    else:
        out["home_xg_proxy"] = np.nan

    if has_away:
        ash = pd.to_numeric(out["AS"], errors="coerce")
        ast = pd.to_numeric(out["AST"], errors="coerce")
        out["away_xg_proxy"] = 0.1 * ash + 0.3 * ast
    else:
        out["away_xg_proxy"] = np.nan

    return out


def engineer_rolling_features(
    matches: pd.DataFrame,
    *,
    windows: Sequence[int] = DEFAULT_ROLL_WINDOWS,
    min_prior_matches: int = 0,
) -> pd.DataFrame:
    """Leakage-safe rolling SOT / corners / xG_proxy for every match.

    For each fixture the home/away rolling stats use **only prior** games of
    that team. After features are written, the current match is appended to
    each team's history.

    Output columns (per window ``n`` in ``windows`` when ``n`` is the primary
    window used by callers; also always emits the user-facing aliases for the
    **largest** window and for window=5 when present):

    - ``rolling_sot_home``, ``rolling_sot_away``
    - ``rolling_corners_home``, ``rolling_corners_away``
    - ``corner_total_avg`` (= home + away rolling corners)
    - ``rolling_xg_home``, ``rolling_xg_away``
    - Plus windowed variants ``rolling_sot_home_r{n}`` etc.
    """
    required = {"Date", "HomeTeam", "AwayTeam"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"matches missing columns: {sorted(missing)}")

    df = matches.dropna(subset=list(required)).copy()
    df["Date"] = df["Date"].map(_as_ts)
    df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
    df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()

    for col in ("HS", "AS", "HST", "AST", "HC", "AC"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    df = add_match_xg_proxy(df)
    df = df.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

    wins = tuple(int(w) for w in windows)
    primary = 5 if 5 in wins else wins[0]
    history: dict[str, list[dict[str, float]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        home, away = str(row["HomeTeam"]), str(row["AwayTeam"])
        hh, ah = history[home], history[away]

        feat: dict[str, Any] = {
            "Date": row["Date"],
            "HomeTeam": home,
            "AwayTeam": away,
        }
        if "Match_ID" in df.columns and pd.notna(row.get("Match_ID")):
            feat["Match_ID"] = row["Match_ID"]

        for n in wins:
            feat[f"rolling_sot_home_r{n}"] = _mean_last([x["sot"] for x in hh], n)
            feat[f"rolling_sot_away_r{n}"] = _mean_last([x["sot"] for x in ah], n)
            feat[f"rolling_corners_home_r{n}"] = _mean_last(
                [x["corners"] for x in hh], n
            )
            feat[f"rolling_corners_away_r{n}"] = _mean_last(
                [x["corners"] for x in ah], n
            )
            feat[f"rolling_xg_home_r{n}"] = _mean_last([x["xg"] for x in hh], n)
            feat[f"rolling_xg_away_r{n}"] = _mean_last([x["xg"] for x in ah], n)
            feat[f"corner_total_avg_r{n}"] = (
                feat[f"rolling_corners_home_r{n}"] + feat[f"rolling_corners_away_r{n}"]
            )

        # User-facing aliases (window=5 preferred, else first window).
        feat["rolling_sot_home"] = feat[f"rolling_sot_home_r{primary}"]
        feat["rolling_sot_away"] = feat[f"rolling_sot_away_r{primary}"]
        feat["rolling_corners_home"] = feat[f"rolling_corners_home_r{primary}"]
        feat["rolling_corners_away"] = feat[f"rolling_corners_away_r{primary}"]
        feat["corner_total_avg"] = feat[f"corner_total_avg_r{primary}"]
        feat["rolling_xg_home"] = feat[f"rolling_xg_home_r{primary}"]
        feat["rolling_xg_away"] = feat[f"rolling_xg_away_r{primary}"]

        # Targets / raw (for training; not used as features for *this* row).
        feat["HC"] = row["HC"] if pd.notna(row["HC"]) else np.nan
        feat["AC"] = row["AC"] if pd.notna(row["AC"]) else np.nan
        feat["home_xg_proxy"] = row["home_xg_proxy"]
        feat["away_xg_proxy"] = row["away_xg_proxy"]

        n_prior = min(len(hh), len(ah))
        if n_prior >= int(min_prior_matches):
            rows.append(feat)

        # Append current match to history (after features → no leakage).
        home_sot = float(row["HST"]) if pd.notna(row["HST"]) else float("nan")
        away_sot = float(row["AST"]) if pd.notna(row["AST"]) else float("nan")
        home_shots = float(row["HS"]) if pd.notna(row["HS"]) else float("nan")
        away_shots = float(row["AS"]) if pd.notna(row["AS"]) else float("nan")
        home_c = float(row["HC"]) if pd.notna(row["HC"]) else float("nan")
        away_c = float(row["AC"]) if pd.notna(row["AC"]) else float("nan")
        home_xg = (
            float(row["home_xg_proxy"])
            if pd.notna(row["home_xg_proxy"])
            else xg_proxy(home_shots, home_sot)
        )
        away_xg = (
            float(row["away_xg_proxy"])
            if pd.notna(row["away_xg_proxy"])
            else xg_proxy(away_shots, away_sot)
        )

        def _ok(v: float) -> bool:
            return v == v  # not NaN

        if _ok(home_sot) or _ok(home_c) or _ok(home_xg):
            history[home].append(
                {
                    "sot": home_sot if _ok(home_sot) else 0.0,
                    "shots": home_shots if _ok(home_shots) else 0.0,
                    "corners": home_c if _ok(home_c) else 0.0,
                    "xg": home_xg if _ok(home_xg) else 0.0,
                }
            )
        if _ok(away_sot) or _ok(away_c) or _ok(away_xg):
            history[away].append(
                {
                    "sot": away_sot if _ok(away_sot) else 0.0,
                    "shots": away_shots if _ok(away_shots) else 0.0,
                    "corners": away_c if _ok(away_c) else 0.0,
                    "xg": away_xg if _ok(away_xg) else 0.0,
                }
            )

    return impute_rolling_with_league_mean(pd.DataFrame(rows))


def impute_rolling_with_league_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN rolling / xG feature columns with column-wise league means.

    Uses ``df.fillna(df.mean(numeric_only=True))`` on rolling_* / *_xg_* /
    corner_total_avg* columns so LightGBM / corner regressors never see NaN
    for debut clubs.
    """
    if df.empty:
        return df.copy()
    out = df.copy()
    feature_cols = [
        c
        for c in out.columns
        if str(c).startswith("rolling_")
        or str(c).startswith("corner_total_avg")
        or str(c) in {"rolling_sot_home", "rolling_sot_away",
                      "rolling_corners_home", "rolling_corners_away",
                      "corner_total_avg", "rolling_xg_home", "rolling_xg_away"}
    ]
    if not feature_cols:
        # Fallback: any numeric column that isn't an ID / target
        skip = {
            "HC",
            "AC",
            "FTHG",
            "FTAG",
            "home_xg_proxy",
            "away_xg_proxy",
            "SeasonStart",
        }
        feature_cols = [
            c
            for c in out.select_dtypes(include=["number", "float", "int"]).columns
            if c not in skip
        ]
    if not feature_cols:
        return out
    means = out[feature_cols].mean(numeric_only=True)
    out[feature_cols] = out[feature_cols].fillna(means)
    # Any remaining NaN (all-NaN columns) → 0
    out[feature_cols] = out[feature_cols].fillna(0.0)
    return out


def feature_matrix_for_corners(
    engineered: pd.DataFrame,
    *,
    windows: Sequence[int] = DEFAULT_ROLL_WINDOWS,
) -> tuple[pd.DataFrame, list[str]]:
    """Select numeric feature columns for corner regressors.

    Returns
    -------
    X, feature_names
    """
    cols: list[str] = []
    for n in windows:
        cols.extend(
            [
                f"rolling_sot_home_r{n}",
                f"rolling_sot_away_r{n}",
                f"rolling_corners_home_r{n}",
                f"rolling_corners_away_r{n}",
                f"rolling_xg_home_r{n}",
                f"rolling_xg_away_r{n}",
                f"corner_total_avg_r{n}",
            ]
        )
    # Deduplicate while preserving order
    seen: set[str] = set()
    names = [c for c in cols if c in engineered.columns and not (c in seen or seen.add(c))]
    X = engineered[names].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.mean(numeric_only=True)).fillna(0.0)
    return X, names


def team_corner_trend(
    matches: pd.DataFrame,
    team: str,
    *,
    last_n: int = 10,
) -> pd.DataFrame:
    """Recent corner-for / against series for UI trend charts.

    Returns a DataFrame with columns ``Date``, ``corners_for``,
    ``corners_against``, ``venue`` (H/A).
    """
    df = matches.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ("HC", "AC"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            return pd.DataFrame(
                columns=["Date", "corners_for", "corners_against", "venue"]
            )

    home = df.loc[df["HomeTeam"] == team, ["Date", "HC", "AC"]].copy()
    home["corners_for"] = home["HC"]
    home["corners_against"] = home["AC"]
    home["venue"] = "H"

    away = df.loc[df["AwayTeam"] == team, ["Date", "HC", "AC"]].copy()
    away["corners_for"] = away["AC"]
    away["corners_against"] = away["HC"]
    away["venue"] = "A"

    out = pd.concat(
        [
            home[["Date", "corners_for", "corners_against", "venue"]],
            away[["Date", "corners_for", "corners_against", "venue"]],
        ],
        ignore_index=True,
    )
    out = out.dropna(subset=["Date"]).sort_values("Date").tail(int(last_n))
    return out.reset_index(drop=True)
