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

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

DEFAULT_ROLL_WINDOWS: tuple[int, ...] = (5, 10)

# ---------------------------------------------------------------------------
# League-tier scaling for rolling attack metrics (multi-comp history)
# ---------------------------------------------------------------------------
# When a recent match is lower-tier / early-round cup vs weak opposition,
# scale that match's contribution by TIER_DOWNWEIGHT (0.8 = −20%) so crushing
# amateurs / early-cup minnows does not inflate xG / goals form.
TIER_DOWNWEIGHT: float = 0.8
# Competitions treated as top-flight for "weak opponent" checks.
TOP_FLIGHT_COMP_IDS: frozenset[str] = frozenset(
    {"EPL", "J1", "LALIGA", "WSL", "BUNDESLIGA", "SERIE_A", "LIGUE_1"}
)
# Match ``league_weight`` below this → candidate for down-weight (J2≈0.75).
LOW_TIER_WEIGHT_THRESHOLD: float = 0.85
# Always down-weight these comps regardless of opponent.
ALWAYS_DOWNWEIGHT_COMPS: frozenset[str] = frozenset(
    {"FRIENDLY", "FLASH_TEAM", "J3"}
)
# Busy schedule threshold for cup rotation risk.
ROTATION_BUSY_MATCHES: int = 4

_AMATEUR_OPP_RE = re.compile(
    r"university|univ\.?|\bsangyo\b|college|high\s*school|amateur|"
    r"\bacademy\b|\bu\d{2}\b",
    re.IGNORECASE,
)
_LATE_CUP_ROUND_RE = re.compile(
    r"final|semi|quarter|round\s*of\s*16|1/8|1/4|1/2|\br16\b|\bqf\b|\bsf\b",
    re.IGNORECASE,
)

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


# ---------------------------------------------------------------------------
# Multi-competition features (cross-league schedule / form)
# ---------------------------------------------------------------------------


def _team_match_history_rows(
    matches_df: pd.DataFrame,
    team_id: int,
    *,
    before: pd.Timestamp,
) -> pd.DataFrame:
    """Rows involving ``team_id`` strictly before ``before``, sorted by date."""
    required = {"Date", "home_team_id", "away_team_id"}
    missing = required - set(matches_df.columns)
    if missing:
        raise ValueError(
            f"matches_df missing {sorted(missing)} "
            "(need global-style ids; use read_matches_as_legacy)"
        )
    df = matches_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    mask = (
        (df["home_team_id"] == int(team_id)) | (df["away_team_id"] == int(team_id))
    ) & (df["Date"] < before)
    out = df.loc[mask].sort_values("Date")
    return out


def _goals_for_team(row: pd.Series, team_id: int) -> float:
    """Goals scored by ``team_id`` in one match row (FTHG/FTAG or home/away_score)."""
    hid = int(row["home_team_id"])
    if "FTHG" in row.index and "FTAG" in row.index:
        home_g, away_g = row["FTHG"], row["FTAG"]
    else:
        home_g, away_g = row.get("home_score"), row.get("away_score")
    if hid == int(team_id):
        val = home_g
    else:
        val = away_g
    if val is None or (isinstance(val, float) and val != val) or pd.isna(val):
        return float("nan")
    return float(val)


def _xg_for_team(row: pd.Series, team_id: int) -> float:
    """Per-team xG for a row; falls back to goals as xG proxy."""
    hid = int(row["home_team_id"])
    is_home = hid == int(team_id)
    if is_home and "home_xg_proxy" in row.index and pd.notna(row.get("home_xg_proxy")):
        return float(row["home_xg_proxy"])
    if (not is_home) and "away_xg_proxy" in row.index and pd.notna(
        row.get("away_xg_proxy")
    ):
        return float(row["away_xg_proxy"])
    # Shot-based proxy when HS/HST present.
    if is_home and "HS" in row.index and "HST" in row.index:
        if pd.notna(row.get("HS")) or pd.notna(row.get("HST")):
            return xg_proxy(row.get("HS"), row.get("HST"))
    if (not is_home) and "AS" in row.index and "AST" in row.index:
        if pd.notna(row.get("AS")) or pd.notna(row.get("AST")):
            return xg_proxy(row.get("AS"), row.get("AST"))
    # Goals as xG proxy (UWCL / sparse stats).
    return _goals_for_team(row, team_id)


# Soft display cap for Lite fatigue labels (avoid off-season gaps like "288 ngày").
REST_DAYS_HARD_CAP: int = 30
REST_DAYS_DISPLAY_PLUS: int = 14


def get_rest_days(
    team_id: int | str,
    upcoming_date: Any,
    *,
    db_path: Path | str | None = None,
) -> float:
    """Instant rest_days from SQLite only — **zero** web requests.

    Uses ``teams.last_match_date`` when set; otherwise falls back to
    ``MAX(matches.match_date)`` for that ``team_id``. Applies the same
    hard-cap as :func:`calculate_multi_comp_features` (``>30`` → ``14``).
    Returns ``NaN`` when no prior match date is known.
    """
    from src.global_db import (
        GLOBAL_DB_PATH,
        connect_global_db,
        get_team_feed_meta,
        resolve_team_id,
    )

    try:
        upcoming = pd.Timestamp(upcoming_date).normalize()
    except (TypeError, ValueError):
        return float("nan")

    path = Path(db_path) if db_path is not None else GLOBAL_DB_PATH
    if str(path) != ":memory:" and not Path(path).is_file():
        return float("nan")

    try:
        conn = connect_global_db(path, init=True)
        try:
            if isinstance(team_id, int) or str(team_id).strip().isdigit():
                tid = int(team_id)
            else:
                tid = resolve_team_id(conn, str(team_id), gender="M", create=False)
                if tid is None:
                    tid = resolve_team_id(conn, str(team_id), gender="W", create=False)
            if tid is None:
                return float("nan")
            meta = get_team_feed_meta(conn, tid)
            last_raw = (meta or {}).get("last_match_date") if meta else None
            if not last_raw:
                row = conn.execute(
                    """
                    SELECT MAX(match_date) AS last_d
                    FROM matches
                    WHERE (home_team_id = ? OR away_team_id = ?)
                      AND home_score IS NOT NULL
                      AND away_score IS NOT NULL
                    """,
                    (tid, tid),
                ).fetchone()
                last_raw = row["last_d"] if row else None
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        return float("nan")

    if not last_raw:
        return float("nan")
    try:
        last = pd.Timestamp(last_raw).normalize()
    except (TypeError, ValueError):
        return float("nan")

    rest_days = float((upcoming - last).days)
    if rest_days < 0:
        return float("nan")
    if rest_days > float(REST_DAYS_HARD_CAP):
        return float(REST_DAYS_DISPLAY_PLUS)
    return rest_days


def attach_rest_days_from_db(
    bets: pd.DataFrame,
    *,
    db_path: Path | str | None = None,
) -> pd.DataFrame:
    """Attach ``home_rest_days`` / ``away_rest_days`` via :func:`get_rest_days`.

    Streamlit-safe: no network I/O. Resolves team names → DB ids when possible.
    """
    if bets is None or getattr(bets, "empty", True):
        return bets if bets is not None else pd.DataFrame()

    from src.global_db import (
        GLOBAL_DB_PATH,
        competition_gender,
        connect_global_db,
        resolve_team_id,
    )

    path = Path(db_path) if db_path is not None else GLOBAL_DB_PATH
    out = bets.copy()
    n = len(out)
    home_rest = [float("nan")] * n
    away_rest = [float("nan")] * n

    # Prefer precomputed upcoming cache columns when present.
    if "home_rest_days" in out.columns and "away_rest_days" in out.columns:
        for i, (_, bet) in enumerate(out.iterrows()):
            try:
                hr = float(bet["home_rest_days"])
                if hr == hr:
                    home_rest[i] = hr
            except (TypeError, ValueError):
                pass
            try:
                ar = float(bet["away_rest_days"])
                if ar == ar:
                    away_rest[i] = ar
            except (TypeError, ValueError):
                pass
        if all(v == v for v in home_rest) and all(v == v for v in away_rest):
            out["home_rest_days"] = home_rest
            out["away_rest_days"] = away_rest
            if "fatigue_label" not in out.columns:
                labels = []
                for i, (_, bet) in enumerate(out.iterrows()):
                    home = str(bet.get("home") or bet.get("home_team") or "")
                    away = str(bet.get("away") or bet.get("away_team") or "")
                    labels.append(
                        format_fatigue_label(
                            home,
                            {"rest_days": home_rest[i], "matches_last_14d": 0},
                            away,
                            {"rest_days": away_rest[i], "matches_last_14d": 0},
                        )
                    )
                out["fatigue_label"] = labels
            return out

    conn = None
    try:
        if str(path) == ":memory:" or Path(path).is_file():
            conn = connect_global_db(path, init=True)
    except Exception:  # noqa: BLE001
        conn = None

    try:
        for i, (_, bet) in enumerate(out.iterrows()):
            if home_rest[i] == home_rest[i] and away_rest[i] == away_rest[i]:
                continue
            home = str(bet.get("home") or bet.get("home_team") or "")
            away = str(bet.get("away") or bet.get("away_team") or "")
            kick = bet.get("kickoff") or bet.get("match_date") or bet.get("Date")
            if kick is None or (isinstance(kick, float) and pd.isna(kick)):
                continue
            comp = str(
                bet.get("league") or bet.get("comp_id") or bet.get("league_id") or ""
            ).strip().upper() or None
            g = competition_gender(comp) if comp else "M"
            hid = aid = None
            if conn is not None:
                try:
                    hid = resolve_team_id(conn, home, gender=g, comp_id=comp, create=False)
                    aid = resolve_team_id(conn, away, gender=g, comp_id=comp, create=False)
                except Exception:  # noqa: BLE001
                    hid = aid = None
            if hid is not None and home_rest[i] != home_rest[i]:
                home_rest[i] = get_rest_days(hid, kick, db_path=path)
            if aid is not None and away_rest[i] != away_rest[i]:
                away_rest[i] = get_rest_days(aid, kick, db_path=path)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    out["home_rest_days"] = home_rest
    out["away_rest_days"] = away_rest
    labels = []
    for i, (_, bet) in enumerate(out.iterrows()):
        home = str(bet.get("home") or bet.get("home_team") or "")
        away = str(bet.get("away") or bet.get("away_team") or "")
        labels.append(
            format_fatigue_label(
                home,
                {"rest_days": home_rest[i], "matches_last_14d": 0}
                if home_rest[i] == home_rest[i]
                else None,
                away,
                {"rest_days": away_rest[i], "matches_last_14d": 0}
                if away_rest[i] == away_rest[i]
                else None,
            )
        )
    out["fatigue_label"] = labels
    return out


def is_early_cup_round(round_val: Any) -> bool:
    """True for early national-cup rounds (or unknown round → conservative).

    Late markers (R16 / QF / SF / Final) return False. Missing / blank Round
    is treated as early so Emperor's Cup early ties vs amateurs down-weight.
    """
    if round_val is None or (isinstance(round_val, float) and round_val != round_val):
        return True
    if isinstance(round_val, float) and pd.isna(round_val):
        return True
    text = str(round_val).strip()
    if not text:
        return True
    if _LATE_CUP_ROUND_RE.search(text):
        return False
    return True


def _comp_id_from_row(row: Mapping[str, Any] | pd.Series) -> str:
    for key in ("comp_id", "league_id", "league"):
        try:
            val = row[key] if key in row else None  # type: ignore[operator]
        except Exception:  # noqa: BLE001
            val = None
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        text = str(val).strip()
        if text:
            return text.upper()
    return ""


def _league_weight_for_comp(comp_id: str) -> float:
    """Best-effort ``league_weight`` from registry / global defaults."""
    code = str(comp_id or "").strip().upper()
    if not code:
        return 1.0
    try:
        from src.league_registry import get_league_entry, resolve_league_config

        entry = get_league_entry(code)
        if entry is not None and entry.get("league_weight") is not None:
            return float(entry["league_weight"])
        try:
            _c, cfg = resolve_league_config(code)
            return float(cfg.get("league_weight") or 1.0)
        except ValueError:
            pass
    except Exception:  # noqa: BLE001
        pass
    # Mirror src.global_db.DEFAULT_COMPETITIONS when registry misses team-feed comps.
    defaults = {
        "EPL": 1.0,
        "UWCL": 0.85,
        "LALIGA": 0.95,
        "EMPERORS_CUP": 0.8,
        "J1": 0.9,
        "J2": 0.75,
        "J3": 0.65,
        "ACL": 0.95,
        "J_LEAGUE_CUP": 0.7,
        "FRIENDLY": 0.3,
        "FLASH_TEAM": 0.5,
    }
    return float(defaults.get(code, 1.0))


def _opponent_name_from_row(row: pd.Series, team_id: int) -> str:
    hid = int(row["home_team_id"])
    if hid == int(team_id):
        for key in ("AwayTeam", "away_team", "away"):
            if key in row.index and pd.notna(row.get(key)):
                return str(row[key])
        return ""
    for key in ("HomeTeam", "home_team", "home"):
        if key in row.index and pd.notna(row.get(key)):
            return str(row[key])
    return ""


def _opponent_id_from_row(row: pd.Series, team_id: int) -> int | None:
    try:
        hid = int(row["home_team_id"])
        aid = int(row["away_team_id"])
    except (TypeError, ValueError, KeyError):
        return None
    if hid == int(team_id):
        return aid
    if aid == int(team_id):
        return hid
    return None


def _is_amateur_opponent_name(name: str) -> bool:
    text = str(name or "").strip()
    if not text:
        return False
    if _AMATEUR_OPP_RE.search(text):
        return True
    # Stable codes for known university / lower sides in JP aliases.
    upper = text.upper().replace(" ", "_")
    return upper in {"JP_KYOTO_SANGYO"} or "SANGYO" in upper


def match_tier_scale(
    row: pd.Series,
    team_id: int,
    *,
    top_flight_team_ids: set[int] | None = None,
) -> float:
    """Return ``1.0`` or ``TIER_DOWNWEIGHT`` (0.8) for one historical match.

    Tier rules (any → 0.8)
    ----------------------
    1. Competition in ``ALWAYS_DOWNWEIGHT_COMPS`` (friendly / J3 / unmapped).
    2. ``league_weight < LOW_TIER_WEIGHT_THRESHOLD`` (e.g. J2 0.75) **and**
       cup or amateur/non-top-flight opponent.
    3. National cup (``EMPERORS_CUP`` / ``is_cup``) that is early-round
       **or** vs opponent not in the top-flight id set / amateur name.
    4. Opponent name looks like university / amateur.

    Top-flight league matches (EPL / J1 / LaLiga) vs professional sides keep
    scale ``1.0`` so EPL/UWCL paths are unchanged.
    """
    from src.league_registry import is_cup_competition

    comp = _comp_id_from_row(row)
    weight = float(row["league_weight"]) if "league_weight" in row.index and pd.notna(
        row.get("league_weight")
    ) else _league_weight_for_comp(comp)
    opp_name = _opponent_name_from_row(row, int(team_id))
    opp_id = _opponent_id_from_row(row, int(team_id))
    amateur = _is_amateur_opponent_name(opp_name)
    in_top = False
    if top_flight_team_ids is not None and opp_id is not None:
        in_top = int(opp_id) in top_flight_team_ids
    elif opp_id is None and not amateur:
        # No id map → only amateur names / low-weight comps trigger.
        in_top = True

    if comp in ALWAYS_DOWNWEIGHT_COMPS:
        return float(TIER_DOWNWEIGHT)
    if amateur:
        return float(TIER_DOWNWEIGHT)

    cup = is_cup_competition(comp)
    early = is_early_cup_round(row.get("Round") if "Round" in row.index else None)
    weak_opp = amateur or (top_flight_team_ids is not None and not in_top)

    if cup and (early or weak_opp):
        return float(TIER_DOWNWEIGHT)
    if weight < float(LOW_TIER_WEIGHT_THRESHOLD) and (cup or weak_opp or not in_top):
        return float(TIER_DOWNWEIGHT)
    return 1.0


def _top_flight_team_ids(matches_df: pd.DataFrame) -> set[int]:
    """Team ids that have appeared in a top-flight competition in ``matches_df``."""
    if matches_df is None or matches_df.empty:
        return set()
    if "comp_id" not in matches_df.columns and "league_id" not in matches_df.columns:
        return set()
    comp_col = "comp_id" if "comp_id" in matches_df.columns else "league_id"
    mask = matches_df[comp_col].astype(str).str.upper().isin(TOP_FLIGHT_COMP_IDS)
    ids: set[int] = set()
    for col in ("home_team_id", "away_team_id"):
        if col not in matches_df.columns:
            continue
        for val in matches_df.loc[mask, col].dropna():
            try:
                ids.add(int(val))
            except (TypeError, ValueError):
                continue
    return ids


def get_opponent_adjusted_rolling_features(
    team_id: int,
    n_matches: int = 5,
    *,
    match_date: Any | None = None,
    matches_df: pd.DataFrame | None = None,
    conn: Any | None = None,
) -> dict[str, float]:
    """Rolling xG / goals over last ``n_matches`` with tier down-weighting.

    Each prior match contributes ``value * scale`` where ``scale`` is
    :data:`TIER_DOWNWEIGHT` (0.8) for lower-tier / early-cup vs weak sides,
    else ``1.0``. The rolling mean is ``sum(value*scale) / n`` over the last
    N scored matches (same window length as an unweighted mean).

    See :func:`match_tier_scale` docstring for tier rules.

    Returns
    -------
    dict
        ``rolling_xg_5``, ``rolling_gf_5``, ``n_used``, ``mean_tier_scale``.
    """
    before = _as_ts(match_date) if match_date is not None else None
    hist: pd.DataFrame
    if matches_df is None:
        if conn is None:
            raise ValueError("Provide matches_df or conn")
        sql = """
            SELECT
                m.match_date AS Date,
                m.comp_id,
                m.round AS Round,
                c.league_weight,
                m.home_team_id,
                m.away_team_id,
                th.canonical_name AS HomeTeam,
                ta.canonical_name AS AwayTeam,
                m.home_score AS FTHG,
                m.away_score AS FTAG,
                m.HS, m.AS_ AS "AS", m.HST, m.AST
            FROM matches m
            JOIN teams th ON th.team_id = m.home_team_id
            JOIN teams ta ON ta.team_id = m.away_team_id
            LEFT JOIN competitions c ON c.comp_id = m.comp_id
            WHERE (m.home_team_id = ? OR m.away_team_id = ?)
        """
        params: list[Any] = [int(team_id), int(team_id)]
        if before is not None:
            sql += " AND m.match_date < ?"
            params.append(before.strftime("%Y-%m-%d"))
        sql += " ORDER BY m.match_date"
        hist = pd.read_sql(sql, conn, params=params)
        if not hist.empty:
            hist["Date"] = pd.to_datetime(hist["Date"], errors="coerce")
        top_ids = _top_flight_team_ids(hist)
    else:
        if before is None:
            before = _as_ts(matches_df["Date"].max()) + pd.Timedelta(days=1)
        hist = _team_match_history_rows(matches_df, int(team_id), before=before)
        top_ids = _top_flight_team_ids(matches_df)

    empty = {
        "rolling_xg_5": float("nan"),
        "rolling_gf_5": float("nan"),
        "n_used": 0.0,
        "mean_tier_scale": float("nan"),
    }
    if hist is None or hist.empty:
        return empty

    xg_scaled: list[float] = []
    gf_scaled: list[float] = []
    scales: list[float] = []
    for _, row in hist.iterrows():
        xg = _xg_for_team(row, int(team_id))
        gf = _goals_for_team(row, int(team_id))
        if xg != xg and gf != gf:
            continue
        scale = match_tier_scale(row, int(team_id), top_flight_team_ids=top_ids)
        scales.append(scale)
        if xg == xg:
            xg_scaled.append(float(xg) * scale)
        if gf == gf:
            gf_scaled.append(float(gf) * scale)

    n = max(1, int(n_matches))
    return {
        "rolling_xg_5": _mean_last(xg_scaled, n),
        "rolling_gf_5": _mean_last(gf_scaled, n),
        "n_used": float(min(n, max(len(xg_scaled), len(gf_scaled)))),
        "mean_tier_scale": float(np.mean(scales[-n:])) if scales else float("nan"),
    }


def is_rotation_risk(
    matches_last_14d: float,
    upcoming_comp_id: str | None,
    *,
    busy_threshold: int = ROTATION_BUSY_MATCHES,
) -> bool:
    """True when congested (≥4 in 14d) **and** next match is a national cup."""
    from src.league_registry import is_cup_competition

    try:
        n = float(matches_last_14d)
    except (TypeError, ValueError):
        n = 0.0
    return n >= float(busy_threshold) and is_cup_competition(upcoming_comp_id)


def calculate_multi_comp_features(
    team_id: int,
    match_date: Any,
    matches_df: pd.DataFrame | None = None,
    conn: Any | None = None,
    *,
    rolling_n: int = 5,
    lookback_days: int = 14,
    upcoming_comp_id: str | None = None,
) -> dict[str, float]:
    """Cross-competition form features for one team before a kick-off.

    Aggregates **all competitions of the same gender-scoped** ``team_id``
    with ``Date < match_date``. Men's and women's brands never share an id,
    so ``rest_days`` / ``matches_last_14d`` cannot leak across Nam/Nữ.

    Parameters
    ----------
    team_id:
        Global ``teams.team_id``.
    match_date:
        Kick-off / match day (exclusive upper bound).
    matches_df:
        Legacy-projected frame with ``home_team_id`` / ``away_team_id``
        (from :func:`src.global_db.read_matches_as_legacy`). Preferred.
    conn:
        Optional open SQLite connection to ``global_matches.db`` used when
        ``matches_df`` is None.
    rolling_n:
        Window for ``rolling_xg_5`` (opponent-adjusted mean of last N).
    lookback_days:
        Window for ``matches_last_14d``.
    upcoming_comp_id:
        Competition of the *next* fixture (for ``is_rotation_risk``).
        Cup + busy schedule → 1.0; otherwise 0.0. EPL/UWCL callers may omit.

    Returns
    -------
    dict
        ``rest_days``, ``matches_last_14d``, ``rolling_xg_5``,
        ``rolling_gf_5``, ``is_rotation_risk`` (0/1).
        Missing history → ``rest_days`` / rolling as NaN, counts 0.
        ``rest_days`` > ``REST_DAYS_HARD_CAP`` (off-season / sparse data) is
        clamped to ``REST_DAYS_DISPLAY_PLUS`` so Lite cards never show
        absurd values like ``288``.
    """
    before = _as_ts(match_date)

    if matches_df is None:
        if conn is None:
            raise ValueError("Provide matches_df or conn")
        sql = """
            SELECT
                m.match_date AS Date,
                m.comp_id,
                m.round AS Round,
                c.league_weight,
                m.home_team_id,
                m.away_team_id,
                th.canonical_name AS HomeTeam,
                ta.canonical_name AS AwayTeam,
                m.home_score AS FTHG,
                m.away_score AS FTAG,
                m.HS, m.AS_ AS "AS", m.HST, m.AST
            FROM matches m
            JOIN teams th ON th.team_id = m.home_team_id
            JOIN teams ta ON ta.team_id = m.away_team_id
            LEFT JOIN competitions c ON c.comp_id = m.comp_id
            WHERE (m.home_team_id = ? OR m.away_team_id = ?)
              AND m.match_date < ?
            ORDER BY m.match_date
        """
        hist = pd.read_sql(
            sql,
            conn,
            params=(int(team_id), int(team_id), before.strftime("%Y-%m-%d")),
        )
        if not hist.empty:
            hist["Date"] = pd.to_datetime(hist["Date"], errors="coerce")
        full_for_top = hist
    else:
        hist = _team_match_history_rows(matches_df, int(team_id), before=before)
        full_for_top = matches_df

    empty = {
        "rest_days": float("nan"),
        "matches_last_14d": 0.0,
        "rolling_xg_5": float("nan"),
        "rolling_gf_5": float("nan"),
        "is_rotation_risk": 0.0,
        "last_match_date": "",
        "last_comp_id": "",
    }
    if hist is None or hist.empty:
        return empty

    last_row = hist.iloc[-1]
    last_date = _as_ts(last_row["Date"])
    rest_days = float((before - last_date).days)
    # Off-season / missing recent fixtures → soft default (never show 100+ days).
    if rest_days > float(REST_DAYS_HARD_CAP):
        rest_days = float(REST_DAYS_DISPLAY_PLUS)

    cutoff = before - pd.Timedelta(days=int(lookback_days))
    # Compare on normalized dates so tz-aware kickoffs still count correctly.
    hist_dates = hist["Date"].map(
        lambda d: _as_ts(d) if pd.notna(d) else pd.NaT
    )
    recent = hist.loc[hist_dates >= cutoff]
    matches_last = float(len(recent))

    adjusted = get_opponent_adjusted_rolling_features(
        int(team_id),
        n_matches=int(rolling_n),
        match_date=before,
        matches_df=full_for_top if matches_df is not None else hist,
        conn=None if matches_df is not None else conn,
    )

    rot = is_rotation_risk(matches_last, upcoming_comp_id)
    last_comp = _comp_id_from_row(last_row)

    return {
        "rest_days": rest_days,
        "matches_last_14d": matches_last,
        "rolling_xg_5": float(adjusted["rolling_xg_5"]),
        "rolling_gf_5": float(adjusted["rolling_gf_5"]),
        "is_rotation_risk": 1.0 if rot else 0.0,
        "last_match_date": last_date.strftime("%Y-%m-%d"),
        "last_comp_id": last_comp,
    }


def format_team_fatigue_phrase(
    team_name: str,
    feats: Mapping[str, Any] | None,
    *,
    lookback_days: int = 14,
    busy_threshold: int = 3,
) -> str:
    """One-side fatigue blurb for Lite cards.

    Busy schedule (``matches_last_* >= busy_threshold``) →
    ``\"{team} cày N trận/{lookback} ngày\"``.
    Long rest (``rest_days >= REST_DAYS_DISPLAY_PLUS``) →
    ``\"{team} nghỉ 14+ ngày\"`` (never ``288 ngày``).
    Otherwise when ``rest_days`` is known → ``\"{team} nghỉ N ngày\"``,
    optionally with ``(vừa đá J1 … 20/09)`` when ``last_comp_id`` /
    ``last_match_date`` are present.
    """
    name = str(team_name or "").strip() or "?"
    if not feats:
        return f"{name} (thiếu lịch)"
    try:
        n_busy = float(feats.get("matches_last_14d") or 0.0)
    except (TypeError, ValueError):
        n_busy = 0.0
    if n_busy >= float(busy_threshold):
        base = f"{name} cày {int(n_busy)} trận/{int(lookback_days)} ngày"
        return f"{base}{_last_match_paren(feats)}"
    rest = feats.get("rest_days")
    try:
        rest_f = float(rest) if rest is not None else float("nan")
    except (TypeError, ValueError):
        rest_f = float("nan")
    if rest_f == rest_f:  # not NaN
        if rest_f >= float(REST_DAYS_DISPLAY_PLUS):
            return f"{name} nghỉ {int(REST_DAYS_DISPLAY_PLUS)}+ ngày"
        return f"{name} nghỉ {int(rest_f)} ngày{_last_match_paren(feats)}"
    return f"{name} (thiếu lịch)"


def _last_match_paren(feats: Mapping[str, Any]) -> str:
    """Optional `` (vừa đá J1 … 20/09)`` suffix from last match metadata."""
    last_comp = str(feats.get("last_comp_id") or "").strip().upper()
    last_date = str(feats.get("last_match_date") or "").strip()
    if not last_comp and not last_date:
        return ""
    bits: list[str] = []
    if last_comp:
        bits.append(last_comp)
    if last_date:
        try:
            ts = pd.Timestamp(last_date)
            bits.append(ts.strftime("%d/%m"))
        except Exception:  # noqa: BLE001
            bits.append(last_date)
    if not bits:
        return ""
    if len(bits) == 2:
        return f" (vừa đá {bits[0]} … {bits[1]})"
    return f" (vừa đá {bits[0]})"


def format_fatigue_label(
    home_name: str,
    home_feats: Mapping[str, Any] | None,
    away_name: str,
    away_feats: Mapping[str, Any] | None,
    *,
    lookback_days: int = 14,
    busy_threshold: int = 3,
) -> str:
    """Lite card fatigue line: ``\"A nghỉ 3 ngày / B cày 3 trận/14 ngày\"``."""
    left = format_team_fatigue_phrase(
        home_name,
        home_feats,
        lookback_days=lookback_days,
        busy_threshold=busy_threshold,
    )
    right = format_team_fatigue_phrase(
        away_name,
        away_feats,
        lookback_days=lookback_days,
        busy_threshold=busy_threshold,
    )
    return f"{left} / {right}"


def enrich_bets_with_multi_comp_features(
    bets: pd.DataFrame,
    matches_df: pd.DataFrame | None,
    *,
    lookback_days: int = 14,
    busy_threshold: int = 3,
    refresh_team_feeds: bool = False,
    team_feed_n_matches: int = 10,
) -> pd.DataFrame:
    """Attach rest/fatigue columns to scanner display rows when global history exists.

    Adds ``home_rest_days``, ``away_rest_days``, ``home_matches_last_14d``,
    ``away_matches_last_14d``, ``home_rolling_xg_5``, ``away_rolling_xg_5``,
    ``home_is_rotation_risk``, ``away_is_rotation_risk``, and ``fatigue_label``.
    When ``matches_df`` is empty/None, sets empty fatigue labels and leaves
    numeric columns as NaN / 0 — safe no-op path.

    Parameters
    ----------
    refresh_team_feeds:
        When True, scrape Flashscore team-results pages for fixture sides that
        have a registered hash (``JP_*`` in ``config/leagues.json``
        ``_team_hashes``) and upsert into ``global_matches.db`` before
        computing rest_days. No-op for EPL/UWCL sides without hashes.
    """
    if bets is None or getattr(bets, "empty", True):
        return bets if bets is not None else pd.DataFrame()

    work_matches = matches_df
    if refresh_team_feeds:
        work_matches = _maybe_refresh_team_feeds_for_bets(
            bets, matches_df, n_matches=int(team_feed_n_matches)
        )

    out = bets.copy()
    n = len(out)
    home_rest = [float("nan")] * n
    away_rest = [float("nan")] * n
    home_n14 = [0.0] * n
    away_n14 = [0.0] * n
    home_xg = [float("nan")] * n
    away_xg = [float("nan")] * n
    home_rot = [0.0] * n
    away_rot = [0.0] * n
    labels = [""] * n

    # Instant path: prefer precomputed rest columns (upcoming_fixtures cache).
    has_cached_rest = (
        "home_rest_days" in out.columns and "away_rest_days" in out.columns
    )
    if has_cached_rest:
        for i, (_, bet) in enumerate(out.iterrows()):
            try:
                hr = float(bet["home_rest_days"])
                if hr == hr:
                    home_rest[i] = hr
            except (TypeError, ValueError):
                pass
            try:
                ar = float(bet["away_rest_days"])
                if ar == ar:
                    away_rest[i] = ar
            except (TypeError, ValueError):
                pass
            if "home_matches_last_14d" in out.columns:
                try:
                    home_n14[i] = float(bet["home_matches_last_14d"] or 0)
                except (TypeError, ValueError):
                    pass
            if "away_matches_last_14d" in out.columns:
                try:
                    away_n14[i] = float(bet["away_matches_last_14d"] or 0)
                except (TypeError, ValueError):
                    pass
            # Still build fatigue labels from cached rest when present.
            h_feats = (
                {"rest_days": home_rest[i], "matches_last_14d": home_n14[i]}
                if home_rest[i] == home_rest[i]
                else None
            )
            a_feats = (
                {"rest_days": away_rest[i], "matches_last_14d": away_n14[i]}
                if away_rest[i] == away_rest[i]
                else None
            )
            home = str(bet.get("home") or bet.get("home_team") or "")
            away = str(bet.get("away") or bet.get("away_team") or "")
            labels[i] = format_fatigue_label(
                home,
                h_feats,
                away,
                a_feats,
                lookback_days=lookback_days,
                busy_threshold=busy_threshold,
            )
        # If every row had cached rest, skip expensive history join.
        if all(v == v for v in home_rest) and all(v == v for v in away_rest):
            out["home_rest_days"] = home_rest
            out["away_rest_days"] = away_rest
            out["home_matches_last_14d"] = home_n14
            out["away_matches_last_14d"] = away_n14
            if "home_rolling_xg_5" not in out.columns:
                out["home_rolling_xg_5"] = home_xg
            if "away_rolling_xg_5" not in out.columns:
                out["away_rolling_xg_5"] = away_xg
            if "home_is_rotation_risk" not in out.columns:
                out["home_is_rotation_risk"] = home_rot
            if "away_is_rotation_risk" not in out.columns:
                out["away_is_rotation_risk"] = away_rot
            out["fatigue_label"] = labels
            return out

    has_ids = (
        work_matches is not None
        and not work_matches.empty
        and {"Date", "HomeTeam", "AwayTeam", "home_team_id", "away_team_id"}.issubset(
            work_matches.columns
        )
    )

    # (canonical_name, gender) → team_id — never collapse Nam/Nữ brands.
    name_gender_to_id: dict[tuple[str, str], int] = {}
    if has_ids:
        assert work_matches is not None
        try:
            from src.data_loader import normalize_team_name
            from src.global_db import competition_gender
        except Exception:  # noqa: BLE001

            def normalize_team_name(n: str, *_a: Any, **_k: Any) -> str:  # type: ignore[misc]
                return str(n)

            def competition_gender(c: str | None) -> str:  # type: ignore[misc]
                return "W" if str(c or "").upper() in {"UWCL", "WSL"} else "M"

        for _, row in work_matches.iterrows():
            try:
                g = competition_gender(
                    str(row.get("comp_id") or row.get("league_id") or "")
                )
                if "home_gender" in row.index and pd.notna(row.get("home_gender")):
                    g = str(row["home_gender"]).strip().upper()[:1] or g
                h_name = normalize_team_name(str(row["HomeTeam"]))
                a_name = normalize_team_name(str(row["AwayTeam"]))
                name_gender_to_id[(h_name, g)] = int(row["home_team_id"])
                name_gender_to_id[(a_name, g)] = int(row["away_team_id"])
                name_gender_to_id[(str(row["HomeTeam"]), g)] = int(row["home_team_id"])
                name_gender_to_id[(str(row["AwayTeam"]), g)] = int(row["away_team_id"])
            except (TypeError, ValueError, KeyError):
                continue

    rest_conn = None
    try:
        from src.global_db import GLOBAL_DB_PATH, connect_global_db, get_team_rest_cache

        if Path(GLOBAL_DB_PATH).is_file():
            rest_conn = connect_global_db(GLOBAL_DB_PATH, init=True)
    except Exception:  # noqa: BLE001
        rest_conn = None

    try:
        for i, (_, bet) in enumerate(out.iterrows()):
            home = str(bet.get("home") or bet.get("home_team") or "")
            away = str(bet.get("away") or bet.get("away_team") or "")
            kick = bet.get("kickoff") or bet.get("match_date") or bet.get("Date")
            # Keep pre-filled cache values when history lookup fails.
            cached_h = home_rest[i]
            cached_a = away_rest[i]
            h_feats: dict[str, float] | None = None
            a_feats: dict[str, float] | None = None
            if has_ids and kick is not None and not (isinstance(kick, float) and pd.isna(kick)):
                try:
                    from src.data_loader import normalize_team_name as _norm
                    from src.global_db import competition_gender as _comp_g
                except Exception:  # noqa: BLE001
                    def _norm(n: str, *_a: Any, **_k: Any) -> str:
                        return str(n)

                    def _comp_g(c: str | None) -> str:
                        return "W" if str(c or "").upper() in {"UWCL", "WSL"} else "M"

                bet_g = _comp_g(str(bet.get("league") or bet.get("comp_id") or ""))
                upcoming_comp = str(
                    bet.get("league") or bet.get("comp_id") or bet.get("league_id") or ""
                ).strip().upper() or None
                hid = name_gender_to_id.get((home, bet_g)) or name_gender_to_id.get(
                    (_norm(home), bet_g)
                )
                aid = name_gender_to_id.get((away, bet_g)) or name_gender_to_id.get(
                    (_norm(away), bet_g)
                )
                if rest_conn is not None:
                    try:
                        if hid is not None and cached_h != cached_h:
                            c = get_team_rest_cache(
                                rest_conn, hid, kick, max_age_minutes=180
                            )
                            if c is not None:
                                h_feats = {
                                    "rest_days": float(c["rest_days"]),
                                    "matches_last_14d": float(c["matches_last_14d"]),
                                    "rolling_xg_5": float("nan"),
                                    "is_rotation_risk": 0.0,
                                }
                        if aid is not None and cached_a != cached_a:
                            c = get_team_rest_cache(
                                rest_conn, aid, kick, max_age_minutes=180
                            )
                            if c is not None:
                                a_feats = {
                                    "rest_days": float(c["rest_days"]),
                                    "matches_last_14d": float(c["matches_last_14d"]),
                                    "rolling_xg_5": float("nan"),
                                    "is_rotation_risk": 0.0,
                                }
                    except Exception:  # noqa: BLE001
                        pass
                try:
                    if hid is not None and h_feats is None:
                        h_feats = calculate_multi_comp_features(
                            hid,
                            kick,
                            work_matches,
                            lookback_days=lookback_days,
                            upcoming_comp_id=upcoming_comp,
                        )
                    if aid is not None and a_feats is None:
                        a_feats = calculate_multi_comp_features(
                            aid,
                            kick,
                            work_matches,
                            lookback_days=lookback_days,
                            upcoming_comp_id=upcoming_comp,
                        )
                except (ValueError, TypeError, KeyError):
                    pass

            if h_feats:
                home_rest[i] = float(h_feats["rest_days"])
                home_n14[i] = float(h_feats["matches_last_14d"])
                home_xg[i] = float(h_feats.get("rolling_xg_5") or float("nan"))
                home_rot[i] = float(h_feats.get("is_rotation_risk") or 0.0)
            elif cached_h == cached_h:
                home_rest[i] = cached_h
            if a_feats:
                away_rest[i] = float(a_feats["rest_days"])
                away_n14[i] = float(a_feats["matches_last_14d"])
                away_xg[i] = float(a_feats.get("rolling_xg_5") or float("nan"))
                away_rot[i] = float(a_feats.get("is_rotation_risk") or 0.0)
            elif cached_a == cached_a:
                away_rest[i] = cached_a

            labels[i] = format_fatigue_label(
                home,
                h_feats
                or (
                    {"rest_days": home_rest[i], "matches_last_14d": home_n14[i]}
                    if home_rest[i] == home_rest[i]
                    else None
                ),
                away,
                a_feats
                or (
                    {"rest_days": away_rest[i], "matches_last_14d": away_n14[i]}
                    if away_rest[i] == away_rest[i]
                    else None
                ),
                lookback_days=lookback_days,
                busy_threshold=busy_threshold,
            )
    finally:
        if rest_conn is not None:
            try:
                rest_conn.close()
            except Exception:  # noqa: BLE001
                pass

    out["home_rest_days"] = home_rest
    out["away_rest_days"] = away_rest
    out["home_matches_last_14d"] = home_n14
    out["away_matches_last_14d"] = away_n14
    out["home_rolling_xg_5"] = home_xg
    out["away_rolling_xg_5"] = away_xg
    out["home_is_rotation_risk"] = home_rot
    out["away_is_rotation_risk"] = away_rot
    out["fatigue_label"] = labels
    return out


def _maybe_refresh_team_feeds_for_bets(
    bets: pd.DataFrame,
    matches_df: pd.DataFrame | None,
    *,
    n_matches: int = 10,
) -> pd.DataFrame | None:
    """Refresh Flashscore team feeds for known JP_* sides; reload global history."""
    try:
        from src.fetchers.flashscore_team import (
            ensure_jp_aliases_registered,
            hash_for_team_id,
            refresh_team_feeds_for_sides,
            team_hash_registry,
        )
        from src.global_db import GLOBAL_DB_PATH, read_matches_as_legacy
    except Exception:  # noqa: BLE001
        return matches_df

    ensure_jp_aliases_registered()
    registry = team_hash_registry()
    # Collect stable codes from bets (JP_* or aliased display names).
    codes: set[str] = set()
    try:
        from src.fetchers.flashscore_league import resolve_league_team_name
    except Exception:  # noqa: BLE001
        resolve_league_team_name = None  # type: ignore[assignment]

    for _, bet in bets.iterrows():
        for key in ("home", "home_team", "away", "away_team"):
            raw = str(bet.get(key) or "").strip()
            if not raw:
                continue
            if raw.upper() in registry or hash_for_team_id(raw):
                codes.add(raw.upper())
                continue
            if resolve_league_team_name is not None:
                mapped = resolve_league_team_name(raw, "EMPERORS_CUP")
                if mapped in registry or hash_for_team_id(mapped):
                    codes.add(mapped)

    if not codes:
        return matches_df

    try:
        refresh_team_feeds_for_sides(sorted(codes), n_matches=n_matches)
        return read_matches_as_legacy(GLOBAL_DB_PATH)
    except Exception:  # noqa: BLE001
        return matches_df


def calculate_rest_days_for_fixture(
    home_team: str,
    away_team: str,
    match_date: Any,
    *,
    matches_df: pd.DataFrame | None = None,
    refresh_team_feeds: bool = True,
    n_matches: int = 10,
    lookback_days: int = 14,
) -> dict[str, Any]:
    """Compute rest / busy for one upcoming fixture (multi-comp JP_* aware).

    Optionally refreshes Flashscore team-results feeds for sides with known
    hashes, then reads ``global_matches.db``. EPL/UWCL paths without hashes are
    unchanged (uses ``matches_df`` / existing DB rows only).
    """
    from src.fetchers.flashscore_league import resolve_league_team_name
    from src.fetchers.flashscore_team import (
        ensure_jp_aliases_registered,
        hash_for_team_id,
        refresh_team_feeds_for_sides,
    )
    from src.global_db import (
        GLOBAL_DB_PATH,
        competition_gender,
        connect_global_db,
        read_matches_as_legacy,
        resolve_team_id,
    )

    ensure_jp_aliases_registered()
    home = resolve_league_team_name(str(home_team), "EMPERORS_CUP")
    away = resolve_league_team_name(str(away_team), "EMPERORS_CUP")
    # Fall back to raw when not a JP alias target (EPL names).
    if not str(home_team).startswith("JP_") and not home.startswith("JP_"):
        home = str(home_team)
    if not str(away_team).startswith("JP_") and not away.startswith("JP_"):
        away = str(away_team)

    if refresh_team_feeds:
        sides = [t for t in (home, away) if hash_for_team_id(t)]
        if sides:
            refresh_team_feeds_for_sides(sides, n_matches=n_matches)

    hist = matches_df
    if hist is None or refresh_team_feeds:
        try:
            hist = read_matches_as_legacy(GLOBAL_DB_PATH)
        except Exception:  # noqa: BLE001
            if hist is None:
                hist = pd.DataFrame()

    # Resolve numeric team ids (men's comps for JP / default).
    conn = connect_global_db(GLOBAL_DB_PATH, init=True)
    try:
        # Prefer EMPERORS_CUP / J1 gender (men).
        hid = resolve_team_id(conn, home, comp_id="EMPERORS_CUP", create=False)
        aid = resolve_team_id(conn, away, comp_id="EMPERORS_CUP", create=False)
        if hid is None:
            hid = resolve_team_id(conn, home, gender="M", create=False)
        if aid is None:
            aid = resolve_team_id(conn, away, gender="M", create=False)
    finally:
        conn.close()

    h_feats = (
        calculate_multi_comp_features(
            hid,
            match_date,
            hist,
            lookback_days=lookback_days,
            upcoming_comp_id="EMPERORS_CUP",
        )
        if hid is not None and hist is not None and not hist.empty
        else {
            "rest_days": float("nan"),
            "matches_last_14d": 0.0,
            "rolling_xg_5": float("nan"),
            "rolling_gf_5": float("nan"),
            "is_rotation_risk": 0.0,
            "last_match_date": "",
            "last_comp_id": "",
        }
    )
    a_feats = (
        calculate_multi_comp_features(
            aid,
            match_date,
            hist,
            lookback_days=lookback_days,
            upcoming_comp_id="EMPERORS_CUP",
        )
        if aid is not None and hist is not None and not hist.empty
        else {
            "rest_days": float("nan"),
            "matches_last_14d": 0.0,
            "rolling_xg_5": float("nan"),
            "rolling_gf_5": float("nan"),
            "is_rotation_risk": 0.0,
            "last_match_date": "",
            "last_comp_id": "",
        }
    )
    label = format_fatigue_label(home, h_feats, away, a_feats, lookback_days=lookback_days)
    return {
        "home": home,
        "away": away,
        "match_date": str(pd.Timestamp(match_date).date()),
        "home_team_id": hid,
        "away_team_id": aid,
        "home_feats": h_feats,
        "away_feats": a_feats,
        "fatigue_label": label,
        "gender_scope": competition_gender("EMPERORS_CUP"),
    }