"""Unit tests — shot/corner columns, rolling features, CornerPredictor."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.corner_model import CornerModel, CornerPredictor
from src.data_loader import (
    STAT_COLUMNS,
    clean_matches,
    read_matches_from_db,
    save_matches_to_db,
)
from src.features import (
    add_match_xg_proxy,
    engineer_rolling_features,
    xg_proxy,
)


def _synthetic_matches(n: int = 60) -> pd.DataFrame:
    """Build a tiny chronological EPL-like table with shots + corners."""
    rng = np.random.default_rng(42)
    teams = [f"Team{i}" for i in range(6)]
    rows = []
    day0 = pd.Timestamp("2024-08-01")
    for i in range(n):
        home, away = teams[i % 6], teams[(i + 1 + i // 6) % 6]
        if home == away:
            away = teams[(i + 2) % 6]
        hs, ash = int(rng.integers(5, 20)), int(rng.integers(5, 18))
        hst, ast = int(rng.integers(1, hs + 1)), int(rng.integers(1, ash + 1))
        hc, ac = int(rng.integers(2, 12)), int(rng.integers(2, 11))
        fthg, ftag = int(rng.integers(0, 4)), int(rng.integers(0, 4))
        ftr = "H" if fthg > ftag else ("A" if fthg < ftag else "D")
        rows.append(
            {
                "Date": day0 + pd.Timedelta(days=i),
                "HomeTeam": home,
                "AwayTeam": away,
                "FTHG": fthg,
                "FTAG": ftag,
                "FTR": ftr,
                "HS": hs,
                "AS": ash,
                "HST": hst,
                "AST": ast,
                "HC": hc,
                "AC": ac,
                "Season": "2024/25",
                "SeasonStart": 2024,
            }
        )
    return pd.DataFrame(rows)


def test_xg_proxy_formula() -> None:
    assert xg_proxy(10, 4) == pytest.approx(0.1 * 10 + 0.3 * 4)
    assert xg_proxy(0, 0) == 0.0


def test_clean_matches_keeps_shot_and_corner_cols() -> None:
    raw = _synthetic_matches(12)
    # Simulate football-data raw (string dates)
    raw["Date"] = raw["Date"].dt.strftime("%d/%m/%Y")
    cleaned = clean_matches(raw)
    for col in STAT_COLUMNS:
        assert col in cleaned.columns, f"missing {col}"
        assert cleaned[col].notna().all()


def test_sqlite_persists_stat_columns(tmp_path: Path) -> None:
    db = tmp_path / "epl_matches.db"
    data = clean_matches(
        _synthetic_matches(20).assign(
            Date=lambda d: d["Date"].dt.strftime("%d/%m/%Y")
        )
    )
    n = save_matches_to_db(data, db)
    assert n >= 20

    with sqlite3.connect(db) as conn:
        cols = {
            r[1]
            for r in conn.execute("PRAGMA table_info(matches)").fetchall()
        }
    for col in STAT_COLUMNS:
        assert col in cols

    loaded = read_matches_from_db(db)
    assert not loaded.empty
    for col in STAT_COLUMNS:
        assert col in loaded.columns
        assert loaded[col].notna().sum() > 0


def test_rolling_features_no_lookahead() -> None:
    data = _synthetic_matches(40)
    feat = engineer_rolling_features(data, windows=(5, 10), min_prior_matches=0)
    assert "rolling_sot_home" in feat.columns
    assert "rolling_corners_away" in feat.columns
    assert "corner_total_avg" in feat.columns

    # Debut rows are imputed with league means (LightGBM-safe); values are
    # finite and must not equal the *current* match's own SOT (no lookahead).
    first = feat.iloc[0]
    assert np.isfinite(first["rolling_sot_home"])
    if "HST" in data.columns and pd.notna(data.iloc[0]["HST"]):
        assert float(first["rolling_sot_home"]) != float(data.iloc[0]["HST"])

    # After enough games, rolling means are finite.
    late = feat.dropna(subset=["rolling_sot_home", "rolling_corners_home"]).tail(5)
    assert not late.empty
    assert (late["corner_total_avg"] >= 0).all()


def test_add_match_xg_proxy() -> None:
    df = add_match_xg_proxy(_synthetic_matches(5))
    assert "home_xg_proxy" in df.columns
    row = df.iloc[0]
    assert row["home_xg_proxy"] == pytest.approx(
        0.1 * float(row["HS"]) + 0.3 * float(row["HST"])
    )


def test_corner_predictor_fit_ou_and_ev() -> None:
    data = _synthetic_matches(80)
    pred = CornerPredictor(backend="poisson", min_prior_matches=1).fit(data)
    teams = pred.teams
    home, away = teams[0], teams[1]
    rates = pred.expected_corners(home, away)
    assert rates["total"] == pytest.approx(rates["hc"] + rates["ac"])
    assert rates["hc"] > 0 and rates["ac"] > 0

    ou = pred.predict_over_under(home, away, line=10.5)
    assert 0.0 <= ou["over"] <= 1.0
    assert 0.0 <= ou["under"] <= 1.0
    assert abs(ou["over"] + ou["under"] - 1.0) < 1e-6

    lines = pred.predict_lines(home, away)
    assert set(lines["line"]) >= {9.5, 10.5, 11.5}

    ah = pred.predict_handicap(home, away, handicap=-0.5)
    assert abs(ah["home"] + ah["away"] - 1.0) < 1e-6

    # Soft book → at least one positive-ish EV possible
    legs = pred.predict_corner_ev(
        home,
        away,
        {"over": 2.40, "under": 2.40},
        line=10.5,
        min_ev=-1.0,
    )
    assert len(legs) == 2
    assert all("ev_pct" in x for x in legs)

    trend = pred.trend(home, last_n=5)
    assert len(trend) <= 5
    assert {"corners_for", "corners_against", "venue"} <= set(trend.columns)


def test_legacy_corner_model_still_fits() -> None:
    data = _synthetic_matches(50)
    model = CornerModel().fit(data)
    home, away = model.teams[0], model.teams[1]
    exp = model.expected_corners(home, away)
    assert exp["total"] > 0
