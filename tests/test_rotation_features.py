"""Unit tests for rest_days / density / rotation risk / tier down-weight."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features import (
    TIER_DOWNWEIGHT,
    calculate_multi_comp_features,
    get_opponent_adjusted_rolling_features,
    is_early_cup_round,
    is_rotation_risk,
    match_tier_scale,
)
from src.global_db import connect_global_db, import_legacy_matches_df, resolve_team_id
from src.league_registry import clear_leagues_cache, is_cup_competition


@pytest.fixture(autouse=True)
def _clear_league_cache() -> None:
    clear_leagues_cache()
    yield
    clear_leagues_cache()


def _busy_cup_frame() -> pd.DataFrame:
    """Four matches in 14 days for Kobe, then an Emperor's Cup kickoff."""
    rows = [
        {
            "Date": "2026-09-08",
            "HomeTeam": "JP_VISSEL_KOBE",
            "AwayTeam": "JP_URAWA",
            "FTHG": 1,
            "FTAG": 0,
            "Match_ID": "j1_1",
            "HS": 10,
            "HST": 4,
            "AS": 5,
            "AST": 1,
        },
        {
            "Date": "2026-09-11",
            "HomeTeam": "JP_KASHIMA",
            "AwayTeam": "JP_VISSEL_KOBE",
            "FTHG": 0,
            "FTAG": 2,
            "Match_ID": "j1_2",
            "HS": 6,
            "HST": 2,
            "AS": 11,
            "AST": 5,
        },
        {
            "Date": "2026-09-14",
            "HomeTeam": "JP_VISSEL_KOBE",
            "AwayTeam": "JP_KYOTO_SANGYO",
            "FTHG": 5,
            "FTAG": 0,
            "Match_ID": "cup_early",
            "HS": 18,
            "HST": 9,
            "AS": 2,
            "AST": 0,
            "Round": "2nd Round",
        },
        {
            "Date": "2026-09-17",
            "HomeTeam": "JP_SAGAN_TOSU",
            "AwayTeam": "JP_VISSEL_KOBE",
            "FTHG": 1,
            "FTAG": 1,
            "Match_ID": "j1_3",
            "HS": 8,
            "HST": 3,
            "AS": 9,
            "AST": 3,
        },
    ]
    return pd.DataFrame(rows)


def test_is_cup_competition_emperors() -> None:
    assert is_cup_competition("EMPERORS_CUP") is True
    assert is_cup_competition("EPL") is False
    assert is_cup_competition("J1") is False


def test_is_early_cup_round() -> None:
    assert is_early_cup_round("2nd Round") is True
    assert is_early_cup_round(None) is True
    assert is_early_cup_round("Semi-finals") is False
    assert is_early_cup_round("Final") is False


def test_rest_days_and_matches_last_14d() -> None:
    conn = connect_global_db(":memory:", init=True)
    # Seed J1 then cup crush then J1 again under different comps.
    frame = _busy_cup_frame()
    import_legacy_matches_df(conn, frame.iloc[:2], comp_id="J1")
    import_legacy_matches_df(conn, frame.iloc[[2]], comp_id="EMPERORS_CUP")
    import_legacy_matches_df(conn, frame.iloc[[3]], comp_id="J1")

    tid = resolve_team_id(conn, "JP_VISSEL_KOBE", comp_id="J1")
    assert tid is not None

    kick = pd.Timestamp("2026-09-21")
    feats = calculate_multi_comp_features(
        tid,
        kick,
        conn=conn,
        upcoming_comp_id="EMPERORS_CUP",
    )
    assert feats["rest_days"] == 4.0  # 21 - 17
    assert feats["matches_last_14d"] == 4.0
    assert feats["is_rotation_risk"] == 1.0
    conn.close()


def test_is_rotation_risk_requires_cup_and_busy() -> None:
    assert is_rotation_risk(4, "EMPERORS_CUP") is True
    assert is_rotation_risk(3, "EMPERORS_CUP") is False
    assert is_rotation_risk(5, "EPL") is False
    assert is_rotation_risk(5, None) is False


def test_tier_scale_0_8_early_cup_vs_amateur() -> None:
    conn = connect_global_db(":memory:", init=True)
    frame = _busy_cup_frame()
    import_legacy_matches_df(conn, frame.iloc[:2], comp_id="J1")
    import_legacy_matches_df(conn, frame.iloc[[2]], comp_id="EMPERORS_CUP")
    import_legacy_matches_df(conn, frame.iloc[[3]], comp_id="J1")

    legacy = pd.read_sql(
        """
        SELECT
            m.match_date AS Date,
            m.comp_id,
            m.round AS Round,
            c.league_weight,
            th.canonical_name AS HomeTeam,
            ta.canonical_name AS AwayTeam,
            m.home_team_id,
            m.away_team_id,
            m.home_score AS FTHG,
            m.away_score AS FTAG,
            m.HS, m.AS_ AS "AS", m.HST, m.AST
        FROM matches m
        JOIN teams th ON th.team_id = m.home_team_id
        JOIN teams ta ON ta.team_id = m.away_team_id
        LEFT JOIN competitions c ON c.comp_id = m.comp_id
        """,
        conn,
    )
    legacy["Date"] = pd.to_datetime(legacy["Date"])
    kobe_id = int(
        legacy.loc[legacy["HomeTeam"] == "JP_VISSEL_KOBE", "home_team_id"].iloc[0]
    )
    cup_row = legacy.loc[legacy["comp_id"] == "EMPERORS_CUP"].iloc[0]
    scale = match_tier_scale(cup_row, kobe_id, top_flight_team_ids={kobe_id})
    assert scale == pytest.approx(TIER_DOWNWEIGHT)

    # Unweighted mean of last-5 xG would treat the 5-0 university win at full value;
    # adjusted rolling must be strictly lower than unscaled mean of the same window.
    adjusted = get_opponent_adjusted_rolling_features(
        kobe_id,
        n_matches=5,
        match_date="2026-09-21",
        matches_df=legacy,
    )
    # Build unscaled mean of xG for comparison.
    from src.features import _xg_for_team, _mean_last

    raw: list[float] = []
    for _, row in legacy.sort_values("Date").iterrows():
        if int(row["home_team_id"]) != kobe_id and int(row["away_team_id"]) != kobe_id:
            continue
        xg = _xg_for_team(row, kobe_id)
        if xg == xg:
            raw.append(float(xg))
    unscaled = _mean_last(raw, 5)
    assert adjusted["rolling_xg_5"] == adjusted["rolling_xg_5"]  # not NaN
    assert adjusted["rolling_xg_5"] < unscaled
    assert adjusted["mean_tier_scale"] < 1.0
    conn.close()


def test_epl_path_no_rotation_without_cup() -> None:
    """EPL busy schedule must not flip is_rotation_risk."""
    conn = connect_global_db(":memory:", init=True)
    opponents = ["Chelsea", "Liverpool", "Everton", "Tottenham"]
    rows = []
    for i, day in enumerate(["2026-09-08", "2026-09-11", "2026-09-14", "2026-09-17"]):
        rows.append(
            {
                "Date": day,
                "HomeTeam": "Arsenal" if i % 2 == 0 else opponents[i],
                "AwayTeam": opponents[i] if i % 2 == 0 else "Arsenal",
                "FTHG": 1,
                "FTAG": 0,
                "Match_ID": f"e{i}",
                "HS": 8,
                "HST": 3,
                "AS": 6,
                "AST": 2,
            }
        )
    import_legacy_matches_df(conn, pd.DataFrame(rows), comp_id="EPL")
    tid = resolve_team_id(conn, "Arsenal", comp_id="EPL")
    assert tid is not None
    feats = calculate_multi_comp_features(
        tid, "2026-09-21", conn=conn, upcoming_comp_id="EPL"
    )
    assert feats["matches_last_14d"] == 4.0
    assert feats["is_rotation_risk"] == 0.0
    conn.close()


def test_dixon_coles_time_weights_tz_naive() -> None:
    """Timezone-aware Date must not break decay_factor = exp(-xi * t)."""
    from src.dixon_coles import DixonColesModel

    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2026-01-01", "2026-06-01", "2026-09-01"]
            ).tz_localize("Asia/Tokyo"),
            "HomeTeam": ["A", "A", "B"],
            "AwayTeam": ["B", "B", "A"],
            "FTHG": [1, 2, 0],
            "FTAG": [0, 1, 1],
        }
    )
    model = DixonColesModel(xi=0.0018)
    w = model._time_weights(df)
    assert len(w) == 3
    assert w[-1] == pytest.approx(1.0)
    assert w[0] < w[1] < w[2]
    # Same weights as naive civil dates (no UTC day-shift).
    df_naive = df.copy()
    df_naive["Date"] = pd.to_datetime(["2026-01-01", "2026-06-01", "2026-09-01"])
    w2 = model._time_weights(df_naive)
    assert w == pytest.approx(w2)
