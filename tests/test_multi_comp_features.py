"""Tests for multi-comp rest/fatigue features and Lite enrichment."""

from __future__ import annotations

import pandas as pd

from src.features import (
    calculate_multi_comp_features,
    enrich_bets_with_multi_comp_features,
    format_fatigue_label,
    format_team_fatigue_phrase,
)
from src.global_db import (
    apply_league_weight_to_rates,
    connect_global_db,
    import_legacy_matches_df,
)
from src.models import LeagueWeightDixonColesProxy


def _legacy_frame() -> pd.DataFrame:
    """Tiny cross-comp history for team_ids after import."""
    rows = [
        # Arsenal played 3 matches in 10 days, last on 2026-09-18
        {
            "Date": "2026-09-10",
            "HomeTeam": "Arsenal",
            "AwayTeam": "Chelsea",
            "FTHG": 2,
            "FTAG": 1,
            "Match_ID": "e1",
            "HS": 10,
            "HST": 4,
            "AS": 8,
            "AST": 3,
        },
        {
            "Date": "2026-09-14",
            "HomeTeam": "Liverpool",
            "AwayTeam": "Arsenal",
            "FTHG": 0,
            "FTAG": 1,
            "Match_ID": "e2",
            "HS": 7,
            "HST": 2,
            "AS": 9,
            "AST": 5,
        },
        {
            "Date": "2026-09-18",
            "HomeTeam": "Arsenal",
            "AwayTeam": "Everton",
            "FTHG": 3,
            "FTAG": 0,
            "Match_ID": "e3",
            "HS": 12,
            "HST": 6,
            "AS": 4,
            "AST": 1,
        },
        # Chelsea last played earlier → more rest
        {
            "Date": "2026-09-05",
            "HomeTeam": "Chelsea",
            "AwayTeam": "Everton",
            "FTHG": 1,
            "FTAG": 1,
            "Match_ID": "e0",
            "HS": 8,
            "HST": 3,
            "AS": 6,
            "AST": 2,
        },
    ]
    return pd.DataFrame(rows)


def test_calculate_multi_comp_features_rest_and_busy() -> None:
    conn = connect_global_db(":memory:", init=True)
    import_legacy_matches_df(conn, _legacy_frame(), comp_id="EPL")
    legacy = pd.read_sql(
        """
        SELECT
            m.match_date AS Date,
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
        """,
        conn,
    )
    legacy["Date"] = pd.to_datetime(legacy["Date"])
    arsenal_id = int(
        legacy.loc[legacy["HomeTeam"] == "Arsenal", "home_team_id"].iloc[0]
    )
    chelsea_id = int(
        legacy.loc[legacy["AwayTeam"] == "Chelsea", "away_team_id"].iloc[0]
    )

    kick = pd.Timestamp("2026-09-21")
    a = calculate_multi_comp_features(arsenal_id, kick, legacy, lookback_days=14)
    c = calculate_multi_comp_features(chelsea_id, kick, legacy, lookback_days=14)

    assert a["rest_days"] == 3.0  # 21 - 18
    assert a["matches_last_14d"] == 3.0
    assert c["rest_days"] == 11.0  # 21 - 10 (vs Arsenal) — wait Chelsea played 9/10 and 9/5
    # Chelsea's last match is 2026-09-10 vs Arsenal
    assert c["rest_days"] == 11.0
    assert c["matches_last_14d"] == 1.0

    label = format_fatigue_label("Arsenal", a, "Chelsea", c, lookback_days=14)
    assert "Arsenal cày 3 trận/14 ngày" in label
    assert "Chelsea nghỉ 11 ngày" in label
    conn.close()


def test_format_team_fatigue_phrase_fallbacks() -> None:
    assert "nghỉ 3 ngày" in format_team_fatigue_phrase(
        "A", {"rest_days": 3, "matches_last_14d": 1}
    )
    assert "vừa đá J1" in format_team_fatigue_phrase(
        "Machida",
        {
            "rest_days": 3,
            "matches_last_14d": 1,
            "last_comp_id": "J1",
            "last_match_date": "2026-09-20",
        },
    )
    assert "20/09" in format_team_fatigue_phrase(
        "Machida",
        {
            "rest_days": 3,
            "matches_last_14d": 1,
            "last_comp_id": "J1",
            "last_match_date": "2026-09-20",
        },
    )
    assert "thiếu lịch" in format_team_fatigue_phrase("B", None)
    # Long rest (off-season / capped) → never print raw huge integers.
    assert "14+ ngày" in format_team_fatigue_phrase(
        "C", {"rest_days": 14, "matches_last_14d": 0}
    )
    assert "14+ ngày" in format_team_fatigue_phrase(
        "D", {"rest_days": 288, "matches_last_14d": 0}
    )
    assert "288" not in format_team_fatigue_phrase(
        "D", {"rest_days": 288, "matches_last_14d": 0}
    )


def test_rest_days_hard_cap_off_season() -> None:
    """Gaps > 30 days clamp to 14 so Lite never shows e.g. 288."""
    from src.features import calculate_multi_comp_features
    from src.global_db import connect_global_db, import_legacy_matches_df, resolve_team_id

    conn = connect_global_db(":memory:", init=True)
    import_legacy_matches_df(
        conn,
        pd.DataFrame(
            [
                {
                    "Date": "2025-05-01",
                    "HomeTeam": "Arsenal",
                    "AwayTeam": "Chelsea",
                    "FTHG": 1,
                    "FTAG": 0,
                    "Match_ID": "old1",
                }
            ]
        ),
        comp_id="EPL",
    )
    tid = resolve_team_id(conn, "Arsenal", comp_id="EPL")
    assert tid is not None
    # ~288 days later (new season kickoff)
    feats = calculate_multi_comp_features(tid, "2026-02-14", conn=conn)
    assert feats["rest_days"] == 14.0
    assert "14+ ngày" in format_team_fatigue_phrase("Arsenal", feats)
    conn.close()


def test_enrich_bets_noop_without_global() -> None:
    bets = pd.DataFrame(
        [
            {
                "home": "Arsenal",
                "away": "Chelsea",
                "kickoff": "2026-09-21",
                "league": "EPL",
                "ev": 0.1,
            }
        ]
    )
    out = enrich_bets_with_multi_comp_features(bets, None)
    assert "fatigue_label" in out.columns
    assert "thiếu lịch" in str(out.iloc[0]["fatigue_label"])


def test_apply_league_weight_rates() -> None:
    lam, mu = apply_league_weight_to_rates(2.0, 1.0, 0.9, 0.85, w_ref=1.0)
    assert abs(lam - 1.8) < 1e-9
    assert abs(mu - 0.85) < 1e-9


def test_league_weight_proxy_scales_expected_goals() -> None:
    class _Inner:
        fitted_ = True
        max_goals = 5
        rho = 0.0

        def expected_goals(self, home: str, away: str) -> tuple[float, float]:
            return 2.0, 1.0

    proxy = LeagueWeightDixonColesProxy(_Inner(), lambda _t: 0.5, w_ref=1.0)  # type: ignore[arg-type]
    lam, mu = proxy.expected_goals("A", "B")
    assert abs(lam - 1.0) < 1e-9
    assert abs(mu - 0.5) < 1e-9
    probs = proxy.predict_match_probs("A", "B")
    assert abs(sum(probs.values()) - 1.0) < 1e-6
