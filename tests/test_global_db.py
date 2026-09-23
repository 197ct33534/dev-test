"""Tests for global multi-competition DB, features, and league weights."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import normalize_team_name
from src.dixon_coles import DixonColesModel
from src.features import calculate_multi_comp_features
from src.global_db import (
    apply_league_weight_to_rates,
    build_global_match_id,
    canonical_team_name,
    connect_global_db,
    get_competition_weight,
    import_legacy_matches_df,
    read_matches_as_legacy,
    resolve_team_id,
    upsert_team,
)
from src.models import expected_goals_multi_comp


@pytest.fixture()
def mem_db():
    conn = connect_global_db(":memory:", init=True)
    yield conn
    conn.close()


def test_alias_resolves_to_same_team_id(mem_db) -> None:
    """Leuven / Oud-Heverlee Leuven / OH Leuven → one team_id (same gender)."""
    tid = upsert_team(mem_db, "Oud-Heverlee Leuven", gender="W")
    mem_db.commit()

    assert resolve_team_id(mem_db, "Leuven", gender="W") == tid
    assert resolve_team_id(mem_db, "OH Leuven", gender="W") == tid
    assert resolve_team_id(mem_db, "Oud-Heverlee Leuven", gender="W") == tid
    assert canonical_team_name("OH Leuven") == normalize_team_name("Leuven")
    assert canonical_team_name("Oud-Heverlee Leuven") == "Leuven"


def test_men_women_same_brand_get_distinct_team_ids(mem_db) -> None:
    """Arsenal EPL (M) and Arsenal UWCL (W) must not share team_id."""
    from src.global_db import make_team_key

    mid = upsert_team(mem_db, "Arsenal", gender="M", comp_id="EPL")
    wid = upsert_team(mem_db, "Arsenal", gender="W", comp_id="UWCL")
    mem_db.commit()
    assert mid != wid
    assert make_team_key("Arsenal", "M") == "ARSENAL_M"
    assert make_team_key("Arsenal", "W") == "ARSENAL_W"
    assert resolve_team_id(mem_db, "Arsenal", comp_id="EPL") == mid
    assert resolve_team_id(mem_db, "Arsenal", comp_id="UWCL") == wid


def test_rest_days_do_not_mix_men_women_schedules(mem_db) -> None:
    """Men's Arsenal rest_days ignores women's UWCL fixtures."""
    rows_epl = [
        {
            "Date": "2024-09-01",
            "HomeTeam": "Arsenal",
            "AwayTeam": "Chelsea",
            "FTHG": 2,
            "FTAG": 1,
            "Match_ID": "e1",
        },
        {
            "Date": "2024-09-10",
            "HomeTeam": "Liverpool",
            "AwayTeam": "Arsenal",
            "FTHG": 0,
            "FTAG": 1,
            "Match_ID": "e2",
        },
    ]
    rows_uwcl = [
        {
            "Date": "2024-09-18",
            "HomeTeam": "Arsenal",
            "AwayTeam": "Barcelona",
            "FTHG": 3,
            "FTAG": 0,
            "Match_ID": "w1",
        },
    ]
    import_legacy_matches_df(mem_db, pd.DataFrame(rows_epl), comp_id="EPL")
    import_legacy_matches_df(mem_db, pd.DataFrame(rows_uwcl), comp_id="UWCL")

    arsenal_m = resolve_team_id(mem_db, "Arsenal", comp_id="EPL")
    arsenal_w = resolve_team_id(mem_db, "Arsenal", comp_id="UWCL")
    assert arsenal_m is not None and arsenal_w is not None
    assert arsenal_m != arsenal_w

    feats_m = calculate_multi_comp_features(arsenal_m, "2024-09-20", conn=mem_db)
    feats_w = calculate_multi_comp_features(arsenal_w, "2024-09-20", conn=mem_db)
    # Men's last match 09-10 → rest 10 days; 14d window from 09-20 starts 09-06
    # so only the 09-10 EPL match counts (09-01 is outside).
    assert feats_m["rest_days"] == 10.0
    assert feats_m["matches_last_14d"] == 1.0
    # Women's only the 09-18 UWCL match
    assert feats_w["rest_days"] == 2.0
    assert feats_w["matches_last_14d"] == 1.0


def test_league_weight_defaults(mem_db) -> None:
    assert get_competition_weight(mem_db, "EPL") == 1.0
    assert get_competition_weight(mem_db, "WSL") == 0.9
    assert get_competition_weight(mem_db, "UWCL") == 0.85


def test_apply_league_weight_to_rates() -> None:
    lam, mu = 1.5, 1.0
    # Equal weights at baseline → unchanged.
    assert apply_league_weight_to_rates(lam, mu, 1.0, 1.0) == (1.5, 1.0)
    # Stronger home domestic league boosts λ relative to μ.
    lam2, mu2 = apply_league_weight_to_rates(lam, mu, 0.9, 0.7)
    assert lam2 == pytest.approx(1.5 * 0.9)
    assert mu2 == pytest.approx(1.0 * 0.7)
    assert lam2 / mu2 > lam / mu


def test_multi_comp_features_across_competitions(mem_db) -> None:
    """Form aggregates same-gender comps only (EPL men stay on ARSENAL_M)."""
    home = upsert_team(mem_db, "Chelsea", gender="M", comp_id="EPL")
    upsert_team(mem_db, "Arsenal", gender="M", comp_id="EPL")
    upsert_team(mem_db, "Brighton", gender="M", comp_id="EPL")
    mem_db.commit()

    rows = [
        {
            "Date": "2024-09-01",
            "HomeTeam": "Chelsea",
            "AwayTeam": "Arsenal",
            "FTHG": 2,
            "FTAG": 1,
            "league_id": "EPL",
            "Match_ID": "2024-09-01|Chelsea|Arsenal",
        },
        {
            "Date": "2024-09-08",
            "HomeTeam": "Arsenal",
            "AwayTeam": "Chelsea",
            "FTHG": 0,
            "FTAG": 1,
            "league_id": "EPL",
            "Match_ID": "2024-09-08|Arsenal|Chelsea",
        },
        {
            "Date": "2024-09-12",
            "HomeTeam": "Chelsea",
            "AwayTeam": "Brighton",
            "FTHG": 3,
            "FTAG": 0,
            "league_id": "EPL",
            "Match_ID": "2024-09-12|Chelsea|Brighton",
        },
    ]
    for r in rows:
        import_legacy_matches_df(
            mem_db, pd.DataFrame([r]), comp_id=r["league_id"], dry_run=False
        )

    feats = calculate_multi_comp_features(home, "2024-09-20", conn=mem_db)
    # lookback 14d from 09-20 → since 09-06 → two matches (09-08, 09-12)
    assert feats["matches_last_14d"] == 2.0
    assert feats["rest_days"] == 8.0  # 20 - 12
    assert feats["rolling_xg_5"] == pytest.approx(2.0)

    empty = calculate_multi_comp_features(home, "2024-08-01", conn=mem_db)
    assert empty["matches_last_14d"] == 0.0
    assert empty["rest_days"] != empty["rest_days"]  # NaN


def test_migration_round_trip_toy(tmp_path: Path) -> None:
    """Legacy flat DB → global → legacy projection preserves scores & teams."""
    import importlib.util

    from src.data_loader import save_matches_to_db

    epl = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2024-08-16"),
                "HomeTeam": "Man United",
                "AwayTeam": "Fulham",
                "FTHG": 1,
                "FTAG": 0,
                "FTR": "H",
                "Season": "2024/25",
                "SeasonStart": 2024,
                "league_id": "EPL",
                "HC": 7,
                "AC": 8,
            },
            {
                "Date": pd.Timestamp("2024-08-17"),
                "HomeTeam": "Oud-Heverlee Leuven",
                "AwayTeam": "PSG",
                "FTHG": 0,
                "FTAG": 2,
                "FTR": "A",
                "Season": "2024/25",
                "SeasonStart": 2024,
                "league_id": "EPL",
            },
        ]
    )
    uwcl = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2024-09-01"),
                "HomeTeam": "Leuven",
                "AwayTeam": "Barcelona",
                "FTHG": 1,
                "FTAG": 3,
                "FTR": "A",
                "Season": "2024/2025",
                "SeasonStart": 2024,
                "league_id": "UWCL",
            }
        ]
    )

    epl_db = tmp_path / "epl_matches.db"
    uwcl_db = tmp_path / "uwcl_matches.db"
    global_db = tmp_path / "global_matches.db"
    save_matches_to_db(epl, epl_db)
    save_matches_to_db(uwcl, uwcl_db)

    script = Path(__file__).resolve().parents[1] / "scripts" / "migrate_to_global_db.py"
    spec = importlib.util.spec_from_file_location("migrate_to_global_db", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    stats = mod.migrate(
        target=global_db,
        sources=[("EPL", epl_db), ("UWCL", uwcl_db)],
        dry_run=False,
    )
    assert stats["EPL"]["matches_upserted"] == 2
    assert stats["UWCL"]["matches_upserted"] == 1

    stats2 = mod.migrate(
        target=global_db,
        sources=[("EPL", epl_db), ("UWCL", uwcl_db)],
        dry_run=False,
    )
    assert stats2["EPL"]["matches_upserted"] == 2

    out_epl = read_matches_as_legacy(global_db, comp_id="EPL")
    out_uwcl = read_matches_as_legacy(global_db, comp_id="UWCL")
    assert len(out_epl) == 2
    assert len(out_uwcl) == 1
    assert "Leuven" in set(out_epl["HomeTeam"]) | set(out_epl["AwayTeam"])
    assert out_uwcl.iloc[0]["HomeTeam"] == "Leuven"
    assert int(out_epl.iloc[0]["FTHG"]) == 1

    conn = connect_global_db(global_db, init=False)
    try:
        # EPL Leuven is men's key; UWCL Leuven is women's — distinct ids.
        epl_leuven = resolve_team_id(conn, "Leuven", comp_id="EPL")
        epl_oh = resolve_team_id(conn, "Oud-Heverlee Leuven", comp_id="EPL")
        epl_oh2 = resolve_team_id(conn, "OH Leuven", comp_id="EPL")
        uwcl_leuven = resolve_team_id(conn, "Leuven", comp_id="UWCL")
        assert epl_leuven == epl_oh == epl_oh2
        assert epl_leuven is not None
        assert uwcl_leuven is not None
        assert epl_leuven != uwcl_leuven
    finally:
        conn.close()


def test_dc_league_weight_optional_path() -> None:
    """Weighted expected_goals differs when domestic weights differ; 1.0 is no-op."""
    matches = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22"]
            ),
            "HomeTeam": ["Alpha", "Beta", "Alpha", "Beta"],
            "AwayTeam": ["Beta", "Alpha", "Beta", "Alpha"],
            "FTHG": [2, 1, 3, 0],
            "FTAG": [1, 1, 0, 2],
        }
    )
    model = DixonColesModel(xi=0.0).fit(matches)
    lam0, mu0 = model.expected_goals("Alpha", "Beta")
    lam1, mu1 = model.expected_goals_with_league_weights(
        "Alpha", "Beta", home_weight=1.0, away_weight=1.0
    )
    assert lam1 == pytest.approx(lam0)
    assert mu1 == pytest.approx(mu0)

    lam_w, mu_w = expected_goals_multi_comp(
        model,
        "Alpha",
        "Beta",
        home_league_weight=0.9,
        away_league_weight=0.7,
    )
    assert lam_w == pytest.approx(lam0 * 0.9)
    assert mu_w == pytest.approx(mu0 * 0.7)


def test_build_global_match_id_stable() -> None:
    mid = build_global_match_id(
        "UWCL", "2024-09-12", "Chelsea", "Barcelona", legacy_match_id="2024-09-12|Chelsea|Barcelona"
    )
    assert mid == "UWCL|2024-09-12|Chelsea|Barcelona"
    # Re-prefix is idempotent
    assert (
        build_global_match_id(
            "UWCL", "2024-09-12", "Chelsea", "Barcelona", legacy_match_id=mid
        )
        == mid
    )


def test_read_matches_prefer_global(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src import data_loader as dl
    from src import global_db as gdb
    from src.data_loader import save_matches_to_db

    epl_rows = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2024-08-16"),
                "HomeTeam": "Man United",
                "AwayTeam": "Fulham",
                "FTHG": 1,
                "FTAG": 0,
                "FTR": "H",
                "SeasonStart": 2024,
                "league_id": "EPL",
            }
        ]
    )
    league_db = tmp_path / "epl_matches.db"
    global_path = tmp_path / "global_matches.db"
    save_matches_to_db(epl_rows, league_db)

    conn = connect_global_db(global_path, init=True)
    import_legacy_matches_df(conn, epl_rows, comp_id="EPL")
    conn.close()

    monkeypatch.setattr(gdb, "GLOBAL_DB_PATH", global_path)
    monkeypatch.setattr(dl, "DEFAULT_DB_PATH", league_db)
    monkeypatch.setitem(dl.LEAGUE_CONFIG["EPL"], "db_path", league_db)

    from src.data_loader import read_matches_prefer_global

    got = read_matches_prefer_global("EPL", prefer_global=True)
    assert not got.empty
    assert got.attrs.get("data_source") == "sqlite_global"
    assert len(got) == 1


def test_upcoming_fixtures_db_roundtrip(tmp_path: Path) -> None:
    """save_upcoming_to_db → load_upcoming_from_db preserves odds + kickoff."""
    from datetime import datetime, timezone

    from src.global_db import (
        load_upcoming_from_db,
        save_upcoming_to_db,
        upcoming_cache_age_minutes,
    )

    db = tmp_path / "global_matches.db"
    now = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
    fx = pd.DataFrame(
        [
            {
                "HomeTeam": "Arsenal",
                "AwayTeam": "Chelsea",
                "Kickoff": pd.Timestamp("2026-03-21 15:00:00"),
                "Date": pd.Timestamp("2026-03-21"),
                "FlashscoreEventId": "abc123",
                "B365H": 1.90,
                "B365D": 3.50,
                "B365A": 4.00,
                "OddsProvider": "Flashscore/bet365",
            }
        ]
    )
    fx.attrs["odds_sources"] = ["Flashscore"]
    fx.attrs["league"] = "EPL"

    n = save_upcoming_to_db("EPL", fx, db_path=db, updated_at=now)
    assert n == 1

    loaded, last_ts = load_upcoming_from_db("EPL", db_path=db)
    assert not loaded.empty
    assert last_ts is not None
    assert loaded.iloc[0]["HomeTeam"] == "Arsenal"
    assert float(loaded.iloc[0]["B365H"]) == pytest.approx(1.90)
    assert loaded.attrs.get("odds_sources") == ["Flashscore"]
    assert upcoming_cache_age_minutes(last_ts, now=now) == pytest.approx(0.0)

    # Strict max_age: empty frame when older than threshold.
    later = datetime(2026, 3, 20, 12, 30, tzinfo=timezone.utc)
    stale, ts2 = load_upcoming_from_db(
        "EPL", db_path=db, max_age_minutes=15, now=later
    )
    assert stale.empty
    assert ts2 == last_ts

    # Soft read still returns rows when max_age is None.
    fresh, _ = load_upcoming_from_db("EPL", db_path=db, now=later)
    assert len(fresh) == 1


def test_fetch_flashscore_odds_for_events_parallel(monkeypatch: pytest.MonkeyPatch) -> None:
    """ThreadPoolExecutor path gathers per-event odds without live HTTP."""
    from src import data_loader as dl

    calls: list[str] = []

    def _fake_odds(event_id: str, **kwargs):
        calls.append(str(event_id))
        if event_id == "bad":
            raise RuntimeError("boom")
        if event_id == "empty":
            return {}
        return {
            "FlashscoreEventId": event_id,
            "B365H": 2.0,
            "B365D": 3.0,
            "B365A": 4.0,
        }

    monkeypatch.setattr(dl, "fetch_flashscore_match_odds", _fake_odds)
    out = dl.fetch_flashscore_odds_for_events(
        ["a", "bad", "empty", "b"], max_workers=3
    )
    assert set(out) == {"a", "b"}
    assert out["a"]["B365H"] == 2.0
    assert set(calls) == {"a", "bad", "empty", "b"}
