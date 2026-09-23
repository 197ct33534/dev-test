"""Badge criterion: any finished history in global DB via has_team_history."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data_loader import has_sufficient_team_data, has_team_history
from src.features import get_rest_days
from src.global_db import (
    GLOBAL_DB_PATH,
    clear_team_history_cache,
    connect_global_db,
    has_team_history as has_history_global,
    set_team_feed_meta,
    upsert_match,
    upsert_team,
)
from src.models import teams_needing_data_badge


@pytest.fixture()
def mem_db(tmp_path: Path):
    clear_team_history_cache()
    db = tmp_path / "suff.db"
    conn = connect_global_db(db, init=True)
    yield db, conn
    conn.close()
    clear_team_history_cache()


def test_has_team_history_true_with_one_finished_match(mem_db) -> None:
    db, conn = mem_db
    home = upsert_team(conn, "JP_FUJIEDA", gender="M")
    away = upsert_team(conn, "JP_OMIYA", gender="M")
    upsert_match(
        conn,
        match_id="J2|2026-09-19|JP_FUJIEDA|JP_OMIYA",
        comp_id="J2",
        home_team_id=home,
        away_team_id=away,
        match_date="2026-09-19",
        home_score=1,
        away_score=1,
    )
    conn.commit()

    assert has_history_global("JP_FUJIEDA", db_path=db)
    assert has_team_history("JP_FUJIEDA", db_path=db)
    # Any finished row counts — same-day / upcoming_date is irrelevant.
    assert has_sufficient_team_data(
        "JP_FUJIEDA", upcoming_date="2026-09-19", min_matches=1, db_path=db
    )


def test_has_team_history_false_when_empty(mem_db) -> None:
    db, conn = mem_db
    upsert_team(conn, "JP_UNKNOWN", gender="M")
    conn.commit()
    assert not has_history_global("JP_UNKNOWN", db_path=db)
    assert not has_team_history("NoSuchClubXYZ", db_path=db)
    assert not has_sufficient_team_data(
        "NoSuchClubXYZ", upcoming_date="2026-09-23", db_path=db
    )


def test_has_team_history_uses_team_id_columns(mem_db) -> None:
    """History lookup must use home_team_id/away_team_id, not name strings."""
    db, conn = mem_db
    home = upsert_team(
        conn,
        "JP_FUJIEDA",
        gender="M",
        aliases=["Fujieda MYFC", "Fujieda"],
    )
    away = upsert_team(conn, "JP_OMIYA", gender="M")
    upsert_match(
        conn,
        match_id="J2|2026-09-19|JP_FUJIEDA|JP_OMIYA",
        comp_id="J2",
        home_team_id=home,
        away_team_id=away,
        match_date="2026-09-19",
        home_score=2,
        away_score=0,
    )
    conn.commit()
    # Integer PK and alias both resolve via teams.aliases / team_id columns.
    assert has_team_history(int(home), db_path=db)
    assert has_team_history("Fujieda MYFC", db_path=db)


def test_teams_needing_data_badge_ignores_dc_thin(mem_db) -> None:
    """Fujieda-like: history in DB → not flagged even if DC would treat as thin."""
    db, conn = mem_db
    home = upsert_team(conn, "JP_FUJIEDA", gender="M")
    away = upsert_team(conn, "JP_OMIYA", gender="M")
    upsert_match(
        conn,
        match_id="J2|2026-09-19|JP_FUJIEDA|JP_OMIYA",
        comp_id="J2",
        home_team_id=home,
        away_team_id=away,
        match_date="2026-09-19",
        home_score=1,
        away_score=1,
    )
    conn.commit()

    class _ThinDC:
        teams = ["JP_FUJIEDA", "JP_OMIYA"]
        min_team_matches = 5

        def team_match_count(self, team: str) -> int:
            return 0  # would be thin under old DC criterion

        def is_thin_team(self, team: str) -> bool:
            return True

    flagged = teams_needing_data_badge(
        _ThinDC(),  # type: ignore[arg-type]
        "JP_FUJIEDA",
        "JP_OMIYA",
        upcoming_date="2026-09-23",
        min_matches=1,
        db_path=db,
    )
    assert flagged == []


def test_stale_thin_names_do_not_override_history(mem_db) -> None:
    """UI must not prefer scanner thin_names over live has_team_history."""
    db, conn = mem_db
    home = upsert_team(conn, "JP_FUJIEDA", gender="M")
    away = upsert_team(conn, "JP_OMIYA", gender="M")
    upsert_match(
        conn,
        match_id="J2|2026-09-19|JP_FUJIEDA|JP_OMIYA",
        comp_id="J2",
        home_team_id=home,
        away_team_id=away,
        match_date="2026-09-19",
        home_score=1,
        away_score=1,
    )
    conn.commit()

    # Simulate stale scanner flags saying Fujieda is thin.
    assert has_team_history("JP_FUJIEDA", db_path=db)
    flagged = teams_needing_data_badge(
        None,
        "JP_FUJIEDA",
        "JP_OMIYA",
        db_path=db,
    )
    assert "JP_FUJIEDA" not in flagged


def test_fujieda_rest_days_four_when_data_present() -> None:
    """Live DB: JP_FUJIEDA last match 2026-09-19 → rest_days=4 on 2026-09-23."""
    if not GLOBAL_DB_PATH.is_file():
        pytest.skip("global_matches.db not present")
    clear_team_history_cache()
    if not has_history_global("JP_FUJIEDA"):
        pytest.skip("JP_FUJIEDA has no finished history in global DB")
    assert has_team_history("JP_FUJIEDA")
    assert has_team_history("Fujieda MYFC")
    rd = get_rest_days("JP_FUJIEDA", "2026-09-23")
    assert rd == 4.0


def test_fujieda_like_rest_days_from_feed_meta(mem_db) -> None:
    db, conn = mem_db
    tid = upsert_team(conn, "JP_FUJIEDA", gender="M")
    set_team_feed_meta(conn, tid, last_match_date="2026-09-19")
    conn.commit()
    assert get_rest_days("JP_FUJIEDA", "2026-09-23", db_path=db) == 4.0


def test_rest_days_from_finished_match_row(mem_db) -> None:
    db, conn = mem_db
    home = upsert_team(conn, "JP_FUJIEDA", gender="M")
    away = upsert_team(conn, "JP_OMIYA", gender="M")
    upsert_match(
        conn,
        match_id="J2|2026-09-19|JP_FUJIEDA|JP_OMIYA",
        comp_id="J2",
        home_team_id=home,
        away_team_id=away,
        match_date="2026-09-19",
        home_score=1,
        away_score=1,
    )
    conn.commit()
    assert has_team_history("JP_FUJIEDA", db_path=db)
    assert get_rest_days("JP_FUJIEDA", "2026-09-23", db_path=db) == 4.0
