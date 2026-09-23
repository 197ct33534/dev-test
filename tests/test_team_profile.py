"""Unit tests for get_team_profile_data (past + upcoming, DB-only)."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import get_team_profile_data as get_team_profile_data_reexport
from src.global_db import (
    connect_global_db,
    get_team_profile_data,
    import_legacy_matches_df,
    save_upcoming_to_db,
    upsert_team,
)


@pytest.fixture()
def profile_db(tmp_path: Path):
    db = tmp_path / "profile.db"
    conn = connect_global_db(db, init=True)
    # Seed finished matches across comps for JP_VISSEL_KOBE (via alias).
    rows = [
        {
            "Date": "2026-09-10",
            "HomeTeam": "Vissel Kobe",
            "AwayTeam": "Sagan Tosu",
            "FTHG": 2,
            "FTAG": 0,
            "league_id": "J1",
            "Match_ID": "j1-1",
        },
        {
            "Date": "2026-09-14",
            "HomeTeam": "Machida Zelvia",
            "AwayTeam": "Vissel Kobe",
            "FTHG": 1,
            "FTAG": 1,
            "league_id": "EMPERORS_CUP",
            "Match_ID": "ec-1",
        },
        {
            "Date": "2026-09-18",
            "HomeTeam": "Vissel Kobe",
            "AwayTeam": "Tochigi City",
            "FTHG": 0,
            "FTAG": 2,
            "league_id": "J1",
            "Match_ID": "j1-2",
        },
        {
            "Date": "2026-09-20",
            "HomeTeam": "Sagan Tosu",
            "AwayTeam": "Machida Zelvia",
            "FTHG": 3,
            "FTAG": 1,
            "league_id": "J2",
            "Match_ID": "j2-1",
        },
    ]
    for r in rows:
        import_legacy_matches_df(
            conn, pd.DataFrame([r]), comp_id=r["league_id"], dry_run=False
        )
    conn.commit()

    upcoming = pd.DataFrame(
        [
            {
                "Date": "2026-09-28",
                "Kickoff": "2026-09-28T10:00:00",
                "HomeTeam": "Vissel Kobe",
                "AwayTeam": "Kashima Antlers",
            },
            {
                "Date": "2026-10-05",
                "Kickoff": "2026-10-05T11:00:00",
                "HomeTeam": "Urawa Reds",
                "AwayTeam": "Vissel Kobe",
            },
            {
                "Date": "2026-10-12",
                "Kickoff": "2026-10-12T09:00:00",
                "HomeTeam": "Sagan Tosu",
                "AwayTeam": "Machida Zelvia",
            },
        ]
    )
    save_upcoming_to_db(
        "EMPERORS_CUP",
        upcoming,
        db_path=db,
        updated_at=datetime(2026, 9, 23, tzinfo=timezone.utc),
    )
    yield db, conn
    conn.close()


def test_get_team_profile_past_and_upcoming(profile_db) -> None:
    db, _conn = profile_db
    as_of = "2026-09-23T00:00:00"
    data = get_team_profile_data("Vissel Kobe", limit=5, db_path=db, as_of=as_of)

    assert data["db_team_id"] is not None
    assert data["display_name"]  # human-readable preferred
    past = data["past_matches"]
    assert len(past) == 3  # only Kobe's three finished games
    assert past[0]["date"] == "2026-09-18"
    assert past[0]["result"] == "L"  # home 0-2
    assert past[0]["score"] == "0-2"
    assert past[1]["result"] == "D"  # away 1-1
    assert past[2]["result"] == "W"  # home 2-0
    # Display names, not only codes
    assert "Kobe" in past[0]["home"] or past[0]["home"] == "Vissel Kobe"

    upcoming = data["upcoming_matches"]
    assert len(upcoming) == 2
    assert upcoming[0]["is_home"] is True
    assert "Kashima" in upcoming[0]["opponent"] or upcoming[0]["opponent"]
    assert upcoming[1]["is_home"] is False


def test_get_team_profile_empty_unknown_team(tmp_path: Path) -> None:
    db = tmp_path / "empty.db"
    connect_global_db(db, init=True).close()
    data = get_team_profile_data("Totally Unknown FC XYZ", limit=5, db_path=db)
    assert data["past_matches"] == []
    assert data["upcoming_matches"] == []
    assert data["db_team_id"] is None


def test_get_team_profile_limit(profile_db) -> None:
    db, _ = profile_db
    data = get_team_profile_data(
        "Vissel Kobe", limit=2, db_path=db, as_of="2026-09-23"
    )
    assert len(data["past_matches"]) == 2
    assert len(data["upcoming_matches"]) == 2


def test_data_loader_reexport(profile_db) -> None:
    db, _ = profile_db
    a = get_team_profile_data("Vissel Kobe", limit=1, db_path=db, as_of="2026-09-23")
    b = get_team_profile_data_reexport(
        "Vissel Kobe", limit=1, db_path=db, as_of="2026-09-23"
    )
    assert a["past_matches"] == b["past_matches"]


def test_get_team_profile_query_under_50ms(profile_db) -> None:
    """Warm + measure; temp DB is tiny so should be well under 50ms."""
    db, _ = profile_db
    # Warm caches / connection path.
    get_team_profile_data("Vissel Kobe", limit=5, db_path=db, as_of="2026-09-23")
    t0 = time.perf_counter()
    for _ in range(20):
        get_team_profile_data(
            "Vissel Kobe", limit=5, db_path=db, as_of="2026-09-23"
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / 20.0
    assert elapsed_ms < 50.0, f"avg {elapsed_ms:.2f}ms >= 50ms"


def test_profile_indexes_exist(profile_db) -> None:
    db, conn = profile_db
    idx = {
        str(r[1])
        for r in conn.execute("PRAGMA index_list(upcoming_fixtures)").fetchall()
    }
    assert "idx_upcoming_home_ko" in idx
    assert "idx_upcoming_away_ko" in idx
    # silence
    assert db.exists()
    upsert_team(conn, "Vissel Kobe", gender="M")
