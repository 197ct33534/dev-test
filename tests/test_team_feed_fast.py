"""Tests for lazy Team Feed sync + DB-only rest_days."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.fetchers import flashscore_team as ft
from src.features import attach_rest_days_from_db, get_rest_days
from src.global_db import (
    connect_global_db,
    set_team_feed_meta,
    upsert_team,
    upsert_team_flashscore_hash,
)


@pytest.fixture()
def mem_db(tmp_path: Path):
    db = tmp_path / "g.db"
    conn = connect_global_db(db, init=True)
    yield db, conn
    conn.close()


def test_schema_has_feed_cache_columns(mem_db) -> None:
    db, conn = mem_db
    cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(teams)").fetchall()}
    assert "flashscore_hash" in cols
    assert "last_match_date" in cols
    assert "feed_updated_at" in cols


def test_should_update_team_feed_24h(mem_db) -> None:
    db, conn = mem_db
    tid = upsert_team(conn, "JP_MACHIDA", gender="M")
    conn.commit()
    # Missing feed_updated_at → must update.
    assert ft.should_update_team_feed(tid, db_path=db) is True

    now = datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc)
    set_team_feed_meta(
        conn,
        tid,
        last_match_date="2026-09-20",
        feed_updated_at=now - timedelta(hours=2),
    )
    conn.commit()
    assert ft.should_update_team_feed(tid, db_path=db, now=now) is False

    set_team_feed_meta(
        conn,
        tid,
        feed_updated_at=now - timedelta(hours=25),
    )
    conn.commit()
    assert ft.should_update_team_feed(tid, db_path=db, now=now) is True


def test_get_rest_days_db_only(mem_db) -> None:
    db, conn = mem_db
    tid = upsert_team(conn, "JP_MACHIDA", gender="M")
    set_team_feed_meta(
        conn,
        tid,
        last_match_date="2026-09-20",
        feed_updated_at="2026-09-22T00:00:00+00:00",
    )
    conn.commit()
    assert get_rest_days(tid, "2026-09-23", db_path=db) == 3.0
    assert get_rest_days("JP_MACHIDA", "2026-09-23", db_path=db) == 3.0
    # Hard-cap: 40 days → display 14.
    set_team_feed_meta(conn, tid, last_match_date="2026-08-01")
    conn.commit()
    assert get_rest_days(tid, "2026-09-23", db_path=db) == 14.0


def test_sync_upcoming_teams_fast_respects_max_and_fresh(
    mem_db, monkeypatch: pytest.MonkeyPatch
) -> None:
    db, conn = mem_db
    now = datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc)
    # Fresh team — should skip scrape.
    t_fresh = upsert_team_flashscore_hash(
        conn, "JP_MACHIDA", "CUSC1dab", flashscore_slug="machida-zelvia"
    )
    set_team_feed_meta(
        conn,
        t_fresh,
        last_match_date="2026-09-20",
        feed_updated_at=now - timedelta(hours=1),
    )
    # Stale team — should fetch.
    t_stale = upsert_team_flashscore_hash(
        conn, "JP_VISSEL_KOBE", "698tGI9q", flashscore_slug="vissel-kobe"
    )
    set_team_feed_meta(
        conn,
        t_stale,
        last_match_date="2026-09-10",
        feed_updated_at=now - timedelta(hours=30),
    )
    conn.commit()

    calls: list[str] = []

    def _fake_fetch(team_id: str, n_matches: int = 10, **_kw):
        calls.append(str(team_id).upper())
        return [
            {
                "match_date": pd.Timestamp("2026-09-20"),
                "home_team": "JP_G_OSAKA",
                "away_team": "JP_VISSEL_KOBE",
                "score": "0-1",
                "home_goals": 0,
                "away_goals": 1,
                "competition_name": "JAPAN: J1 League",
                "comp_id": "J1",
                "flashscore_event_id": "fake1",
                "source": "flashscore_team",
            }
        ]

    monkeypatch.setattr(ft, "fetch_team_recent_matches_by_id", _fake_fetch)
    monkeypatch.setattr(
        ft, "persist_team_matches_to_global_db", lambda *a, **k: {"matches_upserted": 1}
    )

    upcoming = pd.DataFrame(
        [
            {"HomeTeam": "JP_MACHIDA", "AwayTeam": "JP_TOCHIGI_CITY", "Kickoff": "2026-09-23"},
            {"HomeTeam": "JP_VISSEL_KOBE", "AwayTeam": "JP_SAGAN_TOSU", "Kickoff": "2026-09-23"},
            {"HomeTeam": "Extra1", "AwayTeam": "Extra2", "Kickoff": "2026-09-24"},
        ]
    )
    # max_teams=1 → only first match sides considered (Machida + Tochigi).
    result = ft.sync_upcoming_teams_fast(
        upcoming,
        max_teams=1,
        db_path=db,
        now=now,
        timeout=3.0,
        max_workers=3,
    )
    assert "JP_MACHIDA" in (result["skipped_fresh"] + result["to_fetch"] + result["skipped_no_hash"])
    assert "JP_VISSEL_KOBE" not in result["to_fetch"]
    assert "JP_VISSEL_KOBE" not in calls

    # max_teams=2 includes Kobe (stale) → fetch once.
    result2 = ft.sync_upcoming_teams_fast(
        upcoming,
        max_teams=2,
        db_path=db,
        now=now,
        timeout=3.0,
        max_workers=3,
    )
    assert "JP_VISSEL_KOBE" in result2["to_fetch"]
    assert "JP_MACHIDA" in result2["skipped_fresh"]
    assert calls.count("JP_VISSEL_KOBE") == 1


def test_attach_rest_days_from_db_no_network(mem_db) -> None:
    db, conn = mem_db
    tid = upsert_team(conn, "JP_MACHIDA", gender="M")
    set_team_feed_meta(conn, tid, last_match_date="2026-09-20")
    upsert_team(conn, "JP_TOCHIGI_CITY", gender="M")
    conn.commit()

    bets = pd.DataFrame(
        [
            {
                "home": "JP_MACHIDA",
                "away": "JP_TOCHIGI_CITY",
                "kickoff": "2026-09-23",
                "league": "EMPERORS_CUP",
            }
        ]
    )
    out = attach_rest_days_from_db(bets, db_path=db)
    assert float(out.iloc[0]["home_rest_days"]) == 3.0
    assert "fatigue_label" in out.columns
