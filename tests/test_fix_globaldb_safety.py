"""Regression tests: global DB init-once, safe migration, upcoming cache guards."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

import src.global_db as gdb
from src.global_db import (
    GLOBAL_DB_VERSION,
    connect_global_db,
    load_upcoming_from_db,
    save_upcoming_to_db,
)


def _fixtures(*kickoffs: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "HomeTeam": f"H{i}",
                "AwayTeam": f"A{i}",
                "Kickoff": pd.Timestamp(k),
                "B365H": 2.0,
                "B365D": 3.2,
                "B365A": 3.8,
            }
            for i, k in enumerate(kickoffs)
        ]
    )


def _tables(db: Path) -> set[str]:
    with closing(sqlite3.connect(db)) as conn:
        return {
            str(r[0])
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }


# ---------------------------------------------------------------------------
# Connections: WAL, init once, closed by ``with``
# ---------------------------------------------------------------------------


def test_connect_sets_wal_and_inits_schema_once(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "g.db"
    calls: list[int] = []
    real = gdb.init_schema

    def spy(conn: sqlite3.Connection) -> None:
        calls.append(1)
        real(conn)

    monkeypatch.setattr(gdb, "init_schema", spy)
    for _ in range(3):
        conn = connect_global_db(db, init=True)
        conn.close()
    assert len(calls) == 1

    with connect_global_db(db, init=True) as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
    assert str(mode).lower() == "wal"
    assert int(timeout) >= 30000


def test_init_reruns_when_file_recreated(tmp_path: Path) -> None:
    db = tmp_path / "g.db"
    connect_global_db(db, init=True).close()
    db.unlink()
    with connect_global_db(db, init=True) as conn:
        ver = conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()[0]
    assert int(ver) == GLOBAL_DB_VERSION


def test_with_block_closes_connection(tmp_path: Path) -> None:
    db = tmp_path / "g.db"
    with connect_global_db(db, init=True) as conn:
        conn.execute("SELECT 1")
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


# ---------------------------------------------------------------------------
# Migration never drops a v2+ schema
# ---------------------------------------------------------------------------


def _seed_v2_db(db: Path) -> None:
    conn = connect_global_db(db, init=True)
    try:
        tid = gdb.upsert_team(conn, "Arsenal", gender="M", comp_id="EPL")
        gdb.set_team_rest_cache(conn, tid, "2026-09-20", rest_days=3.0, matches_last_14d=2)
        conn.commit()
    finally:
        conn.close()


@pytest.mark.parametrize("meta_state", ["missing_row", "garbage", "old_version", "no_meta"])
def test_migration_keeps_tables_when_team_key_present(
    tmp_path: Path, meta_state: str
) -> None:
    db = tmp_path / "g.db"
    _seed_v2_db(db)
    with closing(sqlite3.connect(db)) as conn:
        if meta_state == "missing_row":
            conn.execute("DELETE FROM meta WHERE key='schema_version'")
        elif meta_state == "garbage":
            conn.execute("UPDATE meta SET value='x' WHERE key='schema_version'")
        elif meta_state == "old_version":
            conn.execute("UPDATE meta SET value='1' WHERE key='schema_version'")
        else:
            conn.execute("DROP TABLE meta")
        conn.commit()

    gdb._SCHEMA_READY.clear()
    with connect_global_db(db, init=True) as conn:
        n_teams = conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0]
        n_rest = conn.execute("SELECT COUNT(*) FROM team_rest_cache").fetchone()[0]
        ver = conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()[0]
    assert n_teams >= 1
    assert n_rest == 1
    assert int(ver) == GLOBAL_DB_VERSION


def test_legacy_schema_moved_aside_not_dropped(tmp_path: Path) -> None:
    db = tmp_path / "legacy.db"
    with closing(sqlite3.connect(db)) as conn:
        conn.executescript(
            """
            CREATE TABLE teams (team_id INTEGER PRIMARY KEY, canonical_name TEXT);
            CREATE INDEX idx_teams_canonical ON teams (canonical_name);
            CREATE TABLE matches (match_id TEXT PRIMARY KEY, home_team_id INTEGER);
            CREATE INDEX idx_matches_home ON matches (home_team_id);
            INSERT INTO teams VALUES (1, 'Arsenal');
            INSERT INTO matches VALUES ('m1', 1);
            """
        )
        conn.commit()

    with connect_global_db(db, init=True) as conn:
        cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(teams)")}
    assert "team_key" in cols
    tables = _tables(db)
    backups = sorted(t for t in tables if "_legacy_v1_" in t)
    assert any(t.startswith("teams_legacy_v1_") for t in backups)
    assert any(t.startswith("matches_legacy_v1_") for t in backups)
    with closing(sqlite3.connect(db)) as conn:
        bt = next(t for t in backups if t.startswith("teams_legacy_v1_"))
        assert conn.execute(f'SELECT COUNT(*) FROM "{bt}"').fetchone()[0] == 1
        idx = {
            str(r[0])
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='teams'"
            )
        }
    assert "idx_teams_canonical" in idx


# ---------------------------------------------------------------------------
# Upcoming cache: no wipe on empty frame, future_only filter
# ---------------------------------------------------------------------------


def test_save_upcoming_empty_frame_keeps_cache(tmp_path: Path) -> None:
    db = tmp_path / "g.db"
    t0 = datetime(2026, 9, 20, 10, 0, tzinfo=timezone.utc)
    assert save_upcoming_to_db("EPL", _fixtures("2026-09-21 15:00"), db_path=db, updated_at=t0) == 1
    _, ts0 = load_upcoming_from_db("EPL", db_path=db)

    t1 = datetime(2026, 9, 20, 11, 0, tzinfo=timezone.utc)
    assert save_upcoming_to_db("EPL", pd.DataFrame(), db_path=db, updated_at=t1) == 0
    assert save_upcoming_to_db("EPL", None, db_path=db, updated_at=t1) == 0  # type: ignore[arg-type]
    df, ts = load_upcoming_from_db("EPL", db_path=db)
    assert len(df) == 1
    assert ts == ts0  # freshness stamp not bumped

    assert (
        save_upcoming_to_db(
            "EPL", pd.DataFrame(), db_path=db, updated_at=t1, allow_empty=True
        )
        == 0
    )
    df2, _ = load_upcoming_from_db("EPL", db_path=db)
    assert df2.empty


def test_load_upcoming_future_only_and_max_age(tmp_path: Path) -> None:
    db = tmp_path / "g.db"
    saved_at = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)
    fx = _fixtures("2026-09-20 11:00", "2026-09-20 12:00", "2026-09-20 15:00")
    save_upcoming_to_db("EPL", fx, db_path=db, updated_at=saved_at)

    now = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)
    all_rows, _ = load_upcoming_from_db("EPL", db_path=db, now=now)
    assert len(all_rows) == 3
    fut, ts = load_upcoming_from_db("EPL", db_path=db, now=now, future_only=True)
    assert list(fut["HomeTeam"]) == ["H2"]  # kickoff == now counts as started
    assert ts is not None
    # Naive ``now`` is UTC.
    fut_naive, _ = load_upcoming_from_db(
        "EPL", db_path=db, now=datetime(2026, 9, 20, 11, 30), future_only=True
    )
    assert list(fut_naive["HomeTeam"]) == ["H1", "H2"]

    stale, _ = load_upcoming_from_db(
        "EPL",
        db_path=db,
        now=datetime(2026, 9, 20, 13, 0, tzinfo=timezone.utc),
        max_age_minutes=30,
        future_only=True,
    )
    assert stale.empty


def test_load_japan_upcoming_drops_started(tmp_path: Path) -> None:
    from src.japan_schedule import load_japan_upcoming

    db = tmp_path / "g.db"
    save_upcoming_to_db(
        "EMPERORS_CUP",
        _fixtures("2026-09-20 05:00", "2026-09-20 10:00"),
        db_path=db,
        updated_at=datetime(2026, 9, 20, 4, 0, tzinfo=timezone.utc),
    )
    now = datetime(2026, 9, 20, 6, 0, tzinfo=timezone.utc)
    df = load_japan_upcoming(db_path=db, now=now)
    assert list(df["HomeTeam"]) == ["H1"]
    assert set(df["comp_id"]) == {"EMPERORS_CUP"}
    both = load_japan_upcoming(db_path=db, now=now, include_started=True)
    assert len(both) == 2
