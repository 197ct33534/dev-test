"""Regression tests: journal settlement window, CAS settle, atomic dedup, SQLite setup."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import closing
from pathlib import Path

import pandas as pd
import pytest

from src.journal import (
    JOURNAL_HISTORY_VIEW,
    LIVE_BETS_TABLE,
    PENDING_UNIQUE_INDEX,
    add_live_bet,
    add_to_journal,
    ensure_live_bets_table,
    find_match_result,
    load_live_bets,
    settle_live_bet,
)


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    path = tmp_path / "journal.db"
    ensure_live_bets_table(path)
    return path


def _results(*rows: tuple[str, str, str, int, int, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Date": pd.Timestamp(d),
                "HomeTeam": h,
                "AwayTeam": a,
                "FTHG": hg,
                "FTAG": ag,
                "FTR": ftr,
            }
            for d, h, a, hg, ag, ftr in rows
        ]
    )


# ---------------------------------------------------------------------------
# Bug 1: find_match_result must not settle against another date's fixture
# ---------------------------------------------------------------------------


def test_find_match_result_ignores_last_season_fixture() -> None:
    res = _results(("2025-09-20 15:00", "Arsenal", "Chelsea", 3, 0, "H"))
    assert (
        find_match_result(
            res, home_team="Arsenal", away_team="Chelsea", match_date="2026-09-20 15:00"
        )
        is None
    )


def test_find_match_result_accepts_vn_utc_day_shift() -> None:
    # Bet stored with VN date (Sep 21), result row keyed on UTC date (Sep 20).
    res = _results(
        ("2025-09-21 15:00", "Arsenal", "Chelsea", 3, 0, "H"),
        ("2026-09-20 19:00", "Arsenal", "Chelsea", 1, 2, "A"),
    )
    row = find_match_result(
        res, home_team="Arsenal", away_team="Chelsea", match_date="2026-09-21 02:00"
    )
    assert row is not None
    assert int(row["FTHG"]) == 1 and str(row["FTR"]) == "A"
    assert "_day_diff" not in row.index


def test_find_match_result_outside_window_and_missing_date() -> None:
    res = _results(("2026-09-17 15:00", "Arsenal", "Chelsea", 1, 1, "D"))
    assert (
        find_match_result(
            res, home_team="Arsenal", away_team="Chelsea", match_date="2026-09-20"
        )
        is None
    )
    assert find_match_result(res, home_team="Arsenal", away_team="Chelsea") is None
    # Wider window is opt-in.
    assert (
        find_match_result(
            res,
            home_team="Arsenal",
            away_team="Chelsea",
            match_date="2026-09-20",
            max_day_diff=3,
        )
        is not None
    )


def test_find_match_result_prefers_exact_day() -> None:
    res = _results(
        ("2026-09-19 15:00", "Arsenal", "Chelsea", 0, 0, "D"),
        ("2026-09-20 15:00", "Arsenal", "Chelsea", 2, 1, "H"),
    )
    row = find_match_result(
        res, home_team="Arsenal", away_team="Chelsea", match_date="2026-09-20"
    )
    assert row is not None and str(row["FTR"]) == "H"


# ---------------------------------------------------------------------------
# Bug 2: settle_live_bet is compare-and-set on status='PENDING'
# ---------------------------------------------------------------------------


def _new_bet(db: Path, **kw) -> int:
    base = dict(
        home_team="Arsenal",
        away_team="Chelsea",
        selection="Home",
        odds=2.0,
        stake_amount=10.0,
        match_date="2026-09-20 15:00",
        db_path=db,
    )
    base.update(kw)
    bet_id = add_live_bet(**base)
    assert bet_id is not None
    return int(bet_id)


def test_settle_twice_raises_and_keeps_first(tmp_db: Path) -> None:
    bet_id = _new_bet(tmp_db)
    assert settle_live_bet(bet_id, "WIN", tmp_db) == pytest.approx(10.0)
    with pytest.raises(RuntimeError):
        settle_live_bet(bet_id, "LOSS", tmp_db)
    row = load_live_bets(tmp_db).iloc[0]
    assert row["status"] == "WIN"
    assert float(row["pnl"]) == pytest.approx(10.0)


def test_concurrent_settle_only_one_wins(tmp_db: Path) -> None:
    bet_id = _new_bet(tmp_db)
    n = 6
    barrier = threading.Barrier(n)
    outcomes: list[str] = []
    lock = threading.Lock()

    def worker(status: str) -> None:
        barrier.wait()
        try:
            settle_live_bet(bet_id, status, tmp_db)
            res = f"ok:{status}"
        except RuntimeError:
            res = "dup"
        with lock:
            outcomes.append(res)

    threads = [
        threading.Thread(target=worker, args=("WIN" if i % 2 else "LOSS",))
        for i in range(n)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    winners = [o for o in outcomes if o.startswith("ok:")]
    assert len(winners) == 1
    row = load_live_bets(tmp_db).iloc[0]
    assert row["status"] == winners[0].split(":", 1)[1]


# ---------------------------------------------------------------------------
# Bug 3: atomic dedup + UNIQUE partial index
# ---------------------------------------------------------------------------


def test_add_to_journal_concurrent_creates_single_row(tmp_db: Path) -> None:
    n = 8
    barrier = threading.Barrier(n)
    results: list[dict] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        r = add_to_journal(
            home_team="Arsenal",
            away_team="Chelsea",
            selection="Home",
            odds=2.1,
            stake_amount=5.0,
            match_date="2026-09-20 15:00",
            db_path=tmp_db,
        )
        with lock:
            results.append(r)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(1 for r in results if r["created"]) == 1
    ids = {r["id"] for r in results}
    assert len(ids) == 1 and None not in ids
    assert len(load_live_bets(tmp_db, status="PENDING")) == 1


def test_add_to_journal_dedup_and_skip_if_pending(tmp_db: Path) -> None:
    first = add_to_journal(
        home_team="A", away_team="B", selection="Home", odds=2.0,
        stake_amount=1.0, match_date="2026-09-20 15:00", db_path=tmp_db,
    )
    second = add_to_journal(
        home_team="A", away_team="B", selection="Chủ nhà", odds=2.2,
        stake_amount=1.0, match_date="2026-09-20T18:00:00", db_path=tmp_db,
    )
    assert first["created"] and not second["created"]
    assert second["id"] == first["id"]
    assert second["reason"] == f"duplicate_pending:{first['id']}"
    assert (
        add_live_bet(
            home_team="A", away_team="B", selection="Home", odds=2.0,
            stake_amount=1.0, match_date="2026-09-20", db_path=tmp_db,
            skip_if_pending=True,
        )
        is None
    )
    # Different selection is a distinct bet.
    third = add_to_journal(
        home_team="A", away_team="B", selection="Away", odds=3.0,
        stake_amount=1.0, match_date="2026-09-20 15:00", db_path=tmp_db,
    )
    assert third["created"]


def test_unique_pending_index_created_on_fresh_db(tmp_db: Path) -> None:
    with closing(sqlite3.connect(tmp_db)) as conn:
        names = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert PENDING_UNIQUE_INDEX in names
    assert "idx_live_bets_pending_dedup" in names
    assert str(mode).lower() == "wal"


def test_existing_duplicates_are_kept_and_index_skipped(tmp_path: Path) -> None:
    db = tmp_path / "legacy_dupes.db"
    with closing(sqlite3.connect(db)) as conn:
        conn.execute(
            f"""
            CREATE TABLE {LIVE_BETS_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_date TEXT, home_team TEXT NOT NULL, away_team TEXT NOT NULL,
                market TEXT DEFAULT '1X2', selection TEXT NOT NULL,
                odds REAL NOT NULL, stake_amount REAL NOT NULL, stake_pct REAL,
                ev REAL, p_model REAL, status TEXT NOT NULL DEFAULT 'PENDING',
                pnl REAL DEFAULT 0, closing_odds REAL, clv_pct REAL,
                created_at TEXT, settled_at TEXT
            )
            """
        )
        for _ in range(2):
            conn.execute(
                f"INSERT INTO {LIVE_BETS_TABLE} (match_date, home_team, away_team, "
                "market, selection, odds, stake_amount, status) "
                "VALUES ('2026-09-20 15:00', 'A', 'B', '1X2', 'Home', 2.0, 1.0, 'PENDING')"
            )
        conn.commit()

    ensure_live_bets_table(db)  # must not raise
    assert len(load_live_bets(db)) == 2
    with closing(sqlite3.connect(db)) as conn:
        names = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
    assert PENDING_UNIQUE_INDEX not in names
    assert "idx_live_bets_pending_dedup" in names
    # Atomic dedup still works without the unique index.
    r = add_to_journal(
        home_team="A", away_team="B", selection="Home", odds=2.0,
        stake_amount=1.0, match_date="2026-09-20 15:00", db_path=db,
    )
    assert r["created"] is False


# ---------------------------------------------------------------------------
# Bug 4: schema setup once per path, view recreated only when stale
# ---------------------------------------------------------------------------


def test_history_view_upgraded_when_sql_differs(tmp_path: Path) -> None:
    db = tmp_path / "old_view.db"
    with closing(sqlite3.connect(db)) as conn:
        conn.execute(
            f"CREATE TABLE {LIVE_BETS_TABLE} (id INTEGER PRIMARY KEY, match_date TEXT, "
            "home_team TEXT NOT NULL, away_team TEXT NOT NULL, market TEXT, "
            "selection TEXT NOT NULL, odds REAL NOT NULL, stake_amount REAL NOT NULL, "
            "status TEXT NOT NULL DEFAULT 'PENDING', pnl REAL DEFAULT 0)"
        )
        conn.execute(
            f"CREATE VIEW {JOURNAL_HISTORY_VIEW} AS SELECT * FROM {LIVE_BETS_TABLE} "
            "WHERE status IN ('WIN','LOSS','PUSH')"
        )
        conn.commit()
    ensure_live_bets_table(db)
    with closing(sqlite3.connect(db)) as conn:
        sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='view' AND name=?",
            (JOURNAL_HISTORY_VIEW,),
        ).fetchone()[0]
    assert "HALF_WIN" in sql and "VOID" in sql


def test_ensure_runs_ddl_once_per_path(tmp_path: Path, monkeypatch) -> None:
    import src.journal as journal

    db = tmp_path / "once.db"
    calls: list[Path] = []
    real = journal._ensure_live_bets_schema

    def spy(path: Path) -> None:
        calls.append(path)
        real(path)

    monkeypatch.setattr(journal, "_ensure_live_bets_schema", spy)
    for _ in range(3):
        ensure_live_bets_table(db)
    load_live_bets(db)
    assert len(calls) == 1
