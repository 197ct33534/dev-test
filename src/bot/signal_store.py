"""SQLite persistence for Telegram subscribers and sent value-signal keys.

Uses ``data/notified_signals.db`` (separate from ``global_matches.db``) so the
scheduler can push without Redis and without coupling to match history schema.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from src.data_loader import DATA_DIR

DEFAULT_DB_PATH = DATA_DIR / "notified_signals.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS telegram_subscribers (
    chat_id TEXT PRIMARY KEY,
    username TEXT,
    first_name TEXT,
    subscribed_at TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS notified_signals (
    signal_key TEXT PRIMARY KEY,
    match_id TEXT,
    market TEXT,
    selection TEXT,
    notified_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_notified_match
    ON notified_signals (match_id);
"""

# Paths whose schema has been created in this process (run DDL once, not per call).
_INITIALISED: set[str] = set()
_INIT_LOCK = threading.Lock()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _resolve(db_path: Path | str | None) -> Path:
    return Path(db_path) if db_path else DEFAULT_DB_PATH


def _ensure_schema(conn: sqlite3.Connection, path: Path) -> None:
    key = str(path.resolve())
    if key in _INITIALISED and path.exists():
        return
    with _INIT_LOCK:
        if key in _INITIALISED and path.exists():
            return
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        _INITIALISED.add(key)


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Open (and initialise once) the signal/subscriber database.

    The caller owns the returned connection and must ``close()`` it. Internal
    helpers use :func:`_session`, which always closes.
    """
    path = _resolve(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        _ensure_schema(conn, path)
    except Exception:
        conn.close()
        raise
    return conn


@contextmanager
def _session(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    """Connection that commits on success, rolls back on error, always closes."""
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def upsert_subscriber(
    chat_id: str | int,
    *,
    username: str | None = None,
    first_name: str | None = None,
    db_path: Path | str | None = None,
) -> None:
    """Persist ``chat_id`` on ``/start`` (idempotent; re-activates if disabled)."""
    cid = str(chat_id).strip()
    if not cid:
        raise ValueError("chat_id is required")
    with _session(db_path) as conn:
        conn.execute(
            """
            INSERT INTO telegram_subscribers
                (chat_id, username, first_name, subscribed_at, active)
            VALUES (?, ?, ?, ?, 1)
            ON CONFLICT(chat_id) DO UPDATE SET
                username = COALESCE(excluded.username, telegram_subscribers.username),
                first_name = COALESCE(excluded.first_name, telegram_subscribers.first_name),
                active = 1
            """,
            (cid, username, first_name, _utc_now_iso()),
        )
        conn.commit()


def list_active_chat_ids(db_path: Path | str | None = None) -> list[str]:
    """Return active subscriber chat ids."""
    with _session(db_path) as conn:
        rows = conn.execute(
            "SELECT chat_id FROM telegram_subscribers WHERE active = 1 ORDER BY chat_id"
        ).fetchall()
    return [str(r["chat_id"]) for r in rows]


def deactivate_subscriber(chat_id: str | int, *, db_path: Path | str | None = None) -> None:
    """Soft-disable a chat (e.g. after bot blocked)."""
    with _session(db_path) as conn:
        conn.execute(
            "UPDATE telegram_subscribers SET active = 0 WHERE chat_id = ?",
            (str(chat_id).strip(),),
        )
        conn.commit()


def make_signal_key(
    *,
    match_id: str | None,
    market: str | None,
    selection: str | None,
    home: str | None = None,
    away: str | None = None,
    kickoff: str | None = None,
) -> str:
    """Stable dedupe key: prefer match_id+market+selection."""
    mid = (match_id or "").strip()
    mkt = (market or "").strip().upper() or "?"
    sel = (selection or "").strip() or "?"
    if mid:
        return f"{mid}|{mkt}|{sel}"
    h = (home or "?").strip()
    a = (away or "?").strip()
    ko = (kickoff or "").strip() or "?"
    return f"{h}|{a}|{ko}|{mkt}|{sel}"


def already_notified(signal_key: str, *, db_path: Path | str | None = None) -> bool:
    with _session(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM notified_signals WHERE signal_key = ? LIMIT 1",
            (signal_key,),
        ).fetchone()
    return row is not None


def mark_notified(
    signal_key: str,
    *,
    match_id: str | None = None,
    market: str | None = None,
    selection: str | None = None,
    db_path: Path | str | None = None,
) -> bool:
    """Insert signal key. Returns True if newly recorded, False if duplicate."""
    with _session(db_path) as conn:
        try:
            conn.execute(
                """
                INSERT INTO notified_signals
                    (signal_key, match_id, market, selection, notified_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    signal_key,
                    match_id,
                    market,
                    selection,
                    _utc_now_iso(),
                ),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False


def filter_new_keys(
    keys: Iterable[str],
    *,
    db_path: Path | str | None = None,
) -> list[str]:
    """Return keys not yet present in ``notified_signals``."""
    key_list = [k for k in keys if k]
    if not key_list:
        return []
    with _session(db_path) as conn:
        placeholders = ",".join("?" * len(key_list))
        rows = conn.execute(
            f"SELECT signal_key FROM notified_signals WHERE signal_key IN ({placeholders})",
            key_list,
        ).fetchall()
    seen = {str(r["signal_key"]) for r in rows}
    return [k for k in key_list if k not in seen]
