"""Global multi-competition SQLite schema and helpers.

Unifies fixtures across EPL / UWCL / WSL (etc.) into ``data/global_matches.db``
with three tables:

* ``teams`` — display name + gender-scoped ``team_key`` (``ARSENAL_M`` / ``ARSENAL_W``)
* ``competitions`` — competition id, name, and ``league_weight``
* ``matches`` — fixtures keyed by ``match_id``, linking team ids + comp

Critical: men's and women's clubs that share a brand name **must not** share
``team_id``. Rest / fatigue features therefore only see the correct schedule.

Default ``league_weight`` values (EPL baseline = 1.0)
----------------------------------------------------
=======  ==============================  ======
comp_id  comp_name                       weight
=======  ==============================  ======
EPL      Premier League                  1.00
WSL      Women's Super League            0.90
UWCL     UEFA Women's Champions League   0.85
=======  ==============================  ======

League-weight math (optional Dixon–Coles path)
----------------------------------------------
When two clubs from different domestic strengths meet (e.g. UWCL)::

    λ' = λ · (w_home / w_ref)
    μ' = μ · (w_away / w_ref)

with ``w_ref = 1.0`` (EPL). Single-league Streamlit fits leave weights at 1.0.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from src.data_loader import (
    DATA_DIR,
    STAT_COLUMNS,
    TEAM_ALIASES,
    canonicalize_team_columns,
    normalize_team_name,
)

logger = logging.getLogger(__name__)

GLOBAL_DB_PATH = DATA_DIR / "global_matches.db"
# v2: gender-scoped team_key (ARSENAL_M / ARSENAL_W) — no shared Nam/Nữ ids.
# v3: teams.flashscore_hash / flashscore_slug + team_rest_cache (additive).
# v4: teams.last_match_date + feed_updated_at for lazy Team Feed (additive).
# v5: upcoming_fixtures home/away+kickoff indexes for team profile queries.
GLOBAL_DB_VERSION = 5

GENDER_MEN = "M"
GENDER_WOMEN = "W"

# Documented defaults — keep in sync with module docstring.
DEFAULT_COMPETITIONS: tuple[tuple[str, str, float], ...] = (
    ("EPL", "Premier League", 1.0),
    ("WSL", "Women's Super League", 0.9),
    ("UWCL", "UEFA Women's Champions League", 0.85),
    ("LALIGA", "LaLiga - Spain", 0.95),
    ("EMPERORS_CUP", "Emperor's Cup - Japan", 0.8),
    # Populated by Flashscore team-results feeds (multi-comp rest_days).
    ("J1", "J1 League - Japan", 0.9),
    ("J2", "J2 League - Japan", 0.75),
    ("J3", "J3 League - Japan", 0.65),
    ("ACL", "AFC Champions League", 0.95),
    ("J_LEAGUE_CUP", "J.League Cup - Japan", 0.8),
    ("FRIENDLY", "Club Friendly", 0.3),
    ("FLASH_TEAM", "Flashscore team feed (unmapped)", 0.5),
)

# Competition → gender (men's / women's). Unknown codes infer from name.
COMPETITION_GENDER: dict[str, str] = {
    "EPL": GENDER_MEN,
    "WSL": GENDER_WOMEN,
    "UWCL": GENDER_WOMEN,
    "LALIGA": GENDER_MEN,
    "EMPERORS_CUP": GENDER_MEN,
    "J1": GENDER_MEN,
    "J2": GENDER_MEN,
    "J3": GENDER_MEN,
    "ACL": GENDER_MEN,
    "J_LEAGUE_CUP": GENDER_MEN,
    "FRIENDLY": GENDER_MEN,
    "FLASH_TEAM": GENDER_MEN,
}

# Optional columns copied from legacy league DBs when present.
OPTIONAL_MATCH_COLS: tuple[str, ...] = (
    "FTR",
    "Season",
    "SeasonStart",
    "FotmobMatchId",
    "Round",
    *STAT_COLUMNS,
    "B365H",
    "B365D",
    "B365A",
    "AvgH",
    "AvgD",
    "AvgA",
    "MaxH",
    "MaxD",
    "MaxA",
    "PSH",
    "PSD",
    "PSA",
    "B365_O25",
    "B365_U25",
    "Avg_O25",
    "Avg_U25",
    "AHh",
    "B365AHH",
    "B365AHA",
    "AvgAHH",
    "AvgAHA",
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_key TEXT NOT NULL UNIQUE,
    canonical_name TEXT NOT NULL,
    gender TEXT NOT NULL DEFAULT 'M',
    aliases TEXT NOT NULL DEFAULT '[]',
    country TEXT,
    flashscore_hash TEXT,
    flashscore_slug TEXT,
    last_match_date TEXT,
    feed_updated_at TEXT
);

CREATE TABLE IF NOT EXISTS competitions (
    comp_id TEXT PRIMARY KEY,
    comp_name TEXT NOT NULL,
    league_weight REAL NOT NULL DEFAULT 1.0
);

CREATE TABLE IF NOT EXISTS matches (
    match_id TEXT PRIMARY KEY,
    comp_id TEXT NOT NULL,
    home_team_id INTEGER NOT NULL,
    away_team_id INTEGER NOT NULL,
    match_date TEXT NOT NULL,
    home_score INTEGER,
    away_score INTEGER,
    ftr TEXT,
    season TEXT,
    season_start INTEGER,
    HC INTEGER,
    AC INTEGER,
    HS INTEGER,
    AS_ INTEGER,
    HST INTEGER,
    AST INTEGER,
    B365H REAL,
    B365D REAL,
    B365A REAL,
    AvgH REAL,
    AvgD REAL,
    AvgA REAL,
    MaxH REAL,
    MaxD REAL,
    MaxA REAL,
    PSH REAL,
    PSD REAL,
    PSA REAL,
    B365_O25 REAL,
    B365_U25 REAL,
    Avg_O25 REAL,
    Avg_U25 REAL,
    AHh REAL,
    B365AHH REAL,
    B365AHA REAL,
    AvgAHH REAL,
    AvgAHA REAL,
    fotmob_match_id TEXT,
    round TEXT,
    FOREIGN KEY (comp_id) REFERENCES competitions(comp_id),
    FOREIGN KEY (home_team_id) REFERENCES teams(team_id),
    FOREIGN KEY (away_team_id) REFERENCES teams(team_id)
);

CREATE INDEX IF NOT EXISTS idx_matches_comp_date
    ON matches (comp_id, match_date);
CREATE INDEX IF NOT EXISTS idx_matches_home
    ON matches (home_team_id, match_date);
CREATE INDEX IF NOT EXISTS idx_matches_away
    ON matches (away_team_id, match_date);
CREATE INDEX IF NOT EXISTS idx_teams_canonical
    ON teams (canonical_name);
CREATE INDEX IF NOT EXISTS idx_teams_key_gender
    ON teams (canonical_name, gender);

CREATE TABLE IF NOT EXISTS upcoming_fixtures (
    comp_id TEXT NOT NULL,
    match_key TEXT NOT NULL,
    kickoff TEXT,
    home_team TEXT,
    away_team TEXT,
    row_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (comp_id, match_key)
);
CREATE INDEX IF NOT EXISTS idx_upcoming_comp
    ON upcoming_fixtures (comp_id);
CREATE INDEX IF NOT EXISTS idx_upcoming_updated
    ON upcoming_fixtures (comp_id, updated_at);
CREATE INDEX IF NOT EXISTS idx_upcoming_home_ko
    ON upcoming_fixtures (home_team, kickoff);
CREATE INDEX IF NOT EXISTS idx_upcoming_away_ko
    ON upcoming_fixtures (away_team, kickoff);
CREATE INDEX IF NOT EXISTS idx_upcoming_kickoff
    ON upcoming_fixtures (kickoff);

CREATE TABLE IF NOT EXISTS team_rest_cache (
    team_id INTEGER NOT NULL,
    ref_date TEXT NOT NULL,
    rest_days REAL,
    matches_last_14d REAL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (team_id, ref_date),
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);
CREATE INDEX IF NOT EXISTS idx_team_rest_cache_updated
    ON team_rest_cache (updated_at);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _jsonable_cell(value: Any) -> Any:
    """Convert a DataFrame cell to a JSON-serializable value."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (pd.Timestamp, datetime)):
        ts = pd.Timestamp(value)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return ts.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    return value


def _upcoming_match_key(home: str, away: str, kickoff: Any) -> str:
    ko = ""
    if kickoff is not None and not (isinstance(kickoff, float) and pd.isna(kickoff)):
        try:
            ts = pd.Timestamp(kickoff)
            if ts.tzinfo is not None:
                ts = ts.tz_convert("UTC").tz_localize(None)
            ko = ts.isoformat()
        except (TypeError, ValueError):
            ko = str(kickoff)
    return f"{home}|{away}|{ko}"


def upsert_upcoming_fixture(
    comp_id: str,
    row: Mapping[str, Any] | pd.Series,
    *,
    db_path: Path | str = GLOBAL_DB_PATH,
    updated_at: datetime | None = None,
) -> str | None:
    """Insert or replace one upcoming fixture without wiping the competition cache.

    Returns the ``match_key`` written, or ``None`` when home/away are missing.
    """
    code = str(comp_id or "").strip().upper()
    if not code:
        return None

    if isinstance(row, pd.Series):
        payload_src: Mapping[str, Any] = row.to_dict()
        home = str(row.get("HomeTeam") or "")
        away = str(row.get("AwayTeam") or "")
        kick = row.get("Kickoff") if "Kickoff" in row.index else row.get("Date")
    else:
        payload_src = dict(row)
        home = str(payload_src.get("HomeTeam") or "")
        away = str(payload_src.get("AwayTeam") or "")
        kick = payload_src.get("Kickoff", payload_src.get("Date"))

    if not home or not away:
        return None

    ts = updated_at or datetime.now(timezone.utc)
    if getattr(ts, "tzinfo", None) is None:
        ts = ts.replace(tzinfo=timezone.utc)
    updated_iso = ts.replace(microsecond=0).isoformat()
    key = _upcoming_match_key(home, away, kick)
    payload = {c: _jsonable_cell(payload_src[c]) for c in payload_src}
    kick_iso = _jsonable_cell(kick)

    path = Path(db_path)
    with connect_global_db(path, init=True) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO upcoming_fixtures
                (comp_id, match_key, kickoff, home_team, away_team, row_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                code,
                key,
                str(kick_iso) if kick_iso is not None else None,
                home,
                away,
                json.dumps(payload, ensure_ascii=False),
                updated_iso,
            ),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (f"upcoming_updated_at:{code}", updated_iso),
        )
        conn.commit()
    return key


def save_upcoming_to_db(
    comp_id: str,
    df: pd.DataFrame,
    *,
    db_path: Path | str = GLOBAL_DB_PATH,
    updated_at: datetime | None = None,
) -> int:
    """Replace cached upcoming fixtures+odds for ``comp_id``.

    Returns number of rows written. Empty ``df`` clears the competition cache.
    """
    code = str(comp_id or "").strip().upper()
    if not code:
        return 0

    ts = updated_at or datetime.now(timezone.utc)
    if getattr(ts, "tzinfo", None) is None:
        ts = ts.replace(tzinfo=timezone.utc)
    updated_iso = ts.replace(microsecond=0).isoformat()

    path = Path(db_path)
    with connect_global_db(path, init=True) as conn:
        conn.execute("DELETE FROM upcoming_fixtures WHERE comp_id = ?", (code,))
        if df is None or getattr(df, "empty", True):
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (f"upcoming_updated_at:{code}", updated_iso),
            )
            conn.commit()
            return 0

        work = df.copy()
        n = 0
        for _, row in work.iterrows():
            home = str(row.get("HomeTeam") or "")
            away = str(row.get("AwayTeam") or "")
            kick = row.get("Kickoff") if "Kickoff" in work.columns else row.get("Date")
            if not home or not away:
                continue
            key = _upcoming_match_key(home, away, kick)
            payload = {c: _jsonable_cell(row[c]) for c in work.columns}
            kick_iso = _jsonable_cell(kick)
            conn.execute(
                """
                INSERT OR REPLACE INTO upcoming_fixtures
                    (comp_id, match_key, kickoff, home_team, away_team, row_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    code,
                    key,
                    str(kick_iso) if kick_iso is not None else None,
                    home,
                    away,
                    json.dumps(payload, ensure_ascii=False),
                    updated_iso,
                ),
            )
            n += 1
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (f"upcoming_updated_at:{code}", updated_iso),
        )
        # Preserve lightweight attrs for UI captions.
        attrs = {
            k: df.attrs.get(k)
            for k in ("odds_sources", "league", "data_source", "source_url")
            if k in df.attrs
        }
        if attrs:
            # odds_sources may be a list — JSON-safe.
            safe_attrs: dict[str, Any] = {}
            for k, v in attrs.items():
                if isinstance(v, (list, dict, str, int, float, bool)) or v is None:
                    safe_attrs[k] = v
                else:
                    safe_attrs[k] = str(v)
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (f"upcoming_attrs:{code}", json.dumps(safe_attrs, ensure_ascii=False)),
            )
        conn.commit()
    return n


def load_upcoming_from_db(
    comp_id: str,
    *,
    db_path: Path | str = GLOBAL_DB_PATH,
    max_age_minutes: float | None = None,
    now: datetime | None = None,
) -> tuple[pd.DataFrame, float | None]:
    """Load cached upcoming fixtures+odds from ``global_matches.db``.

    Parameters
    ----------
    comp_id:
        Competition code (``EPL``, ``UWCL``, …).
    max_age_minutes:
        When set, return an empty frame if the cache is older than this
        (``last_updated_ts`` is still returned when known).
    now:
        Reference time for age checks (default: UTC now).

    Returns
    -------
    (df, last_updated_ts)
        ``last_updated_ts`` is a Unix timestamp (UTC) or ``None`` if missing.
    """
    code = str(comp_id or "").strip().upper()
    path = Path(db_path)
    empty = pd.DataFrame()
    if not code or (str(db_path) != ":memory:" and not path.is_file()):
        return empty, None

    ref = now or datetime.now(timezone.utc)
    if getattr(ref, "tzinfo", None) is None:
        ref = ref.replace(tzinfo=timezone.utc)

    try:
        with connect_global_db(path, init=True) as conn:
            # Prefer meta stamp; fall back to MAX(updated_at).
            last_iso: str | None = None
            meta_row = conn.execute(
                "SELECT value FROM meta WHERE key = ?",
                (f"upcoming_updated_at:{code}",),
            ).fetchone()
            if meta_row is not None:
                last_iso = str(
                    meta_row["value"] if isinstance(meta_row, sqlite3.Row) else meta_row[0]
                )
            if not last_iso:
                max_row = conn.execute(
                    "SELECT MAX(updated_at) FROM upcoming_fixtures WHERE comp_id = ?",
                    (code,),
                ).fetchone()
                if max_row is not None and max_row[0]:
                    last_iso = str(max_row[0])

            last_ts: float | None = None
            if last_iso:
                try:
                    parsed = pd.Timestamp(last_iso)
                    if parsed.tzinfo is None:
                        parsed = parsed.tz_localize("UTC")
                    else:
                        parsed = parsed.tz_convert("UTC")
                    last_ts = float(parsed.timestamp())
                except (TypeError, ValueError):
                    last_ts = None

            if max_age_minutes is not None and last_ts is not None:
                age_min = (ref.timestamp() - last_ts) / 60.0
                if age_min > float(max_age_minutes):
                    return empty, last_ts

            rows = conn.execute(
                """
                SELECT row_json FROM upcoming_fixtures
                WHERE comp_id = ?
                ORDER BY kickoff
                """,
                (code,),
            ).fetchall()
            attrs_row = conn.execute(
                "SELECT value FROM meta WHERE key = ?",
                (f"upcoming_attrs:{code}",),
            ).fetchone()
    except sqlite3.Error:
        return empty, None

    if not rows:
        return empty, last_ts

    records: list[dict[str, Any]] = []
    for r in rows:
        raw = r["row_json"] if isinstance(r, sqlite3.Row) else r[0]
        try:
            records.append(json.loads(raw))
        except (TypeError, json.JSONDecodeError):
            continue
    if not records:
        return empty, last_ts

    df = pd.DataFrame.from_records(records)
    for col in ("Kickoff", "Date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in (
        "B365H",
        "B365D",
        "B365A",
        "B365_O25",
        "B365_U25",
        "OU_Line",
        "OddsOver",
        "OddsUnder",
        "AHh",
        "B365AHH",
        "B365AHA",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if attrs_row is not None:
        raw_attrs = attrs_row["value"] if isinstance(attrs_row, sqlite3.Row) else attrs_row[0]
        try:
            parsed_attrs = json.loads(str(raw_attrs))
            if isinstance(parsed_attrs, dict):
                df.attrs.update(parsed_attrs)
        except (TypeError, json.JSONDecodeError):
            pass
    df.attrs.setdefault("league", code)
    df.attrs.setdefault("data_source", "sqlite_upcoming")
    if last_ts is not None:
        df.attrs["db_updated_at"] = last_ts
    return df.reset_index(drop=True), last_ts


def upcoming_cache_age_minutes(
    last_updated_ts: float | None,
    *,
    now: datetime | None = None,
) -> float | None:
    """Minutes since ``last_updated_ts`` (Unix UTC), or ``None`` if unknown."""
    if last_updated_ts is None:
        return None
    ref = now or datetime.now(timezone.utc)
    if getattr(ref, "tzinfo", None) is None:
        ref = ref.replace(tzinfo=timezone.utc)
    return max(0.0, (ref.timestamp() - float(last_updated_ts)) / 60.0)


def ensure_data_dir(db_path: Path | str = GLOBAL_DB_PATH) -> Path:
    path = Path(db_path)
    if str(db_path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def competition_gender(comp_id: str | None) -> str:
    """Return ``M`` or ``W`` for a competition code."""
    cid = str(comp_id or "").strip().upper()
    if cid in COMPETITION_GENDER:
        return COMPETITION_GENDER[cid]
    # Prefer JSON registry when present.
    try:
        from src.league_registry import get_league_entry

        entry = get_league_entry(cid)
        if entry is not None:
            name = str(entry.get("name") or "")
            if any(tok in name.upper() for tok in ("WOMEN", "WSL", "UWCL")):
                return GENDER_WOMEN
            return GENDER_MEN
    except Exception:  # noqa: BLE001
        pass
    # Heuristic for future women's comps not yet seeded.
    if any(tok in cid for tok in ("WOMEN", "WSL", "UWCL", "_W", "WCL")):
        return GENDER_WOMEN
    return GENDER_MEN


def get_available_leagues() -> list[dict]:
    """Leagues from ``config/leagues.json`` (UI / scanners)."""
    from src.league_registry import get_available_leagues as _registry_leagues

    return _registry_leagues()


def seed_competitions_from_registry(conn: sqlite3.Connection) -> None:
    """Upsert competitions listed in ``config/leagues.json``."""
    try:
        leagues = get_available_leagues()
    except Exception:  # noqa: BLE001
        return
    for row in leagues:
        code = str(row.get("code") or "").strip().upper()
        if not code:
            continue
        name = str(row.get("name") or code)
        weight = float(row.get("league_weight") or 1.0)
        upsert_competition(conn, code, name, weight)


def normalize_gender(gender: str | None) -> str:
    """Map free-form gender / suffix → ``M`` or ``W``."""
    raw = str(gender or GENDER_MEN).strip().upper()
    if raw in {"W", "F", "FEMALE", "WOMEN", "WOMAN", "WOMENS"}:
        return GENDER_WOMEN
    if raw.endswith("_W") or raw.endswith("-W"):
        return GENDER_WOMEN
    return GENDER_MEN


def make_team_key(name: str, gender: str | None = GENDER_MEN) -> str:
    """Build stable slug ``ARSENAL_M`` / ``CHELSEA_W`` from display name + gender."""
    canon = canonical_team_name(name)
    slug = re.sub(r"[^A-Za-z0-9]+", "", canon).upper()
    if not slug:
        slug = "UNKNOWN"
    return f"{slug}_{normalize_gender(gender)}"


def connect_global_db(
    db_path: Path | str = GLOBAL_DB_PATH,
    *,
    init: bool = True,
) -> sqlite3.Connection:
    """Open ``global_matches.db``, optionally creating schema + seed comps."""
    if str(db_path) == ":memory:":
        conn = sqlite3.connect(":memory:")
    else:
        path = ensure_data_dir(db_path)
        conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if init:
        init_schema(conn)
    return conn


def _read_schema_version(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = ?", ("schema_version",)
        ).fetchone()
        if row is None:
            return 0
        return int(row["value"] if isinstance(row, sqlite3.Row) else row[0])
    except (sqlite3.Error, TypeError, ValueError):
        return 0


def _teams_has_team_key(conn: sqlite3.Connection) -> bool:
    try:
        cols = {
            str(r[1])
            for r in conn.execute("PRAGMA table_info(teams)").fetchall()
        }
        return "team_key" in cols
    except sqlite3.Error:
        return False


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {
            str(r[1])
            for r in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
    except sqlite3.Error:
        return set()


def _ensure_flashscore_team_columns(conn: sqlite3.Connection) -> None:
    """Additive v3/v4 migration: flashscore_* + feed cache columns on teams."""
    tables = {
        str(r[0])
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if "teams" not in tables:
        return
    cols = _table_columns(conn, "teams")
    alters: list[tuple[str, str]] = [
        ("flashscore_hash", "TEXT"),
        ("flashscore_slug", "TEXT"),
        ("last_match_date", "TEXT"),
        ("feed_updated_at", "TEXT"),
    ]
    for col, typ in alters:
        if col not in cols:
            conn.execute(f"ALTER TABLE teams ADD COLUMN {col} {typ}")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS team_rest_cache (
            team_id INTEGER NOT NULL,
            ref_date TEXT NOT NULL,
            rest_days REAL,
            matches_last_14d REAL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (team_id, ref_date),
            FOREIGN KEY (team_id) REFERENCES teams(team_id)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_team_rest_cache_updated
            ON team_rest_cache (updated_at)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_teams_flashscore_hash
            ON teams (flashscore_hash)
        """
    )
    _ensure_team_profile_indexes(conn)


def _ensure_team_profile_indexes(conn: sqlite3.Connection) -> None:
    """Additive indexes for fast team past/upcoming profile lookups."""
    tables = {
        str(r[0])
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if "upcoming_fixtures" in tables:
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_upcoming_home_ko
                ON upcoming_fixtures (home_team, kickoff)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_upcoming_away_ko
                ON upcoming_fixtures (away_team, kickoff)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_upcoming_kickoff
                ON upcoming_fixtures (kickoff)
            """
        )
    if "matches" in tables:
        # Already in SCHEMA_SQL; re-assert for older DBs.
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_matches_home
                ON matches (home_team_id, match_date)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_matches_away
                ON matches (away_team_id, match_date)
            """
        )


def _migrate_schema_if_needed(conn: sqlite3.Connection) -> None:
    """Rebuild teams/matches when pre-v2 schema lacks gender-scoped keys.

    League SQLite files remain the source of truth; callers re-sync after.
    v2→v3 is additive (see :func:`_ensure_flashscore_team_columns`).
    """
    tables = {
        str(r[0])
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if "teams" not in tables:
        return
    ver = _read_schema_version(conn)
    # Pre-v2 rebuild only — do not drop on v2→v3 (flashscore columns).
    if ver >= 2 and _teams_has_team_key(conn):
        return
    logger.warning(
        "global_matches.db schema v%s → v%s (gender team_key); rebuilding tables",
        ver,
        GLOBAL_DB_VERSION,
    )
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.executescript(
        """
        DROP TABLE IF EXISTS matches;
        DROP TABLE IF EXISTS teams;
        """
    )
    conn.execute("PRAGMA foreign_keys = ON")


def init_schema(conn: sqlite3.Connection) -> None:
    """Create tables/indexes and seed default competitions (idempotent)."""
    _migrate_schema_if_needed(conn)
    conn.executescript(SCHEMA_SQL)
    # Additive ALTERs for existing DBs created before flashscore_* columns.
    _ensure_flashscore_team_columns(conn)
    for comp_id, comp_name, weight in DEFAULT_COMPETITIONS:
        upsert_competition(conn, comp_id, comp_name, weight)
    seed_competitions_from_registry(conn)
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        ("schema_version", str(GLOBAL_DB_VERSION)),
    )
    conn.commit()


def upsert_competition(
    conn: sqlite3.Connection,
    comp_id: str,
    comp_name: str,
    league_weight: float = 1.0,
) -> str:
    """Insert or update a competition row. Returns ``comp_id``."""
    cid = str(comp_id).strip().upper()
    conn.execute(
        """
        INSERT INTO competitions (comp_id, comp_name, league_weight)
        VALUES (?, ?, ?)
        ON CONFLICT(comp_id) DO UPDATE SET
            comp_name = excluded.comp_name,
            league_weight = excluded.league_weight
        """,
        (cid, str(comp_name), float(league_weight)),
    )
    return cid


def get_competition_weight(
    conn: sqlite3.Connection,
    comp_id: str,
    *,
    default: float = 1.0,
) -> float:
    row = conn.execute(
        "SELECT league_weight FROM competitions WHERE comp_id = ?",
        (str(comp_id).strip().upper(),),
    ).fetchone()
    if row is None:
        return float(default)
    return float(row["league_weight"])


def _aliases_to_json(aliases: Iterable[str] | None) -> str:
    cleaned = sorted({str(a).strip() for a in (aliases or []) if str(a).strip()})
    return json.dumps(cleaned, ensure_ascii=False)


def _aliases_from_json(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [str(x) for x in data]


def reverse_alias_map(
    aliases: dict[str, str] | None = None,
) -> dict[str, set[str]]:
    """Map canonical_name → set of alias strings (including self)."""
    src = aliases if aliases is not None else TEAM_ALIASES
    out: dict[str, set[str]] = {}
    for alias, canon in src.items():
        out.setdefault(str(canon), set()).add(str(alias))
        out.setdefault(str(canon), set()).add(str(canon))
    return out


def canonical_team_name(name: str) -> str:
    """Normalize via :func:`normalize_team_name` (TEAM_ALIASES)."""
    return normalize_team_name(str(name or "").strip())


def upsert_team(
    conn: sqlite3.Connection,
    name: str,
    *,
    gender: str = GENDER_MEN,
    aliases: Iterable[str] | None = None,
    country: str | None = None,
    comp_id: str | None = None,
) -> int:
    """Get-or-create a gender-scoped team; merge aliases. Returns ``team_id``.

    Men's Arsenal (``ARSENAL_M``) and women's Arsenal (``ARSENAL_W``) are
    distinct rows — never share an id across genders.
    """
    g = (
        competition_gender(comp_id)
        if comp_id is not None
        else normalize_gender(gender)
    )
    canon = canonical_team_name(name)
    if not canon:
        raise ValueError("Cannot upsert team with empty name")
    key = make_team_key(canon, g)

    extra = {canonical_team_name(a) for a in (aliases or []) if str(a).strip()}
    rev = reverse_alias_map()
    extra |= rev.get(canon, set())
    extra.add(canon)
    extra.discard("")

    row = conn.execute(
        "SELECT team_id, aliases, country FROM teams WHERE team_key = ?",
        (key,),
    ).fetchone()
    if row is None:
        cur = conn.execute(
            """
            INSERT INTO teams (team_key, canonical_name, gender, aliases, country)
            VALUES (?, ?, ?, ?, ?)
            """,
            (key, canon, g, _aliases_to_json(extra), country),
        )
        return int(cur.lastrowid)

    team_id = int(row["team_id"])
    merged = set(_aliases_from_json(row["aliases"])) | extra
    new_country = country if country is not None else row["country"]
    conn.execute(
        "UPDATE teams SET aliases = ?, country = ?, canonical_name = ?, gender = ? "
        "WHERE team_id = ?",
        (_aliases_to_json(merged), new_country, canon, g, team_id),
    )
    return team_id


def resolve_team_id(
    conn: sqlite3.Connection,
    name: str,
    *,
    gender: str | None = None,
    comp_id: str | None = None,
    create: bool = False,
) -> int | None:
    """Resolve a display / aliased name to ``team_id`` within one gender.

    Lookup order: ``team_key`` → canonical+gender → alias JSON (same gender)
    → create (optional).
    """
    raw = str(name or "").strip()
    if not raw:
        return None
    g = (
        competition_gender(comp_id)
        if comp_id is not None
        else normalize_gender(gender if gender is not None else GENDER_MEN)
    )
    canon = canonical_team_name(raw)
    key = make_team_key(canon or raw, g)

    row = conn.execute(
        "SELECT team_id FROM teams WHERE team_key = ?",
        (key,),
    ).fetchone()
    if row:
        return int(row["team_id"])

    row = conn.execute(
        "SELECT team_id FROM teams WHERE canonical_name = ? AND gender = ?",
        (canon, g),
    ).fetchone()
    if row:
        return int(row["team_id"])

    # Scan aliases within the same gender only (avoid Nam↔Nữ collisions).
    for r in conn.execute(
        "SELECT team_id, canonical_name, aliases, gender FROM teams WHERE gender = ?",
        (g,),
    ).fetchall():
        alias_list = _aliases_from_json(r["aliases"])
        alias_set = {canonical_team_name(a) for a in alias_list}
        alias_set |= set(alias_list)
        alias_set.add(str(r["canonical_name"]))
        if canon in alias_set or raw in alias_set:
            return int(r["team_id"])

    if create:
        return upsert_team(conn, canon or raw, gender=g)
    return None


def _coerce_db_team_id(
    conn: sqlite3.Connection,
    team_id: int | str,
    *,
    gender: str | None = None,
    comp_id: str | None = None,
) -> int | None:
    """Accept integer ``team_id`` or stable code / display name → int PK."""
    if isinstance(team_id, int):
        return int(team_id)
    raw = str(team_id or "").strip()
    if not raw:
        return None
    if raw.isdigit():
        return int(raw)
    return resolve_team_id(
        conn, raw, gender=gender, comp_id=comp_id, create=False
    )


def get_team_flashscore_meta(
    conn: sqlite3.Connection,
    team_id: int | str,
    *,
    gender: str | None = None,
    comp_id: str | None = None,
) -> dict[str, Any] | None:
    """Return ``{team_id, team_key, canonical_name, hash, slug, gender}`` or None."""
    tid = _coerce_db_team_id(conn, team_id, gender=gender, comp_id=comp_id)
    if tid is None:
        return None
    cols = _table_columns(conn, "teams")
    hash_col = "flashscore_hash" if "flashscore_hash" in cols else None
    slug_col = "flashscore_slug" if "flashscore_slug" in cols else None
    select_hash = hash_col or "NULL"
    select_slug = slug_col or "NULL"
    row = conn.execute(
        f"""
        SELECT team_id, team_key, canonical_name, gender,
               {select_hash} AS flashscore_hash,
               {select_slug} AS flashscore_slug
        FROM teams WHERE team_id = ?
        """,
        (tid,),
    ).fetchone()
    if row is None:
        return None
    h = str(row["flashscore_hash"] or "").strip()
    slug = str(row["flashscore_slug"] or "").strip().strip("/")
    return {
        "team_id": int(row["team_id"]),
        "team_key": str(row["team_key"]),
        "canonical_name": str(row["canonical_name"]),
        "gender": str(row["gender"] or GENDER_MEN),
        "hash": h or None,
        "slug": slug or None,
        "flashscore_hash": h or None,
        "flashscore_slug": slug or None,
    }


def set_team_flashscore_hash(
    conn: sqlite3.Connection,
    team_id: int | str,
    flashscore_hash: str,
    *,
    flashscore_slug: str | None = None,
    gender: str | None = None,
    comp_id: str | None = None,
) -> int | None:
    """Set ``teams.flashscore_hash`` (+ optional slug) by integer id or code.

    Returns the integer ``team_id`` updated, or ``None`` if the team is missing.
    Does not clear an existing hash when ``flashscore_hash`` is empty.
    """
    _ensure_flashscore_team_columns(conn)
    tid = _coerce_db_team_id(conn, team_id, gender=gender, comp_id=comp_id)
    if tid is None:
        return None
    h = str(flashscore_hash or "").strip()
    if not h:
        return tid
    slug = str(flashscore_slug or "").strip().strip("/") or None
    if slug:
        conn.execute(
            """
            UPDATE teams
            SET flashscore_hash = ?,
                flashscore_slug = COALESCE(?, flashscore_slug)
            WHERE team_id = ?
            """,
            (h, slug, tid),
        )
    else:
        conn.execute(
            "UPDATE teams SET flashscore_hash = ? WHERE team_id = ?",
            (h, tid),
        )
    return tid


def set_team_feed_meta(
    conn: sqlite3.Connection,
    team_id: int | str,
    *,
    last_match_date: str | pd.Timestamp | None = None,
    feed_updated_at: datetime | str | None = None,
    gender: str | None = None,
    comp_id: str | None = None,
) -> int | None:
    """Update ``last_match_date`` / ``feed_updated_at`` on a teams row."""
    _ensure_flashscore_team_columns(conn)
    tid = _coerce_db_team_id(conn, team_id, gender=gender, comp_id=comp_id)
    if tid is None:
        return None
    sets: list[str] = []
    params: list[Any] = []
    if last_match_date is not None and not (
        isinstance(last_match_date, float) and pd.isna(last_match_date)
    ):
        try:
            date_s = pd.Timestamp(last_match_date).strftime("%Y-%m-%d")
        except (TypeError, ValueError):
            date_s = str(last_match_date)[:10]
        sets.append("last_match_date = ?")
        params.append(date_s)
    if feed_updated_at is not None:
        if isinstance(feed_updated_at, datetime):
            ts = feed_updated_at
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            iso = ts.replace(microsecond=0).isoformat()
        else:
            iso = str(feed_updated_at).strip()
        if iso:
            sets.append("feed_updated_at = ?")
            params.append(iso)
    if not sets:
        return tid
    params.append(tid)
    conn.execute(
        f"UPDATE teams SET {', '.join(sets)} WHERE team_id = ?",
        tuple(params),
    )
    return tid


def get_team_feed_meta(
    conn: sqlite3.Connection,
    team_id: int | str,
    *,
    gender: str | None = None,
    comp_id: str | None = None,
) -> dict[str, Any] | None:
    """Return ``{team_id, last_match_date, feed_updated_at, hash, slug, ...}``."""
    _ensure_flashscore_team_columns(conn)
    tid = _coerce_db_team_id(conn, team_id, gender=gender, comp_id=comp_id)
    if tid is None:
        return None
    row = conn.execute(
        """
        SELECT team_id, team_key, canonical_name, gender,
               flashscore_hash, flashscore_slug,
               last_match_date, feed_updated_at
        FROM teams WHERE team_id = ?
        """,
        (tid,),
    ).fetchone()
    if row is None:
        return None
    return {
        "team_id": int(row["team_id"]),
        "team_key": str(row["team_key"]),
        "canonical_name": str(row["canonical_name"]),
        "gender": str(row["gender"] or GENDER_MEN),
        "hash": str(row["flashscore_hash"] or "").strip() or None,
        "slug": str(row["flashscore_slug"] or "").strip().strip("/") or None,
        "last_match_date": str(row["last_match_date"] or "").strip() or None,
        "feed_updated_at": str(row["feed_updated_at"] or "").strip() or None,
    }


def upsert_team_flashscore_hash(
    conn: sqlite3.Connection,
    name: str,
    flashscore_hash: str,
    *,
    flashscore_slug: str | None = None,
    gender: str | None = None,
    comp_id: str | None = None,
    country: str | None = None,
) -> int:
    """Get-or-create team by name and store Flashscore hash/slug.

    Keeps stable codes (``JP_*``, ``ES_*``) via :func:`canonical_team_name`.
    """
    _ensure_flashscore_team_columns(conn)
    g = (
        competition_gender(comp_id)
        if comp_id is not None
        else normalize_gender(gender if gender is not None else GENDER_MEN)
    )
    tid = upsert_team(
        conn, name, gender=g, country=country, comp_id=comp_id
    )
    set_team_flashscore_hash(
        conn,
        tid,
        flashscore_hash,
        flashscore_slug=flashscore_slug,
    )
    return tid


def find_team_id_by_flashscore_hash(
    conn: sqlite3.Connection,
    flashscore_hash: str,
    *,
    gender: str | None = None,
) -> int | None:
    """Look up ``team_id`` by stored Flashscore hash (optional gender filter)."""
    h = str(flashscore_hash or "").strip()
    if not h:
        return None
    _ensure_flashscore_team_columns(conn)
    if gender is not None:
        row = conn.execute(
            """
            SELECT team_id FROM teams
            WHERE flashscore_hash = ? AND gender = ?
            """,
            (h, normalize_gender(gender)),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT team_id FROM teams WHERE flashscore_hash = ?",
            (h,),
        ).fetchone()
    return int(row["team_id"]) if row else None


def set_team_rest_cache(
    conn: sqlite3.Connection,
    team_id: int,
    ref_date: str | pd.Timestamp,
    *,
    rest_days: float | None,
    matches_last_14d: float | None = None,
    updated_at: datetime | None = None,
) -> None:
    """Upsert rest/fatigue snapshot for ``team_id`` as of ``ref_date``."""
    _ensure_flashscore_team_columns(conn)
    date_s = pd.Timestamp(ref_date).strftime("%Y-%m-%d")
    ts = updated_at or datetime.now(timezone.utc)
    if getattr(ts, "tzinfo", None) is None:
        ts = ts.replace(tzinfo=timezone.utc)
    updated_iso = ts.replace(microsecond=0).isoformat()
    rd = None if rest_days is None or (
        isinstance(rest_days, float) and rest_days != rest_days
    ) else float(rest_days)
    n14 = None if matches_last_14d is None else float(matches_last_14d)
    conn.execute(
        """
        INSERT INTO team_rest_cache
            (team_id, ref_date, rest_days, matches_last_14d, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(team_id, ref_date) DO UPDATE SET
            rest_days = excluded.rest_days,
            matches_last_14d = excluded.matches_last_14d,
            updated_at = excluded.updated_at
        """,
        (int(team_id), date_s, rd, n14, updated_iso),
    )


def get_team_rest_cache(
    conn: sqlite3.Connection,
    team_id: int,
    ref_date: str | pd.Timestamp,
    *,
    max_age_minutes: float | None = None,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Load cached rest_days for ``team_id`` / ``ref_date``, or None."""
    date_s = pd.Timestamp(ref_date).strftime("%Y-%m-%d")
    try:
        row = conn.execute(
            """
            SELECT rest_days, matches_last_14d, updated_at
            FROM team_rest_cache
            WHERE team_id = ? AND ref_date = ?
            """,
            (int(team_id), date_s),
        ).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    updated_raw = str(row["updated_at"] or "")
    if max_age_minutes is not None and updated_raw:
        try:
            updated = datetime.fromisoformat(updated_raw.replace("Z", "+00:00"))
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            ref = now or datetime.now(timezone.utc)
            if getattr(ref, "tzinfo", None) is None:
                ref = ref.replace(tzinfo=timezone.utc)
            age_min = (ref - updated).total_seconds() / 60.0
            if age_min > float(max_age_minutes):
                return None
        except (TypeError, ValueError):
            pass
    rd = row["rest_days"]
    n14 = row["matches_last_14d"]
    return {
        "team_id": int(team_id),
        "ref_date": date_s,
        "rest_days": float(rd) if rd is not None else float("nan"),
        "matches_last_14d": float(n14) if n14 is not None else 0.0,
        "updated_at": updated_raw,
    }


def build_global_match_id(
    comp_id: str,
    match_date: str | pd.Timestamp,
    home_canon: str,
    away_canon: str,
    *,
    legacy_match_id: str | None = None,
) -> str:
    """Stable PK: ``{COMP}|{YYYY-MM-DD}|{Home}|{Away}``.

    When ``legacy_match_id`` is provided it is embedded after the comp prefix
    so re-imports stay idempotent with football-data / Fotmob keys.
    """
    cid = str(comp_id).strip().upper()
    if legacy_match_id:
        # Avoid double-prefix on re-run.
        mid = str(legacy_match_id)
        if mid.upper().startswith(f"{cid}|"):
            return mid
        return f"{cid}|{mid}"
    date_s = pd.Timestamp(match_date).strftime("%Y-%m-%d")
    return f"{cid}|{date_s}|{home_canon}|{away_canon}"


def _sql_null(value: Any) -> Any:
    if value is None or (isinstance(value, float) and value != value):
        return None
    if pd.isna(value):
        return None
    return value


def upsert_match(
    conn: sqlite3.Connection,
    *,
    match_id: str,
    comp_id: str,
    home_team_id: int,
    away_team_id: int,
    match_date: str | pd.Timestamp,
    home_score: int | None,
    away_score: int | None,
    extras: dict[str, Any] | None = None,
) -> str:
    """Idempotent insert/replace of one match row. Returns ``match_id``."""
    cid = str(comp_id).strip().upper()
    date_s = pd.Timestamp(match_date).strftime("%Y-%m-%d")
    ex = extras or {}

    # Map legacy column names → schema (AS is reserved → AS_).
    col_map = {
        "FTR": "ftr",
        "Season": "season",
        "SeasonStart": "season_start",
        "HC": "HC",
        "AC": "AC",
        "HS": "HS",
        "AS": "AS_",
        "HST": "HST",
        "AST": "AST",
        "B365H": "B365H",
        "B365D": "B365D",
        "B365A": "B365A",
        "AvgH": "AvgH",
        "AvgD": "AvgD",
        "AvgA": "AvgA",
        "MaxH": "MaxH",
        "MaxD": "MaxD",
        "MaxA": "MaxA",
        "PSH": "PSH",
        "PSD": "PSD",
        "PSA": "PSA",
        "B365_O25": "B365_O25",
        "B365_U25": "B365_U25",
        "Avg_O25": "Avg_O25",
        "Avg_U25": "Avg_U25",
        "AHh": "AHh",
        "B365AHH": "B365AHH",
        "B365AHA": "B365AHA",
        "AvgAHH": "AvgAHH",
        "AvgAHA": "AvgAHA",
        "FotmobMatchId": "fotmob_match_id",
        "Round": "round",
        "ftr": "ftr",
        "season": "season",
        "season_start": "season_start",
        "fotmob_match_id": "fotmob_match_id",
        "round": "round",
    }

    fields: dict[str, Any] = {
        "match_id": match_id,
        "comp_id": cid,
        "home_team_id": int(home_team_id),
        "away_team_id": int(away_team_id),
        "match_date": date_s,
        "home_score": _sql_null(home_score),
        "away_score": _sql_null(away_score),
    }
    for src, dest in col_map.items():
        if src in ex and dest not in fields:
            fields[dest] = _sql_null(ex[src])

    columns = list(fields.keys())
    placeholders = ", ".join("?" for _ in columns)
    col_sql = ", ".join(columns)
    updates = ", ".join(f"{c} = excluded.{c}" for c in columns if c != "match_id")
    conn.execute(
        f"""
        INSERT INTO matches ({col_sql})
        VALUES ({placeholders})
        ON CONFLICT(match_id) DO UPDATE SET {updates}
        """,
        tuple(fields[c] for c in columns),
    )
    return match_id


def import_legacy_matches_df(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    *,
    comp_id: str,
    dry_run: bool = False,
) -> dict[str, int]:
    """Upsert a legacy flat matches DataFrame into the global schema.

    Returns counts: ``teams_touched``, ``matches_upserted``, ``skipped``.
    """
    if df.empty:
        return {"teams_touched": 0, "matches_upserted": 0, "skipped": 0}

    work = canonicalize_team_columns(df.copy())
    if "Date" not in work.columns:
        raise ValueError("legacy matches missing Date column")

    teams_touched: set[int] = set()
    upserted = 0
    skipped = 0
    gender = competition_gender(comp_id)

    for _, row in work.iterrows():
        home = canonical_team_name(str(row.get("HomeTeam", "")))
        away = canonical_team_name(str(row.get("AwayTeam", "")))
        if not home or not away:
            skipped += 1
            continue
        date = row["Date"]
        if pd.isna(date):
            skipped += 1
            continue

        if dry_run:
            upserted += 1
            continue

        home_id = upsert_team(conn, home, gender=gender, comp_id=comp_id)
        away_id = upsert_team(conn, away, gender=gender, comp_id=comp_id)
        teams_touched.add(home_id)
        teams_touched.add(away_id)

        legacy_id = None
        if "Match_ID" in work.columns and pd.notna(row.get("Match_ID")):
            legacy_id = str(row["Match_ID"])
        match_id = build_global_match_id(
            comp_id, date, home, away, legacy_match_id=legacy_id
        )

        extras = {c: row[c] for c in OPTIONAL_MATCH_COLS if c in work.columns}
        # Prefer FTR from row; if missing derive.
        if "FTR" not in extras or _sql_null(extras.get("FTR")) is None:
            try:
                hs = int(row["FTHG"])
                aws = int(row["FTAG"])
                extras["FTR"] = "H" if hs > aws else ("A" if hs < aws else "D")
            except (TypeError, ValueError, KeyError):
                pass

        upsert_match(
            conn,
            match_id=match_id,
            comp_id=comp_id,
            home_team_id=home_id,
            away_team_id=away_id,
            match_date=date,
            home_score=int(row["FTHG"]) if pd.notna(row.get("FTHG")) else None,
            away_score=int(row["FTAG"]) if pd.notna(row.get("FTAG")) else None,
            extras=extras,
        )
        upserted += 1

    if not dry_run:
        conn.commit()
    return {
        "teams_touched": len(teams_touched) if not dry_run else 0,
        "matches_upserted": upserted,
        "skipped": skipped,
    }


def global_db_has_matches(
    db_path: Path | str = GLOBAL_DB_PATH,
    *,
    comp_id: str | None = None,
) -> bool:
    path = Path(db_path)
    if not path.is_file():
        return False
    try:
        with sqlite3.connect(path) as conn:
            if comp_id:
                row = conn.execute(
                    "SELECT COUNT(*) FROM matches WHERE comp_id = ?",
                    (str(comp_id).strip().upper(),),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM matches").fetchone()
        return bool(row and int(row[0]) > 0)
    except sqlite3.Error:
        return False


def read_matches_as_legacy(
    db_path: Path | str = GLOBAL_DB_PATH,
    *,
    comp_id: str | None = None,
    seasons: Sequence[int] | None = None,
    n_seasons: int | None = None,
) -> pd.DataFrame:
    """Load global matches projected to the legacy flat column layout.

    Columns mirror league DBs: ``Date``, ``HomeTeam``, ``AwayTeam``,
    ``FTHG``, ``FTAG``, ``FTR``, stats/odds, plus ``home_team_id`` /
    ``away_team_id`` / ``comp_id`` for multi-comp features.
    """
    path = Path(db_path)
    if not path.is_file():
        return pd.DataFrame()

    sql = """
        SELECT
            m.match_id AS Match_ID,
            m.comp_id AS league_id,
            m.comp_id,
            m.match_date AS Date,
            th.canonical_name AS HomeTeam,
            ta.canonical_name AS AwayTeam,
            th.team_key AS home_team_key,
            ta.team_key AS away_team_key,
            th.gender AS home_gender,
            ta.gender AS away_gender,
            m.home_team_id,
            m.away_team_id,
            m.home_score AS FTHG,
            m.away_score AS FTAG,
            m.ftr AS FTR,
            m.season AS Season,
            m.season_start AS SeasonStart,
            m.HC, m.AC, m.HS, m.AS_ AS "AS", m.HST, m.AST,
            m.B365H, m.B365D, m.B365A,
            m.AvgH, m.AvgD, m.AvgA,
            m.MaxH, m.MaxD, m.MaxA,
            m.PSH, m.PSD, m.PSA,
            m.B365_O25, m.B365_U25, m.Avg_O25, m.Avg_U25,
            m.AHh, m.B365AHH, m.B365AHA, m.AvgAHH, m.AvgAHA,
            m.fotmob_match_id AS FotmobMatchId,
            m.round AS Round
        FROM matches m
        JOIN teams th ON th.team_id = m.home_team_id
        JOIN teams ta ON ta.team_id = m.away_team_id
    """
    params: list[Any] = []
    if comp_id:
        sql += " WHERE m.comp_id = ?"
        params.append(str(comp_id).strip().upper())

    with sqlite3.connect(path) as conn:
        try:
            df = pd.read_sql(sql, conn, params=params or None)
        except (sqlite3.Error, pd.errors.DatabaseError):
            return pd.DataFrame()

    if df.empty:
        return df

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ("FTHG", "FTAG"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(
        subset=[c for c in ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG") if c in df.columns]
    )
    if "FTHG" in df.columns:
        df["FTHG"] = df["FTHG"].astype(int)
    if "FTAG" in df.columns:
        df["FTAG"] = df["FTAG"].astype(int)
    for col in STAT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    if "SeasonStart" in df.columns:
        df["SeasonStart"] = pd.to_numeric(df["SeasonStart"], errors="coerce").astype(
            "Int64"
        )

    if seasons is not None and "SeasonStart" in df.columns:
        starts = {int(s) for s in seasons}
        df = df.loc[df["SeasonStart"].isin(starts)].copy()
    elif n_seasons is not None and "SeasonStart" in df.columns:
        from src.data_loader import default_season_starts

        starts = set(default_season_starts(n_seasons))
        df = df.loc[df["SeasonStart"].isin(starts)].copy()

    df = df.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    return canonicalize_team_columns(df)


def apply_league_weight_to_rates(
    lam: float,
    mu: float,
    home_weight: float,
    away_weight: float,
    *,
    w_ref: float = 1.0,
) -> tuple[float, float]:
    """Scale expected goals by competition strength ratios.

    Math
    ----
    ``λ' = λ · (w_home / w_ref)``, ``μ' = μ · (w_away / w_ref)``.

    With ``w_ref = 1.0`` (EPL baseline) this is a no-op when both weights are
    1.0. Relative domestic strength in cups is captured by differing
    ``w_home`` / ``w_away`` (e.g. WSL 0.9 vs weaker domestic 0.7).
    """
    ref = float(w_ref) if float(w_ref) > 0 else 1.0
    wh = max(float(home_weight), 1e-6)
    wa = max(float(away_weight), 1e-6)
    return float(lam) * (wh / ref), float(mu) * (wa / ref)


def sync_to_global_db(
    db_path: Path | str = GLOBAL_DB_PATH,
    *,
    force: bool = True,
    sources: Sequence[tuple[str, Path]] | None = None,
) -> dict[str, Any]:
    """Upsert all league SQLite DBs into ``global_matches.db`` (gender-scoped).

    Parameters
    ----------
    force:
        When False and the global DB already has rows, skip re-import.
    sources:
        Optional ``(comp_id, path)`` pairs; default = every
        :data:`LEAGUE_CONFIG` DB that exists on disk.

    Returns
    -------
    dict
        ``ok``, ``per_comp`` stats, ``matches_upserted`` total.
    """
    path = Path(db_path) if str(db_path) != ":memory:" else Path(":memory:")
    out: dict[str, Any] = {
        "ok": False,
        "per_comp": {},
        "matches_upserted": 0,
        "skipped": False,
    }
    try:
        if not force and str(db_path) != ":memory:" and global_db_has_matches(path):
            out["ok"] = True
            out["skipped"] = True
            return out

        from src.data_loader import LEAGUE_CONFIG, read_matches_from_db

        pairs: list[tuple[str, Path]]
        if sources is not None:
            pairs = [(str(c).upper(), Path(p)) for c, p in sources]
        else:
            pairs = []
            for code, cfg in LEAGUE_CONFIG.items():
                legacy = Path(cfg["db_path"])  # type: ignore[arg-type]
                if legacy.is_file():
                    pairs.append((str(code).upper(), legacy))

        conn = connect_global_db(db_path, init=True)
        try:
            total = 0
            for code, legacy in pairs:
                df = read_matches_from_db(legacy)
                if df.empty:
                    out["per_comp"][code] = {
                        "matches_upserted": 0,
                        "skipped": 0,
                        "teams_touched": 0,
                        "source_rows": 0,
                    }
                    continue
                name = next(
                    (n for c, n, _ in DEFAULT_COMPETITIONS if c == code),
                    str(code),
                )
                weight = next(
                    (w for c, _, w in DEFAULT_COMPETITIONS if c == code),
                    1.0,
                )
                upsert_competition(conn, code, name, float(weight))
                stats = import_legacy_matches_df(conn, df, comp_id=code)
                stats["source_rows"] = int(len(df))
                out["per_comp"][code] = stats
                total += int(stats.get("matches_upserted", 0))
            conn.commit()
            out["matches_upserted"] = total
            out["ok"] = global_db_has_matches(db_path) if str(db_path) != ":memory:" else total > 0
            return out
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        logger.exception("sync_to_global_db failed: %s", exc)
        out["error"] = str(exc)
        out["ok"] = global_db_has_matches(path) if str(db_path) != ":memory:" else False
        return out


def ensure_global_db_from_legacy(
    db_path: Path | str = GLOBAL_DB_PATH,
    *,
    force: bool = False,
) -> bool:
    """Create/populate ``global_matches.db`` from league SQLite files if needed.

    Returns True when the global DB has at least one match afterwards.
    Never raises — Streamlit callers can treat False as “use legacy fallback”.

    When ``force=True``, re-imports every available league DB (after schema
    init / gender migration) so Top-20 / Lite always see fresh schedules.
    """
    path = Path(db_path)
    try:
        if not force and global_db_has_matches(path):
            return True
        stats = sync_to_global_db(db_path=path, force=True)
        return bool(stats.get("ok")) and global_db_has_matches(path)
    except Exception:  # noqa: BLE001
        return global_db_has_matches(path)


def team_weight_for_match(
    conn: sqlite3.Connection,
    team_id: int,
    match_comp_id: str,
    *,
    default: float = 1.0,
) -> float:
    """Best-effort domestic weight for a team in a cup fixture.

    Preference order:
    1. Most frequent non-cup competition the team has played
       (anything other than ``UWCL``).
    2. Weight of ``match_comp_id`` itself.
    3. ``default``.
    """
    cup_like = {"UWCL", "EMPERORS_CUP", "J_LEAGUE_CUP"}
    rows = conn.execute(
        """
        SELECT comp_id, COUNT(*) AS n
        FROM matches
        WHERE (home_team_id = ? OR away_team_id = ?)
          AND comp_id NOT IN ({cups})
        GROUP BY comp_id
        ORDER BY n DESC
        LIMIT 1
        """.format(cups=",".join("?" for _ in cup_like)),
        (team_id, team_id, *sorted(cup_like)),
    ).fetchall()
    if rows:
        return get_competition_weight(conn, rows[0]["comp_id"], default=default)
    return get_competition_weight(conn, match_comp_id, default=default)


# ---------------------------------------------------------------------------
# Team profile (past + upcoming) — DB-only, no network
# ---------------------------------------------------------------------------

_STABLE_CODE_RE = re.compile(r"^[A-Z]{2,}_[A-Z0-9_]+$")
_DISPLAY_NAME_CACHE: dict[str, str] = {}
_REGISTRY_DISPLAY: dict[str, str] | None = None


def _looks_like_stable_code(name: str) -> bool:
    return bool(_STABLE_CODE_RE.match(str(name or "").strip()))


def _humanize_stable_code(code: str) -> str:
    """``JP_VISSEL_KOBE`` → ``Vissel Kobe`` (best-effort fallback)."""
    raw = str(code or "").strip()
    if not raw:
        return raw
    parts = raw.split("_", 1)
    body = parts[1] if len(parts) == 2 and len(parts[0]) <= 3 else raw
    return " ".join(p.capitalize() for p in body.split("_") if p)


def _registry_display_map() -> dict[str, str]:
    """Lazy JP_*/ES_* → display name map (cached)."""
    global _REGISTRY_DISPLAY
    if _REGISTRY_DISPLAY is not None:
        return _REGISTRY_DISPLAY
    out: dict[str, str] = {}
    try:
        from src.fetchers.flashscore_team import team_hash_registry

        for code, meta in team_hash_registry().items():
            nm = str(meta.get("name") or "").strip()
            if nm:
                out[str(code).upper()] = nm
    except Exception:  # noqa: BLE001
        pass
    try:
        from src.fetchers.flashscore_league import (
            EMPERORS_CUP_TEAM_ALIASES,
            LALIGA_TEAM_ALIASES,
        )

        for alias_map in (EMPERORS_CUP_TEAM_ALIASES, LALIGA_TEAM_ALIASES):
            for display, code in alias_map.items():
                c = str(code).upper()
                d = str(display).strip()
                if d and not _looks_like_stable_code(d):
                    # Prefer longer display names.
                    prev = out.get(c, "")
                    if len(d) >= len(prev):
                        out[c] = d
    except Exception:  # noqa: BLE001
        pass
    _REGISTRY_DISPLAY = out
    return out


def _comp_display_name(conn: sqlite3.Connection, comp_id: str) -> str:
    cid = str(comp_id or "").strip().upper()
    if not cid:
        return ""
    row = conn.execute(
        "SELECT comp_name FROM competitions WHERE comp_id = ?", (cid,)
    ).fetchone()
    if row and str(row["comp_name"] or "").strip():
        return str(row["comp_name"]).strip()
    try:
        from src.data_loader import league_label

        return str(league_label(cid))
    except Exception:  # noqa: BLE001
        return cid


def _display_name_for_canonical(
    conn: sqlite3.Connection,
    canonical: str,
    *,
    aliases: Sequence[str] | None = None,
) -> str:
    """Prefer a human display name over ``JP_*`` / ``ES_*`` stable codes."""
    canon = str(canonical or "").strip()
    if not canon:
        return ""
    cached = _DISPLAY_NAME_CACHE.get(canon)
    if cached is not None and not aliases:
        return cached

    candidates: list[str] = []
    for a in aliases or []:
        s = str(a or "").strip()
        if s and not _looks_like_stable_code(s):
            candidates.append(s)
    rev = reverse_alias_map()
    for a in rev.get(canon, set()):
        if a and not _looks_like_stable_code(a) and a not in candidates:
            candidates.append(a)
    if _looks_like_stable_code(canon):
        reg = _registry_display_map().get(canon.upper())
        if reg:
            candidates.insert(0, reg)

    if candidates:
        candidates.sort(key=lambda s: (-len(s), s))
        result = candidates[0]
    elif _looks_like_stable_code(canon):
        result = _humanize_stable_code(canon)
    else:
        result = canon
    if not aliases:
        _DISPLAY_NAME_CACHE[canon] = result
    return result


def _team_name_variants(
    conn: sqlite3.Connection,
    db_team_id: int,
    canonical: str,
    aliases: Sequence[str],
) -> list[str]:
    """All strings that may appear as home/away in upcoming_fixtures."""
    names: set[str] = {canonical, str(db_team_id)}
    names.update(str(a).strip() for a in aliases if str(a).strip())
    display = _display_name_for_canonical(conn, canonical, aliases=aliases)
    if display:
        names.add(display)
    rev = reverse_alias_map()
    names |= {str(a) for a in rev.get(canonical, set()) if str(a).strip()}
    # Drop empties; keep order stable for tests.
    out = sorted({n for n in names if n})
    return out


def _result_from_perspective(
    *,
    perspective_team_id: int,
    home_team_id: int,
    away_team_id: int,
    home_score: int | None,
    away_score: int | None,
) -> str | None:
    """Return ``W`` / ``D`` / ``L`` from ``perspective_team_id``'s point of view."""
    if home_score is None or away_score is None:
        return None
    try:
        hs = int(home_score)
        ags = int(away_score)
    except (TypeError, ValueError):
        return None
    if perspective_team_id == int(home_team_id):
        if hs > ags:
            return "W"
        if hs < ags:
            return "L"
        return "D"
    if perspective_team_id == int(away_team_id):
        if ags > hs:
            return "W"
        if ags < hs:
            return "L"
        return "D"
    return None


# Light in-process cache for card rows that call history twice (home+away / re-render).
_TEAM_HISTORY_CACHE: dict[tuple[Any, ...], bool] = {}
_TEAM_HISTORY_CACHE_MAX = 512


def clear_sufficient_team_data_cache() -> None:
    """Drop the in-process :func:`has_team_history` cache (tests / after import)."""
    _TEAM_HISTORY_CACHE.clear()


clear_team_history_cache = clear_sufficient_team_data_cache


def count_team_finished_matches(
    conn: sqlite3.Connection,
    db_team_id: int,
    *,
    before_date: str | None = None,
    on_or_before: str | None = None,
) -> int:
    """Count finished matches for ``db_team_id`` across all competitions.

    Finished = both ``home_score`` and ``away_score`` present. Date filter is
    exclusive (``match_date < before_date``) or inclusive
    (``match_date <= on_or_before``). Uses home/away indexes (UNION), not OR.
    """
    tid = int(db_team_id)
    if before_date:
        cutoff = str(before_date)[:10]
        op, bound = "<", cutoff
    elif on_or_before:
        cutoff = str(on_or_before)[:10]
        op, bound = "<=", cutoff
    else:
        op, bound = None, None

    date_clause = f" AND m.match_date {op} ?" if op else ""
    params_home: tuple[Any, ...] = (tid, bound) if bound else (tid,)
    params_away: tuple[Any, ...] = (tid, bound) if bound else (tid,)
    sql = f"""
        SELECT COUNT(*) AS n FROM (
            SELECT m.match_id
            FROM matches m
            WHERE m.home_team_id = ?
              {date_clause}
              AND m.home_score IS NOT NULL
              AND m.away_score IS NOT NULL
            UNION
            SELECT m.match_id
            FROM matches m
            WHERE m.away_team_id = ?
              {date_clause}
              AND m.home_score IS NOT NULL
              AND m.away_score IS NOT NULL
        )
    """
    row = conn.execute(sql, params_home + params_away).fetchone()
    if row is None:
        return 0
    try:
        return int(row["n"] if hasattr(row, "keys") else row[0])
    except (TypeError, ValueError, KeyError, IndexError):
        return 0


def _resolve_team_pk(
    conn: sqlite3.Connection,
    team_id: int | str,
    *,
    gender: str | None = None,
    comp_id: str | None = None,
) -> int | None:
    """Resolve display name / code / int → ``teams.team_id`` (try M then W)."""
    tid = _coerce_db_team_id(conn, team_id, gender=gender, comp_id=comp_id)
    if tid is not None:
        return int(tid)
    if isinstance(team_id, int):
        return None
    label = str(team_id).strip()
    if not label or gender is not None or comp_id is not None:
        return None
    tid = resolve_team_id(conn, label, gender="M", create=False)
    if tid is None:
        tid = resolve_team_id(conn, label, gender="W", create=False)
    return int(tid) if tid is not None else None


def has_team_history(
    team_id: int | str,
    *,
    db_path: Path | str | None = None,
    gender: str | None = None,
    comp_id: str | None = None,
) -> bool:
    """True if **any** finished match exists for this team in ``global_matches.db``.

    Resolves ``team_id`` (int PK, stable code, or display alias) → ``teams.team_id``,
    then counts rows where the team is ``home_team_id`` or ``away_team_id`` with
    both scores present. No date filter — any history hides ``[⚠️ Thiếu Data Đội]``.
    Independent of Dixon–Coles thin priors and ``flashscore_hash``.
    """
    label = str(team_id).strip() if not isinstance(team_id, int) else str(int(team_id))
    if not label:
        return False

    path = Path(db_path) if db_path is not None else GLOBAL_DB_PATH
    path_key = str(path)
    cache_key = (path_key, label, gender or "", comp_id or "")
    cached = _TEAM_HISTORY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if path_key != ":memory:" and not Path(path).is_file():
        _TEAM_HISTORY_CACHE[cache_key] = False
        return False

    try:
        conn = connect_global_db(path, init=True)
    except Exception:  # noqa: BLE001
        _TEAM_HISTORY_CACHE[cache_key] = False
        return False

    try:
        tid = _resolve_team_pk(conn, team_id, gender=gender, comp_id=comp_id)
        ok = (
            count_team_finished_matches(conn, int(tid)) >= 1
            if tid is not None
            else False
        )
    finally:
        conn.close()

    if len(_TEAM_HISTORY_CACHE) >= _TEAM_HISTORY_CACHE_MAX:
        _TEAM_HISTORY_CACHE.clear()
    _TEAM_HISTORY_CACHE[cache_key] = ok
    return ok


def has_sufficient_team_data(
    team_id: int | str,
    *,
    upcoming_date: str | datetime | pd.Timestamp | None = None,
    min_matches: int = 1,
    db_path: Path | str | None = None,
    gender: str | None = None,
    comp_id: str | None = None,
) -> bool:
    """Backward-compatible wrapper → :func:`has_team_history` for badge use.

    When ``min_matches <= 1`` (default), ignores ``upcoming_date`` and returns
    whether any finished history exists. Higher ``min_matches`` still counts
    finished rows (optionally before ``upcoming_date``).
    """
    threshold = max(0, int(min_matches))
    if threshold == 0:
        return True
    if threshold <= 1:
        return has_team_history(
            team_id, db_path=db_path, gender=gender, comp_id=comp_id
        )

    label = str(team_id).strip() if not isinstance(team_id, int) else str(int(team_id))
    if not label:
        return False

    path = Path(db_path) if db_path is not None else GLOBAL_DB_PATH
    path_key = str(path)

    before: str | None = None
    if upcoming_date is not None and not (
        isinstance(upcoming_date, float) and pd.isna(upcoming_date)
    ):
        try:
            before = pd.Timestamp(upcoming_date).strftime("%Y-%m-%d")
        except (TypeError, ValueError):
            before = None

    cache_key = (
        "count",
        path_key,
        label,
        before or "",
        threshold,
        gender or "",
        comp_id or "",
    )
    cached = _TEAM_HISTORY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if path_key != ":memory:" and not Path(path).is_file():
        _TEAM_HISTORY_CACHE[cache_key] = False
        return False

    try:
        conn = connect_global_db(path, init=True)
    except Exception:  # noqa: BLE001
        _TEAM_HISTORY_CACHE[cache_key] = False
        return False

    try:
        tid = _resolve_team_pk(conn, team_id, gender=gender, comp_id=comp_id)
        if tid is None:
            ok = False
        else:
            n = count_team_finished_matches(
                conn,
                int(tid),
                before_date=before,
            )
            ok = n >= threshold
    finally:
        conn.close()

    if len(_TEAM_HISTORY_CACHE) >= _TEAM_HISTORY_CACHE_MAX:
        _TEAM_HISTORY_CACHE.clear()
    _TEAM_HISTORY_CACHE[cache_key] = ok
    return ok


def get_team_profile_data(
    team_id: int | str,
    limit: int = 5,
    *,
    db_path: Path | str | None = None,
    as_of: str | datetime | pd.Timestamp | None = None,
    gender: str | None = None,
    comp_id: str | None = None,
) -> dict[str, Any]:
    """Load last/next fixtures for a team from ``global_matches.db`` (no network).

    Parameters
    ----------
    team_id:
        Integer ``teams.team_id`` or stable code / display name (``JP_VISSEL_KOBE``).
    limit:
        Max rows per list (past and upcoming), default 5.
    as_of:
        Reference date/time (default: now UTC). Past = ``match_date <= as_of``
        with finished scores; upcoming = ``kickoff > as_of``.

    Returns
    -------
    dict
        ``team_id``, ``canonical_name``, ``display_name``, ``past_matches``,
        ``upcoming_matches``. Each past row has ``date``, ``competition``,
        ``home``, ``away``, ``score``, ``result`` (W/D/L). Each upcoming row
        has ``kickoff``, ``competition``, ``opponent``, ``home``, ``away``.
    """
    n = max(0, int(limit))
    path = Path(db_path) if db_path is not None else GLOBAL_DB_PATH
    ref = as_of if as_of is not None else datetime.now(timezone.utc)
    try:
        ref_ts = pd.Timestamp(ref)
        if getattr(ref_ts, "tzinfo", None) is not None:
            ref_ts = ref_ts.tz_convert("UTC").tz_localize(None)
    except (TypeError, ValueError):
        ref_ts = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)
    as_of_date = ref_ts.strftime("%Y-%m-%d")
    as_of_iso = ref_ts.strftime("%Y-%m-%dT%H:%M:%S")

    empty: dict[str, Any] = {
        "team_id": team_id,
        "db_team_id": None,
        "canonical_name": str(team_id),
        "display_name": str(team_id),
        "past_matches": [],
        "upcoming_matches": [],
        "as_of": as_of_iso,
    }
    if not str(team_id or "").strip() and not isinstance(team_id, int):
        return empty

    # Skip heavy init_schema on existing files (CREATE INDEX IF NOT EXISTS adds
    # latency); still create schema for missing / in-memory DBs.
    need_init = str(path) == ":memory:" or not Path(path).is_file()
    conn = connect_global_db(path, init=need_init)
    try:
        # Ensure profile indexes once if an older DB is missing them.
        has_idx = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' "
            "AND name='idx_upcoming_home_ko' LIMIT 1"
        ).fetchone()
        if has_idx is None:
            _ensure_team_profile_indexes(conn)
            conn.commit()
        tid = _coerce_db_team_id(conn, team_id, gender=gender, comp_id=comp_id)
        if tid is None and gender is None and not isinstance(team_id, int):
            tid = resolve_team_id(conn, str(team_id), gender="M", create=False)
            if tid is None:
                tid = resolve_team_id(conn, str(team_id), gender="W", create=False)
        if tid is None:
            empty["display_name"] = _display_name_for_canonical(
                conn, str(team_id), aliases=[]
            )
            return empty

        row = conn.execute(
            """
            SELECT team_id, team_key, canonical_name, aliases
            FROM teams WHERE team_id = ?
            """,
            (int(tid),),
        ).fetchone()
        if row is None:
            return empty

        canon = str(row["canonical_name"] or "")
        aliases = _aliases_from_json(row["aliases"])
        display = _display_name_for_canonical(conn, canon, aliases=aliases)
        variants = _team_name_variants(conn, int(tid), canon, aliases)

        # Past: UNION of home/away indexed lookups (avoids OR index miss).
        past_sql = """
            SELECT * FROM (
                SELECT
                    m.match_date AS match_date,
                    m.comp_id AS comp_id,
                    m.home_team_id AS home_team_id,
                    m.away_team_id AS away_team_id,
                    m.home_score AS home_score,
                    m.away_score AS away_score,
                    th.canonical_name AS home_canon,
                    ta.canonical_name AS away_canon,
                    th.aliases AS home_aliases,
                    ta.aliases AS away_aliases,
                    COALESCE(c.comp_name, m.comp_id) AS comp_name
                FROM matches m
                JOIN teams th ON th.team_id = m.home_team_id
                JOIN teams ta ON ta.team_id = m.away_team_id
                LEFT JOIN competitions c ON c.comp_id = m.comp_id
                WHERE m.home_team_id = ?
                  AND m.match_date <= ?
                  AND m.home_score IS NOT NULL
                  AND m.away_score IS NOT NULL
                UNION ALL
                SELECT
                    m.match_date,
                    m.comp_id,
                    m.home_team_id,
                    m.away_team_id,
                    m.home_score,
                    m.away_score,
                    th.canonical_name,
                    ta.canonical_name,
                    th.aliases,
                    ta.aliases,
                    COALESCE(c.comp_name, m.comp_id)
                FROM matches m
                JOIN teams th ON th.team_id = m.home_team_id
                JOIN teams ta ON ta.team_id = m.away_team_id
                LEFT JOIN competitions c ON c.comp_id = m.comp_id
                WHERE m.away_team_id = ?
                  AND m.match_date <= ?
                  AND m.home_score IS NOT NULL
                  AND m.away_score IS NOT NULL
            )
            ORDER BY match_date DESC
            LIMIT ?
        """
        past_rows = conn.execute(
            past_sql, (int(tid), as_of_date, int(tid), as_of_date, n)
        ).fetchall()

        disp_cache: dict[str, str] = {canon: display}
        past_matches: list[dict[str, Any]] = []
        for pr in past_rows:
            hs = pr["home_score"]
            ags = pr["away_score"]
            hc = str(pr["home_canon"])
            ac = str(pr["away_canon"])
            if hc not in disp_cache:
                disp_cache[hc] = _display_name_for_canonical(
                    conn, hc, aliases=_aliases_from_json(pr["home_aliases"])
                )
            if ac not in disp_cache:
                disp_cache[ac] = _display_name_for_canonical(
                    conn, ac, aliases=_aliases_from_json(pr["away_aliases"])
                )
            result = _result_from_perspective(
                perspective_team_id=int(tid),
                home_team_id=int(pr["home_team_id"]),
                away_team_id=int(pr["away_team_id"]),
                home_score=hs,
                away_score=ags,
            )
            past_matches.append(
                {
                    "date": str(pr["match_date"])[:10],
                    "competition": str(pr["comp_name"] or pr["comp_id"] or ""),
                    "comp_id": str(pr["comp_id"] or "").upper(),
                    "home": disp_cache[hc],
                    "away": disp_cache[ac],
                    "home_score": int(hs) if hs is not None else None,
                    "away_score": int(ags) if ags is not None else None,
                    "score": f"{int(hs)}-{int(ags)}" if hs is not None and ags is not None else "",
                    "result": result,
                }
            )

        upcoming_matches: list[dict[str, Any]] = []
        if variants and n > 0:
            placeholders = ", ".join("?" for _ in variants)
            up_sql = f"""
                SELECT u.kickoff, u.comp_id, u.home_team, u.away_team,
                       COALESCE(c.comp_name, u.comp_id) AS comp_name
                FROM upcoming_fixtures u
                LEFT JOIN competitions c ON c.comp_id = u.comp_id
                WHERE u.kickoff > ?
                  AND (u.home_team IN ({placeholders}) OR u.away_team IN ({placeholders}))
                ORDER BY u.kickoff ASC
                LIMIT ?
            """
            params: list[Any] = [as_of_iso, *variants, *variants, n]
            up_rows = conn.execute(up_sql, params).fetchall()
            if not up_rows:
                up_sql_date = f"""
                    SELECT u.kickoff, u.comp_id, u.home_team, u.away_team,
                           COALESCE(c.comp_name, u.comp_id) AS comp_name
                    FROM upcoming_fixtures u
                    LEFT JOIN competitions c ON c.comp_id = u.comp_id
                    WHERE date(u.kickoff) > date(?)
                      AND (u.home_team IN ({placeholders})
                           OR u.away_team IN ({placeholders}))
                    ORDER BY u.kickoff ASC
                    LIMIT ?
                """
                up_rows = conn.execute(
                    up_sql_date, [as_of_date, *variants, *variants, n]
                ).fetchall()

            variant_cf = {v.casefold() for v in variants}
            for ur in up_rows:
                home_raw = str(ur["home_team"] or "").strip()
                away_raw = str(ur["away_team"] or "").strip()
                if home_raw not in disp_cache:
                    disp_cache[home_raw] = _display_name_for_canonical(conn, home_raw)
                if away_raw not in disp_cache:
                    disp_cache[away_raw] = _display_name_for_canonical(conn, away_raw)
                home_disp = disp_cache[home_raw]
                away_disp = disp_cache[away_raw]
                is_home = home_raw.casefold() in variant_cf
                opponent = away_disp if is_home else home_disp
                upcoming_matches.append(
                    {
                        "kickoff": str(ur["kickoff"] or ""),
                        "competition": str(ur["comp_name"] or ur["comp_id"] or ""),
                        "comp_id": str(ur["comp_id"] or "").upper(),
                        "home": home_disp,
                        "away": away_disp,
                        "opponent": opponent,
                        "is_home": is_home,
                    }
                )

        return {
            "team_id": team_id,
            "db_team_id": int(tid),
            "team_key": str(row["team_key"] or ""),
            "canonical_name": canon,
            "display_name": display,
            "past_matches": past_matches,
            "upcoming_matches": upcoming_matches,
            "as_of": as_of_iso,
        }
    finally:
        conn.close()
