r"""Paper-trading journal: persist live bets in SQLite ``live_bets``.

Tracks PENDING → WIN/LOSS/PUSH settlements, Closing-Line Value (CLV),
and realised PnL against a starting bankroll.

Schema notes
------------
* Canonical table: ``live_bets`` (used by Streamlit journal UI).
* Compatibility views (created on ensure):
  - ``paper_trades`` → alias of ``live_bets``
  - ``journal_history`` → settled rows only (WIN/LOSS/PUSH)
There is no separate SCHEDULED fixtures table; post-match automation
operates on PENDING journal bets whose ``match_date``/kickoff has passed,
matched against refreshed historical rows in the league SQLite DB.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.calibration import brier_score
from src.config import (
    CLOSE_ODDS_WINDOW_MAX_MINUTES,
    CLOSE_ODDS_WINDOW_MIN_MINUTES,
)
from src.data_loader import DEFAULT_DB_PATH

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

LIVE_BETS_TABLE = "live_bets"
PAPER_TRADES_VIEW = "paper_trades"
JOURNAL_HISTORY_VIEW = "journal_history"
BetStatus = Literal["PENDING", "WIN", "LOSS", "PUSH"]

# Match considered finished this long after kickoff (FT + buffer).
DEFAULT_SETTLE_GRACE = timedelta(hours=2)

# Canonical selection labels used for dedup / CLV column mapping.
_SELECTION_ALIASES: dict[str, str] = {
    "home": "Home",
    "h": "Home",
    "chu": "Home",
    "chủ": "Home",
    "chu nha": "Home",
    "chủ nhà": "Home",
    "draw": "Draw",
    "d": "Draw",
    "hoa": "Draw",
    "hòa": "Draw",
    "away": "Away",
    "a": "Away",
    "khach": "Away",
    "khách": "Away",
}

_SELECTION_TO_ODDS_COL: dict[str, tuple[str, ...]] = {
    "Home": ("B365H", "AvgH", "PSH", "MaxH"),
    "Draw": ("B365D", "AvgD", "PSD", "MaxD"),
    "Away": ("B365A", "AvgA", "PSA", "MaxA"),
}

# Fixture / live-odds column aliases (upcoming feed) → 1X2 selection.
_FIXTURE_ODDS_COL: dict[str, tuple[str, ...]] = {
    "Home": ("B365H", "AvgH", "OddsH", "PSH"),
    "Draw": ("B365D", "AvgD", "OddsD", "PSD"),
    "Away": ("B365A", "AvgA", "OddsA", "PSA"),
}


def normalize_selection(selection: str) -> str:
    """Map UI / model selection text → Home | Draw | Away (else original)."""
    raw = str(selection or "").strip()
    key = " ".join(raw.lower().replace("thắng", "").replace("thang", "").split())
    if key in _SELECTION_ALIASES:
        return _SELECTION_ALIASES[key]
    # "Chủ thắng (Home)" / "Home" already
    for alias, canon in _SELECTION_ALIASES.items():
        if alias in key:
            return canon
    if raw in {"Home", "Draw", "Away"}:
        return raw
    return raw


def _match_day_key(match_date: Any) -> str | None:
    """YYYY-MM-DD for dedup across slightly different kickoff strings."""
    if match_date is None or (isinstance(match_date, float) and pd.isna(match_date)):
        return None
    try:
        return pd.Timestamp(match_date).strftime("%Y-%m-%d")
    except Exception:
        text = str(match_date).strip()
        return text[:10] if len(text) >= 10 else text or None


def ensure_live_bets_table(db_path: Path | str = DEFAULT_DB_PATH) -> Path:
    """Create ``live_bets`` if missing and migrate CLV columns + alias views."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {LIVE_BETS_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_date TEXT,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                market TEXT DEFAULT '1X2',
                selection TEXT NOT NULL,
                odds REAL NOT NULL,
                stake_amount REAL NOT NULL,
                stake_pct REAL,
                ev REAL,
                p_model REAL,
                status TEXT NOT NULL DEFAULT 'PENDING',
                pnl REAL DEFAULT 0,
                closing_odds REAL,
                clv_pct REAL,
                created_at TEXT,
                settled_at TEXT
            )
            """
        )
        cols = {
            str(r[1])
            for r in conn.execute(f"PRAGMA table_info({LIVE_BETS_TABLE})").fetchall()
        }
        if "closing_odds" not in cols:
            conn.execute(
                f"ALTER TABLE {LIVE_BETS_TABLE} ADD COLUMN closing_odds REAL"
            )
        if "clv_pct" not in cols:
            conn.execute(f"ALTER TABLE {LIVE_BETS_TABLE} ADD COLUMN clv_pct REAL")
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_live_bets_status
            ON {LIVE_BETS_TABLE} (status)
            """
        )
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_live_bets_pending_dedup
            ON {LIVE_BETS_TABLE} (home_team, away_team, market, selection, status)
            """
        )
        # Compatibility aliases requested by post-match pipeline / external tools.
        conn.execute(
            f"""
            CREATE VIEW IF NOT EXISTS {PAPER_TRADES_VIEW} AS
            SELECT * FROM {LIVE_BETS_TABLE}
            """
        )
        conn.execute(
            f"""
            CREATE VIEW IF NOT EXISTS {JOURNAL_HISTORY_VIEW} AS
            SELECT * FROM {LIVE_BETS_TABLE}
            WHERE status IN ('WIN', 'LOSS', 'PUSH')
            """
        )
        conn.commit()
    return path


def pending_bet_exists(
    *,
    home_team: str,
    away_team: str,
    selection: str,
    market: str = "1X2",
    match_date: str | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> int | None:
    """Return existing PENDING bet id if same fixture/selection already queued."""
    path = ensure_live_bets_table(db_path)
    sel = normalize_selection(selection)
    day = _match_day_key(match_date)
    with sqlite3.connect(path) as conn:
        rows = conn.execute(
            f"""
            SELECT id, match_date, selection FROM {LIVE_BETS_TABLE}
            WHERE status = 'PENDING'
              AND home_team = ?
              AND away_team = ?
              AND UPPER(market) = UPPER(?)
            """,
            (str(home_team).strip(), str(away_team).strip(), str(market).strip()),
        ).fetchall()
    for bet_id, md, sel_db in rows:
        if normalize_selection(str(sel_db)) != sel:
            continue
        if day is None or _match_day_key(md) in {None, day}:
            return int(bet_id)
    return None


def add_live_bet(
    *,
    home_team: str,
    away_team: str,
    selection: str,
    odds: float,
    stake_amount: float,
    match_date: str | None = None,
    market: str = "1X2",
    stake_pct: float | None = None,
    ev: float | None = None,
    p_model: float | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
    skip_if_pending: bool = False,
) -> int | None:
    """Insert a PENDING paper bet. Returns new row ``id`` (or ``None`` if deduped)."""
    if skip_if_pending:
        existing = pending_bet_exists(
            home_team=home_team,
            away_team=away_team,
            selection=selection,
            market=market,
            match_date=match_date,
            db_path=db_path,
        )
        if existing is not None:
            return None

    path = ensure_live_bets_table(db_path)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    md = match_date
    if md is not None and not isinstance(md, str):
        md = pd.Timestamp(md).strftime("%Y-%m-%d %H:%M")
    sel = normalize_selection(selection)

    with sqlite3.connect(path) as conn:
        cur = conn.execute(
            f"""
            INSERT INTO {LIVE_BETS_TABLE} (
                match_date, home_team, away_team, market, selection,
                odds, stake_amount, stake_pct, ev, p_model,
                status, pnl, closing_odds, clv_pct, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', 0, NULL, NULL, ?)
            """,
            (
                md,
                str(home_team).strip(),
                str(away_team).strip(),
                str(market).strip(),
                sel,
                float(odds),
                float(stake_amount),
                float(stake_pct) if stake_pct is not None else None,
                float(ev) if ev is not None else None,
                float(p_model) if p_model is not None else None,
                now,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def add_to_journal(
    *,
    home_team: str,
    away_team: str,
    selection: str,
    odds: float,
    stake_amount: float,
    match_date: str | None = None,
    market: str = "1X2",
    stake_pct: float | None = None,
    ev: float | None = None,
    p_model: float | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Dedup-aware journal insert for the auto scanner / UI.

    Returns
    -------
    dict
        ``{\"id\": int|None, \"created\": bool, \"reason\": str}``.
    """
    existing = pending_bet_exists(
        home_team=home_team,
        away_team=away_team,
        selection=selection,
        market=market,
        match_date=match_date,
        db_path=db_path,
    )
    if existing is not None:
        return {
            "id": existing,
            "created": False,
            "reason": f"duplicate_pending:{existing}",
        }

    new_id = add_live_bet(
        home_team=home_team,
        away_team=away_team,
        selection=selection,
        odds=odds,
        stake_amount=stake_amount,
        match_date=match_date,
        market=market,
        stake_pct=stake_pct,
        ev=ev,
        p_model=p_model,
        db_path=db_path,
        skip_if_pending=False,
    )
    return {"id": new_id, "created": True, "reason": "created"}


def add_recommendations_to_journal(
    recommendations: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    db_path: Path | str = DEFAULT_DB_PATH,
    bankroll: float = 1000.0,
    markets: Sequence[str] = ("1X2",),
) -> dict[str, Any]:
    """Bulk-insert Value Bet rows into ``live_bets`` with PENDING dedup.

    Returns counts: ``created``, ``duplicates``, ``ids``.
    """
    if recommendations is None:
        return {"created": 0, "duplicates": 0, "ids": []}
    if isinstance(recommendations, pd.DataFrame):
        rows = recommendations.to_dict(orient="records")
    else:
        rows = list(recommendations)

    allowed = {str(m).upper() for m in markets} if markets else set()
    created = 0
    duplicates = 0
    ids: list[int] = []

    for row in rows:
        market = str(row.get("market", "1X2") or "1X2")
        if allowed and market.upper() not in allowed:
            continue
        home = str(row.get("home_team") or row.get("HomeTeam") or "").strip()
        away = str(row.get("away_team") or row.get("AwayTeam") or "").strip()
        selection = str(row.get("selection") or "")
        odds = float(row.get("bookmaker_odds") or row.get("odds") or 0)
        if not home or not away or odds <= 1.0:
            continue

        stake = row.get("stake") or row.get("stake_amount")
        stake_pct = row.get("kelly_fraction") or row.get("stake_pct")
        if stake is None:
            stake = float(bankroll) * float(stake_pct or 0.0)
        if stake_pct is None and float(bankroll) > 0:
            stake_pct = float(stake) / float(bankroll)

        ev = row.get("ev")
        if ev is None and row.get("ev_pct") is not None:
            ev = float(row["ev_pct"]) / 100.0

        kick = row.get("match_date") or row.get("kickoff") or row.get("Kickoff")
        result = add_to_journal(
            home_team=home,
            away_team=away,
            selection=selection,
            odds=odds,
            stake_amount=float(stake),
            match_date=str(kick) if kick is not None else None,
            market=market,
            stake_pct=float(stake_pct) if stake_pct is not None else None,
            ev=float(ev) if ev is not None else None,
            p_model=float(row["p_model"]) if row.get("p_model") is not None else None,
            db_path=db_path,
        )
        if result["created"] and result["id"] is not None:
            created += 1
            ids.append(int(result["id"]))
        else:
            duplicates += 1
            if result["id"] is not None:
                ids.append(int(result["id"]))

    return {"created": created, "duplicates": duplicates, "ids": ids}


def compute_clv_pct(odds_entered: float, closing_odds: float) -> float:
    """CLV = (odds_entered / closing_odds) − 1  (fraction; 0.05 = +5%).

    ``closing_odds`` is the canonical store; callers may pass ``odds_close``.
    """
    entered = float(odds_entered)
    closing = float(closing_odds)
    if entered <= 1.0 or closing <= 1.0:
        raise ValueError("odds_entered and closing_odds must be > 1")
    return (entered / closing) - 1.0


def closing_odds_from_row(
    selection: str,
    row: Mapping[str, Any] | pd.Series,
    *,
    prefer_fixture: bool = False,
) -> float | None:
    """Extract a closing / settlement price for a 1X2 selection from a match row.

    Tries B365 → Avg → PS → Max (results) or fixture aliases when
    ``prefer_fixture``. Returns ``None`` when no price > 1.0 exists.
    """
    sel = normalize_selection(selection)
    cols = (
        _FIXTURE_ODDS_COL.get(sel, ())
        if prefer_fixture
        else _SELECTION_TO_ODDS_COL.get(sel, ())
    )
    if not cols:
        # Fall back to the other map when selection is 1X2 but source differs.
        cols = _SELECTION_TO_ODDS_COL.get(sel, ()) or _FIXTURE_ODDS_COL.get(sel, ())
    if isinstance(row, pd.Series):
        index = row.index
        get = row.get
    else:
        index = row.keys()  # type: ignore[assignment]
        get = row.get  # type: ignore[assignment]
    for c in cols:
        if c not in index:
            continue
        val = get(c)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        try:
            price = float(val)
        except (TypeError, ValueError):
            continue
        if price > 1.0:
            return price
    return None


def update_closing_odds(
    bet_id: int,
    closing_odds: float | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
    *,
    odds_close: float | None = None,
) -> float:
    """Set closing odds for a bet and store ``clv_pct``.

    ``clv_pct = (odds_entered / closing_odds) - 1``

    Accepts ``odds_close`` as an alias for ``closing_odds`` (canonical column).
    Positive CLV ⇒ got a better (higher) price than the closing market.
    Returns the stored ``clv_pct`` fraction.
    """
    price = closing_odds if closing_odds is not None else odds_close
    if price is None:
        raise ValueError("closing_odds / odds_close is required")

    path = ensure_live_bets_table(db_path)
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            f"SELECT odds FROM {LIVE_BETS_TABLE} WHERE id = ?",
            (int(bet_id),),
        ).fetchone()
        if row is None:
            raise KeyError(f"live_bets id={bet_id} not found")
        entered = float(row[0])
        clv = compute_clv_pct(entered, float(price))
        conn.execute(
            f"""
            UPDATE {LIVE_BETS_TABLE}
            SET closing_odds = ?, clv_pct = ?
            WHERE id = ?
            """,
            (float(price), float(clv), int(bet_id)),
        )
        conn.commit()
    return float(clv)


def sync_closing_odds_from_results(
    results: pd.DataFrame,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> dict[str, int]:
    """Fill ``closing_odds`` / ``clv_pct`` from finished-match odds columns.

    Matches journal rows (any status) that still lack ``closing_odds`` against
    ``results`` on (date, home, away) and selection → B365H/D/A (or Avg/PS).
    """
    if results is None or results.empty:
        return {"updated": 0, "skipped": 0}

    path = ensure_live_bets_table(db_path)
    df = results.copy()
    if "Date" in df.columns:
        df["_day"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    else:
        return {"updated": 0, "skipped": 0}

    updated = 0
    skipped = 0
    bets = load_live_bets(db_path)
    if bets.empty:
        return {"updated": 0, "skipped": 0}

    need = bets.loc[bets["closing_odds"].isna()] if "closing_odds" in bets.columns else bets
    for _, bet in need.iterrows():
        day = _match_day_key(bet.get("match_date"))
        if not day:
            skipped += 1
            continue
        home = str(bet["home_team"]).strip()
        away = str(bet["away_team"]).strip()
        hit = df.loc[
            (df["_day"] == day)
            & (df["HomeTeam"].astype(str).str.strip() == home)
            & (df["AwayTeam"].astype(str).str.strip() == away)
        ]
        if hit.empty:
            skipped += 1
            continue
        closing = closing_odds_from_row(str(bet["selection"]), hit.iloc[0])
        if closing is None:
            skipped += 1
            continue
        try:
            update_closing_odds(int(bet["id"]), closing, db_path=path)
            updated += 1
        except Exception:
            skipped += 1

    return {"updated": updated, "skipped": skipped}


def count_bets_on_vn_day(
    db_path: Path | str = DEFAULT_DB_PATH,
    *,
    now: datetime | None = None,
) -> int:
    """Count journal bets created on the current VN (Asia/Ho_Chi_Minh) calendar day."""
    path = ensure_live_bets_table(db_path)
    ref = now or datetime.now(VN_TZ)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=VN_TZ)
    else:
        ref = ref.astimezone(VN_TZ)
    day = ref.strftime("%Y-%m-%d")
    bets = load_live_bets(path)
    if bets.empty or "created_at" not in bets.columns:
        return 0
    n = 0
    for raw in bets["created_at"]:
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            continue
        try:
            ts = pd.Timestamp(raw)
            if ts.tzinfo is None:
                # Stored as UTC ``…Z`` or naive UTC.
                text = str(raw)
                if text.endswith("Z") or "+00" in text:
                    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
                else:
                    ts = ts.tz_localize("UTC")
            vn = ts.tz_convert(VN_TZ)
            if vn.strftime("%Y-%m-%d") == day:
                n += 1
        except Exception:
            continue
    return n


def snapshot_closing_odds_near_kickoff(
    fixtures: pd.DataFrame,
    db_path: Path | str = DEFAULT_DB_PATH,
    *,
    now: datetime | None = None,
    window_min_minutes: int = CLOSE_ODDS_WINDOW_MIN_MINUTES,
    window_max_minutes: int = CLOSE_ODDS_WINDOW_MAX_MINUTES,
    overwrite: bool = False,
) -> dict[str, int]:
    """Snapshot live book odds into ``closing_odds`` for PENDING near-KO bets.

    When the scanner runs ``window_min``–``window_max`` minutes before kickoff,
    copies current fixture 1X2 prices into the journal so CLV is available even
    if post-match results lack Avg/B365 columns.

    Returns ``{\"updated\": n, \"skipped\": m}``.
    """
    if fixtures is None or fixtures.empty:
        return {"updated": 0, "skipped": 0}

    path = ensure_live_bets_table(db_path)
    pending = load_live_bets(path, status="PENDING")
    if pending.empty:
        return {"updated": 0, "skipped": 0}

    ref = now or datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)

    fx = fixtures.copy()
    if "HomeTeam" not in fx.columns or "AwayTeam" not in fx.columns:
        return {"updated": 0, "skipped": 0}

    updated = 0
    skipped = 0
    for _, bet in pending.iterrows():
        if (
            not overwrite
            and "closing_odds" in bet.index
            and pd.notna(bet.get("closing_odds"))
        ):
            skipped += 1
            continue
        md = bet.get("match_date")
        if md is None or (isinstance(md, float) and pd.isna(md)):
            skipped += 1
            continue
        try:
            ko = pd.Timestamp(md)
        except Exception:
            skipped += 1
            continue
        if pd.isna(ko):
            skipped += 1
            continue
        if ko.tzinfo is None:
            ko_utc = ko.tz_localize("UTC")
        else:
            ko_utc = ko.tz_convert("UTC")
        minutes_to_ko = (ko_utc.to_pydatetime() - ref).total_seconds() / 60.0
        if minutes_to_ko < float(window_min_minutes) or minutes_to_ko > float(
            window_max_minutes
        ):
            skipped += 1
            continue

        home = str(bet["home_team"]).strip()
        away = str(bet["away_team"]).strip()
        hit = fx.loc[
            (fx["HomeTeam"].astype(str).str.strip() == home)
            & (fx["AwayTeam"].astype(str).str.strip() == away)
        ]
        if hit.empty:
            skipped += 1
            continue
        closing = closing_odds_from_row(
            str(bet["selection"]), hit.iloc[0], prefer_fixture=True
        )
        if closing is None:
            skipped += 1
            continue
        try:
            update_closing_odds(int(bet["id"]), closing, db_path=path)
            updated += 1
        except Exception:
            skipped += 1

    return {"updated": updated, "skipped": skipped}


def load_live_bets(
    db_path: Path | str = DEFAULT_DB_PATH,
    *,
    status: BetStatus | None = None,
) -> pd.DataFrame:
    """Load journal rows, optionally filtered by ``status``."""
    path = ensure_live_bets_table(db_path)
    with sqlite3.connect(path) as conn:
        if status:
            df = pd.read_sql(
                f"SELECT * FROM {LIVE_BETS_TABLE} WHERE status = ? ORDER BY id DESC",
                conn,
                params=(status,),
            )
        else:
            df = pd.read_sql(
                f"SELECT * FROM {LIVE_BETS_TABLE} ORDER BY id DESC",
                conn,
            )
    return df


def load_live_bets_all_leagues(
    leagues: Sequence[str] = ("EPL", "UWCL"),
    *,
    status: BetStatus | None = None,
) -> pd.DataFrame:
    """Load ``live_bets`` from each league SQLite DB and tag ``league``.

    Paper bets live in per-league databases (``epl_matches.db`` /
    ``uwcl_matches.db``); there is no ``league`` column on the table itself.
    """
    from src.data_loader import league_db_path, normalize_league

    frames: list[pd.DataFrame] = []
    for raw in leagues:
        code = normalize_league(raw)
        part = load_live_bets(league_db_path(code), status=status)
        if part.empty:
            continue
        part = part.copy()
        part["league"] = code
        frames.append(part)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def settle_live_bet(
    bet_id: int,
    status: BetStatus,
    db_path: Path | str = DEFAULT_DB_PATH,
    *,
    closing_odds: float | None = None,
    odds_close: float | None = None,
    pnl: float | None = None,
) -> float:
    """Mark bet WIN/LOSS/PUSH and write PnL. Optionally set closing odds/CLV.

    Parameters
    ----------
    pnl:
        Optional cash PnL override (needed for half-win / half-lose Asian lines).
        When ``None``, PnL is derived from full stake × (odds − 1) / −stake / 0.
    closing_odds / odds_close:
        Optional closing price (aliases). Sets ``clv_pct`` when provided.
    """
    if status not in {"WIN", "LOSS", "PUSH"}:
        raise ValueError(f"status must be WIN/LOSS/PUSH, got {status!r}")
    if status == "PENDING":
        raise ValueError("Cannot settle as PENDING")

    path = ensure_live_bets_table(db_path)
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            f"SELECT odds, stake_amount, status FROM {LIVE_BETS_TABLE} WHERE id = ?",
            (int(bet_id),),
        ).fetchone()
        if row is None:
            raise KeyError(f"live_bets id={bet_id} not found")
        odds, stake, cur_status = float(row[0]), float(row[1]), str(row[2])
        if cur_status != "PENDING":
            raise RuntimeError(f"Bet #{bet_id} already settled as {cur_status}")

        if pnl is not None:
            realised = float(pnl)
        elif status == "WIN":
            realised = stake * (odds - 1.0)
        elif status == "LOSS":
            realised = -stake
        else:
            realised = 0.0

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        conn.execute(
            f"""
            UPDATE {LIVE_BETS_TABLE}
            SET status = ?, pnl = ?, settled_at = ?
            WHERE id = ?
            """,
            (status, float(realised), now, int(bet_id)),
        )
        conn.commit()

    close_price = closing_odds if closing_odds is not None else odds_close
    if close_price is not None:
        try:
            update_closing_odds(int(bet_id), float(close_price), db_path=path)
        except Exception:
            pass
    return float(realised)


def delete_live_bet(bet_id: int, db_path: Path | str = DEFAULT_DB_PATH) -> None:
    """Remove a journal row (any status)."""
    path = ensure_live_bets_table(db_path)
    with sqlite3.connect(path) as conn:
        conn.execute(f"DELETE FROM {LIVE_BETS_TABLE} WHERE id = ?", (int(bet_id),))
        conn.commit()


def journal_summary_from_frame(
    df: pd.DataFrame,
    initial_bankroll: float,
) -> dict[str, Any]:
    """Summarise paper bankroll / hit-rate / CLV from an in-memory journal frame.

    Useful when the UI filters by league or concatenates multi-DB loads.
    """
    if df is None or df.empty:
        settled = pending = df if df is not None else pd.DataFrame()
    else:
        settled = df.loc[df["status"].isin(["WIN", "LOSS", "PUSH"])]
        pending = df.loc[df["status"] == "PENDING"]

    realised_pnl = float(settled["pnl"].sum()) if not settled.empty else 0.0
    pending_stake = (
        float(pending["stake_amount"].sum()) if not pending.empty else 0.0
    )
    wins = int((settled["status"] == "WIN").sum()) if not settled.empty else 0
    losses = int((settled["status"] == "LOSS").sum()) if not settled.empty else 0
    pushes = int((settled["status"] == "PUSH").sum()) if not settled.empty else 0
    current = float(initial_bankroll) + realised_pnl

    stake_settled = (
        float(pd.to_numeric(settled["stake_amount"], errors="coerce").fillna(0).sum())
        if not settled.empty
        else 0.0
    )
    roi = (realised_pnl / stake_settled) if stake_settled > 0 else 0.0

    avg_clv = float("nan")
    n_clv = 0
    if df is not None and not df.empty and "clv_pct" in df.columns:
        clv_series = pd.to_numeric(df["clv_pct"], errors="coerce").dropna()
        n_clv = int(len(clv_series))
        if n_clv:
            avg_clv = float(clv_series.mean())

    brier = compute_settled_brier(settled) if not settled.empty else float("nan")

    return {
        "initial_bankroll": float(initial_bankroll),
        "current_bankroll": current,
        "realised_pnl": realised_pnl,
        "pending_stake": pending_stake,
        "n_pending": int(len(pending)),
        "n_settled": int(len(settled)),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "hit_rate": (wins / (wins + losses)) if (wins + losses) else 0.0,
        "stake_settled": stake_settled,
        "roi": roi,
        "roi_pct": roi * 100.0,
        "avg_clv": avg_clv,
        "avg_clv_pct": avg_clv * 100.0 if avg_clv == avg_clv else float("nan"),
        "n_clv": n_clv,
        "brier": brier,
    }


def journal_bankroll_summary(
    initial_bankroll: float,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Summarise paper bankroll, hit-rate, and average CLV."""
    return journal_summary_from_frame(load_live_bets(db_path), initial_bankroll)


# ---------------------------------------------------------------------------
# Post-match settle / CLV / Brier helpers
# ---------------------------------------------------------------------------


_LINE_RE = re.compile(
    r"(?P<label>Over|Under|AH\s+Home|AH\s+Away|Home|Away)\s*(?P<line>[+-]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def parse_selection_line(selection: str) -> tuple[str, float | None]:
    """Parse ``Over 2.5`` / ``AH Home -0.25`` → (label, line).

    Returns ``(selection, None)`` when no numeric line is present (1X2).
    """
    text = str(selection or "").strip()
    m = _LINE_RE.search(text)
    if not m:
        return text, None
    label = re.sub(r"\s+", " ", m.group("label")).strip()
    # Canonical casing
    low = label.lower()
    if low.startswith("ah home"):
        label = "AH Home"
    elif low.startswith("ah away"):
        label = "AH Away"
    elif low.startswith("over"):
        label = "Over"
    elif low.startswith("under"):
        label = "Under"
    else:
        label = label.title()
    return label, float(m.group("line"))


def kickoff_has_passed(
    match_date: Any,
    *,
    now: datetime | None = None,
    grace: timedelta = DEFAULT_SETTLE_GRACE,
) -> bool:
    """True when kickoff + grace is strictly before ``now`` (UTC-aware)."""
    if match_date is None or (isinstance(match_date, float) and pd.isna(match_date)):
        return False
    try:
        ts = pd.Timestamp(match_date)
    except Exception:
        return False
    if pd.isna(ts):
        return False
    # Date-only strings ("2026-09-20") → assume 15:00 UTC so same-day
    # morning rechecks do not settle before a typical afternoon kickoff.
    text = str(match_date).strip()
    if len(text) <= 10 and ts.hour == 0 and ts.minute == 0 and ts.second == 0:
        ts = ts + pd.Timedelta(hours=15)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    ref = now or datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    return (ts.to_pydatetime() + grace) <= ref


def list_pending_past_kickoff(
    db_path: Path | str = DEFAULT_DB_PATH,
    *,
    now: datetime | None = None,
    grace: timedelta = DEFAULT_SETTLE_GRACE,
) -> pd.DataFrame:
    """PENDING journal rows whose kickoff (+ grace) has already passed."""
    pending = load_live_bets(db_path, status="PENDING")
    if pending.empty:
        return pending
    mask = pending["match_date"].map(
        lambda md: kickoff_has_passed(md, now=now, grace=grace)
    )
    return pending.loc[mask].copy()


def find_match_result(
    results: pd.DataFrame,
    *,
    home_team: str,
    away_team: str,
    match_date: Any = None,
) -> pd.Series | None:
    """Locate a finished match row by (day, home, away)."""
    if results is None or results.empty:
        return None
    df = results
    if "_day" not in df.columns:
        if "Date" not in df.columns:
            return None
        df = df.copy()
        df["_day"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    home = str(home_team).strip()
    away = str(away_team).strip()
    day = _match_day_key(match_date)
    hit = df.loc[
        (df["HomeTeam"].astype(str).str.strip() == home)
        & (df["AwayTeam"].astype(str).str.strip() == away)
    ]
    if day:
        day_hit = hit.loc[hit["_day"] == day]
        if not day_hit.empty:
            hit = day_hit
    if hit.empty:
        return None
    # Prefer rows that already have an FTR / goals.
    scored = hit.loc[hit["FTR"].notna()] if "FTR" in hit.columns else hit
    row = scored.iloc[-1] if not scored.empty else hit.iloc[-1]
    if "FTR" in row.index and (pd.isna(row["FTR"]) or str(row["FTR"]).strip() == ""):
        return None
    return row


def resolve_bet_outcome(
    *,
    market: str,
    selection: str,
    odds: float,
    stake: float,
    fthg: int,
    ftag: int,
    ftr: str,
    hc: float | int | None = None,
    ac: float | int | None = None,
) -> tuple[BetStatus, float]:
    """Settle one paper bet against full-time (and optional corner) scores.

    Supports markets ``1X2``, ``OU``, ``AH``, ``Corners`` (OU or AH on HC/AC).
    Reuses backtester Asian settlement (incl. quarter-line half outcomes).
    """
    from src.backtester import settle_1x2, settle_ah, settle_ou

    mkt = str(market or "1X2").strip().upper()
    sel = str(selection or "").strip()
    label, line = parse_selection_line(sel)
    o, s = float(odds), float(stake)

    if mkt in {"1X2", "MATCH", "H2H"}:
        status = settle_1x2(normalize_selection(sel), str(ftr))
        if status == "WIN":
            return "WIN", s * (o - 1.0)
        if status == "LOSS":
            return "LOSS", -s
        return "PUSH", 0.0

    if mkt in {"OU", "O/U", "OVER/UNDER"}:
        if line is None:
            raise ValueError(f"OU selection missing line: {sel!r}")
        return settle_ou(label, int(fthg), int(ftag), float(line), o, s)

    if mkt in {"AH", "ASIAN", "ASIAN HANDICAP"}:
        if line is None:
            raise ValueError(f"AH selection missing handicap: {sel!r}")
        # Recommender stores away as ``AH Away {-home_hand:+g}``; convert
        # back to home-convention before ``settle_ah`` (which mirrors Away).
        if "away" in label.lower():
            return settle_ah("AH Away", int(fthg), int(ftag), -float(line), o, s)
        return settle_ah("AH Home", int(fthg), int(ftag), float(line), o, s)

    if mkt in {"CORNERS", "CORNER"}:
        if hc is None or ac is None or (isinstance(hc, float) and np.isnan(hc)):
            raise ValueError("Corners settlement requires HC/AC")
        hc_i, ac_i = int(hc), int(ac)
        if label in {"Over", "Under"}:
            if line is None:
                raise ValueError(f"Corners OU missing line: {sel!r}")
            return settle_ou(label, hc_i, ac_i, float(line), o, s)
        if "AH" in label.upper() or label in {"Home", "Away"}:
            if line is None:
                raise ValueError(f"Corners AH missing line: {sel!r}")
            if "away" in label.lower():
                return settle_ah("AH Away", hc_i, ac_i, -float(line), o, s)
            return settle_ah("AH Home", hc_i, ac_i, float(line), o, s)
        raise ValueError(f"Unsupported Corners selection: {sel!r}")

    # Fallback: treat unknown market as 1X2 if selection normalises.
    status = settle_1x2(normalize_selection(sel), str(ftr))
    if status == "WIN":
        return "WIN", s * (o - 1.0)
    if status == "LOSS":
        return "LOSS", -s
    return "PUSH", 0.0


def compute_settled_brier(
    settled: pd.DataFrame | None = None,
    *,
    db_path: Path | str | None = None,
) -> float:
    """Binary overnight Brier on settled WIN/LOSS rows with ``p_model``.

    ``y = 1`` for WIN, ``0`` for LOSS; PUSH and missing ``p_model`` skipped.
    """
    if settled is None:
        if db_path is None:
            return float("nan")
        df = load_live_bets(db_path)
        settled = df.loc[df["status"].isin(["WIN", "LOSS", "PUSH"])] if not df.empty else df
    if settled is None or settled.empty:
        return float("nan")
    if "p_model" not in settled.columns:
        return float("nan")

    rows = settled.loc[settled["status"].isin(["WIN", "LOSS"])].copy()
    p = pd.to_numeric(rows["p_model"], errors="coerce")
    mask = p.notna() & np.isfinite(p.to_numpy(dtype=float))
    if not mask.any():
        return float("nan")
    y = (rows.loc[mask, "status"] == "WIN").astype(float)
    return brier_score(y.to_numpy(), p.loc[mask].to_numpy(dtype=float))


def evaluate_pending_against_results(
    results: pd.DataFrame,
    db_path: Path | str = DEFAULT_DB_PATH,
    *,
    now: datetime | None = None,
    grace: timedelta = DEFAULT_SETTLE_GRACE,
    sync_clv: bool = True,
) -> dict[str, Any]:
    """Settle PENDING past-kickoff paper bets against finished match results.

    Returns session stats: settled count, wins/losses/pushes, PnL, skipped,
    per-bet details, and CLV sync counts when ``sync_clv``.
    """
    ensure_live_bets_table(db_path)
    pending = list_pending_past_kickoff(db_path, now=now, grace=grace)

    clv_info = {"updated": 0, "skipped": 0}
    if sync_clv and results is not None and not results.empty:
        try:
            clv_info = sync_closing_odds_from_results(results, db_path)
        except Exception:
            pass

    details: list[dict[str, Any]] = []
    settled_n = wins = losses = pushes = 0
    pnl_sum = 0.0
    skipped = 0
    skipped_reasons: list[str] = []

    if pending.empty:
        return {
            "settled": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "pnl": 0.0,
            "skipped": 0,
            "skipped_reasons": [],
            "details": [],
            "clv_sync": clv_info,
            "pending_past_kickoff": 0,
        }

    # Precompute day key once
    res = results.copy() if results is not None and not results.empty else pd.DataFrame()
    if not res.empty and "Date" in res.columns and "_day" not in res.columns:
        res["_day"] = pd.to_datetime(res["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    for _, bet in pending.iterrows():
        bet_id = int(bet["id"])
        home = str(bet["home_team"])
        away = str(bet["away_team"])
        match = find_match_result(
            res, home_team=home, away_team=away, match_date=bet.get("match_date")
        )
        if match is None:
            skipped += 1
            skipped_reasons.append(f"#{bet_id} no result yet ({home} vs {away})")
            continue

        try:
            fthg = int(match["FTHG"])
            ftag = int(match["FTAG"])
            ftr = str(match["FTR"]).strip().upper()
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            skipped_reasons.append(f"#{bet_id} bad scoreboard: {exc}")
            continue

        hc = match["HC"] if "HC" in match.index and pd.notna(match.get("HC")) else None
        ac = match["AC"] if "AC" in match.index and pd.notna(match.get("AC")) else None
        market = str(bet.get("market") or "1X2")

        try:
            status, pnl = resolve_bet_outcome(
                market=market,
                selection=str(bet["selection"]),
                odds=float(bet["odds"]),
                stake=float(bet["stake_amount"]),
                fthg=fthg,
                ftag=ftag,
                ftr=ftr,
                hc=hc,
                ac=ac,
            )
        except ValueError as exc:
            skipped += 1
            skipped_reasons.append(f"#{bet_id} {exc}")
            continue

        closing = None
        if "closing_odds" in bet.index and pd.notna(bet.get("closing_odds")):
            closing = float(bet["closing_odds"])
        elif "odds_close" in bet.index and pd.notna(bet.get("odds_close")):
            closing = float(bet["odds_close"])
        if closing is None:
            # Settlement fallback: B365 / Avg from results so CLV is never n/a
            # when a closing price exists on the finished-match row.
            closing = closing_odds_from_row(str(bet["selection"]), match)

        try:
            settle_live_bet(
                bet_id, status, db_path, closing_odds=closing, pnl=pnl
            )
        except RuntimeError:
            # Race / already settled — idempotent skip
            skipped += 1
            skipped_reasons.append(f"#{bet_id} already settled")
            continue

        settled_n += 1
        pnl_sum += float(pnl)
        if status == "WIN":
            wins += 1
        elif status == "LOSS":
            losses += 1
        else:
            pushes += 1
        details.append(
            {
                "id": bet_id,
                "home_team": home,
                "away_team": away,
                "market": market,
                "selection": str(bet["selection"]),
                "status": status,
                "pnl": float(pnl),
                "score": f"{fthg}-{ftag}",
            }
        )

    return {
        "settled": settled_n,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "pnl": pnl_sum,
        "skipped": skipped,
        "skipped_reasons": skipped_reasons,
        "details": details,
        "clv_sync": clv_info,
        "pending_past_kickoff": int(len(pending)),
    }


def summarise_evaluation_session(
    session: Mapping[str, Any],
    *,
    db_path: Path | str = DEFAULT_DB_PATH,
    initial_bankroll: float = 1000.0,
) -> dict[str, Any]:
    """Merge a settle session with full-journal bankroll / CLV / Brier stats."""
    bank = journal_bankroll_summary(initial_bankroll, db_path)
    hit = session.get("wins", 0) + session.get("losses", 0)
    session_hit = (
        float(session["wins"]) / hit if hit else float("nan")
    )
    return {
        **dict(session),
        "session_hit_rate": session_hit,
        "journal": bank,
        "avg_clv": bank["avg_clv"],
        "avg_clv_pct": bank["avg_clv_pct"],
        "brier": bank["brier"],
        "roi": bank["roi"],
        "roi_pct": bank["roi_pct"],
        "realised_pnl": bank["realised_pnl"],
        "hit_rate": bank["hit_rate"],
    }
