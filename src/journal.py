r"""Paper-trading journal: persist live bets in SQLite ``live_bets``.

Tracks PENDING → WIN/LOSS/PUSH settlements, Closing-Line Value (CLV),
and realised PnL against a starting bankroll.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import pandas as pd

from src.data_loader import DEFAULT_DB_PATH

LIVE_BETS_TABLE = "live_bets"
BetStatus = Literal["PENDING", "WIN", "LOSS", "PUSH"]

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
    """Create ``live_bets`` if missing and migrate CLV columns."""
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
    """CLV = (odds_entered / closing_odds) − 1  (fraction; 0.05 = +5%)."""
    entered = float(odds_entered)
    closing = float(closing_odds)
    if entered <= 1.0 or closing <= 1.0:
        raise ValueError("odds_entered and closing_odds must be > 1")
    return (entered / closing) - 1.0


def update_closing_odds(
    bet_id: int,
    closing_odds: float,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> float:
    """Set closing odds for a bet and store ``clv_pct``.

    ``clv_pct = (odds_entered / closing_odds) - 1``

    Positive CLV ⇒ got a better (higher) price than the closing market.
    Returns the stored ``clv_pct`` fraction.
    """
    path = ensure_live_bets_table(db_path)
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            f"SELECT odds FROM {LIVE_BETS_TABLE} WHERE id = ?",
            (int(bet_id),),
        ).fetchone()
        if row is None:
            raise KeyError(f"live_bets id={bet_id} not found")
        entered = float(row[0])
        clv = compute_clv_pct(entered, float(closing_odds))
        conn.execute(
            f"""
            UPDATE {LIVE_BETS_TABLE}
            SET closing_odds = ?, clv_pct = ?
            WHERE id = ?
            """,
            (float(closing_odds), float(clv), int(bet_id)),
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
        row = hit.iloc[0]
        sel = normalize_selection(str(bet["selection"]))
        cols = _SELECTION_TO_ODDS_COL.get(sel, ())
        closing = None
        for c in cols:
            if c in row.index and pd.notna(row[c]) and float(row[c]) > 1.0:
                closing = float(row[c])
                break
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


def settle_live_bet(
    bet_id: int,
    status: BetStatus,
    db_path: Path | str = DEFAULT_DB_PATH,
    *,
    closing_odds: float | None = None,
) -> float:
    """Mark bet WIN/LOSS/PUSH and write PnL. Optionally set closing odds/CLV."""
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

        if status == "WIN":
            pnl = stake * (odds - 1.0)
        elif status == "LOSS":
            pnl = -stake
        else:
            pnl = 0.0

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        conn.execute(
            f"""
            UPDATE {LIVE_BETS_TABLE}
            SET status = ?, pnl = ?, settled_at = ?
            WHERE id = ?
            """,
            (status, float(pnl), now, int(bet_id)),
        )
        conn.commit()

    if closing_odds is not None:
        try:
            update_closing_odds(int(bet_id), float(closing_odds), db_path=path)
        except Exception:
            pass
    return float(pnl)


def delete_live_bet(bet_id: int, db_path: Path | str = DEFAULT_DB_PATH) -> None:
    """Remove a journal row (any status)."""
    path = ensure_live_bets_table(db_path)
    with sqlite3.connect(path) as conn:
        conn.execute(f"DELETE FROM {LIVE_BETS_TABLE} WHERE id = ?", (int(bet_id),))
        conn.commit()


def journal_bankroll_summary(
    initial_bankroll: float,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Summarise paper bankroll, hit-rate, and average CLV."""
    df = load_live_bets(db_path)
    settled = df.loc[df["status"].isin(["WIN", "LOSS", "PUSH"])] if not df.empty else df
    pending = df.loc[df["status"] == "PENDING"] if not df.empty else df

    realised_pnl = float(settled["pnl"].sum()) if not settled.empty else 0.0
    pending_stake = (
        float(pending["stake_amount"].sum()) if not pending.empty else 0.0
    )
    wins = int((settled["status"] == "WIN").sum()) if not settled.empty else 0
    losses = int((settled["status"] == "LOSS").sum()) if not settled.empty else 0
    current = float(initial_bankroll) + realised_pnl

    avg_clv = float("nan")
    n_clv = 0
    if not df.empty and "clv_pct" in df.columns:
        clv_series = pd.to_numeric(df["clv_pct"], errors="coerce").dropna()
        n_clv = int(len(clv_series))
        if n_clv:
            avg_clv = float(clv_series.mean())

    return {
        "initial_bankroll": float(initial_bankroll),
        "current_bankroll": current,
        "realised_pnl": realised_pnl,
        "pending_stake": pending_stake,
        "n_pending": int(len(pending)),
        "n_settled": int(len(settled)),
        "wins": wins,
        "losses": losses,
        "hit_rate": (wins / (wins + losses)) if (wins + losses) else 0.0,
        "avg_clv": avg_clv,
        "avg_clv_pct": avg_clv * 100.0 if avg_clv == avg_clv else float("nan"),
        "n_clv": n_clv,
    }
