"""Vietnam (Asia/Ho_Chi_Minh) kickoff window helpers for Top-20 scanning."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
UTC = ZoneInfo("UTC")


def _to_naive_utc(ts: datetime | pd.Timestamp) -> pd.Timestamp:
    """Normalize any timestamp to naive UTC (matches fixture Kickoff storage)."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        # Assume already naive UTC (Fotmob / Flashscore pipeline convention).
        return t.tz_localize(None)
    return t.tz_convert(UTC).tz_localize(None)


def get_vn_today_tomorrow_window(
    *,
    now: datetime | pd.Timestamp | None = None,
    as_naive_utc: bool = True,
    horizon_hours: float = 48.0,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return kickoff window starting at VN midnight today.

    The end bound is the later of:

    * calendar end of tomorrow (``23:59:59`` VN), and
    * ``now + horizon_hours`` (default 48h).

    That way European evening kickoffs that land early on day+2 VN (common for
    UWCL midweek slots) still appear in Top-20 while morning scans keep the
    full today→tomorrow calendar coverage.

    Parameters
    ----------
    now:
        Reference instant (default: current time). Timezone-aware preferred;
        naive values are interpreted as Vietnam local time.
    as_naive_utc:
        If True (default), convert bounds to naive UTC for comparison against
        fixture ``Kickoff`` columns stored as UTC-naive timestamps.
    horizon_hours:
        Rolling horizon from ``now`` (default 48).

    Returns
    -------
    (today_start, window_end)
        Inclusive window endpoints.
    """
    if now is None:
        ref = datetime.now(VN_TZ)
    else:
        ref = pd.Timestamp(now).to_pydatetime()
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=VN_TZ)
        else:
            ref = ref.astimezone(VN_TZ)

    today = ref.date()
    tomorrow = today + timedelta(days=1)
    today_start = datetime(today.year, today.month, today.day, 0, 0, 0, tzinfo=VN_TZ)
    tomorrow_end = datetime(
        tomorrow.year, tomorrow.month, tomorrow.day, 23, 59, 59, tzinfo=VN_TZ
    )
    horizon_end = ref + timedelta(hours=float(horizon_hours))
    window_end = max(tomorrow_end, horizon_end)

    start = pd.Timestamp(today_start)
    end = pd.Timestamp(window_end)
    if as_naive_utc:
        return _to_naive_utc(start), _to_naive_utc(end)
    return start, end


def kickoff_to_vn(kickoff: Any) -> pd.Timestamp | pd.NaT:
    """Convert a (naive-UTC or aware) kickoff to Asia/Ho_Chi_Minh."""
    if kickoff is None or (isinstance(kickoff, float) and pd.isna(kickoff)):
        return pd.NaT
    ts = pd.Timestamp(kickoff)
    if pd.isna(ts):
        return pd.NaT
    if ts.tzinfo is None:
        ts = ts.tz_localize(UTC)
    return ts.tz_convert(VN_TZ)


def format_kickoff_vn(kickoff: Any, fmt: str = "%d/%m %H:%M") -> str:
    """Format kickoff for VN display; empty string if missing."""
    vn = kickoff_to_vn(kickoff)
    if pd.isna(vn):
        return ""
    return vn.strftime(fmt)


def filter_matches_today_tomorrow(
    df: pd.DataFrame,
    kickoff_col: str = "Kickoff",
    *,
    now: datetime | pd.Timestamp | None = None,
    horizon_hours: float = 48.0,
) -> pd.DataFrame:
    """Keep rows whose kickoff falls in the VN today→tomorrow / ~48h window.

    See :func:`get_vn_today_tomorrow_window`. ``Kickoff`` values are treated as
    naive UTC (project convention). Aware timestamps are converted to naive UTC
    before the window comparison.
    """
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()
    if kickoff_col not in df.columns:
        raise ValueError(f"DataFrame missing kickoff column {kickoff_col!r}")

    start, end = get_vn_today_tomorrow_window(
        now=now, as_naive_utc=True, horizon_hours=horizon_hours
    )
    out = df.copy()
    ko = pd.to_datetime(out[kickoff_col], errors="coerce", utc=False)
    # Aware → naive UTC; naive stays as-is (assumed UTC).
    def _norm(x: pd.Timestamp) -> pd.Timestamp:
        if pd.isna(x):
            return x
        return _to_naive_utc(x)

    ko_utc = ko.map(_norm)
    mask = ko_utc.notna() & (ko_utc >= start) & (ko_utc <= end)
    return out.loc[mask].reset_index(drop=True)


def get_today_tomorrow_matches(
    db_path: str | Path | None = None,
    *,
    fixtures: pd.DataFrame | None = None,
    kickoff_col: str = "Kickoff",
    now: datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return today/tomorrow fixtures from an in-memory frame or SQLite.

    Primary path is filtering ``fixtures`` (from ``load_upcoming_fixtures``).
    If ``fixtures`` is None and ``db_path`` is given, attempts to read a
    ``fixtures`` / ``upcoming`` table when present; otherwise returns empty.
    """
    if fixtures is not None:
        return filter_matches_today_tomorrow(
            fixtures, kickoff_col=kickoff_col, now=now
        )

    if db_path is None:
        return pd.DataFrame()

    path = Path(db_path)
    if not path.exists():
        return pd.DataFrame()

    try:
        with sqlite3.connect(path) as conn:
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            table = None
            for candidate in ("fixtures", "upcoming", "upcoming_fixtures"):
                if candidate in tables:
                    table = candidate
                    break
            if table is None:
                return pd.DataFrame()
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    except (sqlite3.Error, pd.errors.DatabaseError, ValueError):
        return pd.DataFrame()

    if df.empty or kickoff_col not in df.columns:
        # Try Date as fallback
        if "Date" in df.columns and kickoff_col not in df.columns:
            df = df.copy()
            df[kickoff_col] = pd.to_datetime(df["Date"], errors="coerce")
        else:
            return pd.DataFrame()

    return filter_matches_today_tomorrow(df, kickoff_col=kickoff_col, now=now)
