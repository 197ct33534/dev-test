"""Data refresh + global DB sync orchestration.

Used by Streamlit (``Tải dữ liệu & huấn luyện lại``), auto_scanner, and
migration scripts so ``global_matches.db`` never lags behind league SQLite.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_loader import LEAGUE_CONFIG, load_league_data, normalize_league
from src.global_db import GLOBAL_DB_PATH, sync_to_global_db as _sync_global

logger = logging.getLogger(__name__)

__all__ = [
    "sync_to_global_db",
    "refresh_data",
]


def sync_to_global_db(
    db_path: Path | str = GLOBAL_DB_PATH,
    *,
    force: bool = True,
) -> dict[str, Any]:
    """Re-export :func:`src.global_db.sync_to_global_db` for callers / hooks."""
    return _sync_global(db_path=db_path, force=force)


def refresh_data(
    league: str | None = None,
    *,
    n_seasons: int | None = None,
    force_refresh: bool = True,
    sync_global: bool = True,
    global_db_path: Path | str = GLOBAL_DB_PATH,
) -> dict[str, Any]:
    """Refresh league CSV/API → league SQLite, then sync ``global_matches.db``.

    Parameters
    ----------
    league:
        Single league code, or ``None`` to refresh every configured league.
    force_refresh:
        Passed to :func:`load_league_data` (bypass local cache / re-download).
    sync_global:
        When True (default), call :func:`sync_to_global_db` after league loads.
    """
    codes: list[str]
    if league is None:
        codes = list(LEAGUE_CONFIG.keys())
    else:
        codes = [normalize_league(league)]

    loaded: dict[str, Any] = {}
    for code in codes:
        cfg = LEAGUE_CONFIG[code]
        seasons = n_seasons
        if seasons is None:
            seasons = 5 if code == "UWCL" else 3
        try:
            df = load_league_data(
                league=code,
                n_seasons=int(seasons),
                force_refresh=bool(force_refresh),
            )
            loaded[code] = {
                "ok": True,
                "n_matches": int(len(df)) if isinstance(df, pd.DataFrame) else 0,
                "source": (
                    df.attrs.get("data_source")
                    if isinstance(df, pd.DataFrame)
                    else None
                ),
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("refresh_data failed for %s: %s", code, exc)
            loaded[code] = {"ok": False, "error": str(exc)}

    sync_stats: dict[str, Any] | None = None
    if sync_global:
        sync_stats = sync_to_global_db(db_path=global_db_path, force=True)
        logger.info(
            "sync_to_global_db: ok=%s upserted=%s",
            sync_stats.get("ok"),
            sync_stats.get("matches_upserted"),
        )

    return {
        "leagues": loaded,
        "global_sync": sync_stats,
        "ok": any(v.get("ok") for v in loaded.values()),
    }
