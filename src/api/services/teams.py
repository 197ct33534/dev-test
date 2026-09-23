"""Team profile + Flashscore import services (DB / network)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data_loader import get_team_profile_data
from src.fetchers.flashscore_team import import_team_from_flashscore_url
from src.global_db import GLOBAL_DB_PATH


def fetch_team_profile(
    team_id: str | int,
    *,
    limit: int = 5,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Return past/upcoming lists from ``global_matches.db`` (no network)."""
    return get_team_profile_data(
        team_id,
        limit=limit,
        db_path=db_path if db_path is not None else GLOBAL_DB_PATH,
    )


def team_profile_found(data: dict[str, Any]) -> bool:
    """True when the team resolved to a DB row."""
    return data.get("db_team_id") is not None


def import_team_url(
    team_id: str,
    flashscore_url: str,
    *,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Upsert Flashscore hash and fetch recent matches into the global DB."""
    return import_team_from_flashscore_url(
        team_id,
        flashscore_url,
        db_path=db_path if db_path is not None else GLOBAL_DB_PATH,
    )
