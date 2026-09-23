"""Data fetchers for external match sources (Flashscore, etc.)."""

from src.fetchers.flashscore_league import (
    fetch_and_persist_league,
    fetch_and_persist_match_from_url,
    fetch_league_history,
    fetch_league_upcoming,
    parse_flashscore_match_url,
    persist_league_matches,
    persist_league_upcoming,
    resolve_league_from_flashscore_path,
    resolve_league_team_name,
)
from src.fetchers.flashscore_team import (
    fetch_team_recent_matches,
    fetch_team_recent_matches_by_id,
    import_team_from_flashscore_url,
    parse_flashscore_team_url,
    persist_team_matches_to_global_db,
    refresh_team_feeds_for_sides,
    resolve_team_hash,
    should_update_team_feed,
    sync_upcoming_teams_fast,
    team_hash_registry,
    update_all_upcoming_teams_rest_days,
)

__all__ = [
    "fetch_and_persist_league",
    "fetch_and_persist_match_from_url",
    "fetch_league_history",
    "fetch_league_upcoming",
    "fetch_team_recent_matches",
    "fetch_team_recent_matches_by_id",
    "import_team_from_flashscore_url",
    "parse_flashscore_match_url",
    "parse_flashscore_team_url",
    "persist_league_matches",
    "persist_league_upcoming",
    "persist_team_matches_to_global_db",
    "refresh_team_feeds_for_sides",
    "resolve_league_from_flashscore_path",
    "resolve_league_team_name",
    "resolve_team_hash",
    "should_update_team_feed",
    "sync_upcoming_teams_fast",
    "team_hash_registry",
    "update_all_upcoming_teams_rest_days",
]
