"""Tests for config/leagues.json multi-league registry + Flashscore fetcher."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.fetchers.flashscore_league import (
    EMPERORS_CUP_TEAM_ALIASES,
    LALIGA_TEAM_ALIASES,
    apply_league_team_aliases,
    fetch_league_history,
    resolve_league_team_name,
)
from src.global_db import get_available_leagues as global_get_available_leagues
from src.league_registry import (
    LEAGUES_JSON_PATH,
    clear_leagues_cache,
    get_available_leagues,
    get_league_entry,
    load_leagues_json,
    resolve_league_config,
)


@pytest.fixture(autouse=True)
def _clear_registry_cache() -> None:
    clear_leagues_cache()
    yield
    clear_leagues_cache()


def test_leagues_json_loads_and_has_laliga() -> None:
    assert LEAGUES_JSON_PATH.is_file()
    data = load_leagues_json()
    assert "LALIGA" in data
    assert "EPL" in data
    assert "UWCL" in data
    assert "EMPERORS_CUP" in data
    laliga = data["LALIGA"]
    assert laliga["flashscore_id"] == "dWdJXP6U"
    assert "2024-2025" in laliga["seasons"]
    assert laliga["seasons"]["2024-2025"] == "A1MYWy8T"
    assert float(laliga["league_weight"]) == pytest.approx(0.95)
    cup = data["EMPERORS_CUP"]
    assert cup["flashscore_id"] == "AsquizUQ"
    assert cup["flashscore_path"] == "/football/japan/emperors-cup"
    assert cup["time_zone"] == "Asia/Tokyo"
    assert cup["history_source"] == "flashscore"
    assert cup["fd_div"] is None
    assert "2026" in cup["seasons"]
    assert cup["seasons"]["2026"] == "pn1jVG5j"
    assert float(cup["league_weight"]) == pytest.approx(0.8)
    assert cup["team_aliases"]["Gamba Osaka"] == "JP_G_OSAKA"
    assert cup["team_aliases"]["Vissel Kobe"] == "JP_VISSEL_KOBE"


def test_get_available_leagues_includes_laliga() -> None:
    rows = get_available_leagues()
    codes = {r["code"] for r in rows}
    assert {"EPL", "UWCL", "LALIGA", "EMPERORS_CUP"} <= codes
    names = {r["name"] for r in rows}
    assert any("LaLiga" in n for n in names)
    assert any("Emperor" in n for n in names)
    # Re-export from global_db stays in sync
    assert {r["code"] for r in global_get_available_leagues()} == codes


def test_resolve_league_config_laliga() -> None:
    code, cfg = resolve_league_config("LALIGA")
    assert code == "LALIGA"
    assert cfg["flashscore_id"] == "dWdJXP6U"
    assert cfg["fd_div"] == "SP1"
    assert "laliga" in str(cfg.get("flashscore_fixtures_url") or "").lower()


def test_resolve_league_config_emperors_cup() -> None:
    code, cfg = resolve_league_config("EMPERORS_CUP")
    assert code == "EMPERORS_CUP"
    assert cfg["flashscore_id"] == "AsquizUQ"
    assert cfg["fd_div"] is None
    assert "emperors-cup" in str(cfg.get("flashscore_fixtures_url") or "").lower()
    assert float(cfg["league_weight"]) == pytest.approx(0.8)


def test_japanese_aliases_to_jp_codes() -> None:
    assert resolve_league_team_name("Gamba Osaka", "EMPERORS_CUP") == "JP_G_OSAKA"
    assert resolve_league_team_name("Vissel Kobe", "EMPERORS_CUP") == "JP_VISSEL_KOBE"
    assert resolve_league_team_name("Kashima Antlers", "EMPERORS_CUP") == "JP_KASHIMA"
    assert EMPERORS_CUP_TEAM_ALIASES["Gamba Osaka"] == "JP_G_OSAKA"

    frame = pd.DataFrame(
        {
            "HomeTeam": ["Gamba Osaka", "Vissel Kobe"],
            "AwayTeam": ["Urawa Reds", "Kofu"],
            "FTHG": [1, 2],
            "FTAG": [0, 2],
        }
    )
    out = apply_league_team_aliases(frame, "EMPERORS_CUP")
    assert list(out["HomeTeam"]) == ["JP_G_OSAKA", "JP_VISSEL_KOBE"]
    assert list(out["AwayTeam"]) == ["JP_URAWA", "JP_KOFU"]


def test_barcelona_aliases_to_es_barcelona() -> None:
    assert resolve_league_team_name("Barcelona", "LALIGA") == "ES_BARCELONA"
    assert resolve_league_team_name("FC Barcelona", "LALIGA") == "ES_BARCELONA"
    assert resolve_league_team_name("Barça", "LALIGA") == "ES_BARCELONA"
    assert LALIGA_TEAM_ALIASES["Barcelona"] == "ES_BARCELONA"

    frame = pd.DataFrame(
        {
            "HomeTeam": ["Barcelona", "Real Madrid"],
            "AwayTeam": ["Ath Bilbao", "Betis"],
            "FTHG": [1, 2],
            "FTAG": [0, 2],
        }
    )
    out = apply_league_team_aliases(frame, "LALIGA")
    assert list(out["HomeTeam"]) == ["ES_BARCELONA", "ES_REAL_MADRID"]
    assert list(out["AwayTeam"]) == ["ES_ATHLETIC", "ES_BETIS"]


def test_fetch_league_history_uses_mock_no_network() -> None:
    """Fetcher must not hit live Flashscore/FD when dependencies are mocked."""
    fake_hist = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-08-15", "2024-08-16"]),
            "HomeTeam": ["ES_BARCELONA", "ES_REAL_MADRID"],
            "AwayTeam": ["ES_VALENCIA", "ES_SEVILLA"],
            "FTHG": [2, 1],
            "FTAG": [1, 1],
            "FTR": ["H", "D"],
            "Season": ["2024/25", "2024/25"],
            "SeasonStart": [2024, 2024],
            "Source": ["football-data", "football-data"],
            "league_id": ["LALIGA", "LALIGA"],
        }
    )

    with patch(
        "src.fetchers.flashscore_league._download_fd_div_seasons",
        return_value=(fake_hist, []),
    ), patch(
        "src.fetchers.flashscore_league.fetch_flashscore_season_results",
        return_value=pd.DataFrame(),
    ):
        out = fetch_league_history("LALIGA", n_seasons=2, include_flashscore_results=True)

    assert len(out) == 2
    assert set(out["HomeTeam"]) == {"ES_BARCELONA", "ES_REAL_MADRID"}
    assert out.attrs.get("league") == "LALIGA"


def test_get_league_entry_optional_fields_documented() -> None:
    entry = get_league_entry("LALIGA")
    assert entry is not None
    # Optional fields present for LaLiga sample
    assert "fotmob_id" in entry
    assert "fd_div" in entry
    assert "db_path" in entry
