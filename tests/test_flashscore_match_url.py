"""Unit tests for Flashscore match URL parsing (mid= / path forms)."""

from __future__ import annotations

import pytest

from src.fetchers import flashscore_league as fl


@pytest.mark.parametrize(
    "url, mid, home_slug, home_hash, away_slug, away_hash",
    [
        (
            "https://www.flashscore.com/match/football/"
            "gamba-osaka-zLQAGOBK/tokushima-IcwgwCCt/?mid=zg0G0pvA",
            "zg0G0pvA",
            "gamba-osaka",
            "zLQAGOBK",
            "tokushima",
            "IcwgwCCt",
        ),
        (
            "https://www.flashscore.com/match/football/"
            "gamba-osaka-zLQAGOBK/tokushima-IcwgwCCt/?mid=zg0G0pvA&utm=1",
            "zg0G0pvA",
            "gamba-osaka",
            "zLQAGOBK",
            "tokushima",
            "IcwgwCCt",
        ),
        (
            "www.flashscore.com/match/zg0G0pvA/",
            "zg0G0pvA",
            None,
            None,
            None,
            None,
        ),
        (
            "https://www.flashscore.vn/match/football/a-Abc12345/b-Def67890/?mid=xtmHKGT0",
            "xtmHKGT0",
            "a",
            "Abc12345",
            "b",
            "Def67890",
        ),
    ],
)
def test_parse_flashscore_match_url_valid(
    url: str,
    mid: str,
    home_slug: str | None,
    home_hash: str | None,
    away_slug: str | None,
    away_hash: str | None,
) -> None:
    got = fl.parse_flashscore_match_url(url)
    assert got["match_id"] == mid
    if home_slug is None:
        assert "home_slug" not in got
    else:
        assert got.get("home_slug") == home_slug
        assert got.get("home_hash") == home_hash
        assert got.get("away_slug") == away_slug
        assert got.get("away_hash") == away_hash


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "https://www.google.com/?mid=zg0G0pvA",
        "https://www.flashscore.com/team/vissel-kobe/698tGI9q/",
        "https://www.flashscore.com/match/football/gamba-osaka-zLQAGOBK/tokushima/",
        "not-a-url",
        "https://www.flashscore.com/football/japan/emperors-cup/",
    ],
)
def test_parse_flashscore_match_url_invalid(url: str) -> None:
    with pytest.raises(ValueError):
        fl.parse_flashscore_match_url(url)


def test_resolve_league_from_flashscore_path_emperors() -> None:
    assert (
        fl.resolve_league_from_flashscore_path("/football/japan/emperors-cup/")
        == "EMPERORS_CUP"
    )
    assert (
        fl.resolve_league_from_flashscore_path("/football/japan/emperors-cup")
        == "EMPERORS_CUP"
    )
    assert fl.resolve_league_from_flashscore_path("/football/unknown/xyz") is None


def test_fetch_and_persist_match_from_url_mocked(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.global_db import connect_global_db, load_upcoming_from_db

    db = tmp_path / "g.db"
    connect_global_db(db, init=True).close()

    detail = {
        "match_id": "zg0G0pvA",
        "league_id": "EMPERORS_CUP",
        "row": {
            "Date": __import__("pandas").Timestamp("2026-09-23"),
            "Kickoff": __import__("pandas").Timestamp("2026-09-23 08:00:00"),
            "HomeTeam": "Gamba Osaka",
            "AwayTeam": "Tokushima",
            "FlashscoreEventId": "zg0G0pvA",
            "Source": "flashscore_match_link",
            "league_id": "EMPERORS_CUP",
            "B365H": 1.44,
            "B365D": 4.2,
            "B365A": 5.5,
            "HomeFlashscoreHash": "zLQAGOBK",
            "AwayFlashscoreHash": "IcwgwCCt",
        },
        "odds": {"B365H": 1.44},
        "page_url": "https://www.flashscore.com/match/zg0G0pvA/",
        "tournament_path": "/football/japan/emperors-cup/",
        "tournament_name": "Emperors Cup",
    }

    monkeypatch.setattr(
        fl,
        "fetch_flashscore_match_detail",
        lambda *a, **k: detail,
    )
    monkeypatch.setattr(fl, "persist_team_hashes", lambda *a, **k: 0)

    url = (
        "https://www.flashscore.com/match/football/"
        "gamba-osaka-zLQAGOBK/tokushima-IcwgwCCt/?mid=zg0G0pvA"
    )
    out = fl.fetch_and_persist_match_from_url(
        url, default_league="EPL", db_path=db
    )
    assert out["match_id"] == "zg0G0pvA"
    assert out["league"] == "EMPERORS_CUP"
    assert out["home"] == "JP_G_OSAKA"
    assert out["away"] == "JP_TOKUSHIMA"

    fx, _ = load_upcoming_from_db("EMPERORS_CUP", db_path=db)
    assert not fx.empty
    assert str(fx.iloc[0]["FlashscoreEventId"]) == "zg0G0pvA"
