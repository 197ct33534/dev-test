"""Unit tests for share / deep-link helpers."""

from __future__ import annotations

import pandas as pd

from src.share import (
    build_share_text,
    build_share_url,
    fallback_match_id,
    find_fixture_for_share,
    match_id_for_fixture,
    match_slug_from_teams,
    parse_home_vs_away,
    parse_match_query,
    pick_top_bet_for_share,
)


def test_fallback_match_id_and_slug() -> None:
    assert fallback_match_id("Man City", "Oud-Heverlee Leuven") == (
        "Man_City__Oud-Heverlee_Leuven"
    )
    assert match_slug_from_teams("Arsenal", "Chelsea") == "Arsenal_vs_Chelsea"
    assert parse_home_vs_away("Arsenal_vs_Chelsea") == ("Arsenal", "Chelsea")
    assert parse_home_vs_away("Man_City_vs_Oud-Heverlee_Leuven") == (
        "Man City",
        "Oud-Heverlee Leuven",
    )
    assert parse_home_vs_away("broken") is None


def test_match_id_prefers_flashscore() -> None:
    row = pd.Series({"FlashscoreEventId": "fs-123", "HomeTeam": "A", "AwayTeam": "B"})
    assert match_id_for_fixture(row, "A", "B") == "fs-123"
    assert match_id_for_fixture(None, "A", "B") == "A__B"


def test_parse_match_query_list_or_str() -> None:
    assert parse_match_query({"match_id": "fs-1"}) == ("fs-1", None)
    assert parse_match_query({"match_id": ["fs-2"], "match": ["A_vs_B"]}) == (
        "fs-2",
        "A_vs_B",
    )
    assert parse_match_query({}) == (None, None)
    assert parse_match_query(None) == (None, None)


def test_find_fixture_flashscore_style_id() -> None:
    """Real Flashscore event ids look like ``IFfhi4pB`` (alphanumeric)."""
    fx = pd.DataFrame(
        [
            {
                "HomeTeam": "Arsenal",
                "AwayTeam": "Chelsea",
                "Kickoff": "2026-09-22 19:00:00",
                "FlashscoreEventId": "IFfhi4pB",
            },
            {
                "HomeTeam": "Liverpool",
                "AwayTeam": "Everton",
                "Kickoff": "2026-09-23 15:00:00",
                "FlashscoreEventId": "xyz999",
            },
        ]
    )
    hit = find_fixture_for_share(fx, match_id="IFfhi4pB")
    assert hit is not None
    assert hit["home"] == "Arsenal"
    assert hit["away"] == "Chelsea"
    assert hit["match_id"] == "IFfhi4pB"
    assert match_id_for_fixture(hit["row"], "Arsenal", "Chelsea") == "IFfhi4pB"

    # Slug fallback still works alongside Flashscore id
    assert find_fixture_for_share(fx, match_slug="Arsenal_vs_Chelsea")["match_id"] == (
        "IFfhi4pB"
    )
    assert build_share_url(match_id="IFfhi4pB") == "?match_id=IFfhi4pB"


def test_find_fixture_by_id_and_slug() -> None:
    fx = pd.DataFrame(
        [
            {
                "HomeTeam": "Arsenal",
                "AwayTeam": "Chelsea",
                "Kickoff": "2026-09-22 19:00:00",
                "FlashscoreEventId": "eve-99",
            },
            {
                "HomeTeam": "Liverpool",
                "AwayTeam": "Everton",
                "Kickoff": "2026-09-23 15:00:00",
            },
        ]
    )
    hit = find_fixture_for_share(fx, match_id="eve-99")
    assert hit is not None
    assert hit["home"] == "Arsenal"
    assert hit["match_id"] == "eve-99"

    hit2 = find_fixture_for_share(fx, match_slug="Liverpool_vs_Everton")
    assert hit2 is not None
    assert hit2["away"] == "Everton"
    assert hit2["match_id"] == "Liverpool__Everton"

    # Fallback Home__Away also resolves even when Flashscore id exists
    hit3 = find_fixture_for_share(fx, match_id="Arsenal__Chelsea")
    assert hit3 is not None
    assert hit3["match_id"] == "eve-99"

    assert find_fixture_for_share(fx, match_id="missing") is None
    assert find_fixture_for_share(pd.DataFrame(), match_id="x") is None


def test_build_share_url_and_text() -> None:
    assert build_share_url(match_id="A__B") == "?match_id=A__B"
    assert (
        build_share_url(match_id="eve 1", base_url="https://app.example/")
        == "https://app.example?match_id=eve%201"
    )
    assert (
        build_share_url(
            match_id="x",
            base_url="https://app.example/?foo=1&match_id=old",
        )
        == "https://app.example/?foo=1&match_id=x"
    )

    text = build_share_text(
        home="Arsenal",
        away="Chelsea",
        league="EPL",
        kickoff_vn="22/09 19:00",
        model_source="Ensemble (w_ML=40%)",
        p_home=0.48,
        p_draw=0.27,
        p_away=0.25,
        top_selection="Chủ nhà",
        top_odds=2.10,
        top_ev=0.082,
        top_kelly_pct=1.25,
        share_url="?match_id=eve-99",
    )
    assert "⚽ Arsenal vs Chelsea" in text
    assert "1X2: H 48% · D 27% · A 25%" in text
    assert "💎 Top pick: Chủ nhà @ 2.10 · EV +8.2% · Kelly 1.25%" in text
    assert "🔗 ?match_id=eve-99" in text
    assert "không đảm bảo" in text


def test_pick_top_bet_prefers_recommended() -> None:
    class _B:
        def __init__(self, ev: float, recommended: bool) -> None:
            self.ev = ev
            self.recommended = recommended

    top = pick_top_bet_for_share([_B(0.20, False), _B(0.08, True), _B(0.05, True)])
    assert top is not None
    assert top.ev == 0.08
