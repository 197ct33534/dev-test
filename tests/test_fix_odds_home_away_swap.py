"""Flashscore 1X2 / AH home-away mapping when the odds payload has no participants.

Regression: J.League Cup 2026-10-14 Yamaga (home) vs Sanfrecce Hiroshima.
The odds GraphQL payload carried no participant list and unlabeled
selections; feed order put the *away* favourite first, so positional mapping
stored H=1.17 / A=13.0 (Flashscore shows 1 = 12.00, 2 = 1.22).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import data_loader as dl

HOME_PID = "xbYqZ0Vp"  # Yamaga
AWAY_PID = "6FsEvN7N"  # Sanfrecce Hiroshima

# Excerpt of the match page HTML (JSON embedded in a JS string → escaped).
MATCH_HTML = (
    'window.environment = {\\"participantsData\\":{\\"home\\":[{\\"id\\":\\"0lMXpXdr\\",'
    '\\"eventParticipantId\\":\\"' + HOME_PID + '\\",\\"sport_id\\":1,\\"name\\":\\"Yamaga\\",'
    '\\"image_path\\":\\"https:\\/\\/static.flashscore.com\\/x.png\\",\\"rank\\":[]}],'
    '\\"away\\":[{\\"id\\":\\"fX7MRdvd\\",\\"eventParticipantId\\":\\"' + AWAY_PID + '\\",'
    '\\"sport_id\\":1,\\"name\\":\\"Sanfrecce Hiroshima\\"}]}};'
)


def _real_payload() -> dict:
    """Odds payload as served: away selection first, no labels, no participants."""
    hda = {
        "bookmakerId": 16,
        "bettingType": "HOME_DRAW_AWAY",
        "bettingScope": "FULL_TIME",
        "odds": [
            {"selection": None, "eventParticipantId": AWAY_PID, "value": "1.17"},
            {"selection": None, "eventParticipantId": HOME_PID, "value": "13.0"},
            {"selection": None, "eventParticipantId": None, "value": "6.0"},
        ],
    }
    ah = {
        "bookmakerId": 16,
        "bettingType": "ASIAN_HANDICAP",
        "bettingScope": "FULL_TIME",
        "odds": [
            {"eventParticipantId": AWAY_PID, "value": "1.90", "handicap": {"value": "-2"}},
            {"eventParticipantId": HOME_PID, "value": "1.90", "handicap": {"value": "2"}},
        ],
    }
    event = {
        "eventId": "UmDwk8Zr",
        "settings": {"bookmakers": [{"bookmaker": {"id": 16, "name": "bet365"}}]},
        "odds": [hda, ah],
    }
    return {"data": {"findOddsByEventId": event}}


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    dl.clear_flashscore_participant_cache()


def test_parse_participants_from_match_html() -> None:
    assert dl._fs_parse_participants_html(MATCH_HTML) == (HOME_PID, AWAY_PID)
    # Unescaped JSON works too.
    assert dl._fs_parse_participants_html(MATCH_HTML.replace('\\"', '"')) == (
        HOME_PID,
        AWAY_PID,
    )
    assert dl._fs_parse_participants_html("<html>no data</html>") == (None, None)


def test_match_odds_resolves_sides_from_match_page(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dl, "_fetch_json", lambda *a, **k: _real_payload())
    pages: list[str] = []

    def _fake_text(url: str, **kw) -> str:
        pages.append(url)
        return MATCH_HTML

    monkeypatch.setattr(dl, "_fetch_text", _fake_text)
    out = dl.fetch_flashscore_match_odds("UmDwk8Zr")
    assert (out["OddsH"], out["OddsD"], out["OddsA"]) == (13.0, 6.0, 1.17)
    # AH follows the same ids: home +2 @ 1.90.
    assert out["AHh"] == 2.0
    assert pages == ["https://www.flashscore.com/match/UmDwk8Zr/"]

    # Participant ids never change for an event → cached, no second page load.
    dl.fetch_flashscore_match_odds("UmDwk8Zr")
    assert len(pages) == 1


def test_unknown_sides_drop_odds_instead_of_guessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dl, "_fetch_json", lambda *a, **k: _real_payload())

    def _fail(url: str, **kw) -> str:
        raise dl.DataFetchError("HTTP 503")

    monkeypatch.setattr(dl, "_fetch_text", _fail)
    out = dl.fetch_flashscore_match_odds("UmDwk8Zr")
    assert "OddsH" not in out or pd.isna(out["OddsH"])
    assert "AHh" not in out or pd.isna(out["AHh"])


def test_parse_1x2_without_ids_or_labels_is_rejected() -> None:
    hda = _real_payload()["data"]["findOddsByEventId"]["odds"][0]
    h, d, a = dl._fs_parse_1x2(hda)
    assert pd.isna(h) and pd.isna(d) and pd.isna(a)


def test_labelled_selections_still_work_without_ids() -> None:
    market = {
        "bookmakerId": 16,
        "odds": [
            {"selection": "AWAY", "eventParticipantId": AWAY_PID, "value": "1.22"},
            {"selection": "HOME", "eventParticipantId": HOME_PID, "value": "12.0"},
            {"selection": "DRAW", "eventParticipantId": None, "value": "5.65"},
        ],
    }
    assert dl._fs_parse_1x2(market) == (12.0, 5.65, 1.22)
