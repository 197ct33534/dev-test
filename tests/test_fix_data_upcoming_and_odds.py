"""Regression tests: upcoming-cache wipe, 1X2/AH mapping, bookmaker labels, HTTP retry."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError

import pandas as pd
import pytest

import src.data_loader as dl
import src.fetchers.flashscore_league as fl
import src.global_db as gdb

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Bug 1: failed / empty upcoming fetch must not wipe the cache
# ---------------------------------------------------------------------------


@pytest.fixture
def no_global_writes(monkeypatch: pytest.MonkeyPatch) -> list:
    calls: list = []

    class _Conn:
        def commit(self) -> None:
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr(gdb, "connect_global_db", lambda *a, **k: _Conn())
    monkeypatch.setattr(gdb, "upsert_competition", lambda *a, **k: None)
    monkeypatch.setattr(
        gdb, "save_upcoming_to_db", lambda code, df, **k: calls.append((code, df)) or len(df)
    )
    return calls


def test_failed_fixture_fetch_reports_not_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(**_k):
        raise RuntimeError("HTTP 503")

    monkeypatch.setattr(fl, "fetch_flashscore_fixture_events", _boom)
    res = fl.fetch_league_upcoming_result("EPL", include_odds=False)
    assert res.ok is False
    assert res.persistable is False
    assert "503" in (res.error or "")
    # Back-compat wrapper still returns an (empty) frame.
    assert fl.fetch_league_upcoming("EPL", include_odds=False).empty


def test_persist_skips_on_failed_fetch(no_global_writes: list) -> None:
    df = pd.DataFrame([{"HomeTeam": "A", "AwayTeam": "B", "Kickoff": pd.Timestamp("2030-01-01")}])
    stats = fl.persist_league_upcoming("EPL", df, fetch_ok=False)
    assert stats["skipped"] is True and stats["reason"] == "fetch_failed"
    assert no_global_writes == []


def test_persist_skips_empty_frame(no_global_writes: list) -> None:
    stats = fl.persist_league_upcoming("EPL", pd.DataFrame())
    assert stats["skipped"] is True and stats["reason"] == "empty"
    assert no_global_writes == []
    # Explicit opt-in still allows clearing.
    fl.persist_league_upcoming("EPL", pd.DataFrame(), allow_empty=True)
    assert len(no_global_writes) == 1


def test_fetch_and_persist_keeps_cache_when_fetch_fails(
    monkeypatch: pytest.MonkeyPatch, no_global_writes: list
) -> None:
    monkeypatch.setattr(
        fl, "fetch_league_history", lambda *a, **k: pd.DataFrame()
    )

    def _boom(**_k):
        raise TimeoutError("timed out")

    monkeypatch.setattr(fl, "fetch_flashscore_fixture_events", _boom)
    out = fl.fetch_and_persist_league("EPL")
    assert out["upcoming_fetch_ok"] is False
    assert out["persist_upcoming"]["skipped"] is True
    assert no_global_writes == []


# ---------------------------------------------------------------------------
# Bug 3: 1X2 / AH mapping by participant id + overround sanity
# ---------------------------------------------------------------------------


def _hda(bk: int, home: str, draw: str, away: str, order=("A", "D", "H")) -> dict:
    sel = {
        "H": {"eventParticipantId": "pH", "value": home},
        "D": {"eventParticipantId": None, "value": draw},
        "A": {"eventParticipantId": "pA", "value": away},
    }
    return {
        "bookmakerId": bk,
        "bettingType": "HOME_DRAW_AWAY",
        "bettingScope": "FULL_TIME",
        "odds": [sel[k] for k in order],
    }


def test_parse_1x2_by_participant_id_ignores_order() -> None:
    market = _hda(16, "1.50", "4.20", "6.50", order=("A", "D", "H"))
    assert dl._fs_parse_1x2(market, home_pid="pH", away_pid="pA") == (1.5, 4.2, 6.5)


def test_parse_1x2_without_ids_does_not_guess_from_order() -> None:
    # Feed order is not reliable (away favourite often first — J.League Cup
    # 2026-10-14 regression), so no ids + no labels → no prices.
    market = _hda(16, "1.50", "4.20", "6.50", order=("H", "D", "A"))
    h, d, a = dl._fs_parse_1x2(market)
    assert pd.isna(h) and pd.isna(d) and pd.isna(a)


def test_parse_1x2_rejects_bad_overround() -> None:
    too_high = _hda(16, "1.20", "2.00", "2.00")  # ~1.83
    too_low = _hda(16, "3.50", "4.00", "5.00")  # ~0.74
    for m in (too_high, too_low):
        h, d, a = dl._fs_parse_1x2(m, home_pid="pH", away_pid="pA")
        assert pd.isna(h) and pd.isna(d) and pd.isna(a)


def _payload(markets: list[dict], *, participants: bool = True) -> dict:
    event: dict = {
        "settings": {
            "bookmakers": [
                {"bookmaker": {"id": 16, "name": "bet365"}},
                {"bookmaker": {"id": 15, "name": "William Hill"}},
            ]
        },
        "odds": markets,
    }
    if participants:
        event["eventParticipants"] = [
            {"id": "pA", "type": "AWAY"},
            {"id": "pH", "type": "HOME"},
        ]
    return {"data": {"findOddsByEventId": event}}


def _ah_market(bk: int) -> dict:
    return {
        "bookmakerId": bk,
        "bettingType": "ASIAN_HANDICAP",
        "bettingScope": "FULL_TIME",
        "odds": [
            {"eventParticipantId": "pA", "value": "1.95", "handicap": {"value": "1"}},
            {"eventParticipantId": "pH", "value": "1.90", "handicap": {"value": "-1"}},
        ],
    }


def test_match_odds_uses_event_ids_and_records_bookmaker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    markets = [
        _hda(16, "1.20", "2.00", "2.00"),  # bet365 garbage → rejected
        _hda(15, "1.50", "4.20", "6.50"),  # William Hill sane
        _ah_market(16),
    ]
    monkeypatch.setattr(dl, "_fetch_json", lambda *a, **k: _payload(markets))
    out = dl.fetch_flashscore_match_odds("EVT1")
    assert (out["OddsH"], out["OddsD"], out["OddsA"]) == (1.5, 4.2, 6.5)
    assert (out["B365H"], out["B365D"], out["B365A"]) == (1.5, 4.2, 6.5)
    assert out["OddsBookmaker"] == "William Hill"
    assert out["OddsBookmakerId"] == "15"
    assert out["OddsIsBet365"] is False
    assert out["OddsProvider"] == "Flashscore/William Hill"
    # AH mapped via participant ids: home -1 @ 1.90, away +1 @ 1.95.
    assert (out["AHh"], out["B365AHH"], out["B365AHA"]) == (-1.0, 1.9, 1.95)


def test_match_odds_bet365_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    markets = [_hda(16, "1.50", "4.20", "6.50", order=("H", "D", "A"))]
    monkeypatch.setattr(
        dl, "_fetch_json", lambda *a, **k: _payload(markets, participants=False)
    )
    # Sides come from the match page when the odds payload has no participants.
    page = '"participantsData":{"home":[{"id":"t1","eventParticipantId":"pH"}],'         '"away":[{"id":"t2","eventParticipantId":"pA"}]}'
    monkeypatch.setattr(dl, "_fetch_text", lambda *a, **k: page)
    dl.clear_flashscore_participant_cache()
    out = dl.fetch_flashscore_match_odds("EVT2")
    assert out["OddsBookmaker"] == "bet365"
    assert out["OddsIsBet365"] is True
    assert out["B365H"] == 1.5


# ---------------------------------------------------------------------------
# Bug 5: HTTP retry / error wrapping / FSIGN override
# ---------------------------------------------------------------------------


class _Resp:
    def __init__(self, body: bytes) -> None:
        self._body = body
        self.headers = type("H", (), {"get_content_charset": lambda self: "utf-8"})()

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_Resp":
        return self

    def __exit__(self, *exc) -> None:
        return None


def test_fetch_json_retries_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(dl.time, "sleep", lambda s: sleeps.append(s))
    attempts = {"n": 0}

    def _urlopen(req, timeout=None):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise TimeoutError("read timed out")
        if attempts["n"] == 2:
            raise HTTPError(req.full_url, 503, "busy", None, io.BytesIO(b""))
        return _Resp(json.dumps({"ok": 1}).encode())

    monkeypatch.setattr(dl, "urlopen", _urlopen)
    assert dl._fetch_json("https://example.invalid/x") == {"ok": 1}
    assert attempts["n"] == 3
    assert sleeps == [0.5, 1.0]


def test_fetch_json_no_retry_on_404_and_wraps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)
    attempts = {"n": 0}

    def _urlopen(req, timeout=None):
        attempts["n"] += 1
        raise HTTPError(req.full_url, 404, "nf", None, io.BytesIO(b""))

    monkeypatch.setattr(dl, "urlopen", _urlopen)
    with pytest.raises(RuntimeError, match="HTTP 404"):
        dl._fetch_json("https://example.invalid/x")
    assert attempts["n"] == 1


def test_fetch_timeouts_exhausted_raise_module_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)

    def _urlopen(req, timeout=None):
        raise URLError(TimeoutError("timed out"))

    monkeypatch.setattr(dl, "urlopen", _urlopen)
    with pytest.raises(dl.DataFetchError):
        dl._fetch_csv_text("https://example.invalid/x.csv")


def test_fetch_json_decode_error_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dl, "urlopen", lambda req, timeout=None: _Resp(b"<html>"))
    with pytest.raises(dl.DataFetchError) as info:
        dl._fetch_json("https://example.invalid/x")
    assert isinstance(info.value, ValueError)  # back-compat for ValueError catchers


def test_fsign_env_override() -> None:
    env = dict(os.environ, FLASHSCORE_FSIGN="TESTSIGN")
    out = subprocess.run(
        [sys.executable, "-c", "import src.data_loader as d; print(d.FLASHSCORE_FSIGN)"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.stdout.strip().splitlines()[-1] == "TESTSIGN"
    assert dl._FLASHSCORE_FSIGN_DEFAULT == "SW9D1eZo"
