"""Unit tests for Flashscore team-results fetcher (mocked HTML/JSON)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.fetchers import flashscore_team as ft
from src.features import calculate_multi_comp_features, format_fatigue_label
from src.global_db import connect_global_db, read_matches_as_legacy, resolve_team_id


def _kv(k: str, v: str) -> str:
    return f"{k}\xf7{v}"


def _sec(*parts: str) -> str:
    return "\xac".join(parts)


def _comp_header(name: str, short: str = "") -> str:
    return _sec(
        _kv("ZA", name),
        _kv("ZK", short or name),
        _kv("ZL", "/football/japan/j1-league/"),
    )


def _match_sec(
    eid: str,
    ts: int,
    home: str,
    away: str,
    hg: int,
    ag: int,
) -> str:
    return _sec(
        _kv("AA", eid),
        _kv("AB", "3"),
        _kv("AD", str(ts)),
        _kv("AE", home),
        _kv("AF", away),
        _kv("AG", str(hg)),
        _kv("AH", str(ag)),
    )


def _wrap_feed(raw: str, key: str = "results") -> str:
    return f'cjs.initialFeeds["{key}"] = {{ data: `{raw}` }};'


def test_infer_comp_id_mapping() -> None:
    assert ft.infer_comp_id("JAPAN: J1 League") == "J1"
    assert ft.infer_comp_id("JAPAN: J2 League") == "J2"
    assert ft.infer_comp_id("JAPAN: Emperors Cup") == "EMPERORS_CUP"
    assert ft.infer_comp_id("ASIA: AFC Champions League - League phase") == "ACL"
    assert ft.infer_comp_id("JAPAN: J.League Cup - First stage") == "J_LEAGUE_CUP"
    assert ft.infer_comp_id("WORLD: Club Friendly") == "FRIENDLY"
    assert ft.infer_comp_id("SPAIN: Laliga") == "LALIGA"
    assert ft.infer_comp_id("ENGLAND: Premier League") == "EPL"
    assert ft.infer_comp_id("Something Weird Cup") == "FLASH_TEAM"


def test_strip_country_suffix() -> None:
    assert ft.strip_flashscore_country_suffix("Vissel Kobe (Jpn)") == "Vissel Kobe"
    assert ft.strip_flashscore_country_suffix("Port MTI FC (Tha)") == "Port MTI FC"


def test_team_hash_roundtrip() -> None:
    assert ft.hash_for_team_id("JP_VISSEL_KOBE") == "698tGI9q"
    assert ft.team_id_for_hash("698tGI9q") == "JP_VISSEL_KOBE"
    assert ft.hash_for_team_id("JP_SAGAN_TOSU") == "nsRRyAda"
    assert "vissel-kobe" in ft.team_results_url("698tGI9q", team_id="JP_VISSEL_KOBE")


def test_resolve_team_hash_machida_tochigi() -> None:
    machida = ft.resolve_team_hash("JP_MACHIDA")
    assert machida is not None
    assert machida["hash"] == "CUSC1dab"
    assert machida["slug"] == "machida-zelvia"
    assert ft.resolve_team_hash("Machida Zelvia")["hash"] == "CUSC1dab"  # type: ignore[index]
    assert ft.hash_for_team_id("JP_MACHIDA") == "CUSC1dab"

    tochigi = ft.resolve_team_hash("JP_TOCHIGI_CITY")
    assert tochigi is not None
    assert tochigi["hash"] == "4MvNs2n5"
    assert tochigi["slug"] == "tochigi-city"
    assert ft.resolve_team_hash("Tochigi City")["hash"] == "4MvNs2n5"  # type: ignore[index]


def test_parse_and_fetch_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    # 2026-09-20 12:00 UTC and 2026-09-16
    ts1 = int(datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc).timestamp())
    ts2 = int(datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc).timestamp())
    raw = "~".join(
        [
            _sec(_kv("SA", "1")),
            _comp_header("JAPAN: J1 League", "J1 League"),
            _match_sec("eid1", ts1, "Gamba Osaka", "Vissel Kobe", 0, 1),
            _comp_header("ASIA: AFC Champions League - League phase", "ACL"),
            _match_sec("eid2", ts2, "Port MTI FC (Tha)", "Vissel Kobe (Jpn)", 1, 2),
        ]
    )
    html = _wrap_feed(raw, "results")

    monkeypatch.setattr(ft, "_fetch_text_retry", lambda url, **_k: html)

    rows = ft.fetch_team_recent_matches("698tGI9q", n_matches=10, team_id="JP_VISSEL_KOBE")
    assert len(rows) == 2
    assert rows[0]["flashscore_event_id"] == "eid1"
    assert rows[0]["home_team"] == "JP_G_OSAKA"
    assert rows[0]["away_team"] == "JP_VISSEL_KOBE"
    assert rows[0]["comp_id"] == "J1"
    assert rows[0]["score"] == "0-1"
    assert rows[1]["comp_id"] == "ACL"
    assert rows[1]["away_team"] == "JP_VISSEL_KOBE"


def test_persist_and_rest_days(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    ft.ensure_jp_aliases_registered()
    ts_kobe = int(datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc).timestamp())
    ts_tosu = int(datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc).timestamp())
    matches = [
        {
            "match_date": pd.Timestamp("2026-09-20"),
            "home_team": "JP_G_OSAKA",
            "away_team": "JP_VISSEL_KOBE",
            "score": "0-1",
            "home_goals": 0,
            "away_goals": 1,
            "competition_name": "JAPAN: J1 League",
            "comp_id": "J1",
            "flashscore_event_id": "k1",
            "source": "flashscore_team",
        },
        {
            "match_date": pd.Timestamp("2026-09-19"),
            "home_team": "JP_SAGAN_TOSU",
            "away_team": "JP_IWAKI",
            "score": "1-3",
            "home_goals": 1,
            "away_goals": 3,
            "competition_name": "JAPAN: J2 League",
            "comp_id": "J2",
            "flashscore_event_id": "t1",
            "source": "flashscore_team",
        },
    ]
    db = tmp_path / "global_matches.db"
    stats = ft.persist_team_matches_to_global_db(matches, db_path=db)
    assert stats["matches_upserted"] == 2

    # Idempotent re-persist skips duplicates.
    stats2 = ft.persist_team_matches_to_global_db(matches, db_path=db)
    assert stats2["matches_upserted"] == 0
    assert stats2["skipped_duplicate"] >= 2

    hist = read_matches_as_legacy(db)
    conn = connect_global_db(db, init=False)
    try:
        kobe_id = resolve_team_id(conn, "JP_VISSEL_KOBE", comp_id="J1")
        tosu_id = resolve_team_id(conn, "JP_SAGAN_TOSU", comp_id="J2")
    finally:
        conn.close()
    assert kobe_id is not None and tosu_id is not None

    kick = "2026-09-23"
    k = calculate_multi_comp_features(kobe_id, kick, hist)
    t = calculate_multi_comp_features(tosu_id, kick, hist)
    assert k["rest_days"] == 3.0  # 23 - 20
    assert t["rest_days"] == 4.0  # 23 - 19
    label = format_fatigue_label("Vissel Kobe", k, "Sagan Tosu", t)
    assert "Vissel Kobe nghỉ 3 ngày" in label
    assert "Sagan Tosu nghỉ 4 ngày" in label
    # silence unused
    assert ts_kobe and ts_tosu


def test_fetch_graceful_empty_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a, **_k):
        raise RuntimeError("network down")

    monkeypatch.setattr(ft, "_fetch_text_retry", _boom)
    with pytest.warns(UserWarning):
        out = ft.fetch_team_recent_matches("698tGI9q", team_id="JP_VISSEL_KOBE")
    assert out == []


@pytest.mark.parametrize(
    "url,slug,hash_",
    [
        (
            "https://www.flashscore.com/team/vissel-kobe/698tGI9q/",
            "vissel-kobe",
            "698tGI9q",
        ),
        (
            "https://www.flashscore.com/team/vissel-kobe/698tGI9q/results/",
            "vissel-kobe",
            "698tGI9q",
        ),
        (
            "https://www.flashscore.com/team/vissel-kobe/698tGI9q/results",
            "vissel-kobe",
            "698tGI9q",
        ),
        (
            "http://flashscore.com/team/machida-zelvia/CUSC1dab/?utm=1",
            "machida-zelvia",
            "CUSC1dab",
        ),
        (
            "www.flashscore.com/team/tochigi-city/4MvNs2n5/#results",
            "tochigi-city",
            "4MvNs2n5",
        ),
    ],
)
def test_parse_flashscore_team_url_valid(url: str, slug: str, hash_: str) -> None:
    got = ft.parse_flashscore_team_url(url)
    assert got == {"slug": slug, "hash": hash_}


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "https://www.flashscore.com/match/abc123/",
        "https://www.google.com/team/vissel-kobe/698tGI9q/",
        "https://www.flashscore.com/team/vissel-kobe/",
        "not-a-url",
        "https://www.flashscore.com/team/",
    ],
)
def test_parse_flashscore_team_url_invalid(url: str) -> None:
    with pytest.raises(ValueError):
        ft.parse_flashscore_team_url(url)


def test_import_team_from_flashscore_url_mocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "g.db"
    connect_global_db(db, init=True).close()

    ts = int(datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc).timestamp())
    fake_rows = [
        {
            "match_date": datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc),
            "home_team": "JP_VISSEL_KOBE",
            "away_team": "JP_SAGAN_TOSU",
            "score": "2-1",
            "home_goals": 2,
            "away_goals": 1,
            "competition_name": "JAPAN: J1 League",
            "comp_id": "J1",
            "flashscore_event_id": "evt1",
        }
    ]

    monkeypatch.setattr(
        ft,
        "fetch_team_recent_matches",
        lambda *_a, **_k: fake_rows,
    )

    bad = ft.import_team_from_flashscore_url(
        "JP_VISSEL_KOBE",
        "https://example.com/nope",
        db_path=db,
    )
    assert bad["ok"] is False
    assert "hợp lệ" in bad["error"] or "URL" in bad["error"]

    ok = ft.import_team_from_flashscore_url(
        "JP_VISSEL_KOBE",
        "https://www.flashscore.com/team/vissel-kobe/698tGI9q/results/",
        db_path=db,
        upcoming_kickoff="2026-09-23",
        upcoming_comp_id="EMPERORS_CUP",
    )
    assert ok["ok"] is True
    assert ok["hash"] == "698tGI9q"
    assert ok["slug"] == "vissel-kobe"
    assert ok["matches_fetched"] == 1

    status = ft.team_flashscore_data_status("JP_VISSEL_KOBE", db_path=db)
    assert status["has_hash"] is True
    assert status["has_history"] is True
    assert status["ready"] is True
    assert status["needs_import"] is False
    # silence unused
    assert ts > 0
