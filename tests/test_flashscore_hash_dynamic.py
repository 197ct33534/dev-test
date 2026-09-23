"""Tests for dynamic Flashscore team-hash extraction and DB persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.fetchers import flashscore_league as fl
from src.fetchers import flashscore_team as ft
from src.global_db import (
    GLOBAL_DB_VERSION,
    connect_global_db,
    get_team_flashscore_meta,
    set_team_flashscore_hash,
    upsert_team,
    upsert_team_flashscore_hash,
)


def _kv(k: str, v: str) -> str:
    return f"{k}\xf7{v}"


def _sec(*parts: str) -> str:
    return "\xac".join(parts)


def _event_sec(
    *,
    eid: str,
    home: str,
    away: str,
    home_hash: str,
    away_hash: str,
    home_slug: str,
    away_slug: str,
    ts: int = 1_790_150_400,
) -> str:
    return _sec(
        _kv("AA", eid),
        _kv("AB", "1"),
        _kv("AD", str(ts)),
        _kv("AE", home),
        _kv("AF", away),
        _kv("PX", home_hash),
        _kv("PY", away_hash),
        _kv("WU", home_slug),
        _kv("WV", away_slug),
        # JA/JB must NOT be treated as team page hashes.
        _kv("JA", "NOTHASH1"),
        _kv("JB", "NOTHASH2"),
    )


def test_schema_migration_adds_flashscore_columns(tmp_path: Path) -> None:
    db = tmp_path / "g.db"
    conn = connect_global_db(db, init=True)
    try:
        cols = {
            str(r[1]) for r in conn.execute("PRAGMA table_info(teams)").fetchall()
        }
        assert "flashscore_hash" in cols
        assert "flashscore_slug" in cols
        tables = {
            str(r[0])
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "team_rest_cache" in tables
        ver = conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
        assert int(ver[0]) == GLOBAL_DB_VERSION
    finally:
        conn.close()


def test_get_set_flashscore_hash_by_team_id(tmp_path: Path) -> None:
    db = tmp_path / "g.db"
    conn = connect_global_db(db, init=True)
    try:
        tid = upsert_team(conn, "JP_MACHIDA", comp_id="EMPERORS_CUP")
        set_team_flashscore_hash(
            conn, tid, "CUSC1dab", flashscore_slug="machida-zelvia"
        )
        conn.commit()
        meta = get_team_flashscore_meta(conn, tid)
        assert meta is not None
        assert meta["hash"] == "CUSC1dab"
        assert meta["slug"] == "machida-zelvia"
        # Also by stable code string.
        meta2 = get_team_flashscore_meta(conn, "JP_MACHIDA")
        assert meta2 is not None
        assert meta2["hash"] == "CUSC1dab"
        assert int(meta2["team_id"]) == tid
    finally:
        conn.close()


def test_extract_team_hashes_from_feed_uses_px_py_not_ja() -> None:
    raw = "~".join(
        [
            _event_sec(
                eid="e1",
                home="Machida Zelvia",
                away="Tochigi City",
                home_hash="CUSC1dab",
                away_hash="4MvNs2n5",
                home_slug="machida-zelvia",
                away_slug="tochigi-city",
            ),
            _event_sec(
                eid="e2",
                home="Barcelona",
                away="Real Betis",
                home_hash="SKbpIP5D",
                away_hash="h8oAv4Tq",
                home_slug="barcelona",
                away_slug="real-betis",
            ),
        ]
    )
    entries = fl.extract_team_hashes_from_feed(raw)
    by_hash = {e["hash"]: e for e in entries}
    assert "CUSC1dab" in by_hash
    assert by_hash["CUSC1dab"]["slug"] == "machida-zelvia"
    assert by_hash["CUSC1dab"]["name"] == "Machida Zelvia"
    assert "4MvNs2n5" in by_hash
    assert "SKbpIP5D" in by_hash
    assert "NOTHASH1" not in by_hash
    assert "NOTHASH2" not in by_hash


def test_extract_team_hashes_from_html_urls() -> None:
    html = """
    <a href="/team/machida-zelvia/CUSC1dab/">Machida</a>
    <a href="https://www.flashscore.com/team/vissel-kobe/698tGI9q/results/">Kobe</a>
    """
    entries = fl.extract_team_hashes_from_html(html)
    by_hash = {e["hash"]: e for e in entries}
    assert by_hash["CUSC1dab"]["slug"] == "machida-zelvia"
    assert by_hash["698tGI9q"]["slug"] == "vissel-kobe"


def test_persist_hashes_and_resolve_prefers_db(tmp_path: Path) -> None:
    db = tmp_path / "g.db"
    fl.register_league_aliases_globally("EMPERORS_CUP")
    fl.register_league_aliases_globally("LALIGA")
    entries = [
        {
            "hash": "CUSC1dab",
            "slug": "machida-zelvia",
            "name": "Machida Zelvia",
        },
        {
            "hash": "NEWCLUB99",
            "slug": "brand-new-fc",
            "name": "Brand New FC",
        },
    ]
    stats = fl.persist_team_hashes(entries, "EMPERORS_CUP", db_path=db)
    assert stats["upserted"] >= 2

    # DB wins even for a brand-new club with no static map entry.
    meta = ft.resolve_team_hash("Brand New FC", db_path=db)
    assert meta is not None
    assert meta["hash"] == "NEWCLUB99"
    assert meta["slug"] == "brand-new-fc"
    assert meta.get("source") == "db"

    # Static fallback still works when DB has no row (prefer_db looks at empty DB).
    # Seed a code only in static registry — resolve without writing that hash.
    static = ft.resolve_team_hash("JP_VISSEL_KOBE", db_path=db)
    assert static is not None
    assert static["hash"] == "698tGI9q"


def test_events_from_feed_includes_hash_columns() -> None:
    raw = _event_sec(
        eid="e1",
        home="Gamba Osaka",
        away="Tokushima",
        home_hash="zLQAGOBK",
        away_hash="IcwgwCCt",
        home_slug="gamba-osaka",
        away_slug="tokushima",
    )
    rows = fl._events_from_feed(raw, league_key="EMPERORS_CUP", only_unplayed=True)
    assert len(rows) == 1
    assert rows[0]["HomeFlashscoreHash"] == "zLQAGOBK"
    assert rows[0]["AwayFlashscoreHash"] == "IcwgwCCt"
    assert rows[0]["HomeFlashscoreSlug"] == "gamba-osaka"
