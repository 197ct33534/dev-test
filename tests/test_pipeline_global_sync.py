"""Tests for gender-scoped global teams + refresh → sync_to_global_db hook."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import save_matches_to_db
from src.global_db import (
    connect_global_db,
    make_team_key,
    resolve_team_id,
    sync_to_global_db,
)
from src.pipeline import refresh_data


def test_make_team_key_examples() -> None:
    assert make_team_key("Arsenal", "M") == "ARSENAL_M"
    assert make_team_key("Chelsea", "W") == "CHELSEA_W"
    assert make_team_key("Man United", "M") == "MANUNITED_M"


def test_sync_to_global_db_separates_genders(tmp_path: Path) -> None:
    epl = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2024-08-16"),
                "HomeTeam": "Arsenal",
                "AwayTeam": "Chelsea",
                "FTHG": 1,
                "FTAG": 0,
                "FTR": "H",
                "SeasonStart": 2024,
                "league_id": "EPL",
            }
        ]
    )
    uwcl = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2024-09-01"),
                "HomeTeam": "Arsenal",
                "AwayTeam": "Barcelona",
                "FTHG": 2,
                "FTAG": 1,
                "FTR": "H",
                "SeasonStart": 2024,
                "league_id": "UWCL",
            }
        ]
    )
    epl_db = tmp_path / "epl_matches.db"
    uwcl_db = tmp_path / "uwcl_matches.db"
    global_db = tmp_path / "global_matches.db"
    save_matches_to_db(epl, epl_db)
    save_matches_to_db(uwcl, uwcl_db)

    stats = sync_to_global_db(
        db_path=global_db,
        force=True,
        sources=[("EPL", epl_db), ("UWCL", uwcl_db)],
    )
    assert stats["ok"] is True
    assert stats["matches_upserted"] == 2

    conn = connect_global_db(global_db, init=False)
    try:
        m = resolve_team_id(conn, "Arsenal", comp_id="EPL")
        w = resolve_team_id(conn, "Arsenal", comp_id="UWCL")
        assert m is not None and w is not None and m != w
        keys = {
            r[0]
            for r in conn.execute("SELECT team_key FROM teams").fetchall()
        }
        assert "ARSENAL_M" in keys
        assert "ARSENAL_W" in keys
    finally:
        conn.close()


def test_refresh_data_calls_global_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def _fake_load(**kwargs):
        df = pd.DataFrame(
            {
                "Date": [pd.Timestamp("2024-01-01")],
                "HomeTeam": ["A"],
                "AwayTeam": ["B"],
                "FTHG": [1],
                "FTAG": [0],
            }
        )
        df.attrs["data_source"] = "test"
        return df

    def _fake_sync(**kwargs):
        calls.append(kwargs)
        return {"ok": True, "matches_upserted": 1, "per_comp": {}}

    monkeypatch.setattr("src.pipeline.load_league_data", _fake_load)
    monkeypatch.setattr("src.pipeline.sync_to_global_db", _fake_sync)

    out = refresh_data("EPL", force_refresh=True, sync_global=True)
    assert out["ok"] is True
    assert len(calls) == 1
    assert calls[0].get("force") is True
