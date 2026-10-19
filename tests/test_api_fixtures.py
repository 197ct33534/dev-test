"""/api/v1/fixtures: every match in the VN window with its pricing status."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.services import value_bets as vb_svc
from src.scanner import TopScanResult


def test_kickoff_iso_utc_marks_offset() -> None:
    assert vb_svc.kickoff_iso_utc("2026-10-14 10:00:00") == "2026-10-14T10:00:00+00:00"
    aware = pd.Timestamp("2026-10-14 17:00", tz="Asia/Ho_Chi_Minh")
    assert vb_svc.kickoff_iso_utc(aware) == "2026-10-14T10:00:00+00:00"
    assert vb_svc.kickoff_iso_utc(None) is None
    assert vb_svc.kickoff_iso_utc(float("nan")) is None


def test_value_bet_rows_carry_utc_offset() -> None:
    out = vb_svc._row_to_bet({"home": "A", "away": "B", "kickoff": pd.Timestamp("2026-10-14 10:00")})
    assert out["kickoff"] == "2026-10-14T10:00:00+00:00"


class _DC:
    def __init__(self, thin: set[str]) -> None:
        self.thin = thin

    def unknown_teams(self, *teams: str) -> list[str]:
        return [t for t in teams if t in self.thin]


@pytest.fixture()
def fake_league(monkeypatch: pytest.MonkeyPatch) -> None:
    soon = (datetime.now(timezone.utc) + timedelta(hours=2)).replace(tzinfo=None, microsecond=0)
    fx = pd.DataFrame(
        [
            # value pick
            {"Kickoff": soon, "HomeTeam": "A", "AwayTeam": "B", "FlashscoreEventId": "e1",
             "OddsH": 2.2, "OddsD": 3.4, "OddsA": 3.5, "OddsBookmaker": "bet365", "AHh": -0.25},
            # gated (thin side in a cup)
            {"Kickoff": soon, "HomeTeam": "C", "AwayTeam": "Thin", "FlashscoreEventId": "e2",
             "OddsH": 1.8, "OddsD": 3.5, "OddsA": 4.0},
            # no odds at all
            {"Kickoff": soon, "HomeTeam": "E", "AwayTeam": "F", "FlashscoreEventId": "e3"},
            # priced, below threshold
            {"Kickoff": soon, "HomeTeam": "G", "AwayTeam": "H", "FlashscoreEventId": "e4",
             "OddsH": 1.5, "OddsD": 4.0, "OddsA": 6.0},
        ]
    )
    monkeypatch.setattr(
        vb_svc,
        "load_fixtures_with_meta",
        lambda lg: {"df": fx, "source": "db_cache", "stale": False, "age_minutes": 3.0},
    )
    monkeypatch.setattr(vb_svc, "_resolve_models", lambda lg: (_DC({"Thin"}), None, 0.0, "dc_fitted_cold"))

    def _scan(fx_df, dc, ml, **kw):
        pool = pd.DataFrame(
            [
                {"match_id": "e1", "home": "A", "away": "B", "market": "1X2", "selection": "Home",
                 "odds": 2.2, "ev": 0.21, "ev_pct": 21.0, "p_model": 0.55, "kelly_pct": 0.8},
                {"match_id": "e4", "home": "G", "away": "H", "market": "OU", "selection": "Over 2.5",
                 "odds": 1.9, "ev": 0.01, "ev_pct": 1.0, "p_model": 0.53, "kelly_pct": 0.0},
            ]
        )
        return TopScanResult(
            value_bets=pool.iloc[:1], fallback_bets=pool, display_bets=pool.iloc[:1],
            below_threshold=False, n_value=1, excluded_prior=3,
        )

    monkeypatch.setattr(vb_svc, "scan_top_value_bets", _scan)


def test_list_fixtures_statuses(fake_league: None) -> None:
    out = vb_svc.list_fixtures_api(leagues="J_LEAGUE_CUP")
    by_id = {f["match_id"]: f for f in out["fixtures"]}
    assert out["count"] == 4
    assert by_id["e1"]["status"] == "value"
    assert by_id["e1"]["best_pick"]["selection"] == "Home"
    assert by_id["e1"]["odds_home"] == 2.2
    assert by_id["e1"]["ah_line"] == -0.25
    assert by_id["e1"]["kickoff"].endswith("+00:00")
    assert by_id["e2"]["status"] == "gated"
    assert by_id["e2"]["gate_reason"] == "cup_new_team_prior"
    assert by_id["e2"]["best_pick"] is None
    assert by_id["e3"]["status"] == "no_odds"
    assert by_id["e4"]["status"] == "below"
    assert out["league_meta"][0]["excluded_prior"] == 3


def test_fixtures_route(fake_league: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_AUTH_DISABLED", "true")
    monkeypatch.setenv("APP_ENV", "test")
    client = TestClient(create_app())
    r = client.get("/api/v1/fixtures?leagues=J_LEAGUE_CUP")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 4
    assert {f["status"] for f in body["fixtures"]} == {"value", "gated", "no_odds", "below"}
    assert client.get("/api/v1/fixtures?leagues=NOPE").status_code == 400
