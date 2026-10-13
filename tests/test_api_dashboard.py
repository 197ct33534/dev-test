"""Dashboard ↔ API wiring: leagues/config meta, multi-league scan, bearer auth."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import main as api_main
from src.api.main import create_app
from src.api.services import value_bets as vb_svc
from src.config import MAX_KELLY_FRACTION

GOOD_TOKEN = "dash-token-0123456789abcdef"


@pytest.fixture()
def open_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("TELEGRAM_AUTH_DISABLED", "true")
    monkeypatch.setenv("APP_ENV", "test")
    return TestClient(create_app())


@pytest.fixture()
def secured_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("TELEGRAM_AUTH_DISABLED", "false")
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:TEST")
    monkeypatch.setenv("DASHBOARD_API_TOKENS", f"short, {GOOD_TOKEN}")
    monkeypatch.setenv("ADMIN_TELEGRAM_IDS", "42")
    return TestClient(create_app())


def _empty_scan(**_: object) -> dict:
    return {
        "bets": [],
        "count": 0,
        "min_ev_pct": 5.0,
        "markets": ["1X2"],
        "limit": 20,
    }


# ---------------------------------------------------------------------------
# Meta endpoints
# ---------------------------------------------------------------------------


def test_leagues_lists_json_registry(open_client: TestClient) -> None:
    r = open_client.get("/api/v1/leagues")
    assert r.status_code == 200
    body = r.json()
    codes = [lg["code"] for lg in body["leagues"]]
    assert {"EPL", "UWCL", "J_LEAGUE_CUP"} <= set(codes)
    assert body["count"] == len(codes)
    by_code = {lg["code"]: lg for lg in body["leagues"]}
    assert by_code["J_LEAGUE_CUP"]["is_cup"] is True
    assert by_code["EPL"]["is_cup"] is False
    assert by_code["EPL"]["default_scan"] is True
    assert by_code["J_LEAGUE_CUP"]["default_scan"] is False


def test_config_exposes_stake_policy(open_client: TestClient) -> None:
    r = open_client.get("/api/v1/config")
    assert r.status_code == 200
    policy = r.json()["policy"]
    assert policy["min_ev_pct"] == pytest.approx(5.0)
    assert policy["kelly_fraction"] == pytest.approx(0.10)
    assert policy["max_stake_pct"] == pytest.approx(0.01)
    assert policy["new_team_max_stake_pct"] == pytest.approx(0.005)
    assert "1X2" in r.json()["markets"]


# ---------------------------------------------------------------------------
# /value-bets parameters
# ---------------------------------------------------------------------------


def test_value_bets_forwards_leagues_and_kelly(open_client: TestClient) -> None:
    with patch("src.api.routes.vb_svc.scan_value_bets_api", side_effect=_empty_scan) as m:
        r = open_client.get(
            "/api/v1/value-bets?leagues=EPL,J_LEAGUE_CUP&kelly_fraction=0.05"
        )
    assert r.status_code == 200
    kwargs = m.call_args.kwargs
    assert kwargs["leagues"] == "EPL,J_LEAGUE_CUP"
    assert kwargs["kelly_fraction"] == pytest.approx(0.05)


def test_value_bets_rejects_kelly_above_policy(open_client: TestClient) -> None:
    r = open_client.get(f"/api/v1/value-bets?kelly_fraction={MAX_KELLY_FRACTION + 0.05}")
    assert r.status_code == 422


def test_value_bets_unknown_league_is_400(open_client: TestClient) -> None:
    r = open_client.get("/api/v1/value-bets?leagues=NOT_A_LEAGUE")
    assert r.status_code == 400


def test_parse_leagues_normalises_and_dedupes() -> None:
    assert vb_svc.parse_leagues("epl, EPL ,j_league_cup") == ["EPL", "J_LEAGUE_CUP"]
    assert vb_svc.parse_leagues(None) == []
    with pytest.raises(ValueError):
        vb_svc.parse_leagues("NOPE")


def test_scan_reports_league_meta_and_kelly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty fixtures still yield per-league confidence meta + generated_at."""
    monkeypatch.setattr(
        vb_svc,
        "load_fixtures_with_meta",
        lambda lg: {
            "df": pd.DataFrame(),
            "source": "db_cache",
            "stale": True,
            "age_minutes": 95.0,
        },
    )
    out = vb_svc.scan_value_bets_api(leagues="EPL,J_LEAGUE_CUP", kelly_fraction=0.5)
    assert out["leagues"] == ["EPL", "J_LEAGUE_CUP"]
    # Requested fraction is clamped to the policy ceiling.
    assert out["kelly_fraction"] == pytest.approx(MAX_KELLY_FRACTION)
    assert out["generated_at"]
    meta = {m["league"]: m for m in out["league_meta"]}
    assert meta["EPL"]["odds_age_minutes"] == pytest.approx(95.0)
    assert meta["EPL"]["stale"] is True
    assert meta["J_LEAGUE_CUP"]["fixtures"] == 0


def test_model_flags() -> None:
    assert vb_svc._model_flags("dc_pickle+lgbm_pickle") == {
        "ml_enabled": True,
        "model_source": "pickle",
    }
    assert vb_svc._model_flags("dc_fitted_cold") == {
        "ml_enabled": False,
        "model_source": "cold_fit",
    }


# ---------------------------------------------------------------------------
# Bearer token auth (dashboard outside Telegram)
# ---------------------------------------------------------------------------


def test_bearer_token_grants_read(secured_client: TestClient) -> None:
    with patch("src.api.routes.vb_svc.scan_value_bets_api", side_effect=_empty_scan):
        r = secured_client.get(
            "/api/v1/value-bets",
            headers={"Authorization": f"Bearer {GOOD_TOKEN}"},
        )
    assert r.status_code == 200


def test_bearer_wrong_token_401(secured_client: TestClient) -> None:
    r = secured_client.get(
        "/api/v1/leagues",
        headers={"Authorization": "Bearer wrong-token-0123456789"},
    )
    assert r.status_code == 401


def test_short_token_entries_are_ignored(secured_client: TestClient) -> None:
    r = secured_client.get("/api/v1/leagues", headers={"Authorization": "Bearer short"})
    assert r.status_code == 401


def test_no_tokens_configured_rejects_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_AUTH_DISABLED", "false")
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:TEST")
    monkeypatch.delenv("DASHBOARD_API_TOKENS", raising=False)
    client = TestClient(create_app())
    r = client.get("/api/v1/leagues", headers={"Authorization": f"Bearer {GOOD_TOKEN}"})
    assert r.status_code == 401


def test_bearer_token_cannot_write(secured_client: TestClient) -> None:
    r = secured_client.post(
        "/api/v1/teams/import-url",
        json={"team_id": "X", "flashscore_url": "https://www.flashscore.com/team/x/abc/"},
        headers={"Authorization": f"Bearer {GOOD_TOKEN}"},
    )
    assert r.status_code == 403


# ---------------------------------------------------------------------------
# Static dashboard mount
# ---------------------------------------------------------------------------


def test_dashboard_mounted_when_built(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "index.html").write_text("<!doctype html><title>dash</title>", "utf-8")
    monkeypatch.setenv("DASHBOARD_DIST", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_AUTH_DISABLED", "true")
    monkeypatch.setenv("APP_ENV", "test")
    client = TestClient(api_main.create_app())
    r = client.get("/dashboard/")
    assert r.status_code == 200
    assert "dash" in r.text


def test_dashboard_not_mounted_without_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DASHBOARD_DIST", str(tmp_path / "missing"))
    monkeypatch.setenv("TELEGRAM_AUTH_DISABLED", "true")
    monkeypatch.setenv("APP_ENV", "test")
    client = TestClient(api_main.create_app())
    assert client.get("/dashboard/").status_code == 404


def test_row_to_bet_exposes_new_team_fallback() -> None:
    row = {
        "home": "JP_NAGOYA",
        "away": "JP_OKAYAMA",
        "market": "OU",
        "selection": "Over 2.5",
        "odds": 1.85,
        "ev": 0.4,
        "league": "J_LEAGUE_CUP",
        "detail": {"new_team_fallback": True, "ev_method": "settlement"},
    }
    out = vb_svc._row_to_bet(row)
    assert out["new_team_fallback"] is True
    assert out["ev_method"] == "settlement"
    assert out["home_display"] == "Nagoya Grampus"
    assert vb_svc._row_to_bet({**row, "detail": None})["new_team_fallback"] is False
