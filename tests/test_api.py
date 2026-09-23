"""TestClient coverage for FastAPI routes (profile + import + value-bets)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app())


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_team_profile_ok(client: TestClient) -> None:
    fake = {
        "team_id": "Vissel Kobe",
        "db_team_id": 42,
        "team_key": "vissel_kobe",
        "canonical_name": "Vissel Kobe",
        "display_name": "Vissel Kobe",
        "past_matches": [
            {
                "date": "2026-09-10",
                "competition": "J1",
                "home": "Vissel Kobe",
                "away": "Sagan Tosu",
                "score": "2-0",
                "result": "W",
            }
        ],
        "upcoming_matches": [],
        "as_of": "2026-09-23T00:00:00",
    }
    with patch("src.api.routes.teams_svc.fetch_team_profile", return_value=fake):
        with patch("src.api.routes.teams_svc.team_profile_found", return_value=True):
            r = client.get("/api/v1/teams/Vissel%20Kobe/profile")
    assert r.status_code == 200
    body = r.json()
    assert body["db_team_id"] == 42
    assert body["past_matches"][0]["score"] == "2-0"


def test_team_profile_404(client: TestClient) -> None:
    empty = {
        "team_id": "Unknown FC",
        "db_team_id": None,
        "canonical_name": "Unknown FC",
        "display_name": "Unknown FC",
        "past_matches": [],
        "upcoming_matches": [],
    }
    with patch("src.api.routes.teams_svc.fetch_team_profile", return_value=empty):
        with patch("src.api.routes.teams_svc.team_profile_found", return_value=False):
            r = client.get("/api/v1/teams/Unknown%20FC/profile")
    assert r.status_code == 404


def test_import_url_ok(client: TestClient) -> None:
    ok = {
        "ok": True,
        "slug": "vissel-kobe",
        "hash": "abc123",
        "db_team_id": 7,
        "matches_fetched": 10,
    }
    with patch("src.api.routes.teams_svc.import_team_url", return_value=ok):
        r = client.post(
            "/api/v1/teams/import-url",
            json={
                "team_id": "JP_VISSEL_KOBE",
                "flashscore_url": "https://www.flashscore.com/team/vissel-kobe/abc123/",
            },
        )
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert r.json()["hash"] == "abc123"


def test_import_url_bad_url_400(client: TestClient) -> None:
    bad = {"ok": False, "error": "URL Flashscore không hợp lệ."}
    with patch("src.api.routes.teams_svc.import_team_url", return_value=bad):
        r = client.post(
            "/api/v1/teams/import-url",
            json={"team_id": "X", "flashscore_url": "https://example.com/nope"},
        )
    assert r.status_code == 400


def test_value_bets_mocked(client: TestClient) -> None:
    payload = {
        "bets": [
            {
                "pick": "1X2 · Home @ 2.10",
                "selection": "Home",
                "market": "1X2",
                "odds": 2.1,
                "ev_pct": 12.5,
                "home": "Alpha",
                "away": "Beta",
                "match_id": "m1",
                "league": "EPL",
                "ai_reasons": ["EV dương"],
                "home_rest_days": 6.0,
                "away_rest_days": 3.0,
            }
        ],
        "count": 1,
        "min_ev_pct": 5.0,
        "markets": ["1X2", "AH", "OU"],
        "limit": 20,
        "below_threshold": False,
        "odds_missing": False,
        "source": "EPL:db_cache",
        "notes": None,
    }
    with patch("src.api.routes.vb_svc.scan_value_bets_api", return_value=payload):
        r = client.get("/api/v1/value-bets?min_ev=5&limit=20&markets=1X2,AH,OU")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["bets"][0]["market"] == "1X2"
    assert body["bets"][0]["ai_reasons"] == ["EV dương"]


def test_webapp_static_served(client: TestClient) -> None:
    r = client.get("/webapp/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    assert "Soi Kèo" in r.text or "telegram-web-app" in r.text


def test_parse_markets() -> None:
    from src.api.services.value_bets import parse_markets

    assert parse_markets("1X2,AH,OU") == ["1X2", "AH", "OU"]
    assert "Corners" in parse_markets(["1X2", "Corners"])


def test_pick_port_skips_busy_localhost(monkeypatch: pytest.MonkeyPatch) -> None:
    """Laragon-style 127.0.0.1 bind must force an alternate port."""
    import run_api as launcher

    busy = {8000}

    def _free(port: int) -> bool:
        return int(port) not in busy

    monkeypatch.setattr(launcher, "_localhost_free", _free)
    monkeypatch.setattr(launcher, "_who_listens", lambda _p: "php LISTENING")
    assert launcher.pick_port(8000, fixed=False) == 8001


def test_resolve_models_pickle_only_for_epl(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shared pickles must not be applied to UWCL/LaLiga (inflates thin-team EV)."""
    from src.api.services import value_bets as vb

    vb.clear_model_cache()
    sentinel = object()

    monkeypatch.setattr(vb, "_load_joblib", lambda _p: sentinel)
    monkeypatch.setattr(
        vb,
        "maybe_wrap_with_league_weights",
        lambda model, league: ("wrapped", model, league),
    )

    fitted: list[str] = []

    def _fake_fit(league: str, n_seasons: int, n_train: int, xi: float = 0.001):
        fitted.append(str(league))
        return f"fitted:{league}"

    monkeypatch.setattr(vb, "_fit_dc", _fake_fit)

    epl = vb._resolve_models("EPL")
    assert epl[0] == ("wrapped", sentinel, "EPL")
    assert "dc_pickle" in epl[3]
    assert fitted == []

    vb.clear_model_cache()
    uwcl = vb._resolve_models("UWCL")
    assert uwcl[0] == ("wrapped", "fitted:UWCL", "UWCL")
    assert uwcl[3] == "dc_fitted_cold"
    assert fitted == ["UWCL"]
