"""Regression tests: stale odds, auth/admin, EV units, alert delivery, redaction."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch
from urllib.parse import urlencode
from uuid import uuid4

import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.services import scheduler as sched
from src.api.services import value_bets as vb

NOW = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)


def _signed(bot_token: str, **fields: str) -> str:
    payload = dict(fields)
    payload.setdefault("auth_date", str(int(time.time())))
    check = "\n".join(f"{k}={v}" for k, v in sorted(payload.items()))
    secret = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    payload["hash"] = hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()
    return urlencode(payload)


@pytest.fixture(autouse=True)
def _dev_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.delenv("DOMAIN", raising=False)
    vb._LAST_REFRESH.clear()


# --- Bug 1: stale fixtures -------------------------------------------------


def _fixtures(*kickoffs: datetime) -> pd.DataFrame:
    return pd.DataFrame(
        {"HomeTeam": ["A"] * len(kickoffs), "Kickoff": [k.replace(tzinfo=None) for k in kickoffs]}
    )


def test_filter_future_fixtures_drops_started_matches() -> None:
    df = _fixtures(NOW - timedelta(minutes=5), NOW + timedelta(hours=2))
    out = vb.filter_future_fixtures(df, NOW)
    assert len(out) == 1
    assert out["Kickoff"].iloc[0] == (NOW + timedelta(hours=2)).replace(tzinfo=None)


def test_fresh_cache_is_used_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _fixtures(NOW + timedelta(hours=3))
    monkeypatch.setattr(vb, "load_upcoming_from_db", lambda _c: (df, NOW.timestamp() - 600))
    monkeypatch.setattr(
        vb, "refresh_upcoming_fixtures", lambda _c: pytest.fail("network must not be hit")
    )
    meta = vb.load_fixtures_with_meta("EPL", max_age_minutes=60, now=NOW)
    assert meta["source"] == "db_cache" and meta["stale"] is False


def test_stale_cache_refreshes_from_network(monkeypatch: pytest.MonkeyPatch) -> None:
    old = _fixtures(NOW + timedelta(hours=3))
    new = _fixtures(NOW + timedelta(hours=4), NOW + timedelta(hours=5))
    monkeypatch.setattr(vb, "load_upcoming_from_db", lambda _c: (old, NOW.timestamp() - 7200))
    monkeypatch.setattr(vb, "refresh_upcoming_fixtures", lambda _c: new)
    meta = vb.load_fixtures_with_meta("EPL", max_age_minutes=60, now=NOW)
    assert meta["source"] == "network" and meta["stale"] is False
    assert len(meta["df"]) == 2


def test_stale_cache_with_failed_network_is_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    old = _fixtures(NOW + timedelta(hours=3))

    def _boom(_c: str) -> pd.DataFrame:
        raise RuntimeError("offline")

    monkeypatch.setattr(vb, "load_upcoming_from_db", lambda _c: (old, NOW.timestamp() - 7200))
    monkeypatch.setattr(vb, "refresh_upcoming_fixtures", _boom)
    meta = vb.load_fixtures_with_meta("EPL", max_age_minutes=60, now=NOW)
    assert meta["stale"] is True and len(meta["df"]) == 1


def test_scheduler_drops_stale_league_bets() -> None:
    bets = [{"league": "EPL", "match_id": "1"}, {"league": "UWCL", "match_id": "2"}]
    assert [b["match_id"] for b in sched.drop_stale_bets(bets, {"EPL"})] == ["2"]
    assert sched.drop_stale_bets(bets, {"*"}) == []


# --- Bug 8: EV units -------------------------------------------------------


def test_scan_min_ev_is_always_percent(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[float] = []

    def _fake_scan(*_a: Any, **kw: Any) -> Any:
        seen.append(kw["min_ev"])
        raise RuntimeError("stop")

    monkeypatch.setattr(
        vb,
        "load_fixtures_with_meta",
        lambda _lg: {"df": _fixtures(NOW), "source": "db_cache", "stale": False},
    )
    monkeypatch.setattr(vb, "_resolve_models", lambda _lg: (object(), None, 0.0, "x"))
    monkeypatch.setattr(vb, "scan_top_value_bets", _fake_scan)
    with pytest.raises(RuntimeError):
        vb.scan_value_bets_api(min_ev_pct=1.0, league="EPL")
    assert seen == [pytest.approx(0.01)]


def test_v2_min_ev_default_follows_config() -> None:
    from src.api import routes_v2
    from src.config import DEFAULT_MIN_EV

    assert routes_v2.DEFAULT_MIN_EV_PCT == pytest.approx(DEFAULT_MIN_EV * 100)


# --- Bug 2/3/9: auth -------------------------------------------------------


def test_app_refuses_to_start_with_auth_disabled_in_prod(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TELEGRAM_AUTH_DISABLED", "true")
    monkeypatch.setenv("APP_ENV", "prod")
    with pytest.raises(RuntimeError, match="not allowed in production"):
        create_app()
    monkeypatch.delenv("APP_ENV")
    monkeypatch.setenv("DOMAIN", "bets.example.com")
    with pytest.raises(RuntimeError):
        create_app()


def test_prod_hides_docs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("TELEGRAM_AUTH_DISABLED", "false")
    client = TestClient(create_app())
    assert client.get("/docs").status_code == 404
    assert client.get("/health").status_code == 200


def test_auth_date_required() -> None:
    from src.api.auth import verify_telegram_webapp_data

    token = "123:ABC"
    payload = {"query_id": "x"}
    check = "query_id=x"
    secret = hmac.new(b"WebAppData", token.encode(), hashlib.sha256).digest()
    payload["hash"] = hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()
    with pytest.raises(ValueError, match="auth_date"):
        verify_telegram_webapp_data(urlencode(payload), token)


def _auth_client(monkeypatch: pytest.MonkeyPatch, admins: str) -> tuple[TestClient, str]:
    token = "123456:TEST"
    monkeypatch.setenv("TELEGRAM_AUTH_DISABLED", "false")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", token)
    monkeypatch.setenv("ADMIN_TELEGRAM_IDS", admins)
    return TestClient(create_app()), token


def test_import_url_requires_admin(monkeypatch: pytest.MonkeyPatch) -> None:
    client, token = _auth_client(monkeypatch, "42")
    user = _signed(token, user=json.dumps({"id": 7}))
    body = {"team_id": "X", "flashscore_url": "https://www.flashscore.com/team/x/abc/"}
    r = client.post(
        "/api/v1/teams/import-url", json=body, headers={"Authorization": f"tma {user}"}
    )
    assert r.status_code == 403

    admin = _signed(token, user=json.dumps({"id": 42}))
    with patch(
        "src.api.routes.teams_svc.import_team_url",
        return_value={"ok": True, "team_id": "X"},
    ):
        r2 = client.post(
            "/api/v1/teams/import-url", json=body, headers={"Authorization": f"tma {admin}"}
        )
    assert r2.status_code != 403


def test_place_bets_requires_admin(monkeypatch: pytest.MonkeyPatch) -> None:
    client, token = _auth_client(monkeypatch, "")
    user = _signed(token, user=json.dumps({"id": 42}))
    r = client.post(
        "/api/v2/bets/place",
        json={"canonical_match_id": str(uuid4()), "bookmaker_odds": {"H": 2.0}},
        headers={"Authorization": f"tma {user}"},
    )
    assert r.status_code == 403


def test_place_bets_rejects_backdated_persist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_AUTH_DISABLED", "true")
    client = TestClient(create_app())
    old = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    r = client.post(
        "/api/v2/bets/place",
        json={
            "canonical_match_id": str(uuid4()),
            "bookmaker_odds": {"H": 2.0, "D": 3.4, "A": 3.6},
            "as_of_time": old,
            "persist": True,
        },
    )
    assert r.status_code == 400


def test_place_bets_generic_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_AUTH_DISABLED", "true")
    client = TestClient(create_app())
    with patch(
        "src.api.routes_v2.PaperTrader.execute_value_bets",
        new_callable=AsyncMock,
        side_effect=RuntimeError("secret /path/db password=x"),
    ):
        r = client.post(
            "/api/v2/bets/place",
            json={"canonical_match_id": str(uuid4()), "bookmaker_odds": {"H": 2.0}},
        )
    assert r.status_code == 500
    assert "secret" not in r.text


def test_place_default_persist_false() -> None:
    from src.api.schemas_v2 import PlaceBetRequest

    req = PlaceBetRequest(canonical_match_id=uuid4(), bookmaker_odds={"H": 2.0})
    assert req.persist is False


# --- Bug 7: alerts marked only after delivery -----------------------------


def _hot_bet() -> dict[str, Any]:
    return {
        "match_id": "m1",
        "market": "1X2",
        "selection": "H",
        "league": "EPL",
        "ev_pct": 12.0,
        "kickoff": (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat(),
    }


def test_failed_send_is_retried_next_tick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.bot.signal_store import already_notified, make_signal_key

    db = tmp_path / "sig.db"
    monkeypatch.setattr(sched, "_scan_candidates_with_meta", lambda: ([_hot_bet()], set()))
    key = make_signal_key(match_id="m1", market="1X2", selection="H")

    failing = AsyncMock(return_value={"sent": 0, "failed": 1, "errors": ["x"]})
    with patch.object(sched, "broadcast_value_signal", failing):
        out = asyncio.run(sched.run_value_signal_job(db_path=db))
    assert out["retry_later"] == 1
    assert not already_notified(key, db_path=db)

    ok = AsyncMock(return_value={"sent": 2, "failed": 0, "errors": []})
    with patch.object(sched, "broadcast_value_signal", ok):
        out2 = asyncio.run(sched.run_value_signal_job(db_path=db))
    assert out2["pushed"] == 2
    assert already_notified(key, db_path=db)

    with patch.object(sched, "broadcast_value_signal", ok):
        out3 = asyncio.run(sched.run_value_signal_job(db_path=db))
    assert out3["skipped_dup"] == 1


def test_stale_scan_sends_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sched, "_scan_candidates_with_meta", lambda: ([_hot_bet()], {"EPL"}))
    spy = AsyncMock(return_value={"sent": 1})
    with patch.object(sched, "broadcast_value_signal", spy):
        out = asyncio.run(sched.run_value_signal_job(db_path=tmp_path / "s.db"))
    spy.assert_not_called()
    assert out["hot"] == 0


def test_telegram_429_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.bot import notifier as bn

    calls = {"n": 0}

    def handler(_req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                429, json={"ok": False, "parameters": {"retry_after": 0}}
            )
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})

    async def _run() -> dict[str, Any]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await bn.send_telegram_message_async("123:T", "1", "hi", client=c)

    assert asyncio.run(_run())["ok"] is True
    assert calls["n"] == 2


# --- Bug 4: token redaction / HTML escaping -------------------------------


def test_notifier_redacts_token_on_request_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests

    from src import notifier

    token = "999:SECRET-TOKEN"

    def _boom(url: str, **_k: Any) -> Any:
        raise requests.ConnectionError(f"Max retries exceeded with url: {url}")

    monkeypatch.setattr(notifier.requests, "post", _boom)
    with pytest.raises(RuntimeError) as ei:
        notifier.send_telegram_message(token, "1", "hi")
    assert token not in str(ei.value)
    assert ei.value.__cause__ is None


def test_notifier_escapes_team_names() -> None:
    from src.notifier import format_value_bet_alert

    text = format_value_bet_alert(
        home_team="A <b>&", away_team="C>", selection="H", odds=2.0, ev=0.1,
        kelly_pct=1.0, stake_amount=10.0,
    )
    assert "A &lt;b&gt;&amp;" in text and "C&gt;" in text


# --- Bug 6: analytics reads WON/LOST rows ----------------------------------


def test_performance_analytics_normalizes_v2_labels() -> None:
    from src.services.settlement_service import compute_performance_analytics

    df = pd.DataFrame(
        {
            "status": ["WON", "LOST", "WIN"],
            "pnl": [10.0, -10.0, 5.0],
            "stake_amount": [10.0, 10.0, 5.0],
        }
    )
    out = compute_performance_analytics(df)
    assert out["total_bets_settled"] == 3
    assert out["win_rate_percent"] == pytest.approx(66.67, abs=0.01)


# --- Bug 9: model cache --------------------------------------------------


def test_model_cache_reloads_when_pickle_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    vb.clear_model_cache()
    sig = {"v": ((1, 1), None)}
    loads: list[int] = []
    monkeypatch.setattr(vb, "_pickle_signature", lambda: sig["v"])
    monkeypatch.setattr(vb, "_load_joblib", lambda _p: loads.append(1) or object())
    monkeypatch.setattr(vb, "maybe_wrap_with_league_weights", lambda m, _c: m)
    vb._resolve_models("EPL")
    vb._resolve_models("EPL")
    assert len(loads) == 2  # dc + lgbm once
    sig["v"] = ((2, 1), None)
    vb._resolve_models("EPL")
    assert len(loads) == 4
    vb.clear_model_cache()


# --- fixtures worker ------------------------------------------------------


def test_fixtures_worker_cycle_isolates_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.workers import fixtures_worker as fw

    monkeypatch.setenv("FIXTURES_REFRESH_LEAGUES", "epl, uwcl")
    assert fw.refresh_leagues() == ["EPL", "UWCL"]

    def _refresh(code: str) -> pd.DataFrame:
        if code == "EPL":
            raise RuntimeError("down")
        return _fixtures(NOW + timedelta(hours=1))

    out = fw.run_refresh_cycle(refresh_fn=_refresh)
    assert out["ok"] is False
    assert out["leagues"]["EPL"]["ok"] is False
    assert out["leagues"]["UWCL"] == {"ok": True, "rows": 1}
