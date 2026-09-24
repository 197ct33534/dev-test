"""Unit tests for Telegram WebApp initData HMAC verification."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from urllib.parse import urlencode

import pytest
from fastapi.testclient import TestClient

from src.api.auth import verify_telegram_webapp_data
from src.api.main import create_app


def _signed_init_data(bot_token: str, **fields: str) -> str:
    """Build a valid Telegram-style initData query string."""
    payload = dict(fields)
    if "auth_date" not in payload:
        payload["auth_date"] = str(int(time.time()))
    check = "\n".join(f"{k}={v}" for k, v in sorted(payload.items()))
    secret = hmac.new(
        key=b"WebAppData",
        msg=bot_token.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).digest()
    digest = hmac.new(
        key=secret,
        msg=check.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()
    payload["hash"] = digest
    return urlencode(payload)


def test_verify_telegram_webapp_data_ok() -> None:
    token = "123456:ABC-DEF"
    user = json.dumps({"id": 42, "first_name": "Ada"})
    init = _signed_init_data(token, user=user, query_id="AAE")
    parsed = verify_telegram_webapp_data(init, token)
    assert parsed["user"]["id"] == 42
    assert parsed["query_id"] == "AAE"
    assert "_auth_date" in parsed


def test_verify_telegram_webapp_data_bad_hash() -> None:
    token = "123456:ABC-DEF"
    init = _signed_init_data(token, query_id="x")
    pairs = dict(part.split("=", 1) for part in init.split("&"))
    pairs["hash"] = "0" * 64
    bad = urlencode(pairs)
    with pytest.raises(ValueError, match="mismatch"):
        verify_telegram_webapp_data(bad, token)


def test_verify_expired_auth_date() -> None:
    token = "123456:ABC-DEF"
    old = str(int(time.time()) - 100_000)
    init = _signed_init_data(token, auth_date=old, query_id="z")
    with pytest.raises(ValueError, match="expired"):
        verify_telegram_webapp_data(init, token, max_age_seconds=60)


def test_api_rejects_without_init_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_AUTH_DISABLED", "false")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:TEST")
    client = TestClient(create_app())
    r = client.get("/api/v1/value-bets")
    assert r.status_code == 401


def test_api_accepts_valid_tma_header(monkeypatch: pytest.MonkeyPatch) -> None:
    token = "123456:TEST"
    monkeypatch.setenv("TELEGRAM_AUTH_DISABLED", "false")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", token)
    init = _signed_init_data(token, query_id="ok")
    client = TestClient(create_app())
    from unittest.mock import patch

    fake = {
        "generated_at": "2026-01-01T00:00:00",
        "window": "today_tomorrow_vn",
        "min_ev_pct": 5.0,
        "markets": ["1X2"],
        "limit": 20,
        "count": 0,
        "bets": [],
    }
    with patch(
        "src.api.routes.vb_svc.scan_value_bets_api",
        return_value=fake,
    ):
        r = client.get(
            "/api/v1/value-bets",
            headers={"Authorization": f"tma {init}"},
        )
    assert r.status_code == 200
    assert r.json()["count"] == 0


def test_health_stays_public(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_AUTH_DISABLED", "false")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:TEST")
    client = TestClient(create_app())
    assert client.get("/health").status_code == 200
