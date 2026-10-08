"""Telegram Mini App (WebApp) ``initData`` verification.

Implements the official HMAC-SHA256 scheme from
https://core.telegram.org/bots/webapps#validating-data-received-via-the-mini-app

``secret_key = HMAC_SHA256(key="WebAppData", msg=bot_token)``
then ``hash = HMAC_SHA256(secret_key, data_check_string)``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from typing import Any
from urllib.parse import parse_qsl

from fastapi import Depends, HTTPException, Request, status

logger = logging.getLogger(__name__)

# Reject initData older than this (seconds). Override with TELEGRAM_AUTH_MAX_AGE.
DEFAULT_MAX_AGE_SECONDS = 86_400  # 24h


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def telegram_auth_disabled() -> bool:
    """Local bypass when ``TELEGRAM_AUTH_DISABLED=true``."""
    return _env_bool("TELEGRAM_AUTH_DISABLED", default=False)


_PROD_ENVS = frozenset({"prod", "production"})
_NONPROD_ENVS = frozenset({"dev", "development", "local", "test", "testing", "ci"})
_LOCAL_HOSTS = frozenset({"", "localhost", "127.0.0.1", "0.0.0.0", "::1"})


def is_production() -> bool:
    """True for a public deployment.

    ``APP_ENV=prod|production`` → prod; ``APP_ENV=dev|local|test|ci`` → not
    prod. Otherwise a non-local ``DOMAIN`` (set for nginx / Certbot) means prod.
    """
    app_env = (os.getenv("APP_ENV") or "").strip().lower()
    if app_env in _PROD_ENVS:
        return True
    if app_env in _NONPROD_ENVS:
        return False
    domain = (os.getenv("DOMAIN") or "").strip().lower()
    return domain not in _LOCAL_HOSTS


def assert_auth_config_safe() -> None:
    """Refuse to start a production app with Telegram auth disabled."""
    if telegram_auth_disabled() and is_production():
        raise RuntimeError(
            "TELEGRAM_AUTH_DISABLED=true is not allowed in production "
            "(APP_ENV=prod or DOMAIN set). Set TELEGRAM_AUTH_DISABLED=false, "
            "or APP_ENV=dev for a local stack."
        )


def admin_telegram_ids() -> frozenset[int]:
    """Telegram user ids allowed to call write endpoints (``ADMIN_TELEGRAM_IDS``)."""
    raw = os.getenv("ADMIN_TELEGRAM_IDS") or ""
    ids: set[int] = set()
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.add(int(part))
        except ValueError:
            logger.warning("Ignoring non-numeric ADMIN_TELEGRAM_IDS entry")
    return frozenset(ids)


def _max_age_seconds() -> int:
    raw = (os.getenv("TELEGRAM_AUTH_MAX_AGE") or "").strip()
    if not raw:
        return DEFAULT_MAX_AGE_SECONDS
    try:
        return max(60, int(float(raw)))
    except ValueError:
        return DEFAULT_MAX_AGE_SECONDS


def verify_telegram_webapp_data(
    init_data: str,
    bot_token: str,
    *,
    max_age_seconds: int | None = None,
) -> dict[str, Any]:
    """Validate Telegram WebApp ``initData`` and return parsed fields.

    Parameters
    ----------
    init_data:
        Raw query-string from ``Telegram.WebApp.initData``.
    bot_token:
        Bot token from BotFather (same as ``TELEGRAM_BOT_TOKEN``).
    max_age_seconds:
        Reject when ``auth_date`` is older than this. ``None`` uses env/default.

    Returns
    -------
    dict
        Parsed key/value fields. ``user`` / ``receiver`` / ``chat`` are JSON
        objects when present. Includes ``_auth_date`` as int when valid.

    Raises
    ------
    ValueError
        Missing/invalid hash, bad token, or expired ``auth_date``.
    """
    if not init_data or not str(init_data).strip():
        raise ValueError("init_data is empty")
    if not bot_token or not str(bot_token).strip():
        raise ValueError("bot_token is empty")

    pairs = dict(parse_qsl(str(init_data).strip(), keep_blank_values=True))
    received_hash = pairs.pop("hash", None)
    if not received_hash:
        raise ValueError("init_data missing hash")

    # Alphabetical key=<value> joined by \n (hash excluded).
    data_check_string = "\n".join(f"{k}={v}" for k, v in sorted(pairs.items()))

    secret_key = hmac.new(
        key=b"WebAppData",
        msg=str(bot_token).encode("utf-8"),
        digestmod=hashlib.sha256,
    ).digest()
    calculated = hmac.new(
        key=secret_key,
        msg=data_check_string.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(calculated, received_hash):
        raise ValueError("init_data hash mismatch")

    age_limit = (
        DEFAULT_MAX_AGE_SECONDS if max_age_seconds is None else int(max_age_seconds)
    )
    # auth_date is mandatory: without it a leaked initData never expires.
    auth_raw = pairs.get("auth_date")
    if auth_raw is None or not str(auth_raw).strip():
        raise ValueError("init_data missing auth_date")
    try:
        auth_date = int(auth_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid auth_date") from exc
    if age_limit > 0 and (time.time() - auth_date) > age_limit:
        raise ValueError("init_data expired (auth_date too old)")
    pairs["_auth_date"] = auth_date

    out: dict[str, Any] = dict(pairs)
    for json_key in ("user", "receiver", "chat"):
        raw = out.get(json_key)
        if isinstance(raw, str) and raw.strip().startswith("{"):
            try:
                out[json_key] = json.loads(raw)
            except json.JSONDecodeError:
                pass
    return out


def extract_init_data(request: Request) -> str | None:
    """Read initData from ``Authorization: tma …`` or ``X-Telegram-Init-Data``."""
    auth = (request.headers.get("Authorization") or "").strip()
    if auth.lower().startswith("tma "):
        return auth[4:].strip() or None
    header = (request.headers.get("X-Telegram-Init-Data") or "").strip()
    return header or None


async def require_telegram_webapp(request: Request) -> dict[str, Any] | None:
    """FastAPI dependency: verify Mini App initData unless auth is disabled.

    Returns parsed init fields, or ``None`` when ``TELEGRAM_AUTH_DISABLED=true``.
    """
    if telegram_auth_disabled():
        if is_production():
            # Defence in depth: create_app() already refuses to start.
            logger.error("Telegram auth disabled in production — rejecting request")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication misconfigured",
            )
        return None

    cached = getattr(request.state, "telegram_init", None)
    if cached is not None:
        return cached

    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN missing while Telegram auth is enabled")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Telegram auth misconfigured (missing bot token)",
        )

    init_data = extract_init_data(request)
    if not init_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Telegram initData (Authorization: tma … or X-Telegram-Init-Data)",
            headers={"WWW-Authenticate": "tma"},
        )

    try:
        parsed = verify_telegram_webapp_data(
            init_data,
            token,
            max_age_seconds=_max_age_seconds(),
        )
    except ValueError as exc:
        logger.info("Telegram initData rejected: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Telegram initData",
            headers={"WWW-Authenticate": "tma"},
        ) from exc
    request.state.telegram_init = parsed
    return parsed


def _telegram_user_id(init: dict[str, Any] | None) -> int | None:
    if not init:
        return None
    user = init.get("user")
    if isinstance(user, dict) and user.get("id") is not None:
        try:
            return int(user["id"])
        except (TypeError, ValueError):
            return None
    return None


async def require_admin(request: Request) -> dict[str, Any] | None:
    """FastAPI dependency for write endpoints: Telegram user ∈ ``ADMIN_TELEGRAM_IDS``.

    With auth disabled (local / tests — never prod, see
    :func:`assert_auth_config_safe`) the check passes. With auth enabled an
    empty ``ADMIN_TELEGRAM_IDS`` denies everyone (fail closed).
    """
    init = await require_telegram_webapp(request)
    if init is None and telegram_auth_disabled():
        return None
    uid = _telegram_user_id(init)
    if uid is None or uid not in admin_telegram_ids():
        logger.warning("Admin endpoint denied for telegram user id=%s", uid)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return init


# Alias for Depends(...) at call sites
TelegramWebAppUser = dict[str, Any] | None
RequireTelegramAuth = Depends(require_telegram_webapp)
RequireAdmin = Depends(require_admin)
