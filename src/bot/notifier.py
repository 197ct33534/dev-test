"""Telegram HTML value-signal alerts via Bot HTTP API (no PTB app required).

The FastAPI scheduler process pushes with ``TELEGRAM_BOT_TOKEN`` + httpx so it
does not share a ``python-telegram-bot`` Application with the polling bot.
"""

from __future__ import annotations

import html
import logging
import os
from typing import Any, Mapping, Sequence

import httpx

from src.bot.signal_store import deactivate_subscriber, list_active_chat_ids

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
DEFAULT_WEBAPP_URL = "http://127.0.0.1:8000/webapp/"


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=False)


def _ev_as_pct(ev: float | None, ev_pct: float | None) -> float | None:
    if ev_pct is not None:
        try:
            return float(ev_pct)
        except (TypeError, ValueError):
            pass
    if ev is None:
        return None
    try:
        v = float(ev)
    except (TypeError, ValueError):
        return None
    # Fraction (0.08) vs already-percent (8.0).
    return v * 100.0 if abs(v) <= 2.0 else v


def _line_pair(bet: Mapping[str, Any]) -> tuple[str, str]:
    fair = (
        bet.get("model_fair_line")
        or bet.get("fair_ou_line")
        or bet.get("fair_ah_line")
        or "—"
    )
    bookie = (
        bet.get("bookie_market_line")
        or bet.get("bookie_ou_line")
        or bet.get("bookie_ah_line")
        or "—"
    )
    return str(fair), str(bookie)


def _reason_summary(bet: Mapping[str, Any], *, max_chars: int = 280) -> str:
    reasons = bet.get("ai_reasons") or bet.get("insights") or []
    if isinstance(reasons, str):
        text = reasons.strip()
    elif isinstance(reasons, Sequence):
        parts = [str(r).strip() for r in reasons if str(r).strip()]
        text = " · ".join(parts)
    else:
        text = ""
    if not text:
        edge = bet.get("line_edge")
        if edge:
            text = f"Lệch kèo: {edge}"
        else:
            text = "Model đánh giá cửa này có biên giá trị so với nhà cái."
    if len(text) > max_chars:
        return text[: max_chars - 1].rstrip() + "…"
    return text


def format_hot_value_signal(bet: Mapping[str, Any]) -> str:
    """Build the hot value-signal HTML body (parse_mode=HTML)."""
    home = bet.get("home") or bet.get("home_team") or "?"
    away = bet.get("away") or bet.get("away_team") or "?"
    league = (
        bet.get("competition")
        or bet.get("league")
        or bet.get("comp_id")
        or "?"
    )
    when = bet.get("kickoff_vn") or bet.get("kickoff") or bet.get("match_date") or "TBD"
    market = bet.get("market") or bet.get("bet_type") or "?"
    selection = bet.get("selection") or "?"
    odds = bet.get("odds") if bet.get("odds") is not None else bet.get("bookmaker_odds")
    try:
        odds_s = f"{float(odds):.2f}" if odds is not None else "—"
    except (TypeError, ValueError):
        odds_s = "—"

    ev_pct = _ev_as_pct(bet.get("ev"), bet.get("ev_pct"))
    ev_s = f"{ev_pct:+.1f}" if ev_pct is not None else "+0.0"
    fair, bookie = _line_pair(bet)
    reason = _reason_summary(bet)

    return (
        "🚨 <b>TÍN HIỆU VALUE BET NÓNG</b>\n"
        f"⚽ Trận: {_esc(home)} vs {_esc(away)} ({_esc(league)})\n"
        f"⏰ Giờ đá: {_esc(when)}\n"
        f"🎯 AI Pick: {_esc(market)} - {_esc(selection)} @ {_esc(odds_s)}\n"
        f"📈 Chỉ số: EV = {_esc(ev_s)}% | "
        f"Kèo Đề Xuất: {_esc(fair)} vs Nhà cái: {_esc(bookie)}\n"
        f"💡 Lý do: {_esc(reason)}"
    )


def webapp_inline_keyboard(webapp_url: str | None = None) -> dict[str, Any]:
    """InlineKeyboardMarkup JSON with WebApp open button."""
    url = (webapp_url or os.getenv("WEBAPP_URL") or DEFAULT_WEBAPP_URL).strip()
    if not url:
        url = DEFAULT_WEBAPP_URL
    return {
        "inline_keyboard": [
            [
                {
                    "text": "🔥 Mở WebApp Soi Kèo",
                    "web_app": {"url": url},
                }
            ]
        ]
    }


async def send_telegram_message_async(
    bot_token: str,
    chat_id: str,
    text: str,
    *,
    reply_markup: dict[str, Any] | None = None,
    parse_mode: str = "HTML",
    timeout: float = 20.0,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """POST ``sendMessage`` via Bot API. Raises on HTTP/API failure."""
    token = (bot_token or "").strip()
    chat = str(chat_id or "").strip()
    if not token or not chat:
        raise ValueError("Thiếu bot_token hoặc chat_id Telegram.")

    payload: dict[str, Any] = {
        "chat_id": chat,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup

    url = TELEGRAM_API.format(token=token)
    owns = client is None
    http = client or httpx.AsyncClient(timeout=timeout)
    try:
        resp = await http.post(url, json=payload)
        try:
            data = resp.json() if resp.content else {}
        except Exception:  # noqa: BLE001
            data = {"raw": (resp.text or "")[:500]}
        if resp.status_code != 200 or not (isinstance(data, dict) and data.get("ok")):
            desc = ""
            if isinstance(data, dict):
                desc = str(data.get("description") or "")
            desc = desc or resp.reason_phrase or (resp.text or "")[:200]
            raise RuntimeError(f"Telegram API lỗi — HTTP {resp.status_code}: {desc}")
        return data if isinstance(data, dict) else {"ok": True, "result": data}
    finally:
        if owns:
            await http.aclose()


async def broadcast_value_signal(
    bet: Mapping[str, Any],
    *,
    bot_token: str | None = None,
    webapp_url: str | None = None,
    chat_ids: Sequence[str] | None = None,
    db_path: Any = None,
) -> dict[str, Any]:
    """Format + send one signal to all active subscribers."""
    token = (bot_token or os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        logger.warning("broadcast skipped: TELEGRAM_BOT_TOKEN missing")
        return {"sent": 0, "failed": 0, "skipped": True, "errors": ["no_token"]}

    recipients = list(chat_ids) if chat_ids is not None else list_active_chat_ids(db_path)
    if not recipients:
        return {"sent": 0, "failed": 0, "skipped": True, "errors": ["no_subscribers"]}

    text = format_hot_value_signal(bet)
    markup = webapp_inline_keyboard(webapp_url)
    sent = 0
    failed = 0
    errors: list[str] = []

    async with httpx.AsyncClient(timeout=20.0) as client:
        for cid in recipients:
            try:
                await send_telegram_message_async(
                    token,
                    cid,
                    text,
                    reply_markup=markup,
                    client=client,
                )
                sent += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                msg = f"{cid}: {exc}"
                errors.append(msg)
                logger.warning("Telegram push failed: %s", msg)
                err_l = str(exc).lower()
                if "blocked" in err_l or "chat not found" in err_l or "deactivated" in err_l:
                    try:
                        deactivate_subscriber(cid, db_path=db_path)
                    except Exception:  # noqa: BLE001
                        pass

    return {"sent": sent, "failed": failed, "skipped": False, "errors": errors}
