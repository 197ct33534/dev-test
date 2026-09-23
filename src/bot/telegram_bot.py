#!/usr/bin/env python3
"""Telegram bot that opens the Score WebApp.

Requires
--------
- ``TELEGRAM_BOT_TOKEN`` in ``.env``
- ``WEBAPP_URL`` — HTTPS public URL of the WebApp for production
  (default ``http://127.0.0.1:8000/webapp/`` for local smoke tests).

Telegram only launches WebApps over **HTTPS** (or ``http://127.0.0.1`` /
``localhost`` in some BotFather/dev setups). For real chats, tunnel the API
(ngrok / cloudflare) and set ``WEBAPP_URL`` to that HTTPS origin + ``/webapp/``.
If ``python run_api.py`` prints a non-8000 port (Laragon conflict), set
``WEBAPP_URL`` to that port.

Run
---
    python -m src.bot.telegram_bot
    # or
    python src/bot/telegram_bot.py

``/start`` persists ``chat_id`` into ``data/notified_signals.db``
(``telegram_subscribers``) so the FastAPI value-signal scheduler can push
alerts via Bot HTTP API without sharing this polling process.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Repo root on sys.path when executed as a script.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update, WebAppInfo
from telegram.ext import Application, CommandHandler, ContextTypes

from src.bot.signal_store import upsert_subscriber

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("telegram_bot")

DEFAULT_WEBAPP_URL = "http://127.0.0.1:8000/webapp/"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    webapp_url = os.getenv("WEBAPP_URL", DEFAULT_WEBAPP_URL).strip() or DEFAULT_WEBAPP_URL
    user = update.effective_user
    chat = update.effective_chat
    if chat is not None:
        try:
            upsert_subscriber(
                chat.id,
                username=getattr(user, "username", None) if user else None,
                first_name=getattr(user, "first_name", None) if user else None,
            )
            logger.info("Subscribed chat_id=%s", chat.id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist subscriber: %s", exc)

    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "🔥 Soi Kèo Ngay",
                    web_app=WebAppInfo(url=webapp_url),
                )
            ]
        ]
    )
    text = (
        "Chào bạn! Mình là bot Soi Kèo.\n\n"
        "Bấm nút bên dưới để mở WebApp — xem Top kèo hời, EV%, "
        "và hồ sơ đội bóng.\n\n"
        "Bạn đã đăng ký nhận **tín hiệu value bet nóng** "
        "(EV cao / lệch kèo) khi API scheduler bật.\n\n"
        f"WebApp: `{webapp_url}`"
    )
    if update.message:
        await update.message.reply_text(text, reply_markup=keyboard, parse_mode="Markdown")


def build_app(token: str) -> Application:
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    return app


def main() -> None:
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        logger.error("Missing TELEGRAM_BOT_TOKEN in environment / .env")
        sys.exit(1)
    webapp = os.getenv("WEBAPP_URL", DEFAULT_WEBAPP_URL)
    logger.info("Starting bot · WEBAPP_URL=%s", webapp)
    local = "localhost" in webapp or "127.0.0.1" in webapp
    if webapp.startswith("http://") and not local:
        logger.warning(
            "Telegram WebApps require HTTPS in production. "
            "Use a tunnel and set WEBAPP_URL accordingly."
        )
    application = build_app(token)
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
