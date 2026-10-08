#!/usr/bin/env python3
"""Independent Telegram connectivity check (loads .env via python-dotenv).

Usage
-----
python scripts/test_telegram.py

Prints a masked preview of TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID, then POSTs
a test message. On failure prints HTTP status + Telegram API description.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

# Load .env from repo root before reading os.environ
load_dotenv(ROOT / ".env")
load_dotenv()  # also cwd, if different

import requests

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


def mask_secret(value: str, *, keep: int = 4) -> str:
    """Show first/last ``keep`` chars only (never the full secret)."""
    text = str(value or "").strip()
    if not text:
        return "<EMPTY>"
    if len(text) <= keep * 2:
        return text[:1] + "***" + text[-1:]
    return f"{text[:keep]}…{text[-keep:]} (len={len(text)})"


def main() -> int:
    token = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    chat_id = (os.environ.get("TELEGRAM_CHAT_ID") or "").strip()

    print("=== Telegram connectivity test ===")
    print(f".env path tried : {ROOT / '.env'}")
    print(f"Token preview   : {mask_secret(token)}")
    print(f"Chat ID preview : {mask_secret(chat_id)}")

    if not token:
        print("FAIL: TELEGRAM_BOT_TOKEN trống — kiểm tra file .env", file=sys.stderr)
        return 1
    if not chat_id:
        print("FAIL: TELEGRAM_CHAT_ID trống — kiểm tra file .env", file=sys.stderr)
        return 1

    url = TELEGRAM_API.format(token=token)
    payload = {
        "chat_id": chat_id,
        "text": "🧪 [TEST] Kết nối Telegram Bot thành công!",
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    print("\nPOST sendMessage …")
    try:
        resp = requests.post(url, json=payload, timeout=20.0)
    except requests.RequestException as exc:
        print(f"FAIL: HTTP request exception: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    status = resp.status_code
    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text[:500]}

    print(f"HTTP status     : {status}")
    print(f"API response    : {json.dumps(body, ensure_ascii=False, indent=2)}")

    if status == 200 and isinstance(body, dict) and body.get("ok"):
        print("\nOK: Telegram nhận tin test thành công.")
        return 0

    # Human-readable common errors
    desc = ""
    if isinstance(body, dict):
        desc = str(body.get("description") or body.get("error_code") or "")
    hint = ""
    low = (desc or "").lower()
    if status == 401 or "unauthorized" in low:
        hint = " → Token sai / hết hạn. Lấy lại từ @BotFather."
    elif status == 400 and "chat not found" in low:
        hint = " → Chat ID sai, hoặc chưa từng /start bot trong chat đó."
    elif status == 403 or "forbidden" in low:
        hint = " → Bot bị block / chưa được add vào group / thiếu quyền."
    elif status == 400:
        hint = " → Bad Request (chat_id format? thử số nguyên không dấu ngoặc)."

    print(
        f"\nFAIL: {status} {desc or resp.reason}{hint}",
        file=sys.stderr,
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
