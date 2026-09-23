#!/usr/bin/env python3
"""Start the FastAPI server (uvicorn).

Usage
-----
    python run_api.py
    python run_api.py --host 0.0.0.0 --port 8000 --reload

Default port is 8000. If ``127.0.0.1:<port>`` is already taken (common on
Windows with Laragon / ``php -S 127.0.0.1:8000``), the launcher automatically
tries the next free port so ``http://127.0.0.1:<port>/health`` works.

Swagger UI: http://127.0.0.1:<port>/docs
WebApp:     http://127.0.0.1:<port>/webapp/
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

import uvicorn

DEFAULT_PORT = 8000
_PORT_TRIES = 30


def _localhost_free(port: int) -> bool:
    """True when nothing accepts connections on ``127.0.0.1:port``.

    Binding ``0.0.0.0:port`` can succeed even when ``127.0.0.1:port`` is
    already taken (e.g. Laragon PHP). Browsers hitting localhost then miss
    FastAPI — so we always require the loopback address to be free.

    Uses ``connect_ex`` (not ``bind`` + ``SO_REUSEADDR``): on Windows,
    ``SO_REUSEADDR`` can make a busy port look free.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        # 0 → something accepted the TCP handshake (port in use).
        return sock.connect_ex(("127.0.0.1", int(port))) != 0


def _who_listens(port: int) -> str:
    """Best-effort hint for Windows / Unix about what owns the port."""
    try:
        if sys.platform.startswith("win"):
            import subprocess

            out = subprocess.check_output(
                ["netstat", "-ano"],
                text=True,
                errors="replace",
                timeout=5,
            )
            hits = [
                ln.strip()
                for ln in out.splitlines()
                if f":{port} " in ln.replace("\t", " ") and "LISTENING" in ln.upper()
            ]
            return "; ".join(hits[:3]) if hits else ""
    except Exception:  # noqa: BLE001
        pass
    return ""


def pick_port(preferred: int, *, fixed: bool) -> int:
    """Return ``preferred`` if free, else the next free port (unless ``fixed``)."""
    preferred = int(preferred)
    if _localhost_free(preferred):
        return preferred
    hint = _who_listens(preferred)
    msg = (
        f"Port {preferred} is already in use on 127.0.0.1"
        + (f" ({hint})" if hint else "")
        + "."
    )
    if fixed:
        raise SystemExit(
            msg
            + " Stop that process, or re-run without locking the port "
            "(omit a forced busy port / pick another --port)."
        )
    for candidate in range(preferred + 1, preferred + _PORT_TRIES + 1):
        if _localhost_free(candidate):
            print(
                f"WARNING: {msg} Using port {candidate} instead.\n"
                f"         Open http://127.0.0.1:{candidate}/docs "
                f"(not localhost:{preferred}).",
                file=sys.stderr,
            )
            return candidate
    raise SystemExit(
        msg + f" No free port found in {preferred}..{preferred + _PORT_TRIES}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score FastAPI server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Preferred port (default {DEFAULT_PORT}; auto-bumps if busy)",
    )
    parser.add_argument(
        "--strict-port",
        action="store_true",
        help="Fail instead of auto-selecting another port when --port is busy",
    )
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    # Ensure repo root is importable when launched from elsewhere.
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    port = pick_port(int(args.port), fixed=bool(args.strict_port))
    print(
        f"Score API → http://127.0.0.1:{port}/docs  ·  "
        f"WebApp → http://127.0.0.1:{port}/webapp/",
        flush=True,
    )

    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=int(port),
        reload=bool(args.reload),
    )


if __name__ == "__main__":
    main()
