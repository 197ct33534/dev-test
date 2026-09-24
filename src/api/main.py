"""FastAPI application — parallel REST layer (Streamlit unchanged)."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.auth import require_telegram_webapp
from src.api.routes import api_v1, router as root_router
from src.api.routes_analytics import router as analytics_router
from src.data_loader import PROJECT_ROOT


def _cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "*").strip()
    if not raw or raw == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


def _webapp_dir() -> Path:
    """Resolve ``src/webapp`` relative to project root (cwd-independent)."""
    candidates = (
        PROJECT_ROOT / "src" / "webapp",
        Path(__file__).resolve().parent.parent / "webapp",
    )
    for path in candidates:
        if path.is_dir():
            return path
    # Create empty dir so mount still works before files land.
    target = PROJECT_ROOT / "src" / "webapp"
    target.mkdir(parents=True, exist_ok=True)
    return target


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start optional value-signal BackgroundScheduler; stop on shutdown."""
    from src.api.services.scheduler import start_scheduler, stop_scheduler

    scheduler = start_scheduler()
    app.state.scheduler = scheduler
    try:
        yield
    finally:
        stop_scheduler()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Score Value-Bet API",
        description=(
            "REST layer over Dixon–Coles / scanner / global_matches.db. "
            "Streamlit UI remains available separately."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    origins = _cors_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(root_router)
    api_v1.include_router(analytics_router)
    # Protect /api/v1/* with Telegram WebApp initData (bypass via TELEGRAM_AUTH_DISABLED).
    # /health, /docs, /webapp stay public.
    app.include_router(
        api_v1,
        dependencies=[Depends(require_telegram_webapp)],
    )

    webapp = _webapp_dir()
    app.mount(
        "/webapp",
        StaticFiles(directory=str(webapp), html=True),
        name="webapp",
    )
    return app


app = create_app()
