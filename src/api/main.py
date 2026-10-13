"""FastAPI application — parallel REST layer (Streamlit unchanged)."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.auth import assert_auth_config_safe, is_production, require_telegram_webapp
from src.api.routes import api_v1, router as root_router
from src.api.routes_analytics import router as analytics_router
from src.api.routes_v2 import api_v2
from src.data_loader import PROJECT_ROOT


def _cors_origins() -> list[str]:
    """Allowed CORS origins from ``CORS_ORIGINS`` (comma-separated).

    Unset: ``*`` for local/dev; in production only ``https://$DOMAIN`` (the
    WebApp is served same-origin, so no wildcard is needed).
    """
    raw = (os.getenv("CORS_ORIGINS") or "").strip()
    if raw == "*":
        return ["*"]
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    if not is_production():
        return ["*"]
    domain = (os.getenv("DOMAIN") or "").strip()
    return [f"https://{domain}"] if domain else []


def _docs_enabled() -> bool:
    """Swagger / ReDoc / OpenAPI: on locally, off in prod unless ``ENABLE_DOCS=true``."""
    raw = (os.getenv("ENABLE_DOCS") or "").strip().lower()
    if raw:
        return raw in {"1", "true", "yes", "on"}
    return not is_production()


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


def _dashboard_dist() -> Path | None:
    """Built React dashboard (``dashboard/dist``), or ``None`` when not built.

    ``DASHBOARD_DIST`` overrides the location (Docker copies the Node build
    stage output there).
    """
    raw = (os.getenv("DASHBOARD_DIST") or "").strip()
    path = Path(raw) if raw else PROJECT_ROOT / "dashboard" / "dist"
    return path if (path / "index.html").is_file() else None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Optional APScheduler in lifespan (local / single-worker only).

    Docker ``web_api`` sets ENABLE_*_SCHEDULER=false — value alerts run on
    ``telegram_bot``, settlement on ``settlement_worker``.
    """
    from src.api.services.scheduler import start_scheduler, stop_scheduler

    scheduler = start_scheduler()
    app.state.scheduler = scheduler
    try:
        yield
    finally:
        stop_scheduler()


def create_app() -> FastAPI:
    # Fail fast: never serve a public deployment with auth bypassed.
    assert_auth_config_safe()
    docs = _docs_enabled()
    app = FastAPI(
        title="Score Value-Bet API",
        description=(
            "REST layer over Dixon–Coles / scanner / global_matches.db. "
            "Streamlit UI remains available separately."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if docs else None,
        redoc_url="/redoc" if docs else None,
        openapi_url="/openapi.json" if docs else None,
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
    # Protect /api/v1/* and /api/v2/* with Telegram WebApp initData
    # (bypass via TELEGRAM_AUTH_DISABLED). /health, /docs, /webapp, /dashboard stay public.
    app.include_router(
        api_v1,
        dependencies=[Depends(require_telegram_webapp)],
    )
    app.include_router(
        api_v2,
        dependencies=[Depends(require_telegram_webapp)],
    )

    # React dashboard: static shell is public, its data calls /api/v1 with a
    # Bearer token (DASHBOARD_API_TOKENS) or Telegram initData.
    dashboard = _dashboard_dist()
    if dashboard is not None:
        app.mount(
            "/dashboard",
            StaticFiles(directory=str(dashboard), html=True),
            name="dashboard",
        )

    webapp = _webapp_dir()
    app.mount(
        "/webapp",
        StaticFiles(directory=str(webapp), html=True),
        name="webapp",
    )
    return app


app = create_app()
