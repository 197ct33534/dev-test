"""Async SQLAlchemy engine / session factory for Quant Engine v2.

Local default: ``sqlite+aiosqlite`` (file or ``:memory:``).
Production: set ``DATABASE_URL`` to a Postgres async URL, e.g.::

    DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/score_v2

Sync Postgres (psycopg) is also accepted; prefer async URLs for the
Quant Engine path.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import AsyncIterator, Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

DEFAULT_SQLITE_URL = "sqlite+aiosqlite:///./data/quant_engine_v2.db"


def get_database_url(explicit: Optional[str] = None) -> str:
    """Resolve DB URL from argument or ``DATABASE_URL`` env (else local SQLite)."""
    if explicit:
        return explicit
    return os.environ.get("DATABASE_URL", DEFAULT_SQLITE_URL).strip() or DEFAULT_SQLITE_URL


def _normalize_async_url(url: str) -> str:
    """Coerce common sync URLs to async drivers when needed."""
    if url.startswith("sqlite://") and not url.startswith("sqlite+aiosqlite://"):
        return url.replace("sqlite://", "sqlite+aiosqlite://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url


def create_async_engine_from_url(
    url: Optional[str] = None,
    *,
    echo: bool = False,
) -> AsyncEngine:
    """Build an async engine (aiosqlite locally; asyncpg when DATABASE_URL is Postgres)."""
    resolved = _normalize_async_url(get_database_url(url))
    connect_args: dict = {}
    if resolved.startswith("sqlite+aiosqlite"):
        # Needed for check_same_thread=False style usage with aiosqlite.
        connect_args["check_same_thread"] = False
    return create_async_engine(resolved, echo=echo, connect_args=connect_args)


def async_session_factory(
    engine: Optional[AsyncEngine] = None,
    *,
    url: Optional[str] = None,
    echo: bool = False,
) -> async_sessionmaker[AsyncSession]:
    """Return an ``async_sessionmaker`` bound to ``engine`` or a new engine from URL."""
    eng = engine or create_async_engine_from_url(url, echo=echo)
    return async_sessionmaker(eng, expire_on_commit=False, class_=AsyncSession)


@lru_cache(maxsize=1)
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Process-wide cached session factory (uses ``DATABASE_URL`` / default SQLite)."""
    return async_session_factory()


async def session_scope(
    factory: Optional[async_sessionmaker[AsyncSession]] = None,
) -> AsyncIterator[AsyncSession]:
    """Async context helper yielding a session that commits on success."""
    maker = factory or get_session_factory()
    async with maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
