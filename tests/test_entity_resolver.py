"""Async tests for EntityResolver (aiosqlite in-memory)."""

from __future__ import annotations

import logging
from typing import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.data.entity_resolver import UNMAPPED, EntityResolver
from src.db.schema_v2 import CanonicalTeam, TeamAlias, init_schema_v2_async


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    await init_schema_v2_async(engine)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as sess:
        yield sess
    await engine.dispose()


@pytest.mark.asyncio
async def test_duplicate_alias_maps_to_same_canonical_team(
    session: AsyncSession,
) -> None:
    resolver = EntityResolver()

    tid1 = await resolver.get_or_create_canonical_team(
        session, "Arsenal", "football-data", "ARS"
    )
    tid2 = await resolver.get_or_create_canonical_team(
        session, "Arsenal", "football-data", "ARS"
    )

    assert tid1 == tid2

    aliases = (
        await session.execute(
            select(TeamAlias).where(
                TeamAlias.source_type == "football-data",
                TeamAlias.source_team_id == "ARS",
            )
        )
    ).scalars().all()
    assert len(aliases) == 1
    assert aliases[0].canonical_team_id == tid1


@pytest.mark.asyncio
async def test_unmapped_creates_team_alias_and_warning(
    session: AsyncSession,
    caplog: pytest.LogCaptureFixture,
) -> None:
    resolver = EntityResolver()

    with caplog.at_level(logging.WARNING, logger="src.data.entity_resolver"):
        tid = await resolver.get_or_create_canonical_team(
            session, "OH Leuven", "flashscore", "fs-123"
        )

    team = (
        await session.execute(select(CanonicalTeam).where(CanonicalTeam.id == tid))
    ).scalar_one()
    assert team.mapping_status == UNMAPPED
    assert team.name == "OH Leuven"

    alias = (
        await session.execute(
            select(TeamAlias).where(TeamAlias.canonical_team_id == tid)
        )
    ).scalar_one()
    assert alias.mapping_status == UNMAPPED
    assert alias.source_type == "flashscore"
    assert alias.source_team_id == "fs-123"

    assert any("UNMAPPED" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_duplicate_alias_maps_to_same_canonical_competition(
    session: AsyncSession,
) -> None:
    resolver = EntityResolver()

    cid1 = await resolver.get_or_create_canonical_competition(
        session, "Premier League", "football-data", "E0"
    )
    cid2 = await resolver.get_or_create_canonical_competition(
        session, "Premier League", "football-data", source_comp_id="E0"
    )

    assert cid1 == cid2


@pytest.mark.asyncio
async def test_unmapped_competition_warning(
    session: AsyncSession,
    caplog: pytest.LogCaptureFixture,
) -> None:
    resolver = EntityResolver()

    with caplog.at_level(logging.WARNING, logger="src.data.entity_resolver"):
        cid = await resolver.get_or_create_canonical_competition(
            session, "WSL", "flashscore", "wsl-1"
        )

    assert cid is not None
    assert any("UNMAPPED" in r.message for r in caplog.records)
