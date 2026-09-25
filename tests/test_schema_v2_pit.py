"""Tests for Quant Engine v2 schema, entity resolution, and PIT integrity gate."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.data.entity_resolver import UNMAPPED, EntityResolver
from src.data.pit_engine import HARD_GATE_CODE, PITEngine
from src.db.schema_v2 import (
    CanonicalCompetition,
    CanonicalMatch,
    CanonicalTeam,
    RawDataLake,
    init_schema_v2_async,
)
from src.monitoring.data_monitor import DataMonitor


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    await init_schema_v2_async(engine)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as sess:
        yield sess
    await engine.dispose()


async def _seed_match(session: AsyncSession) -> CanonicalMatch:
    home = CanonicalTeam(name="Arsenal", mapping_status="MAPPED")
    away = CanonicalTeam(name="Chelsea", mapping_status="MAPPED")
    comp = CanonicalCompetition(
        name="Premier League", code="EPL", mapping_status="MAPPED"
    )
    session.add_all([home, away, comp])
    await session.flush()
    match = CanonicalMatch(
        home_team_id=home.id,
        away_team_id=away.id,
        competition_id=comp.id,
        kickoff_at=datetime(2025, 9, 20, 15, 0, tzinfo=timezone.utc),
        season="2025/2026",
        status="SCHEDULED",
    )
    session.add(match)
    await session.flush()
    return match


@pytest.mark.asyncio
async def test_schema_create_all_tables(session: AsyncSession) -> None:
    match = await _seed_match(session)
    assert match.id is not None
    assert match.home_team_id is not None


@pytest.mark.asyncio
async def test_entity_resolver_creates_unmapped_and_reuses(
    session: AsyncSession,
) -> None:
    monitor = DataMonitor()
    resolver = EntityResolver(session, monitor=monitor)

    tid1 = await resolver.get_or_create_canonical_team(
        "OH Leuven", "flashscore", source_team_id="fs-123"
    )
    tid2 = await resolver.get_or_create_canonical_team(
        "OH Leuven", "flashscore", source_team_id="fs-123"
    )
    assert tid1 == tid2
    assert any(w["code"] == "ENTITY_UNMAPPED_TEAM" for w in monitor.warnings)

    cid1 = await resolver.get_or_create_canonical_competition(
        "WSL", "football-data", source_competition_id="E0"
    )
    cid2 = await resolver.get_or_create_canonical_competition(
        "WSL", "football-data", source_competition_id="E0"
    )
    assert cid1 == cid2
    assert any(w["code"] == "ENTITY_UNMAPPED_COMPETITION" for w in monitor.warnings)

    team = (
        await session.execute(select(CanonicalTeam).where(CanonicalTeam.id == tid1))
    ).scalar_one()
    assert team.mapping_status == UNMAPPED


async def _raw_records_for_match(
    session: AsyncSession, match_id: object
) -> list[dict]:
    """Load RawDataLake rows as dicts for the dict-based PITEngine API."""
    rows = (
        await session.execute(
            select(RawDataLake).where(RawDataLake.canonical_match_id == match_id)
        )
    ).scalars().all()
    return [
        {
            "source_id": r.source_id,
            "observed_at": r.observed_at,
            "payload": r.payload_json,
        }
        for r in rows
    ]


@pytest.mark.asyncio
async def test_pit_clean_window_passes(session: AsyncSession) -> None:
    match = await _seed_match(session)
    as_of = datetime(2025, 9, 19, 12, 0, tzinfo=timezone.utc)
    session.add(
        RawDataLake(
            payload_json={"xg_home": 1.2},
            observed_at=as_of - timedelta(hours=2),
            source_timestamp=as_of - timedelta(hours=2),
            source_type="odds_api",
            source_id="obs-1",
            canonical_match_id=match.id,
        )
    )
    session.add(
        RawDataLake(
            payload_json={"xg_home": 1.3},
            observed_at=as_of,  # inclusive boundary
            source_timestamp=as_of,
            source_type="odds_api",
            source_id="obs-2",
            canonical_match_id=match.id,
        )
    )
    await session.flush()

    monitor = DataMonitor()
    engine = PITEngine(on_hard_gate=monitor.hard_gate)
    records = await _raw_records_for_match(session, match.id)
    valid, pit_ok = engine.get_valid_raw_records(records, as_of)

    assert pit_ok is True
    assert len(valid) == 2
    assert monitor.hard_gates == []


@pytest.mark.asyncio
async def test_pit_leakage_fails_hard_gate(session: AsyncSession) -> None:
    match = await _seed_match(session)
    as_of = datetime(2025, 9, 19, 12, 0, tzinfo=timezone.utc)
    session.add(
        RawDataLake(
            payload_json={"xg_home": 1.2},
            observed_at=as_of - timedelta(hours=1),
            source_type="odds_api",
            source_id="clean",
            canonical_match_id=match.id,
        )
    )
    session.add(
        RawDataLake(
            payload_json={"xg_home": 9.9, "leaked": True},
            observed_at=as_of + timedelta(minutes=1),
            source_type="odds_api",
            source_id="leak",
            canonical_match_id=match.id,
        )
    )
    await session.flush()

    monitor = DataMonitor()
    engine = PITEngine(on_hard_gate=monitor.hard_gate)
    records = await _raw_records_for_match(session, match.id)
    valid, pit_ok = engine.get_valid_raw_records(records, as_of)

    assert pit_ok is False
    assert len(valid) == 1
    assert len(monitor.hard_gates) == 1
    gate = monitor.hard_gates[0]
    assert gate["code"] == HARD_GATE_CODE
    assert gate["no_bet"] is True
    assert gate["leak_count"] == 1
    # Leaked payload must not appear in valid set
    assert {o["source_id"] for o in valid} == {"clean"}


@pytest.mark.asyncio
async def test_pit_requires_timezone_aware(session: AsyncSession) -> None:
    await _seed_match(session)
    engine = PITEngine()
    with pytest.raises(ValueError, match="timezone-aware"):
        engine.get_valid_raw_records([], datetime(2025, 9, 19, 12, 0))  # naive
