"""Unit tests for CompetitionResolver + BulkFootballDataImporter (no network)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.data.bulk_importer import BulkFootballDataImporter
from src.data.competition_resolver import (
    PENDING_REVIEW,
    SIMILARITY_THRESHOLD,
    CompetitionResolver,
    similarity_ratio,
)
from src.data.entity_resolver import MAPPED
from src.db.schema_v2 import (
    CanonicalCompetition,
    CanonicalMatch,
    CompetitionAlias,
    RawDataLake,
    StagingUnmappedCompetition,
    init_schema_v2_async,
)

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "football_data_tiny.csv"


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    await init_schema_v2_async(engine)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as sess:
        yield sess
    await engine.dispose()


@pytest.mark.asyncio
async def test_fuzzy_resolve_maps_above_threshold(session: AsyncSession) -> None:
    session.add(
        CanonicalCompetition(
            name="Premier League", code="E0", mapping_status=MAPPED
        )
    )
    await session.flush()

    # "Premier Leage" is a close typo — should clear 0.88 with SequenceMatcher.
    assert similarity_ratio("Premier Leage", "Premier League") >= SIMILARITY_THRESHOLD

    resolver = CompetitionResolver(session)
    cid = await resolver.resolve_competition(
        session, "football-data", "Premier Leage"
    )
    assert cid is not None

    alias = (
        await session.execute(
            select(CompetitionAlias).where(
                CompetitionAlias.source_type == "football-data",
                CompetitionAlias.source_name == "Premier Leage",
            )
        )
    ).scalar_one()
    assert alias.canonical_competition_id == cid
    assert alias.mapping_status == MAPPED

    staged = (
        await session.execute(select(StagingUnmappedCompetition))
    ).scalars().all()
    assert staged == []


@pytest.mark.asyncio
async def test_fuzzy_resolve_stages_below_threshold(
    session: AsyncSession,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session.add(
        CanonicalCompetition(
            name="Premier League", code="E0", mapping_status=MAPPED
        )
    )
    await session.flush()

    raw = "XYZ Unknown Cup Absolute Nonsense"
    assert similarity_ratio(raw, "Premier League") < SIMILARITY_THRESHOLD

    resolver = CompetitionResolver(session)
    with caplog.at_level(logging.WARNING, logger="src.data.competition_resolver"):
        cid = await resolver.resolve_competition(session, "football-data", raw)

    assert cid is None

    staged = (
        await session.execute(
            select(StagingUnmappedCompetition).where(
                StagingUnmappedCompetition.raw_name == raw
            )
        )
    ).scalar_one()
    assert staged.status == PENDING_REVIEW
    assert staged.source_type == "football-data"
    assert any("PENDING_REVIEW" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_resolve_hits_existing_alias(session: AsyncSession) -> None:
    comp = CanonicalCompetition(
        name="La Liga", code="SP1", mapping_status=MAPPED
    )
    session.add(comp)
    await session.flush()
    session.add(
        CompetitionAlias(
            canonical_competition_id=comp.id,
            source_type="football-data",
            source_competition_id="SP1",
            source_name="Spanish La Liga",
            mapping_status=MAPPED,
        )
    )
    await session.flush()

    resolver = CompetitionResolver(session)
    cid = await resolver.resolve_competition(
        session, "football-data", "Spanish La Liga"
    )
    assert cid == comp.id


@pytest.mark.asyncio
async def test_bulk_import_fixture_csv(session: AsyncSession) -> None:
    session.add(
        CanonicalCompetition(
            name="Premier League", code="E0", mapping_status=MAPPED
        )
    )
    await session.flush()

    importer = BulkFootballDataImporter(session, batch_size=10)
    stats = await importer.import_csv_season_data(str(FIXTURE_CSV), "E0")

    assert stats.rows_read == 3
    assert stats.rows_skipped == 1  # BadRowClub missing goals
    assert stats.raw_inserted == 2
    assert stats.matches_upserted == 2
    assert stats.competition_id is not None
    assert not stats.errors

    matches = (await session.execute(select(CanonicalMatch))).scalars().all()
    assert len(matches) == 2
    assert all(m.status == "FINISHED" for m in matches)
    assert all(m.ft_home_goals is not None for m in matches)

    raw_rows = (await session.execute(select(RawDataLake))).scalars().all()
    assert len(raw_rows) == 2
    for raw in raw_rows:
        assert raw.source == "football-data"
        assert raw.observed_at == raw.source_timestamp
        assert raw.canonical_match_id is not None
        match = next(m for m in matches if m.canonical_match_id == raw.canonical_match_id)
        assert raw.observed_at == match.kickoff_utc


@pytest.mark.asyncio
async def test_bulk_import_idempotent_upsert(session: AsyncSession) -> None:
    session.add(
        CanonicalCompetition(
            name="Premier League", code="E0", mapping_status=MAPPED
        )
    )
    await session.flush()

    importer = BulkFootballDataImporter(session)
    first = await importer.import_csv_season_data(str(FIXTURE_CSV), "E0")
    second = await importer.import_csv_season_data(str(FIXTURE_CSV), "E0")

    assert first.matches_upserted == 2
    assert second.matches_updated == 2
    assert second.matches_upserted == 0

    matches = (await session.execute(select(CanonicalMatch))).scalars().all()
    assert len(matches) == 2
