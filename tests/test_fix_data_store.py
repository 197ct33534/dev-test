"""Regression tests: legacy SQLite upsert, v2 store dedup, resolver, session, pipeline rollback."""

from __future__ import annotations

import sqlite3
import threading
import uuid
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import AsyncIterator

import pandas as pd
import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.data.bulk_importer import BulkFootballDataImporter, _parse_kickoff
from src.data.entity_resolver import EntityResolver, normalize_team_key
from src.data_loader import read_matches_from_db, save_matches_to_db
from src.db.schema_v2 import (
    CanonicalCompetition,
    CanonicalMatch,
    CanonicalTeam,
    RawDataLake,
    TeamAlias,
    init_schema_v2_async,
)

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "football_data_tiny.csv"


# ---------------------------------------------------------------------------
# Bug 2: save_matches_to_db upsert
# ---------------------------------------------------------------------------


def _rows(*specs: tuple[str, str, str, int, int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Date": pd.Timestamp(d),
                "HomeTeam": h,
                "AwayTeam": a,
                "FTHG": hg,
                "FTAG": ag,
                "FTR": "H" if hg > ag else ("A" if hg < ag else "D"),
                "B365H": 2.0,
            }
            for d, h, a, hg, ag in specs
        ]
    )


def test_save_matches_upserts_and_adds_columns(tmp_path: Path) -> None:
    db = tmp_path / "league.db"
    assert save_matches_to_db(_rows(("2024-08-10", "Arsenal", "Chelsea", 1, 0),
                                    ("2024-08-11", "Leeds", "Spurs", 2, 2)), db) == 2
    upd = _rows(("2024-08-10", "Arsenal", "Chelsea", 3, 0),
                ("2024-08-12", "Fulham", "Wolves", 0, 1))
    upd["HC"] = [7, 4]
    assert save_matches_to_db(upd, db) == 3

    with closing(sqlite3.connect(db)) as conn:
        cols = {r[1]: r[2] for r in conn.execute("PRAGMA table_info(matches)")}
        assert "HC" in cols
        assert cols["FTHG"] == "INTEGER" and cols["B365H"] == "REAL"
        row = conn.execute(
            "SELECT FTHG, typeof(FTHG), HC FROM matches WHERE Match_ID = ?",
            ("2024-08-10|Arsenal|Chelsea",),
        ).fetchone()
        assert row == (3, "integer", 7)
        idx = [r[1] for r in conn.execute("PRAGMA index_list(matches)")]
        assert "idx_matches_pk" in idx
    df = read_matches_from_db(db)
    assert len(df) == 3


def test_save_matches_dedups_legacy_table_without_index(tmp_path: Path) -> None:
    db = tmp_path / "legacy.db"
    with closing(sqlite3.connect(db)) as conn:
        conn.execute(
            'CREATE TABLE matches ("Date" TEXT, "HomeTeam" TEXT, "AwayTeam" TEXT, '
            '"FTHG" INTEGER, "FTAG" INTEGER, "FTR" TEXT, "Match_ID" TEXT)'
        )
        for g in (0, 1):
            conn.execute(
                "INSERT INTO matches VALUES ('2024-08-10','A','B',?,0,'H','2024-08-10|A|B')",
                (g,),
            )
        conn.commit()
    n = save_matches_to_db(_rows(("2024-08-11", "C", "D", 1, 1)), db)
    assert n == 2


def test_save_matches_concurrent_writers_keep_all_rows(tmp_path: Path) -> None:
    db = tmp_path / "concurrent.db"
    save_matches_to_db(_rows(("2024-01-01", "Seed", "Row", 0, 0)), db)
    errors: list[BaseException] = []

    def _writer(k: int) -> None:
        try:
            for i in range(5):
                save_matches_to_db(
                    _rows((f"2024-02-{i + 1:02d}", f"H{k}", f"A{k}", i, 0)), db
                )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_writer, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    with closing(sqlite3.connect(db)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 1 + 4 * 5


# ---------------------------------------------------------------------------
# Bug 6: v2 store
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    await init_schema_v2_async(engine)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as sess:
        yield sess
    await engine.dispose()


def test_parse_kickoff_uk_local_to_utc() -> None:
    # BST (summer): 15:00 London = 14:00 UTC; GMT (winter): unchanged.
    assert _parse_kickoff("12/08/23", "15:00") == datetime(2023, 8, 12, 14, 0, tzinfo=timezone.utc)
    assert _parse_kickoff("13/01/24", "15:00") == datetime(2024, 1, 13, 15, 0, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_reimport_does_not_duplicate_raw_rows(session: AsyncSession) -> None:
    importer = BulkFootballDataImporter(session)
    first = await importer.import_csv_season_data(str(FIXTURE_CSV), "E0")
    second = await importer.import_csv_season_data(str(FIXTURE_CSV), "E0")
    assert first.raw_inserted == 2
    assert second.raw_inserted == 0 and second.raw_duplicates == 2
    raws = (await session.execute(select(RawDataLake))).scalars().all()
    assert len(raws) == 2


@pytest.mark.asyncio
async def test_match_dedup_tolerates_kickoff_shift(session: AsyncSession) -> None:
    importer = BulkFootballDataImporter(session)
    await importer.import_csv_season_data(str(FIXTURE_CSV), "E0")
    # Simulate rows stored by the old importer (UK wall clock tagged as UTC).
    for m in (await session.execute(select(CanonicalMatch))).scalars().all():
        m.kickoff_utc = m.kickoff_utc + timedelta(hours=1)
    await session.flush()

    again = await importer.import_csv_season_data(str(FIXTURE_CSV), "E0")
    assert again.matches_upserted == 0 and again.matches_updated == 2
    matches = (await session.execute(select(CanonicalMatch))).scalars().all()
    assert len(matches) == 2
    kicks = sorted(m.kickoff_utc.replace(tzinfo=None) for m in matches)
    assert kicks[0] == datetime(2023, 8, 12, 14, 0)  # corrected to real UTC


def test_normalize_team_key() -> None:
    assert normalize_team_key("Atlético Madrid CF") == "atletico madrid"
    assert normalize_team_key("FC Barcelona") == normalize_team_key("barcelona")
    assert normalize_team_key("Arsenal W") != normalize_team_key("Arsenal")


@pytest.mark.asyncio
async def test_resolver_matches_normalized_names(session: AsyncSession) -> None:
    resolver = EntityResolver(session)
    tid = await resolver.get_or_create_canonical_team(session, "Atlético Madrid", "flashscore", None)
    same = await resolver.get_or_create_canonical_team(session, "Atletico Madrid CF", "football-data", None)
    assert same == tid
    other = await resolver.get_or_create_canonical_team(session, "Real Madrid", "football-data", None)
    assert other != tid
    teams = (await session.execute(select(CanonicalTeam))).scalars().all()
    assert len(teams) == 2


@pytest.mark.asyncio
async def test_team_alias_null_source_id_is_unique(session: AsyncSession) -> None:
    team = CanonicalTeam(name="X", mapping_status="MAPPED")
    session.add(team)
    await session.flush()
    session.add(TeamAlias(canonical_team_id=team.id, source_type="fd", source_team_id=None, source_name="X"))
    await session.flush()
    session.add(TeamAlias(canonical_team_id=team.id, source_type="fd", source_team_id=None, source_name="X"))
    with pytest.raises(IntegrityError):
        await session.flush()
    await session.rollback()


# ---------------------------------------------------------------------------
# Bug 7 / 8: pipeline rollback, session_scope
# ---------------------------------------------------------------------------


class _FailingSession:
    def __init__(self) -> None:
        self.rolled_back = False

    def add(self, _obj) -> None:
        pass

    async def flush(self) -> None:
        raise RuntimeError("flush failed")

    async def rollback(self) -> None:
        self.rolled_back = True


@pytest.mark.asyncio
async def test_pipeline_rolls_back_on_flush_failure() -> None:
    from src.pipeline.run_pipeline import run_quant_pipeline

    as_of = datetime(2025, 9, 19, 12, 0, tzinfo=timezone.utc)
    sess = _FailingSession()
    out = await run_quant_pipeline(
        uuid.uuid4(),
        as_of,
        raw_records=[{"id": "r1", "observed_at": as_of - timedelta(minutes=5), "source_type": "odds_api", "source_quality": 95.0}],
        dixon_coles_probs={"H": 0.5, "D": 0.25, "A": 0.25},
        lightgbm_probs={"H": 0.5, "D": 0.25, "A": 0.25},
        bookmaker_odds={"H": 2.1, "D": 3.4, "A": 3.6},
        session=sess,  # type: ignore[arg-type]
        persist=True,
    )
    assert out["persisted"] is False
    assert sess.rolled_back is True


@pytest.mark.asyncio
async def test_session_scope_is_async_context_manager(tmp_path: Path) -> None:
    from src.db.session import DEFAULT_SQLITE_PATH, async_session_factory, session_scope

    assert DEFAULT_SQLITE_PATH.is_absolute()
    factory = async_session_factory(url=f"sqlite+aiosqlite:///{(tmp_path / 'v2.db').as_posix()}")
    async with session_scope(factory) as sess:
        assert isinstance(sess, AsyncSession)
    await factory.kw["bind"].dispose()


# ---------------------------------------------------------------------------
# Bug 9: team-feed persist only loads relevant ids
# ---------------------------------------------------------------------------


def test_existing_ids_lookup_is_scoped() -> None:
    from src.fetchers.flashscore_team import _existing_event_match_ids

    with closing(sqlite3.connect(":memory:")) as conn:
        conn.execute("CREATE TABLE matches (match_id TEXT PRIMARY KEY)")
        conn.executemany("INSERT INTO matches VALUES (?)", [("A|1",), ("B|2",), ("C|3",)])
        assert _existing_event_match_ids(conn, match_ids=["B|2", "Z|9"]) == {"B|2"}
        assert _existing_event_match_ids(conn) == {"A|1", "B|2", "C|3"}
