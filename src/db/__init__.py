"""Quant Engine v2 persistence (additive; does not replace global_matches.db)."""

from __future__ import annotations

from src.db.schema_v2 import (
    Base,
    BetSnapshot,
    CanonicalCompetition,
    CanonicalMatch,
    CanonicalTeam,
    CompetitionAlias,
    PitFeatureSnapshot,
    PredictionSnapshot,
    RawDataLake,
    TeamAlias,
    init_schema_v2,
    init_schema_v2_async,
)
from src.db.session import (
    async_session_factory,
    create_async_engine_from_url,
    get_database_url,
    get_session_factory,
)

__all__ = [
    "Base",
    "BetSnapshot",
    "CanonicalCompetition",
    "CanonicalMatch",
    "CanonicalTeam",
    "CompetitionAlias",
    "PitFeatureSnapshot",
    "PredictionSnapshot",
    "RawDataLake",
    "TeamAlias",
    "async_session_factory",
    "create_async_engine_from_url",
    "get_database_url",
    "get_session_factory",
    "init_schema_v2",
    "init_schema_v2_async",
]
