"""Quant Engine Phase P0/P2 SQLAlchemy 2.x schema (PostgreSQL + SQLite).

UUID primary keys and timezone-aware timestamps throughout.
Does not touch ``data/global_matches.db`` — this is an additive v2 store.

JSON columns use SQLAlchemy ``JSON`` (SQLite-compatible). On PostgreSQL the
same columns map to ``JSONB`` via ``TypeEngine.with_variant``.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, synonym
from sqlalchemy.types import JSON, TypeEngine, Uuid

# SQLite / generic: JSON. PostgreSQL: JSONB.
JSONType: TypeEngine[Any] = JSON().with_variant(JSONB(), "postgresql")


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


class Base(DeclarativeBase):
    """Declarative base for Quant Engine v2 tables."""


# ---------------------------------------------------------------------------
# Canonical entity + alias maps
# ---------------------------------------------------------------------------


class CanonicalTeam(Base):
    """Resolved club entity used across sources."""

    __tablename__ = "canonical_teams"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=_uuid
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    gender: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    country: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    mapping_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="MAPPED"
    )  # MAPPED | UNMAPPED
    meta_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    aliases: Mapped[list["TeamAlias"]] = relationship(back_populates="canonical_team")
    home_matches: Mapped[list["CanonicalMatch"]] = relationship(
        back_populates="home_team",
        foreign_keys="CanonicalMatch.home_team_id",
    )
    away_matches: Mapped[list["CanonicalMatch"]] = relationship(
        back_populates="away_team",
        foreign_keys="CanonicalMatch.away_team_id",
    )


class CanonicalCompetition(Base):
    """Resolved competition / league entity."""

    __tablename__ = "canonical_competitions"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=_uuid
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    code: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, index=True)
    country: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    mapping_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="MAPPED"
    )  # MAPPED | UNMAPPED
    meta_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    aliases: Mapped[list["CompetitionAlias"]] = relationship(
        back_populates="canonical_competition"
    )
    matches: Mapped[list["CanonicalMatch"]] = relationship(
        back_populates="competition"
    )


class TeamAlias(Base):
    """Source-specific team name/id → canonical_teams.id."""

    __tablename__ = "team_aliases"
    __table_args__ = (
        UniqueConstraint(
            "source_type",
            "source_team_id",
            "source_name",
            name="uq_team_alias_source",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=_uuid
    )
    canonical_team_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("canonical_teams.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_team_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    source_name: Mapped[str] = mapped_column(String(255), nullable=False)
    mapping_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="MAPPED"
    )  # MAPPED | UNMAPPED
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    canonical_team: Mapped["CanonicalTeam"] = relationship(back_populates="aliases")


class CompetitionAlias(Base):
    """Source-specific competition name/id → canonical_competitions.id."""

    __tablename__ = "competition_aliases"
    __table_args__ = (
        UniqueConstraint(
            "source_type",
            "source_competition_id",
            "source_name",
            name="uq_competition_alias_source",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=_uuid
    )
    canonical_competition_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("canonical_competitions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_competition_id: Mapped[Optional[str]] = mapped_column(
        String(128), nullable=True
    )
    source_name: Mapped[str] = mapped_column(String(255), nullable=False)
    mapping_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="MAPPED"
    )  # MAPPED | UNMAPPED
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    canonical_competition: Mapped["CanonicalCompetition"] = relationship(
        back_populates="aliases"
    )


class StagingUnmappedCompetition(Base):
    """Raw competition names that failed fuzzy auto-mapping (human review)."""

    __tablename__ = "staging_unmapped_competitions"
    __table_args__ = (
        UniqueConstraint(
            "source_type",
            "raw_name",
            name="uq_staging_unmapped_competition_source",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=_uuid
    )
    source_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    raw_name: Mapped[str] = mapped_column(String(255), nullable=False)
    best_match_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    best_match_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="PENDING_REVIEW", index=True
    )  # PENDING_REVIEW | RESOLVED | REJECTED
    meta_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


# ---------------------------------------------------------------------------
# Core snapshots
# ---------------------------------------------------------------------------


class RawDataLake(Base):
    """Immutable raw observation store (JSON payloads + provenance).

    ``payload`` is JSON on SQLite and JSONB on PostgreSQL.
    """

    __tablename__ = "raw_data_lake"

    raw_id: Mapped[uuid.UUID] = mapped_column(
        "raw_id", Uuid(as_uuid=True), primary_key=True, default=_uuid
    )
    source: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(
        String(64), nullable=False, default="observation", index=True
    )
    source_entity_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    source_timestamp: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Optional link for PIT / lineage joins (not required at ingest time).
    canonical_match_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("canonical_matches.canonical_match_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Compat aliases for earlier draft callers (constructor + attribute access).
    id = synonym("raw_id")
    source_type = synonym("source")
    source_id = synonym("source_entity_id")
    payload_json = synonym("payload")


class CanonicalMatch(Base):
    """Canonical fixture linking teams + competition."""

    __tablename__ = "canonical_matches"

    canonical_match_id: Mapped[uuid.UUID] = mapped_column(
        "canonical_match_id", Uuid(as_uuid=True), primary_key=True, default=_uuid
    )
    competition_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("canonical_competitions.id"),
        nullable=False,
        index=True,
    )
    season_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    home_team_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("canonical_teams.id"),
        nullable=False,
        index=True,
    )
    away_team_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("canonical_teams.id"),
        nullable=False,
        index=True,
    )
    kickoff_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="SCHEDULED")
    ft_home_goals: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ft_away_goals: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ht_home_goals: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ht_away_goals: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    home_team: Mapped["CanonicalTeam"] = relationship(
        back_populates="home_matches", foreign_keys=[home_team_id]
    )
    away_team: Mapped["CanonicalTeam"] = relationship(
        back_populates="away_matches", foreign_keys=[away_team_id]
    )
    competition: Mapped["CanonicalCompetition"] = relationship(back_populates="matches")
    pit_snapshots: Mapped[list["PitFeatureSnapshot"]] = relationship(
        back_populates="canonical_match"
    )

    # Compat aliases for earlier draft callers.
    id = synonym("canonical_match_id")
    kickoff_at = synonym("kickoff_utc")
    season = synonym("season_id")


class PitFeatureSnapshot(Base):
    """Point-in-time feature vector with lineage + integrity flag."""

    __tablename__ = "pit_feature_snapshots"

    feature_snapshot_id: Mapped[uuid.UUID] = mapped_column(
        "feature_snapshot_id", Uuid(as_uuid=True), primary_key=True, default=_uuid
    )
    canonical_match_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("canonical_matches.canonical_match_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    as_of_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    feature_schema_version: Mapped[str] = mapped_column(
        String(32), nullable=False, default="v1"
    )
    features_json: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    data_quality_metrics: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSONType, nullable=True
    )
    aggregate_data_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pit_integrity_passed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True
    )
    lineage_mapping: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSONType, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    canonical_match: Mapped["CanonicalMatch"] = relationship(
        back_populates="pit_snapshots"
    )
    predictions: Mapped[list["PredictionSnapshot"]] = relationship(
        back_populates="feature_snapshot"
    )

    # Compat aliases for earlier draft callers.
    id = synonym("feature_snapshot_id")
    feature_json = synonym("features_json")
    data_quality = synonym("data_quality_metrics")


class PredictionSnapshot(Base):
    """Model output at a point in time (raw + calibrated + fair lines)."""

    __tablename__ = "prediction_snapshots"

    prediction_id: Mapped[uuid.UUID] = mapped_column(
        "prediction_id", Uuid(as_uuid=True), primary_key=True, default=_uuid
    )
    feature_snapshot_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("pit_feature_snapshots.feature_snapshot_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    model_tier: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    calibrator_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    market_snapshot_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True), nullable=True, index=True
    )
    raw_probabilities: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    calibrated_probabilities: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSONType, nullable=True
    )
    fair_lines: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONType, nullable=True)
    model_confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    feature_snapshot: Mapped["PitFeatureSnapshot"] = relationship(
        back_populates="predictions"
    )
    bets: Mapped[list["BetSnapshot"]] = relationship(back_populates="prediction")

    id = synonym("prediction_id")


class BetSnapshot(Base):
    """Recommended / paper bet with Kelly sizing, CLV, and settlement."""

    __tablename__ = "bet_snapshots"

    bet_id: Mapped[uuid.UUID] = mapped_column(
        "bet_id", Uuid(as_uuid=True), primary_key=True, default=_uuid
    )
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("prediction_snapshots.prediction_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    market_type: Mapped[str] = mapped_column(String(64), nullable=False)
    selection: Mapped[str] = mapped_column(String(64), nullable=False)
    taken_odds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sharp_closing_odds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    expected_value_percent: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    raw_kelly_fraction: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    data_weight: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    model_weight: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    correlated_discount: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    final_stake_fraction: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    final_stake_amount: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="PENDING", index=True
    )  # PENDING | OPEN | SETTLED | VOID | NO_BET | HARD_GATE
    pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    clv_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    settled_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    prediction: Mapped["PredictionSnapshot"] = relationship(back_populates="bets")

    id = synonym("bet_id")


def init_schema_v2(engine: Engine | AsyncEngine) -> None:
    """Create all v2 tables via ``Base.metadata.create_all``.

    For a sync ``Engine``, tables are created immediately.
    For an ``AsyncEngine``, use :func:`init_schema_v2_async` instead —
    this helper raises if given an async engine so callers do not block
    incorrectly.
    """
    if isinstance(engine, AsyncEngine):
        raise TypeError(
            "AsyncEngine passed to init_schema_v2; use init_schema_v2_async(engine)"
        )
    Base.metadata.create_all(engine)


async def init_schema_v2_async(engine: AsyncEngine) -> None:
    """Async variant of :func:`init_schema_v2` for aiosqlite / asyncpg."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
