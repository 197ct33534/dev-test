"""Pydantic request / response models for Quant Engine API v2."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ValueBetV2Out(BaseModel):
    """Single pipeline-priced value opportunity (Schema/Pipeline V2)."""

    match_id: str = Field(..., description="canonical_match_id (UUID string)")
    prediction_id: str | None = None
    home: str | None = None
    away: str | None = None
    league: str | None = None
    kickoff: str | None = None
    fair_lines: dict[str, float] = Field(default_factory=dict)
    fair_probabilities: dict[str, float] = Field(default_factory=dict)
    bookmaker_odds: dict[str, float] | None = None
    data_score: float = Field(
        0.0, description="aggregate_data_score from quality / PIT gate"
    )
    model_tier: str = Field(..., description="e.g. MODEL_TIER_A / B / C / X")
    model_confidence: float | None = Field(
        default=None,
        description="Ensemble model_confidence_score (0–1)",
    )
    expected_value_percent: float = Field(
        0.0, description="Best single-outcome EV% from ensemble"
    )
    no_bet: bool = False


class ValueBetsV2Response(BaseModel):
    """Filtered upcoming value bets from ``run_quant_pipeline``."""

    bets: list[ValueBetV2Out]
    count: int
    min_ev_pct: float
    limit: int
    league: str | None = None
    scanned: int = Field(
        0, description="Number of upcoming matches priced (before filters)"
    )
    notes: str | None = None


class BetCandidateIn(BaseModel):
    """Optional explicit candidate for paper placement (skips 1X2 discovery)."""

    market_type: str = "1X2"
    selection: str
    probability: float = Field(..., gt=0.0, le=1.0)
    odds: float = Field(..., gt=1.0)


class PlaceBetRequest(BaseModel):
    """WebApp payload → ``PaperTrader.execute_value_bets``."""

    canonical_match_id: UUID
    bookmaker_odds: dict[str, float] = Field(
        ...,
        description="Decimal 1X2 (or market) prices, keys H/D/A preferred",
        min_length=1,
    )
    as_of_time: datetime | None = Field(
        default=None,
        description="PIT cutoff; defaults to UTC now when omitted",
    )
    league: str | None = None
    bankroll: float | None = Field(default=None, gt=0)
    dixon_coles_probs: dict[str, float] | None = None
    lightgbm_probs: dict[str, float] | None = None
    candidates: list[BetCandidateIn] | None = None
    raw_records: list[dict[str, Any]] | None = None
    persist: bool = Field(
        default=True,
        description="Persist bet_snapshots when a DB session is available",
    )


class PlaceBetResponse(BaseModel):
    """Paper-trader placement result (bet_snapshot ids)."""

    ok: bool
    bet_ids: list[str] = Field(default_factory=list)
    count: int = 0
    canonical_match_id: str | None = None
    message: str | None = None


class AuditPredictionOut(BaseModel):
    """Lineage / audit trail for a prediction_id."""

    prediction_id: str
    found: bool
    error: str | None = None
    prediction: dict[str, Any] | None = None
    feature_snapshot: dict[str, Any] | None = None
    lineage_mapping: dict[str, Any] = Field(default_factory=dict)
    raw_ids: list[str] = Field(default_factory=list)
    raw_observations: list[dict[str, Any]] = Field(default_factory=list)
    explanation: str | None = None
