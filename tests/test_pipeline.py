"""End-to-end tests for Worker 5 quant pipeline + EnsembleEngineV2 tiers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from src.data.quality_monitor import ModelTier
from src.models.ensemble_v2 import EnsembleEngineV2
from src.pipeline.run_pipeline import run_quant_pipeline

AS_OF = datetime(2025, 9, 19, 12, 0, tzinfo=timezone.utc)

# High-quality bookmaker prices (vig present) for EV checks.
BOOK_ODDS = {"H": 2.10, "D": 3.40, "A": 3.60}

DC_PROBS = {"H": 0.48, "D": 0.27, "A": 0.25}
LGBM_PROBS = {"H": 0.45, "D": 0.28, "A": 0.27}


def _clean_records(*, source_quality: float = 95.0) -> list[dict]:
    return [
        {
            "id": "r1",
            "observed_at": AS_OF - timedelta(minutes=20),
            "source_type": "odds_api",
            "source_quality": source_quality,
        },
        {
            "id": "r2",
            "observed_at": AS_OF - timedelta(minutes=10),
            "source_type": "stats",
            "source_quality": source_quality,
        },
        {
            "id": "r3",
            "observed_at": AS_OF - timedelta(minutes=5),
            "source_type": "news",
            "source_quality": source_quality,
        },
    ]


@pytest.mark.asyncio
async def test_pipeline_leakage_routes_to_tier_x() -> None:
    """Future-dated raw record → PIT fail → MODEL_TIER_X (NO BET)."""
    match_id = uuid4()
    records = _clean_records() + [
        {
            "id": "leak",
            "observed_at": AS_OF + timedelta(hours=1),
            "source_type": "oracle",
            "source_quality": 99.0,
        }
    ]

    out = await run_quant_pipeline(
        match_id,
        AS_OF,
        raw_records=records,
        dixon_coles_probs=DC_PROBS,
        lightgbm_probs=LGBM_PROBS,
        bookmaker_odds=BOOK_ODDS,
        match_data={
            "expected_sources": ["odds_api", "stats", "news"],
        },
        session=None,
        persist=False,
    )

    assert out["pit_integrity_passed"] is False
    assert out["model_tier"] == ModelTier.X.value
    assert out["no_bet"] is True
    assert out["aggregate_data_score"] == 0.0
    assert out["expected_value_percent"] == 0.0
    assert out["prediction_id"] is not None
    assert out["persisted"] is False


@pytest.mark.asyncio
async def test_pipeline_high_quality_routes_to_tier_a() -> None:
    """Fresh multi-source data → aggregate ≥ 85 → MODEL_TIER_A."""
    match_id = uuid4()
    out = await run_quant_pipeline(
        match_id,
        AS_OF,
        raw_records=_clean_records(source_quality=98.0),
        dixon_coles_probs=DC_PROBS,
        lightgbm_probs=LGBM_PROBS,
        bookmaker_odds=BOOK_ODDS,
        match_data={
            "expected_sources": ["odds_api", "stats", "news"],
            "consistency_conflicts": 0,
        },
        session=None,
        persist=False,
    )

    assert out["pit_integrity_passed"] is True
    assert out["aggregate_data_score"] >= 85.0
    assert out["model_tier"] == ModelTier.A.value
    assert out["no_bet"] is False
    assert set(out["fair_probabilities"]) == {"H", "D", "A"}
    assert abs(sum(out["fair_probabilities"].values()) - 1.0) < 1e-6
    # Full ensemble includes LGBM weight.
    assert out["ensemble"].mix_weights["lgbm"] > 0.0
    assert out["prediction_id"] is not None


@pytest.mark.asyncio
async def test_pipeline_mid_quality_routes_to_tier_b() -> None:
    """Mid aggregate (50–85) → MODEL_TIER_B reduced model (no LGBM)."""
    match_id = uuid4()
    # Sparse / older / lower source_quality to land in the B band without
    # going below 50 (and without PIT leakage).
    records = [
        {
            "id": "only",
            "observed_at": AS_OF - timedelta(hours=10),
            "source_type": "odds_api",
            "source_quality": 40.0,
        }
    ]
    out = await run_quant_pipeline(
        match_id,
        AS_OF,
        raw_records=records,
        dixon_coles_probs=DC_PROBS,
        lightgbm_probs=LGBM_PROBS,
        bookmaker_odds=BOOK_ODDS,
        match_data={
            # Missing expected sources → lower coverage.
            "expected_sources": ["odds_api", "stats", "news", "injury"],
            "consistency_conflicts": 1,
            "required_fields": ["home_xg", "away_xg"],
            "field_values": {"home_xg": None, "away_xg": 1.1},
        },
        session=None,
        persist=False,
    )

    score = out["aggregate_data_score"]
    assert out["pit_integrity_passed"] is True
    assert 50.0 <= score < 85.0, f"expected mid band, got {score}"
    assert out["model_tier"] == ModelTier.B.value
    assert out["no_bet"] is False
    assert out["ensemble"].mix_weights["lgbm"] == 0.0
    assert out["ensemble"].mix_weights["dc"] > 0.0


def test_ensemble_tier_thresholds_unit() -> None:
    """Direct unit checks for Worker-5 score cutoffs."""
    eng = EnsembleEngineV2()
    assert (
        eng.resolve_tier(pit_integrity_passed=False, aggregate_data_score=99.0)
        == ModelTier.X.value
    )
    assert (
        eng.resolve_tier(pit_integrity_passed=True, aggregate_data_score=49.9)
        == ModelTier.X.value
    )
    assert (
        eng.resolve_tier(pit_integrity_passed=True, aggregate_data_score=50.0)
        == ModelTier.B.value
    )
    assert (
        eng.resolve_tier(pit_integrity_passed=True, aggregate_data_score=84.9)
        == ModelTier.B.value
    )
    assert (
        eng.resolve_tier(pit_integrity_passed=True, aggregate_data_score=85.0)
        == ModelTier.A.value
    )


@pytest.mark.asyncio
async def test_pipeline_persist_payloads_ready_without_session() -> None:
    out = await run_quant_pipeline(
        uuid4(),
        AS_OF,
        raw_records=_clean_records(),
        dixon_coles_probs=DC_PROBS,
        bookmaker_odds=BOOK_ODDS,
        session=None,
        persist=False,
    )
    pit = out["pit_feature_snapshot"]
    pred = out["prediction_snapshot"]
    assert pit["canonical_match_id"] == out["canonical_match_id"]
    assert pred["prediction_id"] == out["prediction_id"]
    assert pred["feature_snapshot_id"] == out["feature_snapshot_id"]
    assert "features_json" in pit
    assert "fair_lines" in pred
