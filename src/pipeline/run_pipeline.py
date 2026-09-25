"""Quant Engine async prediction pipeline (Worker 5).

Orchestrates PIT → quality/lineage → lightweight features → EnsembleEngineV2
and optionally persists ``pit_feature_snapshots`` / ``prediction_snapshots``.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.data.lineage_builder import FeatureLineageTracker
from src.data.pit_engine import PITEngine
from src.data.quality_monitor import DataQualityEngine, HardGate
from src.models.ensemble_v2 import (
    CALIBRATOR_VERSION,
    MODEL_VERSION,
    EnsembleEngineV2,
    EnsembleResult,
)
from src.models.vig_removal import VigRemovalEngine

logger = logging.getLogger(__name__)

FEATURE_SCHEMA_VERSION = "v2_stub"


def _ensure_aware(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts


def _market_probs_from_odds(
    bookmaker_odds: Optional[Mapping[str, Any]],
    vig: VigRemovalEngine,
) -> dict[str, float]:
    """Convert H/D/A decimal odds → vig-removed fair probs (multiplicative)."""
    if not bookmaker_odds:
        return {"H": 1.0 / 3.0, "D": 1.0 / 3.0, "A": 1.0 / 3.0}
    ordered_keys = ("H", "D", "A")
    odds_list: list[float] = []
    for k in ordered_keys:
        try:
            o = float(bookmaker_odds[k])
        except (KeyError, TypeError, ValueError):
            return {"H": 1.0 / 3.0, "D": 1.0 / 3.0, "A": 1.0 / 3.0}
        if o <= 1.0:
            return {"H": 1.0 / 3.0, "D": 1.0 / 3.0, "A": 1.0 / 3.0}
        odds_list.append(o)
    fair = vig.multiplicative_margin(odds_list)
    return {k: fair[i] for i, k in enumerate(ordered_keys)}


def _stub_features(
    valid_records: Sequence[Mapping[str, Any]],
    *,
    as_of_time: datetime,
    quality: Mapping[str, Any],
) -> dict[str, Any]:
    """Lightweight feature vector (full feature calc is out of Worker 5 scope)."""
    n = len(valid_records)
    source_types = sorted(
        {str(r.get("source_type")) for r in valid_records if r.get("source_type")}
    )
    return {
        "n_pit_records": float(n),
        "n_sources": float(len(source_types)),
        "aggregate_data_score": float(quality.get("aggregate_score", 0.0)),
        "pit_integrity_component": float(quality.get("pit_integrity", 0.0)),
        "freshness_score": float(quality.get("freshness", 0.0)),
        "as_of_unix": float(as_of_time.timestamp()),
        "source_types": source_types,
    }


def _lineage_sources(
    features: Mapping[str, Any],
    valid_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Map each feature key to contributing raw record ids."""
    raw_ids = [
        {
            "raw_id": str(r.get("id", r.get("raw_id", i))),
            "observed_at": r.get("observed_at"),
        }
        for i, r in enumerate(valid_records)
    ]
    return {key: list(raw_ids) for key in features.keys()}


async def run_quant_pipeline(
    canonical_match_id: UUID,
    as_of_time: datetime,
    *,
    raw_records: Optional[Sequence[Mapping[str, Any]]] = None,
    dixon_coles_probs: Optional[Mapping[str, Any]] = None,
    lightgbm_probs: Optional[Mapping[str, Any]] = None,
    bookmaker_odds: Optional[Mapping[str, Any]] = None,
    market_probs: Optional[Mapping[str, Any]] = None,
    match_data: Optional[Mapping[str, Any]] = None,
    session: Optional[AsyncSession] = None,
    persist: bool = True,
    pit_engine: Optional[PITEngine] = None,
    quality_engine: Optional[DataQualityEngine] = None,
    lineage_tracker: Optional[FeatureLineageTracker] = None,
    ensemble: Optional[EnsembleEngineV2] = None,
    vig_engine: Optional[VigRemovalEngine] = None,
) -> dict[str, Any]:
    """Run PIT → quality → features → ensemble for one canonical match.

    Parameters
    ----------
    canonical_match_id:
        Match UUID in the v2 schema.
    as_of_time:
        Inclusive PIT cutoff.
    raw_records:
        Observation dicts with ``observed_at``. Required unless provided via
        ``match_data['raw_records']``.
    dixon_coles_probs, lightgbm_probs:
        Model probability dicts (``H``/``D``/``A``).
    bookmaker_odds:
        Decimal odds used for vig removal (if ``market_probs`` omitted) and EV%.
    market_probs:
        Optional pre-computed fair market probabilities.
    match_data:
        Extra quality-engine inputs (``expected_sources``, etc.).
    session:
        Optional async SQLAlchemy session. When set and ``persist`` is True,
        writes ``PitFeatureSnapshot`` + ``PredictionSnapshot``.
    persist:
        When False, skip DB writes even if ``session`` is provided.

    Returns
    -------
    dict
        Keys include ``prediction_id``, ``expected_value_percent``,
        ``aggregate_data_score``, ``model_tier``, plus snapshot payloads
        ready to persist when DB is unavailable.
    """
    as_of = _ensure_aware(as_of_time)
    pit = pit_engine or PITEngine(hard_gate=HardGate())
    qe = quality_engine or DataQualityEngine(hard_gate=HardGate())
    lineage = lineage_tracker or FeatureLineageTracker()
    ens = ensemble or EnsembleEngineV2()
    vig = vig_engine or VigRemovalEngine()

    records_in: list[dict[str, Any]] = [
        dict(r) for r in (raw_records or (match_data or {}).get("raw_records") or [])
    ]

    # 1) PIT filter
    valid_records, pit_integrity_passed = pit.get_valid_raw_records(records_in, as_of)

    # 2) Data quality + lineage inputs
    quality_match: dict[str, Any] = dict(match_data or {})
    quality_match["raw_records"] = list(valid_records)
    # Avoid double-counting future leaks already stripped by PIT.
    quality_match.setdefault("as_of_already_filtered", True)
    quality = qe.calculate_quality_metrics(quality_match, as_of)

    # If PIT leaked, force aggregate/score semantics for ensemble routing.
    # DataQualityEngine on filtered records may still score high; Worker-5
    # gate uses the PIT integrity flag from step 1.
    aggregate_score = float(quality.get("aggregate_score", 0.0))
    if not pit_integrity_passed:
        aggregate_score = 0.0
        quality = {
            **quality,
            "aggregate_score": 0.0,
            "pit_integrity": 0.0,
            "model_tier": "MODEL_TIER_X",
        }

    features = _stub_features(valid_records, as_of_time=as_of, quality=quality)
    lineage_map = lineage.build_lineage_map(
        {k: v for k, v in features.items() if k != "source_types"},
        _lineage_sources(
            {k: v for k, v in features.items() if k != "source_types"},
            valid_records,
        ),
    )

    # 3) Market fair probs → ensemble
    mkt = (
        dict(market_probs)
        if market_probs is not None
        else _market_probs_from_odds(bookmaker_odds, vig)
    )
    result: EnsembleResult = ens.predict(
        dixon_coles=dixon_coles_probs,
        lightgbm=lightgbm_probs,
        market=mkt,
        pit_integrity_passed=pit_integrity_passed,
        aggregate_data_score=aggregate_score,
        bookmaker_odds=bookmaker_odds,
        quality_flags={
            "quality_model_tier": quality.get("model_tier"),
            "n_records": quality.get("n_records"),
            "n_leaks": quality.get("n_leaks"),
        },
    )

    feature_snapshot_id = uuid.uuid4()
    prediction_id = uuid.uuid4()

    pit_payload = {
        "feature_snapshot_id": feature_snapshot_id,
        "canonical_match_id": canonical_match_id,
        "as_of_time": as_of,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "features_json": features,
        "data_quality_metrics": quality,
        "aggregate_data_score": aggregate_score,
        "pit_integrity_passed": pit_integrity_passed,
        "lineage_mapping": lineage_map,
    }
    pred_payload = {
        "prediction_id": prediction_id,
        "feature_snapshot_id": feature_snapshot_id,
        "model_tier": result.model_tier,
        "model_version": result.model_version or MODEL_VERSION,
        "calibrator_version": result.calibrator_version or CALIBRATOR_VERSION,
        "raw_probabilities": result.raw_probabilities,
        "calibrated_probabilities": result.calibrated_probabilities,
        "fair_lines": result.fair_lines,
        "model_confidence_score": result.model_confidence_score,
    }

    # 4) Persist when session available
    persisted = False
    if session is not None and persist:
        try:
            from src.db.schema_v2 import PitFeatureSnapshot, PredictionSnapshot

            pit_row = PitFeatureSnapshot(
                feature_snapshot_id=feature_snapshot_id,
                canonical_match_id=canonical_match_id,
                as_of_time=as_of,
                feature_schema_version=FEATURE_SCHEMA_VERSION,
                features_json=features,
                data_quality_metrics=quality,
                aggregate_data_score=aggregate_score,
                pit_integrity_passed=pit_integrity_passed,
                lineage_mapping=lineage_map,
            )
            pred_row = PredictionSnapshot(
                prediction_id=prediction_id,
                feature_snapshot_id=feature_snapshot_id,
                model_tier=result.model_tier,
                model_version=result.model_version or MODEL_VERSION,
                calibrator_version=result.calibrator_version or CALIBRATOR_VERSION,
                raw_probabilities=result.raw_probabilities,
                calibrated_probabilities=result.calibrated_probabilities,
                fair_lines=result.fair_lines,
                model_confidence_score=result.model_confidence_score,
            )
            session.add(pit_row)
            session.add(pred_row)
            await session.flush()
            persisted = True
        except Exception:
            logger.exception(
                "Failed to persist snapshots for match=%s; returning payloads only",
                canonical_match_id,
            )

    # 5) Return summary
    return {
        "prediction_id": prediction_id,
        "feature_snapshot_id": feature_snapshot_id,
        "canonical_match_id": canonical_match_id,
        "expected_value_percent": result.expected_value_percent,
        "aggregate_data_score": aggregate_score,
        "model_tier": result.model_tier,
        "no_bet": result.no_bet,
        "pit_integrity_passed": pit_integrity_passed,
        "fair_probabilities": result.fair_probabilities,
        "fair_lines": result.fair_lines,
        "persisted": persisted,
        "pit_feature_snapshot": pit_payload,
        "prediction_snapshot": pred_payload,
        "ensemble": result,
    }
