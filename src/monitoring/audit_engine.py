"""Prediction audit trail: reverse-lookup from prediction → raw observations."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Optional, Sequence, Union
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.db.schema_v2 import PitFeatureSnapshot, PredictionSnapshot, RawDataLake

JsonDict = dict[str, Any]
FixtureStore = Mapping[str, Any]


def _as_uuid(value: Union[str, UUID, None]) -> Optional[UUID]:
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    return UUID(str(value))


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _collect_raw_ids(lineage_mapping: Optional[Mapping[str, Any]]) -> list[str]:
    """Flatten unique raw_ids from a feature→sources lineage map (stable order)."""
    if not lineage_mapping:
        return []
    seen: set[str] = set()
    ordered: list[str] = []
    for _feature_key, entry in sorted(lineage_mapping.items(), key=lambda kv: str(kv[0])):
        raw_ids: Sequence[Any] = []
        if isinstance(entry, Mapping):
            raw_ids = entry.get("raw_ids") or entry.get("raw_id") or []
            if isinstance(raw_ids, (str, UUID)):
                raw_ids = [raw_ids]
        elif isinstance(entry, (list, tuple)):
            raw_ids = entry
        elif entry is not None:
            raw_ids = [entry]
        for rid in raw_ids:
            key = str(rid)
            if key not in seen:
                seen.add(key)
                ordered.append(key)
    return ordered


def _prediction_to_dict(pred: PredictionSnapshot) -> JsonDict:
    return {
        "prediction_id": str(pred.prediction_id),
        "feature_snapshot_id": str(pred.feature_snapshot_id),
        "model_tier": pred.model_tier,
        "model_version": pred.model_version,
        "calibrator_version": pred.calibrator_version,
        "raw_probabilities": pred.raw_probabilities,
        "calibrated_probabilities": pred.calibrated_probabilities,
        "fair_lines": pred.fair_lines,
        "model_confidence_score": pred.model_confidence_score,
        "created_at": _iso(pred.created_at),
    }


def _pit_to_dict(pit: PitFeatureSnapshot) -> JsonDict:
    return {
        "feature_snapshot_id": str(pit.feature_snapshot_id),
        "canonical_match_id": str(pit.canonical_match_id),
        "as_of_time": _iso(pit.as_of_time),
        "feature_schema_version": pit.feature_schema_version,
        "features": pit.features_json,
        "data_quality_metrics": pit.data_quality_metrics,
        "aggregate_data_score": pit.aggregate_data_score,
        "pit_integrity_passed": pit.pit_integrity_passed,
        "lineage_mapping": pit.lineage_mapping,
        "created_at": _iso(pit.created_at),
    }


def _raw_to_dict(raw: RawDataLake) -> JsonDict:
    return {
        "raw_id": str(raw.raw_id),
        "source": raw.source,
        "entity_type": raw.entity_type,
        "source_entity_id": raw.source_entity_id,
        "observed_at": _iso(raw.observed_at),
        "source_timestamp": _iso(raw.source_timestamp),
        "canonical_match_id": str(raw.canonical_match_id) if raw.canonical_match_id else None,
        "payload": raw.payload,
    }


def _build_explanation(
    *,
    prediction: JsonDict,
    feature_snapshot: JsonDict,
    raw_observations: list[JsonDict],
) -> str:
    """Human-readable summary of why the model produced this result."""
    tier = prediction.get("model_tier", "unknown")
    score = feature_snapshot.get("aggregate_data_score")
    pit_ok = feature_snapshot.get("pit_integrity_passed")
    n_feats = len(feature_snapshot.get("features") or {})
    n_raw = len(raw_observations)
    sources = sorted({str(r.get("source")) for r in raw_observations if r.get("source")})
    conf = prediction.get("model_confidence_score")
    cal = prediction.get("calibrated_probabilities") or prediction.get("raw_probabilities")

    parts = [
        f"Model tier={tier}",
        f"version={prediction.get('model_version')}",
        f"PIT integrity={'passed' if pit_ok else 'FAILED'}",
        f"aggregate_data_score={score}",
        f"features={n_feats}",
        f"raw_observations={n_raw}",
    ]
    if sources:
        parts.append(f"sources={','.join(sources)}")
    if conf is not None:
        parts.append(f"model_confidence={conf}")
    if isinstance(cal, Mapping) and cal:
        keys = ", ".join(f"{k}={v}" for k, v in list(cal.items())[:6])
        parts.append(f"probabilities=[{keys}]")
    return "; ".join(parts)


class AuditEngine:
    """Reverse-audit a prediction through PIT features to raw lake provenance.

    Chain
    -----
    ``prediction_id`` → ``pit_feature_snapshots`` (features & quality)
    → ``lineage_mapping`` → ``raw_ids`` in ``raw_data_lake`` (observed_at, source).

    Prefer an async SQLAlchemy session. For unit tests, pass ``fixtures`` with
    in-memory dicts so no DB is required.
    """

    def __init__(
        self,
        session: Optional[AsyncSession] = None,
        *,
        fixtures: Optional[FixtureStore] = None,
    ) -> None:
        self.session = session
        self.fixtures = fixtures or {}

    async def audit_prediction(self, prediction_id: UUID) -> dict[str, Any]:
        """Return a complete audit report for ``prediction_id``.

        Parameters
        ----------
        prediction_id :
            UUID of the ``prediction_snapshots`` row (or fixture key).

        Returns
        -------
        dict
            Audit report with prediction, feature snapshot, lineage raw_ids,
            raw observations, and a narrative ``explanation``.
        """
        pid = _as_uuid(prediction_id)
        if pid is None:
            raise ValueError("prediction_id is required")

        if self.fixtures:
            return self._audit_from_fixtures(pid)

        if self.session is None:
            raise RuntimeError(
                "AuditEngine requires an AsyncSession or in-memory fixtures"
            )
        return await self._audit_from_db(pid)

    def _audit_from_fixtures(self, prediction_id: UUID) -> dict[str, Any]:
        predictions = self.fixtures.get("predictions") or {}
        pits = self.fixtures.get("pit_feature_snapshots") or {}
        raws = self.fixtures.get("raw_data_lake") or {}

        pred = predictions.get(str(prediction_id)) or predictions.get(prediction_id)
        if pred is None:
            return {
                "prediction_id": str(prediction_id),
                "found": False,
                "error": "prediction_not_found",
                "prediction": None,
                "feature_snapshot": None,
                "lineage_mapping": {},
                "raw_ids": [],
                "raw_observations": [],
                "explanation": f"No prediction found for id={prediction_id}",
            }

        pred_dict = dict(pred)
        pred_dict.setdefault("prediction_id", str(prediction_id))
        fs_id = pred_dict.get("feature_snapshot_id")
        pit = pits.get(str(fs_id)) or pits.get(fs_id) if fs_id is not None else None
        if pit is None:
            pit_dict: JsonDict = {
                "feature_snapshot_id": str(fs_id) if fs_id else None,
                "features": {},
                "data_quality_metrics": None,
                "aggregate_data_score": None,
                "pit_integrity_passed": None,
                "lineage_mapping": {},
            }
        else:
            pit_dict = dict(pit)
            if "features" not in pit_dict and "features_json" in pit_dict:
                pit_dict["features"] = pit_dict["features_json"]

        lineage = pit_dict.get("lineage_mapping") or {}
        raw_ids = _collect_raw_ids(lineage if isinstance(lineage, Mapping) else {})
        observations: list[JsonDict] = []
        for rid in raw_ids:
            row = raws.get(rid) or raws.get(_as_uuid(rid))
            if row is None:
                observations.append(
                    {
                        "raw_id": rid,
                        "source": None,
                        "observed_at": None,
                        "missing": True,
                    }
                )
            else:
                obs = dict(row)
                obs.setdefault("raw_id", rid)
                obs["missing"] = False
                if "observed_at" in obs:
                    obs["observed_at"] = _iso(obs["observed_at"])
                observations.append(obs)

        explanation = _build_explanation(
            prediction=pred_dict,
            feature_snapshot=pit_dict,
            raw_observations=[o for o in observations if not o.get("missing")],
        )
        return {
            "prediction_id": str(prediction_id),
            "found": True,
            "prediction": pred_dict,
            "feature_snapshot": pit_dict,
            "lineage_mapping": lineage,
            "raw_ids": raw_ids,
            "raw_observations": observations,
            "explanation": explanation,
        }

    async def _audit_from_db(self, prediction_id: UUID) -> dict[str, Any]:
        assert self.session is not None
        stmt = (
            select(PredictionSnapshot)
            .where(PredictionSnapshot.prediction_id == prediction_id)
            .options(selectinload(PredictionSnapshot.feature_snapshot))
        )
        result = await self.session.execute(stmt)
        pred = result.scalar_one_or_none()
        if pred is None:
            return {
                "prediction_id": str(prediction_id),
                "found": False,
                "error": "prediction_not_found",
                "prediction": None,
                "feature_snapshot": None,
                "lineage_mapping": {},
                "raw_ids": [],
                "raw_observations": [],
                "explanation": f"No prediction found for id={prediction_id}",
            }

        pit = pred.feature_snapshot
        if pit is None:
            pit_stmt = select(PitFeatureSnapshot).where(
                PitFeatureSnapshot.feature_snapshot_id == pred.feature_snapshot_id
            )
            pit = (await self.session.execute(pit_stmt)).scalar_one_or_none()

        pred_dict = _prediction_to_dict(pred)
        if pit is None:
            pit_dict: JsonDict = {
                "feature_snapshot_id": str(pred.feature_snapshot_id),
                "features": {},
                "data_quality_metrics": None,
                "aggregate_data_score": None,
                "pit_integrity_passed": None,
                "lineage_mapping": {},
            }
            lineage: Mapping[str, Any] = {}
        else:
            pit_dict = _pit_to_dict(pit)
            lineage = pit.lineage_mapping or {}

        raw_ids = _collect_raw_ids(lineage)
        observations: list[JsonDict] = []
        if raw_ids:
            uuid_ids: list[UUID] = []
            for rid in raw_ids:
                try:
                    uuid_ids.append(UUID(str(rid)))
                except (ValueError, TypeError):
                    continue
            if uuid_ids:
                raw_stmt = select(RawDataLake).where(RawDataLake.raw_id.in_(uuid_ids))
                rows = (await self.session.execute(raw_stmt)).scalars().all()
                by_id = {str(r.raw_id): _raw_to_dict(r) for r in rows}
            else:
                by_id = {}
            for rid in raw_ids:
                if rid in by_id:
                    obs = by_id[rid]
                    obs["missing"] = False
                    observations.append(obs)
                else:
                    observations.append(
                        {
                            "raw_id": rid,
                            "source": None,
                            "observed_at": None,
                            "missing": True,
                        }
                    )

        explanation = _build_explanation(
            prediction=pred_dict,
            feature_snapshot=pit_dict,
            raw_observations=[o for o in observations if not o.get("missing")],
        )
        return {
            "prediction_id": str(prediction_id),
            "found": True,
            "prediction": pred_dict,
            "feature_snapshot": pit_dict,
            "lineage_mapping": dict(lineage) if lineage else {},
            "raw_ids": raw_ids,
            "raw_observations": observations,
            "explanation": explanation,
        }


__all__ = ["AuditEngine"]
