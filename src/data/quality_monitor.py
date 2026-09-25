"""Data-quality scoring and local HARD GATE stubs (Worker 2).

Independent of Worker 1 ``DataMonitor`` / schema — call sites may swap in a
real monitor later by passing a compatible ``hard_gate`` callback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

HARD_GATE_CODE = "MODEL_TIER_X"


class ModelTier(str, Enum):
    """Model eligibility tier derived from quality metrics."""

    A = "MODEL_TIER_A"
    B = "MODEL_TIER_B"
    C = "MODEL_TIER_C"
    X = "MODEL_TIER_X"  # hard gate — NO BET


@dataclass
class HardGateEvent:
    """Record of a fired HARD GATE (NO BET)."""

    message: str
    code: str = HARD_GATE_CODE
    no_bet: bool = True
    context: dict[str, Any] = field(default_factory=dict)


class HardGate:
    """In-module HARD GATE logger — does not import Worker 1 monitoring.

    ``trigger`` always records ``MODEL_TIER_X`` / NO BET by default and logs
    at CRITICAL. Events are kept in ``events`` for deterministic tests.
    """

    def __init__(self, *, logger_: Optional[logging.Logger] = None) -> None:
        self._logger = logger_ or logger
        self.events: list[HardGateEvent] = []

    def trigger(
        self,
        message: str,
        *,
        code: str = HARD_GATE_CODE,
        no_bet: bool = True,
        **context: Any,
    ) -> HardGateEvent:
        """Fire HARD GATE and append to ``events``."""
        event = HardGateEvent(
            message=message, code=code, no_bet=no_bet, context=dict(context)
        )
        self.events.append(event)
        self._logger.critical(
            "[HARD_GATE %s] NO_BET=%s %s | %s", code, no_bet, message, context
        )
        return event


# Process-local default used by PITEngine when no callback is injected.
default_hard_gate = HardGate()


def _parse_dt(value: Any) -> Optional[datetime]:
    """Parse ``datetime`` or ISO-8601 string; return None if missing/invalid."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None
    return None


def _clamp_score(value: float) -> float:
    return float(max(0.0, min(100.0, value)))


class DataQualityEngine:
    """Compute six deterministic data-quality scores (0–100) for a match.

    Expected ``match_data`` keys
    ----------------------------
    raw_records : list[dict]
        Each record may include ``observed_at``, ``source_type``,
        ``source_quality`` (0–100), and payload fields.
    expected_sources : list[str], optional
        Source types that should be present (coverage). Default: unique
        ``source_type`` values already in ``raw_records`` → coverage 100.
    required_fields : list[str], optional
        Field names that should be non-null on the match or across records
        (completeness). Default: empty → completeness 100.
    field_values : dict[str, Any], optional
        Top-level field map used for completeness when ``required_fields``
        is set.
    consistency_conflicts : int, optional
        Number of cross-source conflicts (consistency). Default 0.
    unmapped_entity : bool, optional
        If True, force ``aggregate_score=0`` and ``model_tier=MODEL_TIER_X``.
    as_of_already_filtered : bool, optional
        When True, assume records are PIT-clean and skip leak scan for
        ``pit_integrity`` (still 100). Prefer leaving False and letting
        the engine detect ``observed_at > as_of_time``.

    Scoring heuristics (deterministic)
    ----------------------------------
    coverage
        ``100 * n_present_sources / n_expected_sources`` (100 if no expected).
    freshness
        Based on newest ``observed_at`` age vs ``as_of_time``:
        age ≤ 1h → 100; then −5 points per hour (floor 0).
        No timestamps → 0.
    consistency
        ``100 - 25 * consistency_conflicts`` (floor 0).
    pit_integrity
        100 if every ``observed_at <= as_of_time``, else 0.
    source_quality
        Mean of per-record ``source_quality`` (default 70 if omitted).
        Empty records → 0.
    completeness
        ``100 * n_non_null_required / n_required`` (100 if no required).

    Hard rules
    ----------
    If ``pit_integrity < 100`` OR ``unmapped_entity`` is True:
    ``aggregate_score = 0`` and ``model_tier = MODEL_TIER_X`` (NO BET).
    Otherwise ``aggregate_score`` is the arithmetic mean of the six scores
    and tier is A (≥80), B (≥60), else C.
    """

    def __init__(
        self,
        hard_gate: Optional[HardGate] = None,
        on_hard_gate: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.hard_gate = hard_gate or default_hard_gate
        self.on_hard_gate = on_hard_gate

    def calculate_quality_metrics(
        self,
        match_data: dict[str, Any],
        as_of_time: datetime,
    ) -> dict[str, Any]:
        """Return six component scores plus aggregate and model tier.

        Returns
        -------
        dict
            Keys: ``coverage``, ``freshness``, ``consistency``,
            ``pit_integrity``, ``source_quality``, ``completeness``,
            ``aggregate_score``, ``model_tier``, ``unmapped_entity``,
            ``n_records``, ``n_leaks``.
        """
        records = list(match_data.get("raw_records") or [])
        unmapped = bool(match_data.get("unmapped_entity", False))

        coverage = self._score_coverage(match_data, records)
        freshness = self._score_freshness(records, as_of_time)
        consistency = self._score_consistency(match_data)
        pit_integrity, n_leaks = self._score_pit_integrity(records, as_of_time)
        source_quality = self._score_source_quality(records)
        completeness = self._score_completeness(match_data)

        scores = {
            "coverage": coverage,
            "freshness": freshness,
            "consistency": consistency,
            "pit_integrity": pit_integrity,
            "source_quality": source_quality,
            "completeness": completeness,
        }

        hard_fail = pit_integrity < 100.0 or unmapped
        if hard_fail:
            aggregate = 0.0
            tier = ModelTier.X.value
            reason = (
                "PIT integrity failure (future leakage)"
                if pit_integrity < 100.0
                else "Unmapped entity"
            )
            self._fire_hard_gate(reason, pit_integrity=pit_integrity, unmapped=unmapped)
        else:
            aggregate = sum(scores.values()) / 6.0
            tier = self._tier_from_aggregate(aggregate)

        return {
            **scores,
            "aggregate_score": round(aggregate, 4),
            "model_tier": tier,
            "unmapped_entity": unmapped,
            "n_records": len(records),
            "n_leaks": n_leaks,
        }

    def _fire_hard_gate(self, message: str, **context: Any) -> None:
        self.hard_gate.trigger(message, code=HARD_GATE_CODE, no_bet=True, **context)
        if self.on_hard_gate is not None:
            self.on_hard_gate(message, code=HARD_GATE_CODE, no_bet=True, **context)

    @staticmethod
    def _tier_from_aggregate(aggregate: float) -> str:
        if aggregate >= 80.0:
            return ModelTier.A.value
        if aggregate >= 60.0:
            return ModelTier.B.value
        return ModelTier.C.value

    @staticmethod
    def _score_coverage(
        match_data: dict[str, Any], records: list[dict[str, Any]]
    ) -> float:
        expected = match_data.get("expected_sources")
        present = {r.get("source_type") for r in records if r.get("source_type")}
        if not expected:
            return 100.0
        expected_set = set(expected)
        if not expected_set:
            return 100.0
        n_hit = len(expected_set & present)
        return _clamp_score(100.0 * n_hit / len(expected_set))

    @staticmethod
    def _score_freshness(
        records: list[dict[str, Any]], as_of_time: datetime
    ) -> float:
        times = [_parse_dt(r.get("observed_at")) for r in records]
        times = [t for t in times if t is not None]
        if not times:
            return 0.0
        # Normalize naive vs aware for subtraction safety.
        as_of = as_of_time
        if as_of.tzinfo is not None:
            times = [
                t if t.tzinfo is not None else t.replace(tzinfo=timezone.utc)
                for t in times
            ]
        else:
            times = [t.replace(tzinfo=None) if t.tzinfo else t for t in times]
        newest = max(times)
        age_hours = max(0.0, (as_of - newest).total_seconds() / 3600.0)
        if age_hours <= 1.0:
            return 100.0
        return _clamp_score(100.0 - 5.0 * (age_hours - 1.0))

    @staticmethod
    def _score_consistency(match_data: dict[str, Any]) -> float:
        conflicts = int(match_data.get("consistency_conflicts") or 0)
        return _clamp_score(100.0 - 25.0 * conflicts)

    @staticmethod
    def _score_pit_integrity(
        records: list[dict[str, Any]], as_of_time: datetime
    ) -> tuple[float, int]:
        n_leaks = 0
        for rec in records:
            obs = _parse_dt(rec.get("observed_at"))
            if obs is None:
                continue
            # Compare after aligning tz awareness with as_of_time.
            if as_of_time.tzinfo is not None and obs.tzinfo is None:
                obs = obs.replace(tzinfo=timezone.utc)
            elif as_of_time.tzinfo is None and obs.tzinfo is not None:
                obs = obs.replace(tzinfo=None)
            if obs > as_of_time:
                n_leaks += 1
        return (100.0 if n_leaks == 0 else 0.0), n_leaks

    @staticmethod
    def _score_source_quality(records: list[dict[str, Any]]) -> float:
        if not records:
            return 0.0
        values: list[float] = []
        for rec in records:
            raw = rec.get("source_quality", 70.0)
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                values.append(70.0)
        return _clamp_score(sum(values) / len(values))

    @staticmethod
    def _score_completeness(match_data: dict[str, Any]) -> float:
        required = list(match_data.get("required_fields") or [])
        if not required:
            return 100.0
        field_values = dict(match_data.get("field_values") or {})
        # Also allow required keys on match_data itself.
        n_ok = 0
        for key in required:
            val = field_values.get(key, match_data.get(key))
            if val is not None and val != "":
                n_ok += 1
        return _clamp_score(100.0 * n_ok / len(required))
