"""Tests for Worker 2 PITEngine, DataQualityEngine, and FeatureLineageTracker."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.data.lineage_builder import FeatureLineageTracker
from src.data.pit_engine import HARD_GATE_CODE, PITEngine
from src.data.quality_monitor import DataQualityEngine, HardGate, ModelTier


AS_OF = datetime(2025, 9, 19, 12, 0, tzinfo=timezone.utc)


def test_pit_clean_cutoff_passes() -> None:
    gate = HardGate()
    engine = PITEngine(hard_gate=gate)
    records = [
        {"id": "a", "observed_at": AS_OF - timedelta(hours=2), "payload": 1},
        {"id": "b", "observed_at": AS_OF, "payload": 2},  # inclusive boundary
    ]
    valid, pit_integrity_passed = engine.get_valid_raw_records(records, AS_OF)

    assert pit_integrity_passed is True
    assert len(valid) == 2
    assert {r["id"] for r in valid} == {"a", "b"}
    assert gate.events == []


def test_pit_future_leakage_fails_always() -> None:
    """Future leakage must fail integrity 100% of the time (hard assertion)."""
    for _ in range(20):
        gate = HardGate()
        engine = PITEngine(hard_gate=gate)
        records = [
            {
                "id": "clean",
                "observed_at": AS_OF - timedelta(hours=1),
                "payload": {"xg": 1.2},
            },
            {
                "id": "leak",
                "observed_at": AS_OF + timedelta(minutes=1),
                "payload": {"xg": 9.9, "leaked": True},
            },
        ]
        valid, pit_integrity_passed = engine.get_valid_raw_records(records, AS_OF)

        assert pit_integrity_passed is False
        assert len(valid) == 1
        assert valid[0]["id"] == "clean"
        assert len(gate.events) == 1
        assert gate.events[0].code == HARD_GATE_CODE
        assert gate.events[0].no_bet is True
        assert gate.events[0].context["leak_count"] == 1


def test_pit_iso_string_timestamps() -> None:
    gate = HardGate()
    engine = PITEngine(hard_gate=gate)
    records = [
        {"id": "ok", "observed_at": "2025-09-19T11:00:00+00:00"},
        {"id": "leak", "observed_at": "2025-09-19T13:00:00+00:00"},
    ]
    valid, passed = engine.get_valid_raw_records(records, AS_OF)
    assert passed is False
    assert [r["id"] for r in valid] == ["ok"]


def test_quality_metrics_clean_match() -> None:
    gate = HardGate()
    qe = DataQualityEngine(hard_gate=gate)
    match_data = {
        "raw_records": [
            {
                "observed_at": AS_OF - timedelta(minutes=30),
                "source_type": "odds_api",
                "source_quality": 90,
            },
            {
                "observed_at": AS_OF - timedelta(minutes=10),
                "source_type": "football-data",
                "source_quality": 80,
            },
        ],
        "expected_sources": ["odds_api", "football-data"],
        "required_fields": ["home_team", "away_team"],
        "field_values": {"home_team": "Arsenal", "away_team": "Chelsea"},
        "consistency_conflicts": 0,
        "unmapped_entity": False,
    }
    metrics = qe.calculate_quality_metrics(match_data, AS_OF)

    assert metrics["pit_integrity"] == 100.0
    assert metrics["coverage"] == 100.0
    assert metrics["completeness"] == 100.0
    assert metrics["consistency"] == 100.0
    assert metrics["source_quality"] == 85.0
    assert metrics["freshness"] == 100.0
    assert metrics["aggregate_score"] > 0
    assert metrics["model_tier"] == ModelTier.A.value
    assert gate.events == []


def test_quality_metrics_pit_fail_forces_tier_x() -> None:
    gate = HardGate()
    qe = DataQualityEngine(hard_gate=gate)
    match_data = {
        "raw_records": [
            {"observed_at": AS_OF + timedelta(seconds=1), "source_type": "odds_api"},
        ],
        "unmapped_entity": False,
    }
    metrics = qe.calculate_quality_metrics(match_data, AS_OF)

    assert metrics["pit_integrity"] == 0.0
    assert metrics["aggregate_score"] == 0.0
    assert metrics["model_tier"] == ModelTier.X.value
    assert metrics["n_leaks"] == 1
    assert len(gate.events) == 1


def test_quality_metrics_unmapped_forces_tier_x() -> None:
    gate = HardGate()
    qe = DataQualityEngine(hard_gate=gate)
    match_data = {
        "raw_records": [
            {
                "observed_at": AS_OF - timedelta(hours=1),
                "source_type": "odds_api",
                "source_quality": 100,
            },
        ],
        "unmapped_entity": True,
    }
    metrics = qe.calculate_quality_metrics(match_data, AS_OF)

    assert metrics["pit_integrity"] == 100.0
    assert metrics["aggregate_score"] == 0.0
    assert metrics["model_tier"] == ModelTier.X.value
    assert metrics["unmapped_entity"] is True
    assert len(gate.events) == 1


def test_lineage_map_raw_ids_and_max_observed() -> None:
    tracker = FeatureLineageTracker()
    feature_dict = {"home_xg": 1.2, "away_xg": 0.9}
    raw_sources_map = {
        "home_xg": [
            {"raw_id": "r1", "observed_at": AS_OF - timedelta(hours=2)},
            {"raw_id": "r2", "observed_at": AS_OF - timedelta(minutes=5)},
        ],
        "away_xg": [
            {"raw_id": "r3", "observed_at": "2025-09-19T10:00:00+00:00"},
        ],
        "unused_feature": [{"raw_id": "r9", "observed_at": AS_OF}],
    }
    lineage = tracker.build_lineage_map(feature_dict, raw_sources_map)

    assert set(lineage.keys()) == {"away_xg", "home_xg"}
    assert lineage["home_xg"]["raw_ids"] == ["r1", "r2"]
    assert lineage["home_xg"]["observed_at_max"] == (
        AS_OF - timedelta(minutes=5)
    ).isoformat()
    assert lineage["away_xg"]["raw_ids"] == ["r3"]
    assert lineage["away_xg"]["observed_at_max"] == "2025-09-19T10:00:00+00:00"


def test_lineage_map_string_ids_only() -> None:
    tracker = FeatureLineageTracker()
    lineage = tracker.build_lineage_map(
        {"f1": 1.0},
        {"f1": ["raw-a", "raw-b"]},
    )
    assert lineage["f1"]["raw_ids"] == ["raw-a", "raw-b"]
    assert lineage["f1"]["observed_at_max"] is None
