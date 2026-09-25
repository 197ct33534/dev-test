"""Tests for AuditEngine reverse lineage and DriftDetector thresholds."""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from src.monitoring import AuditEngine, DriftDetector
from src.monitoring.audit_engine import _collect_raw_ids


def _dt(days_ago: float, *, now: datetime | None = None) -> datetime:
    base = now or datetime(2025, 9, 25, 12, 0, tzinfo=timezone.utc)
    return base - timedelta(days=days_ago)


@pytest.mark.asyncio
async def test_audit_prediction_reverse_lookup_raw_ids() -> None:
    """prediction_id → PIT lineage → correct raw_ids + lake provenance."""
    prediction_id = uuid.uuid4()
    feature_snapshot_id = uuid.uuid4()
    raw_a = uuid.uuid4()
    raw_b = uuid.uuid4()
    raw_c = uuid.uuid4()
    observed_a = datetime(2025, 9, 18, 10, 0, tzinfo=timezone.utc)
    observed_b = datetime(2025, 9, 18, 11, 0, tzinfo=timezone.utc)

    fixtures = {
        "predictions": {
            str(prediction_id): {
                "prediction_id": str(prediction_id),
                "feature_snapshot_id": str(feature_snapshot_id),
                "model_tier": "MODEL_TIER_A",
                "model_version": "ens-v2",
                "calibrator_version": "cal-1",
                "raw_probabilities": {"home": 0.45, "draw": 0.28, "away": 0.27},
                "calibrated_probabilities": {
                    "home": 0.42,
                    "draw": 0.30,
                    "away": 0.28,
                },
                "fair_lines": {"home": 2.38},
                "model_confidence_score": 0.81,
            }
        },
        "pit_feature_snapshots": {
            str(feature_snapshot_id): {
                "feature_snapshot_id": str(feature_snapshot_id),
                "canonical_match_id": str(uuid.uuid4()),
                "as_of_time": "2025-09-19T12:00:00+00:00",
                "features": {"xg_home": 1.4, "xg_away": 0.9},
                "data_quality_metrics": {"aggregate_score": 88.0},
                "aggregate_data_score": 88.0,
                "pit_integrity_passed": True,
                "lineage_mapping": {
                    "xg_home": {
                        "raw_ids": [str(raw_a), str(raw_b)],
                        "observed_at_max": observed_b.isoformat(),
                    },
                    "xg_away": {
                        "raw_ids": [str(raw_b), str(raw_c)],
                        "observed_at_max": observed_b.isoformat(),
                    },
                },
            }
        },
        "raw_data_lake": {
            str(raw_a): {
                "raw_id": str(raw_a),
                "source": "odds_api",
                "observed_at": observed_a,
                "payload": {"xg_home": 1.3},
            },
            str(raw_b): {
                "raw_id": str(raw_b),
                "source": "flashscore",
                "observed_at": observed_b,
                "payload": {"xg_home": 1.4, "xg_away": 0.9},
            },
            str(raw_c): {
                "raw_id": str(raw_c),
                "source": "football_data",
                "observed_at": observed_a,
                "payload": {"xg_away": 0.85},
            },
        },
    }

    engine = AuditEngine(fixtures=fixtures)
    report = await engine.audit_prediction(prediction_id)

    assert report["found"] is True
    assert report["prediction_id"] == str(prediction_id)
    assert report["prediction"]["model_tier"] == "MODEL_TIER_A"
    assert report["feature_snapshot"]["aggregate_data_score"] == 88.0
    assert report["feature_snapshot"]["features"]["xg_home"] == 1.4

    # Stable unique order from sorted feature keys: xg_away then xg_home
    # but _collect_raw_ids iterates sorted feature keys → xg_away first (b,c) then xg_home (a,b)
    expected_ids = _collect_raw_ids(
        fixtures["pit_feature_snapshots"][str(feature_snapshot_id)]["lineage_mapping"]
    )
    assert report["raw_ids"] == expected_ids
    assert set(report["raw_ids"]) == {str(raw_a), str(raw_b), str(raw_c)}

    by_id = {o["raw_id"]: o for o in report["raw_observations"]}
    assert by_id[str(raw_a)]["source"] == "odds_api"
    assert by_id[str(raw_a)]["observed_at"] == observed_a.isoformat()
    assert by_id[str(raw_b)]["source"] == "flashscore"
    assert by_id[str(raw_c)]["source"] == "football_data"
    assert all(not o.get("missing") for o in report["raw_observations"])
    assert "MODEL_TIER_A" in report["explanation"]
    assert "aggregate_data_score=88.0" in report["explanation"]


@pytest.mark.asyncio
async def test_audit_prediction_not_found() -> None:
    engine = AuditEngine(fixtures={"predictions": {}, "pit_feature_snapshots": {}, "raw_data_lake": {}})
    missing_id = uuid.uuid4()
    report = await engine.audit_prediction(missing_id)
    assert report["found"] is False
    assert report["raw_ids"] == []
    assert report["error"] == "prediction_not_found"


def test_model_drift_brier_increase_alerts() -> None:
    """Brier up >15% vs baseline window → alert."""
    now = datetime(2025, 9, 25, 12, 0, tzinfo=timezone.utc)
    # Baseline window (30–60d ago): sharp probs → low Brier
    baseline_bets = [
        {
            "settled_at": _dt(35 + i * 0.5, now=now),  # 35..54.5 days ago
            "y_true": 1 if i % 2 == 0 else 0,
            "y_prob": 0.95 if i % 2 == 0 else 0.05,
        }
        for i in range(40)
    ]
    # Recent window (0–30d): inverted probs → high Brier
    recent_bets = [
        {
            "settled_at": _dt(2 + i * 0.5, now=now),  # 2..21.5 days ago
            "y_true": 1 if i % 2 == 0 else 0,
            "y_prob": 0.05 if i % 2 == 0 else 0.95,
        }
        for i in range(40)
    ]
    detector = DriftDetector()
    out = detector.check_model_drift(baseline_bets + recent_bets, now=now)

    assert out["n_recent"] == 40
    assert out["n_baseline"] == 40
    assert math.isfinite(out["brier_recent"])
    assert math.isfinite(out["brier_baseline"])
    assert out["brier_recent"] > out["brier_baseline"] * 1.15
    assert out["brier_alert"] is True
    assert out["alert"] is True
    assert any(a["code"] == "MODEL_DRIFT_BRIER" for a in out["alerts"])


def test_model_drift_ece_threshold_alerts() -> None:
    """ECE > 0.10 → alert even if Brier baseline unavailable."""
    now = datetime(2025, 9, 25, 12, 0, tzinfo=timezone.utc)
    # Only recent window: systematically overconfident wrong way → high ECE
    recent_bets = [
        {
            "settled_at": _dt(5 + i * 0.1, now=now),
            "y_true": 0,
            "y_prob": 0.95,
        }
        for i in range(50)
    ]
    detector = DriftDetector()
    out = detector.check_model_drift(recent_bets, now=now)

    assert out["n_recent"] == 50
    assert out["ece_recent"] > 0.10
    assert out["ece_alert"] is True
    assert out["alert"] is True
    assert any(a["code"] == "MODEL_DRIFT_ECE" for a in out["alerts"])


def test_model_drift_healthy_no_alert() -> None:
    """Well-calibrated p=0.5 with balanced outcomes → no Brier/ECE alert."""
    now = datetime(2025, 9, 25, 12, 0, tzinfo=timezone.utc)
    bets = []
    for days_ago_start, n in ((40, 40), (5, 40)):
        for i in range(n):
            bets.append(
                {
                    "settled_at": _dt(days_ago_start + i * 0.4, now=now),
                    "y_true": i % 2,
                    "y_prob": 0.5,
                }
            )
    out = DriftDetector().check_model_drift(bets, now=now)
    assert out["n_recent"] == 40
    assert out["n_baseline"] == 40
    assert out["alert"] is False
    assert out["brier_alert"] is False
    assert out["ece_alert"] is False
    assert out["ece_recent"] <= 0.10


def test_data_drift_score_drop_alerts() -> None:
    """Sudden drop in aggregate_data_score vs baseline → alert."""
    now = datetime(2025, 9, 25, 12, 0, tzinfo=timezone.utc)
    baseline = [
        {
            "as_of_time": _dt(40 + i, now=now),
            "aggregate_data_score": 90.0,
            "features": {"a": 1.0, "b": 2.0},
        }
        for i in range(10)
    ]
    recent = [
        {
            "as_of_time": _dt(5 + i * 0.5, now=now),
            "aggregate_data_score": 60.0,  # ~33% drop > 15%
            "features": {"a": 1.0, "b": 2.0},
        }
        for i in range(10)
    ]
    out = DriftDetector().check_data_drift(baseline + recent, now=now)

    assert out["score_alert"] is True
    assert out["alert"] is True
    assert out["score_drop_ratio"] is not None
    assert out["score_drop_ratio"] > 0.15
    assert any(a["code"] == "DATA_DRIFT_SCORE_DROP" for a in out["alerts"])


def test_data_drift_missing_rate_alerts() -> None:
    now = datetime(2025, 9, 25, 12, 0, tzinfo=timezone.utc)
    snaps = [
        {
            "as_of_time": _dt(3 + i * 0.2, now=now),
            "aggregate_data_score": 85.0,
            "features": {"a": None, "b": None, "c": 1.0, "d": None},  # 75% missing
        }
        for i in range(8)
    ]
    out = DriftDetector().check_data_drift(snaps, now=now)
    assert out["missing_alert"] is True
    assert out["mean_missing_rate_recent"] > 0.25
    assert any(a["code"] == "DATA_DRIFT_MISSING_RATE" for a in out["alerts"])
