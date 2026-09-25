"""TestClient coverage for Quant Engine API v2 (mocked pipeline / paper / audit)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("TELEGRAM_AUTH_DISABLED", "true")
    return TestClient(create_app())


def _pipeline_ok(match_id: Any, *_a: Any, **_k: Any) -> dict[str, Any]:
    return {
        "prediction_id": uuid4(),
        "canonical_match_id": match_id,
        "expected_value_percent": 8.5,
        "aggregate_data_score": 72.0,
        "model_tier": "MODEL_TIER_B",
        "no_bet": False,
        "fair_lines": {"H": 2.05, "D": 3.40, "A": 3.80},
        "fair_probabilities": {"H": 0.488, "D": 0.294, "A": 0.218},
        "prediction_snapshot": {"model_confidence_score": 0.44},
    }


def _pipeline_tier_x(match_id: Any, *_a: Any, **_k: Any) -> dict[str, Any]:
    return {
        "prediction_id": uuid4(),
        "canonical_match_id": match_id,
        "expected_value_percent": 0.0,
        "aggregate_data_score": 0.0,
        "model_tier": "MODEL_TIER_X",
        "no_bet": True,
        "fair_lines": {"H": 3.0, "D": 3.0, "A": 3.0},
        "fair_probabilities": {"H": 1 / 3, "D": 1 / 3, "A": 1 / 3},
        "prediction_snapshot": {"model_confidence_score": 0.0},
    }


def test_v2_value_bets_filters_tier_and_ev(client: TestClient) -> None:
    match_ok = uuid4()
    match_x = uuid4()
    fixtures = [
        {
            "canonical_match_id": match_ok,
            "kickoff_utc": datetime(2026, 10, 1, 15, 0, tzinfo=timezone.utc),
            "home": "Arsenal",
            "away": "Chelsea",
            "league": "EPL",
            "status": "SCHEDULED",
            "bookmaker_odds": {"H": 2.20, "D": 3.40, "A": 3.50},
            "raw_records": [],
        },
        {
            "canonical_match_id": match_x,
            "kickoff_utc": datetime(2026, 10, 1, 17, 0, tzinfo=timezone.utc),
            "home": "Burnley",
            "away": "Wolves",
            "league": "EPL",
            "status": "SCHEDULED",
            "bookmaker_odds": {"H": 2.50, "D": 3.20, "A": 2.90},
            "raw_records": [],
        },
    ]

    async def fake_pipeline(match_id: Any, *_a: Any, **_k: Any) -> dict[str, Any]:
        if match_id == match_x:
            return _pipeline_tier_x(match_id)
        return _pipeline_ok(match_id)

    with (
        patch(
            "src.api.routes_v2.fetch_upcoming_matches",
            new_callable=AsyncMock,
            return_value=fixtures,
        ),
        patch(
            "src.api.routes_v2.get_session_factory",
            return_value=MagicMock(
                return_value=MagicMock(
                    __aenter__=AsyncMock(return_value=MagicMock()),
                    __aexit__=AsyncMock(return_value=None),
                )
            ),
        ),
        patch(
            "src.api.routes_v2.run_quant_pipeline",
            side_effect=fake_pipeline,
        ),
    ):
        # Patch _price_one's default uses run_quant_pipeline from module — already patched.
        # Also patch the name used inside _price_one via module attribute.
        r = client.get("/api/v2/value-bets?min_ev=3&limit=10&league=EPL")

    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["scanned"] == 2
    assert body["min_ev_pct"] == 3.0
    bet = body["bets"][0]
    assert bet["match_id"] == str(match_ok)
    assert bet["model_tier"] == "MODEL_TIER_B"
    assert bet["data_score"] == 72.0
    assert bet["model_confidence"] == 0.44
    assert "H" in bet["fair_lines"]
    assert bet["expected_value_percent"] >= 3.0


def test_v2_value_bets_empty_db(client: TestClient) -> None:
    with patch(
        "src.api.routes_v2.get_session_factory",
        side_effect=RuntimeError("no db"),
    ):
        r = client.get("/api/v2/value-bets")
    assert r.status_code == 200
    body = r.json()
    assert body["bets"] == []
    assert body["count"] == 0
    assert "unavailable" in (body.get("notes") or "").lower()


def test_v2_value_bets_no_fixtures(client: TestClient) -> None:
    with (
        patch(
            "src.api.routes_v2.fetch_upcoming_matches",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "src.api.routes_v2.get_session_factory",
            return_value=MagicMock(
                return_value=MagicMock(
                    __aenter__=AsyncMock(return_value=MagicMock()),
                    __aexit__=AsyncMock(return_value=None),
                )
            ),
        ),
    ):
        r = client.get("/api/v2/value-bets?limit=5")
    assert r.status_code == 200
    assert r.json()["count"] == 0
    assert r.json()["scanned"] == 0


def test_v2_place_bets(client: TestClient) -> None:
    match_id = uuid4()
    bet_a = uuid4()
    bet_b = uuid4()

    with (
        patch("src.api.routes_v2.get_session_factory", side_effect=RuntimeError("skip")),
        patch(
            "src.api.routes_v2.PaperTrader.execute_value_bets",
            new_callable=AsyncMock,
            return_value=[bet_a, bet_b],
        ),
    ):
        r = client.post(
            "/api/v2/bets/place",
            json={
                "canonical_match_id": str(match_id),
                "bookmaker_odds": {"H": 2.1, "D": 3.4, "A": 3.6},
                "league": "EPL",
                "persist": False,
            },
        )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["count"] == 2
    assert set(body["bet_ids"]) == {str(bet_a), str(bet_b)}
    assert body["canonical_match_id"] == str(match_id)


def test_v2_place_bets_none_placed(client: TestClient) -> None:
    with (
        patch("src.api.routes_v2.get_session_factory", side_effect=RuntimeError("skip")),
        patch(
            "src.api.routes_v2.PaperTrader.execute_value_bets",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        r = client.post(
            "/api/v2/bets/place",
            json={
                "canonical_match_id": str(uuid4()),
                "bookmaker_odds": {"H": 1.5, "D": 4.0, "A": 6.0},
                "persist": False,
            },
        )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["count"] == 0
    assert body["message"]


def test_v2_audit_ok(client: TestClient) -> None:
    pred_id = uuid4()
    report = {
        "prediction_id": str(pred_id),
        "found": True,
        "prediction": {"prediction_id": str(pred_id), "model_tier": "MODEL_TIER_A"},
        "feature_snapshot": {"aggregate_data_score": 90.0},
        "lineage_mapping": {"n_pit_records": {"raw_ids": ["r1"]}},
        "raw_ids": ["r1"],
        "raw_observations": [{"raw_id": "r1", "source": "flashscore"}],
        "explanation": "Model tier=MODEL_TIER_A",
    }

    mock_engine = MagicMock()
    mock_engine.audit_prediction = AsyncMock(return_value=report)

    with (
        patch(
            "src.api.routes_v2.get_session_factory",
            return_value=MagicMock(
                return_value=MagicMock(
                    __aenter__=AsyncMock(return_value=MagicMock()),
                    __aexit__=AsyncMock(return_value=None),
                )
            ),
        ),
        patch("src.api.routes_v2.AuditEngine", return_value=mock_engine),
    ):
        r = client.get(f"/api/v2/audit/{pred_id}")

    assert r.status_code == 200
    body = r.json()
    assert body["found"] is True
    assert body["prediction_id"] == str(pred_id)
    assert body["raw_ids"] == ["r1"]
    assert "MODEL_TIER_A" in (body.get("explanation") or "")


def test_v2_audit_not_found(client: TestClient) -> None:
    pred_id = uuid4()
    mock_engine = MagicMock()
    mock_engine.audit_prediction = AsyncMock(
        return_value={
            "prediction_id": str(pred_id),
            "found": False,
            "error": "prediction_not_found",
            "prediction": None,
            "feature_snapshot": None,
            "lineage_mapping": {},
            "raw_ids": [],
            "raw_observations": [],
            "explanation": "missing",
        }
    )
    with (
        patch(
            "src.api.routes_v2.get_session_factory",
            return_value=MagicMock(
                return_value=MagicMock(
                    __aenter__=AsyncMock(return_value=MagicMock()),
                    __aexit__=AsyncMock(return_value=None),
                )
            ),
        ),
        patch("src.api.routes_v2.AuditEngine", return_value=mock_engine),
    ):
        r = client.get(f"/api/v2/audit/{pred_id}")
    assert r.status_code == 404


def test_v1_still_works(client: TestClient) -> None:
    """v2 registration must not break v1 health / mount."""
    r = client.get("/health")
    assert r.status_code == 200
    with patch(
        "src.api.routes.vb_svc.scan_value_bets_api",
        return_value={
            "bets": [],
            "count": 0,
            "min_ev_pct": 5.0,
            "markets": ["1X2"],
            "limit": 20,
            "source": "scan",
        },
    ):
        r2 = client.get("/api/v1/value-bets?limit=5")
    assert r2.status_code == 200
