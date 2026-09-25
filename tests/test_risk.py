"""Unit tests for RiskEngine sizing and ExposureManager caps."""

from __future__ import annotations

import pytest

from src.risk import ExposureManager, RiskEngine, RiskEngineConfig


@pytest.fixture
def engine() -> RiskEngine:
    return RiskEngine()


def test_low_data_score_zero_stake(engine: RiskEngine) -> None:
    """data_score < 50 → NO BET regardless of strong EV."""
    out = engine.calculate_stake(
        probability=0.60,
        odds=2.20,
        data_score=40,
        model_confidence=90,
        existing_match_bets=[],
    )
    assert out["ev"] == pytest.approx(0.60 * 2.20 - 1.0)
    assert out["final_stake"] == 0.0
    assert out["status"] == "NO_BET"
    assert out["reason"] == "data_score_below_threshold"


def test_hard_cap_binds(engine: RiskEngine) -> None:
    """High edge + perfect weights still cannot exceed max_bet_cap_percent."""
    # Full Kelly ≈ (0.7*2.5-1)/(2.5-1) = 0.75/1.5 = 0.5
    # f_kelly = 0.5 * 0.10 = 0.05; after weights 1.0 → 0.05 > 0.01 cap
    out = engine.calculate_stake(
        probability=0.70,
        odds=2.50,
        data_score=100,
        model_confidence=100,
        existing_match_bets=[],
    )
    assert out["status"] == "OK"
    assert out["f_kelly"] == pytest.approx(0.05)
    assert out["f_scaled"] == pytest.approx(0.05)
    assert out["final_stake"] == pytest.approx(0.01)
    assert out["hard_cap_applied"] is True
    assert "hard_cap" in out["reason"]


def test_correlation_discount_home_ml_plus_over(engine: RiskEngine) -> None:
    """Over 2.5 already open + Home ML candidate → ~30% stake discount."""
    existing = [{"market": "OU", "selection": "Over 2.5"}]
    base = engine.calculate_stake(
        probability=0.55,
        odds=2.10,
        data_score=90,
        model_confidence=90,
        existing_match_bets=[],
        market="1X2",
        selection="Home",
    )
    discounted = engine.calculate_stake(
        probability=0.55,
        odds=2.10,
        data_score=90,
        model_confidence=90,
        existing_match_bets=existing,
        market="1X2",
        selection="Home",
    )
    assert base["status"] == "OK"
    assert discounted["status"] == "OK"
    assert discounted["correlation_discount"] == pytest.approx(0.30)
    assert discounted["correlation_multiplier"] == pytest.approx(0.70)
    # Before hard cap, correlated path should be 70% of uncapped scaled stake.
    assert discounted["f_after_correlation"] == pytest.approx(
        base["f_scaled"] * 0.70
    )
    assert "correlated_markets" in discounted.get("correlation_reason", "")


def test_ev_below_threshold_no_bet(engine: RiskEngine) -> None:
    out = engine.calculate_stake(
        probability=0.48,
        odds=2.05,  # EV = 0.48*2.05 - 1 = -0.016
        data_score=80,
        model_confidence=80,
        existing_match_bets=[],
    )
    assert out["final_stake"] == 0.0
    assert out["status"] == "NO_BET"
    assert out["reason"] == "ev_below_threshold"


def test_fractional_kelly_formula(engine: RiskEngine) -> None:
    p, odds = 0.55, 2.20
    # EV = 0.21; full Kelly = 0.21/1.2 = 0.175; ×0.10 = 0.0175
    out = engine.calculate_stake(
        probability=p,
        odds=odds,
        data_score=100,
        model_confidence=100,
        existing_match_bets=[],
    )
    expected_kelly = ((p * odds - 1.0) / (odds - 1.0)) * 0.10
    assert out["f_kelly"] == pytest.approx(expected_kelly)
    assert out["ev"] == pytest.approx(p * odds - 1.0)


def test_custom_cap_config() -> None:
    eng = RiskEngine(
        config=RiskEngineConfig(
            fraction_multiplier=0.10,
            max_bet_cap_percent=0.10,  # above f_scaled so cap does not bind
            min_ev_threshold=0.05,
        )
    )
    out = eng.calculate_stake(
        probability=0.70,
        odds=2.50,
        data_score=100,
        model_confidence=100,
        existing_match_bets=[],
    )
    # f_scaled = 0.05 → passes under 10% cap
    assert out["final_stake"] == pytest.approx(0.05)
    assert out["hard_cap_applied"] is False


@pytest.mark.asyncio
async def test_exposure_manager_daily_and_league_caps() -> None:
    mgr = ExposureManager()
    first = await mgr.accept_bet(0.02, league="EPL", market_type="1X2", selection="Home")
    assert first["accepted"] is True
    # League cap default 0.03 → another 0.02 would exceed remaining 0.01
    blocked = await mgr.accept_bet(0.02, league="EPL")
    assert blocked["accepted"] is False
    assert "max_league_exposure" in blocked["exposure"]["reason"]

    # Different league still limited by daily (used 0.02, max 0.05)
    ok = await mgr.accept_bet(0.02, league="UWCL")
    assert ok["accepted"] is True
    over_daily = await mgr.accept_bet(0.02, league="UWCL")
    assert over_daily["accepted"] is False
    assert "max_daily_exposure" in over_daily["exposure"]["reason"]


@pytest.mark.asyncio
async def test_prepare_bet_snapshot_without_db() -> None:
    eng = RiskEngine()
    sized = eng.calculate_stake(
        probability=0.55,
        odds=2.20,
        data_score=90,
        model_confidence=90,
        existing_match_bets=[],
        market="1X2",
        selection="Home",
    )
    mgr = ExposureManager()
    record = mgr.prepare_bet_snapshot(
        market_type="1X2",
        selection="Home",
        taken_odds=2.20,
        stake_result=sized,
        bankroll=1000.0,
    )
    assert record["final_stake_fraction"] == sized["final_stake"]
    assert record["raw_kelly_fraction"] == sized["f_kelly"]
    assert record["final_stake_amount"] == pytest.approx(
        sized["final_stake"] * 1000.0
    )
    saved = await mgr.save_bet_snapshot(record, session=None)
    assert saved["saved"] is False
    assert saved["reason"] == "schema_or_session_unavailable"
