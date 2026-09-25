"""Unit tests for VigRemovalEngine (multiplicative / additive / Shin)."""

from __future__ import annotations

import pytest

from src.models.vig_removal import VigRemovalEngine


@pytest.fixture
def engine() -> VigRemovalEngine:
    return VigRemovalEngine()


def test_multiplicative_probs_sum_to_one(engine: VigRemovalEngine) -> None:
    odds = [2.10, 3.40, 3.50]
    probs = engine.multiplicative_margin(odds)
    assert len(probs) == 3
    assert sum(probs) == pytest.approx(1.0, abs=1e-6)
    assert all(p > 0 for p in probs)
    # Favoured outcome (lowest odds) gets highest fair prob.
    assert probs[0] == max(probs)


def test_shin_probs_sum_to_one(engine: VigRemovalEngine) -> None:
    odds = [1.90, 3.60, 4.20]
    probs = engine.shin_method(odds)
    assert len(probs) == 3
    assert sum(probs) == pytest.approx(1.0, abs=1e-6)
    assert all(p > 0 for p in probs)


def test_shin_two_way_sums_to_one(engine: VigRemovalEngine) -> None:
    probs = engine.shin_method([1.95, 1.95])
    assert sum(probs) == pytest.approx(1.0, abs=1e-6)
    assert probs[0] == pytest.approx(probs[1], abs=1e-6)


def test_additive_probs_sum_to_one(engine: VigRemovalEngine) -> None:
    probs = engine.additive_margin([2.05, 3.50, 3.60])
    assert sum(probs) == pytest.approx(1.0, abs=1e-6)


def test_shin_vs_multiplicative_longshot_shrink(engine: VigRemovalEngine) -> None:
    """Shin typically shrinks longshot overround vs pure multiplicative."""
    odds = [1.50, 4.50, 8.00]
    multi = engine.multiplicative_margin(odds)
    shin = engine.shin_method(odds)
    assert sum(shin) == pytest.approx(1.0, abs=1e-6)
    # Longshot (index 2) fair prob should not exceed multiplicative.
    assert shin[2] <= multi[2] + 1e-9


def test_convert_ah_smoke(engine: VigRemovalEngine) -> None:
    out = engine.convert_ah_ou_to_fair_prob(
        "AH", line=-0.5, odds_1=1.95, odds_2=1.95, method="shin"
    )
    assert out["market_type"] == "ah"
    assert out["line"] == -0.5
    assert out["p_home"] + out["p_away"] == pytest.approx(1.0, abs=1e-6)
    assert out["sum_fair_prob"] == pytest.approx(1.0, abs=1e-6)
    assert out["method"] == "shin"


def test_convert_ou_multiplicative_smoke(engine: VigRemovalEngine) -> None:
    out = engine.convert_ah_ou_to_fair_prob(
        "OU", line=2.5, odds_1=1.90, odds_2=2.00, method="multiplicative"
    )
    assert out["p_over"] + out["p_under"] == pytest.approx(1.0, abs=1e-6)
    assert out["selection_1"] == "over"
    assert out["fair_odds_1"] == pytest.approx(1.0 / out["p_over"])


def test_invalid_odds_rejected(engine: VigRemovalEngine) -> None:
    with pytest.raises(ValueError):
        engine.shin_method([1.0, 2.0])
    with pytest.raises(ValueError):
        engine.multiplicative_margin([2.0])
