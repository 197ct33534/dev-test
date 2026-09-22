"""Defaults + Kelly strategy + calibrated LightGBM smoke tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import DEFAULT_KELLY_FRACTION, DEFAULT_W_ML, MAX_STAKE_PCT
from src.ml_model import EPLMachineLearningModel
from src.strategy import calculate_kelly_stake, capped_kelly_fraction


def test_config_defaults_from_ablation() -> None:
    assert DEFAULT_W_ML == pytest.approx(0.20)
    assert DEFAULT_KELLY_FRACTION == pytest.approx(0.10)
    assert MAX_STAKE_PCT == pytest.approx(0.01)


def test_kelly_stake_applies_fraction_and_cap() -> None:
    # Full Kelly would be large; 10% Kelly still above 1% → clipped to 1%.
    pct, amount = calculate_kelly_stake(0.55, 2.5, bankroll=1000.0)
    assert pct == pytest.approx(MAX_STAKE_PCT)
    assert amount == pytest.approx(10.0)
    assert capped_kelly_fraction(0.55, 2.5) <= MAX_STAKE_PCT + 1e-12


def _toy_matches(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    teams = [f"T{i}" for i in range(8)]
    rows = []
    base = pd.Timestamp("2023-08-01")
    for i in range(n):
        h, a = teams[i % 8], teams[(i + 3) % 8]
        hg, ag = int(rng.integers(0, 4)), int(rng.integers(0, 4))
        ftr = "H" if hg > ag else ("A" if hg < ag else "D")
        rows.append(
            {
                "Date": base + pd.Timedelta(days=i),
                "HomeTeam": h,
                "AwayTeam": a,
                "FTHG": hg,
                "FTAG": ag,
                "FTR": ftr,
                "HC": int(rng.integers(2, 10)),
                "AC": int(rng.integers(2, 10)),
            }
        )
    return pd.DataFrame(rows)


def test_lightgbm_calibrated_predict_proba() -> None:
    data = _toy_matches(150)
    model = EPLMachineLearningModel(
        calibrate=True,
        calibration_method="sigmoid",
        min_prior_matches=1,
    ).fit(data)
    assert model.calibrated_ is True
    probs = model.predict_proba(data["HomeTeam"].iloc[-1], data["AwayTeam"].iloc[-1])
    assert abs(sum(probs.values()) - 1.0) < 1e-5
    assert all(0.0 < v < 1.0 for v in probs.values())
