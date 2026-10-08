"""Regression tests: LightGBM temporal calibration + predict-path features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from sklearn.model_selection import TimeSeriesSplit  # noqa: E402

from src.ml_model import EPLMachineLearningModel  # noqa: E402


def _matches(n: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    teams = [f"T{i}" for i in range(10)]
    rows = []
    day0 = pd.Timestamp("2024-08-01")
    for i in range(n):
        home, away = teams[i % 10], teams[(i * 3 + 1) % 10]
        if home == away:
            away = teams[(i + 5) % 10]
        rows.append(
            {
                "Date": day0 + pd.Timedelta(days=2 * (i // 5)),
                "HomeTeam": home,
                "AwayTeam": away,
                "FTHG": int(rng.integers(0, 4)),
                "FTAG": int(rng.integers(0, 3)),
                "HC": int(rng.integers(2, 10)),
                "AC": int(rng.integers(2, 9)),
                "comp_id": "EMPERORS_CUP" if i % 7 == 0 else "J1",
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def model() -> EPLMachineLearningModel:
    return EPLMachineLearningModel(calibrate=True).fit(_matches())


def test_calibration_uses_temporal_folds(model: EPLMachineLearningModel) -> None:
    assert model.calibrated_
    assert isinstance(model.model_.cv, TimeSeriesSplit)
    p = model.predict_proba("T0", "T1", match_date="2024-12-20")
    assert sum(p.values()) == pytest.approx(1.0)


def test_rotation_risk_uses_competition_on_predict(
    model: EPLMachineLearningModel,
) -> None:
    last = model.last_match_date_
    assert last is not None
    day = last + pd.Timedelta(days=1)
    # Every team plays ~every 2 days → ≥4 matches in 14d (busy).
    league = model.feature_vector("T0", "T1", match_date=day)
    cup = model.feature_vector("T0", "T1", match_date=day, competition="EMPERORS_CUP")
    forced = model.feature_vector("T0", "T1", match_date=day, is_cup=True)
    assert league["home_matches_last_14d"] >= 4
    assert league["home_is_rotation_risk"] == 0.0
    assert cup["home_is_rotation_risk"] == 1.0
    assert forced["away_is_rotation_risk"] == 1.0
    p = model.predict_proba("T0", "T1", match_date=day, competition="EMPERORS_CUP")
    assert sum(p.values()) == pytest.approx(1.0)


def test_real_fixture_date_drives_rest_days(model: EPLMachineLearningModel) -> None:
    last = model.last_match_date_
    assert last is not None
    fv = model.feature_vector("T0", "T1", match_date=last + pd.Timedelta(days=10))
    assert fv["home_rest_days"] >= 10


def test_missing_date_does_not_fabricate_rest(model: EPLMachineLearningModel) -> None:
    fv = model.feature_vector("T0", "T1")
    assert np.isnan(fv["home_rest_days"]) and np.isnan(fv["away_rest_days"])
    assert np.isnan(fv["home_matches_last_14d"])
    # NaT treated like None.
    fv2 = model.feature_vector("T0", "T1", match_date=pd.NaT)
    assert np.isnan(fv2["home_rest_days"])
    p = model.predict_proba("T0", "T1")
    assert sum(p.values()) == pytest.approx(1.0)
