"""Unit tests for walk-forward splits and QuantMetrics smoke checks."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from src.validation.metrics import QuantMetrics
from src.validation.walk_forward import TimeSplit, WalkForwardValidator


UTC = timezone.utc


def test_generate_time_splits_example_2019_2023() -> None:
    splits = WalkForwardValidator.generate_time_splits(
        start_year=2019,
        end_year=2023,
        train_window_years=3,
        test_window_months=6,
    )
    assert len(splits) == 2

    s0, s1 = splits
    assert s0.train_start == datetime(2019, 1, 1, tzinfo=UTC)
    assert s0.train_end == datetime(2023, 1, 1, tzinfo=UTC)
    assert s0.test_start == datetime(2023, 1, 1, tzinfo=UTC)
    assert s0.test_end == datetime(2023, 7, 1, tzinfo=UTC)

    # Expanding: train grows through previous test end
    assert s1.train_start == datetime(2019, 1, 1, tzinfo=UTC)
    assert s1.train_end == datetime(2023, 7, 1, tzinfo=UTC)
    assert s1.test_start == datetime(2023, 7, 1, tzinfo=UTC)
    assert s1.test_end == datetime(2024, 1, 1, tzinfo=UTC)


def test_every_split_test_strictly_after_train() -> None:
    splits = WalkForwardValidator.generate_time_splits(
        start_year=2018,
        end_year=2024,
        train_window_years=3,
        test_window_months=6,
    )
    assert len(splits) >= 2
    for split in splits:
        assert isinstance(split, TimeSplit)
        assert split.train_start < split.train_end
        assert split.test_start < split.test_end
        # Half-open: train ends at or before test starts (no overlap / leakage)
        assert split.train_end <= split.test_start
        # Any timestamp in test is strictly after every train timestamp
        assert split.test_start >= split.train_end
        assert split.train_start.tzinfo is not None
        assert split.test_end.tzinfo is not None


def test_no_splits_when_horizon_too_short() -> None:
    splits = WalkForwardValidator.generate_time_splits(
        start_year=2020,
        end_year=2022,
        train_window_years=3,
        test_window_months=6,
    )
    # First test would start 2024-01-01, past horizon end 2023-01-01
    assert splits == []


def test_invalid_split_params() -> None:
    with pytest.raises(ValueError):
        WalkForwardValidator.generate_time_splits(2020, 2019)
    with pytest.raises(ValueError):
        WalkForwardValidator.generate_time_splits(2019, 2023, train_window_years=0)
    with pytest.raises(ValueError):
        WalkForwardValidator.generate_time_splits(2019, 2023, test_window_months=0)


def test_metrics_smoke_perfect_and_clv_drawdown() -> None:
    y = [1, 0, 1, 0]
    p = [1.0, 0.0, 1.0, 0.0]
    assert QuantMetrics.brier_score(y, p) == 0.0
    assert QuantMetrics.log_loss_score(y, p) < 1e-6

    ece = QuantMetrics.expected_calibration_error(y, p, n_bins=4)
    assert ece == pytest.approx(0.0)

    clv = QuantMetrics.closing_line_value(2.2, 2.0)
    assert clv == pytest.approx(math.log(2.2 / 2.0))

    # Peak 110 → trough 88 ⇒ drawdown 20%
    mdd = QuantMetrics.max_drawdown([100.0, 110.0, 88.0, 95.0])
    assert mdd == pytest.approx((110.0 - 88.0) / 110.0)


def test_metrics_edge_cases() -> None:
    assert math.isnan(QuantMetrics.brier_score([], []))
    assert math.isnan(QuantMetrics.log_loss_score([], []))
    assert math.isnan(QuantMetrics.expected_calibration_error([], []))
    assert math.isnan(QuantMetrics.max_drawdown([]))
    assert math.isnan(QuantMetrics.closing_line_value(0.0, 2.0))
    assert math.isnan(QuantMetrics.closing_line_value(2.0, -1.0))

    with pytest.raises(ValueError):
        QuantMetrics.brier_score([1, 0], [0.5])
