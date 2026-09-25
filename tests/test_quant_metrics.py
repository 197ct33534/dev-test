"""Focused unit tests for ``QuantMetrics``."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.validation import QuantMetrics


def test_log_loss_clips_extreme_probs() -> None:
    # Without clipping, p=0 / p=1 would blow up; clipped score must be finite
    score = QuantMetrics.log_loss_score([1, 0], [0.0, 1.0])
    assert math.isfinite(score)
    assert score > 0.0


def test_ece_on_calibrated_noise() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0.1, 0.9, size=400)
    y = (rng.random(400) < p).astype(int).tolist()
    ece = QuantMetrics.expected_calibration_error(y, p.tolist(), n_bins=10)
    assert 0.0 <= ece < 0.15


def test_max_drawdown_monotone_up() -> None:
    assert QuantMetrics.max_drawdown([1.0, 2.0, 3.0, 4.0]) == 0.0
