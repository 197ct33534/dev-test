"""Quantitative scoring metrics for probabilistic models and betting PnL.

Provides Brier, log-loss, ECE, closing-line value, and max drawdown
as a cohesive ``QuantMetrics`` suite (pure NumPy / math).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

_EPS = 1e-15


class QuantMetrics:
    """Stateless suite of calibration / risk metrics for walk-forward eval."""

    @staticmethod
    def brier_score(y_true: list[int] | Sequence[int], y_prob: list[float] | Sequence[float]) -> float:
        """Mean squared error between binary outcomes and predicted probabilities.

        Parameters
        ----------
        y_true :
            Binary labels in ``{0, 1}``.
        y_prob :
            Predicted probabilities of the positive class in ``[0, 1]``.

        Returns
        -------
        float
            Average (p - y)^2. ``nan`` when inputs are empty / all non-finite.
        """
        y, p = _aligned_arrays(y_true, y_prob)
        if y.size == 0:
            return float("nan")
        return float(np.mean((p - y) ** 2))

    @staticmethod
    def log_loss_score(
        y_true: list[int] | Sequence[int],
        y_prob: list[float] | Sequence[float],
        *,
        eps: float = _EPS,
    ) -> float:
        """Binary cross-entropy (natural log). Lower is better.

        Probabilities are clipped to ``[eps, 1 - eps]`` to avoid ``log(0)``.
        Returns ``nan`` when inputs are empty / all non-finite.
        """
        y, p = _aligned_arrays(y_true, y_prob)
        if y.size == 0:
            return float("nan")
        p = np.clip(p, eps, 1.0 - eps)
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    @staticmethod
    def expected_calibration_error(
        y_true: list[int] | Sequence[int],
        y_prob: list[float] | Sequence[float],
        n_bins: int = 10,
    ) -> float:
        """Weighted ECE = sum_b (n_b / N) * |p_mean_b - y_freq_b|.

        Equal-width bins on ``[0, 1]``. Empty / non-finite inputs → ``nan``.
        """
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins}")
        y, p = _aligned_arrays(y_true, y_prob)
        if y.size == 0:
            return float("nan")

        edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
        n_total = float(y.size)
        ece = 0.0
        for i in range(int(n_bins)):
            lo, hi = float(edges[i]), float(edges[i + 1])
            if i == int(n_bins) - 1:
                sel = (p >= lo) & (p <= hi)
            else:
                sel = (p >= lo) & (p < hi)
            n = int(sel.sum())
            if n == 0:
                continue
            ece += (n / n_total) * abs(float(p[sel].mean()) - float(y[sel].mean()))
        return float(ece)

    @staticmethod
    def closing_line_value(taken_odds: float, closing_odds: float) -> float:
        """CLV edge = log(taken_odds / closing_odds).

        Positive ⇒ took a better (higher) price than the closing reference.
        Non-positive or non-finite odds → ``nan``.
        """
        taken = float(taken_odds)
        closing = float(closing_odds)
        if (
            not math.isfinite(taken)
            or not math.isfinite(closing)
            or taken <= 0.0
            or closing <= 0.0
        ):
            return float("nan")
        return float(math.log(taken / closing))

    @staticmethod
    def max_drawdown(equity_curve: list[float] | Sequence[float]) -> float:
        """Largest peak-to-trough drawdown as a non-negative fraction of peak.

        For equity path E_t with running peak P_t = max_{s<=t} E_s:
        MDD = max_t (P_t - E_t) / P_t.

        Peaks <= 0 skip the relative term. Empty input → ``nan``.
        """
        arr = np.asarray(list(equity_curve), dtype=float)
        if arr.size == 0:
            return float("nan")
        finite = np.isfinite(arr)
        if not finite.any():
            return float("nan")
        values = arr[finite]

        peak = float(values[0])
        max_dd = 0.0
        for v in values:
            v_f = float(v)
            peak = max(peak, v_f)
            if peak > 0.0:
                dd = (peak - v_f) / peak
                if dd > max_dd:
                    max_dd = dd
        return float(max_dd)


def _aligned_arrays(
    y_true: Sequence[int] | Sequence[float],
    y_prob: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert to float arrays, drop non-finite pairs, enforce equal length."""
    y = np.asarray(list(y_true), dtype=float)
    p = np.asarray(list(y_prob), dtype=float)
    if y.shape != p.shape:
        raise ValueError(
            f"y_true and y_prob length mismatch: {y.size} vs {p.size}"
        )
    mask = np.isfinite(y) & np.isfinite(p)
    return y[mask], p[mask]
