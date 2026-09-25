"""Model and data drift detectors for Quant Engine monitoring."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional, Sequence

from src.validation.metrics import QuantMetrics

JsonDict = dict[str, Any]
BetLike = Mapping[str, Any]
SnapshotLike = Mapping[str, Any]

# Alert thresholds (spec)
BRIER_INCREASE_RATIO = 0.15  # alert if recent Brier > baseline * (1 + 15%)
ECE_ALERT_THRESHOLD = 0.10
DEFAULT_WINDOW_DAYS = 30
# Data drift: score drop vs prior window mean
SCORE_DROP_RATIO = 0.15  # alert if recent mean score < baseline * (1 - 15%)
MISSING_RATE_ALERT = 0.25  # alert if fraction of missing feature values > 25%


def _parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, str):
        text = value.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    return None


def _bet_settled_at(bet: BetLike) -> Optional[datetime]:
    return _parse_dt(
        bet.get("settled_at")
        or bet.get("settled_time")
        or bet.get("created_at")
        or bet.get("timestamp")
    )


def _bet_label_prob(bet: BetLike) -> Optional[tuple[int, float]]:
    """Extract (y_true, y_prob) from a settled bet dict."""
    y_raw = bet.get("y_true", bet.get("outcome", bet.get("won", bet.get("label"))))
    p_raw = bet.get(
        "y_prob",
        bet.get("probability", bet.get("model_prob", bet.get("predicted_prob"))),
    )
    if y_raw is None or p_raw is None:
        return None
    try:
        if isinstance(y_raw, bool):
            y = int(y_raw)
        else:
            y = int(float(y_raw))
        p = float(p_raw)
    except (TypeError, ValueError):
        return None
    if y not in (0, 1) or not math.isfinite(p):
        return None
    return y, max(0.0, min(1.0, p))


def _snapshot_time(snap: SnapshotLike) -> Optional[datetime]:
    return _parse_dt(
        snap.get("as_of_time")
        or snap.get("created_at")
        or snap.get("timestamp")
        or snap.get("observed_at")
    )


def _snapshot_score(snap: SnapshotLike) -> Optional[float]:
    raw = snap.get("aggregate_data_score", snap.get("aggregate_score"))
    if raw is None:
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


def _missing_rate(snap: SnapshotLike) -> Optional[float]:
    """Fraction of missing/null feature values in a snapshot (0–1)."""
    dq = snap.get("data_quality_metrics") or snap.get("data_quality") or {}
    if isinstance(dq, Mapping):
        for key in ("missing_rate", "feature_missing_rate", "pct_missing"):
            if key in dq and dq[key] is not None:
                try:
                    rate = float(dq[key])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(rate):
                    # Accept 0–1 or 0–100 percentage
                    return rate / 100.0 if rate > 1.0 else rate

    features = snap.get("features") or snap.get("features_json") or {}
    if not isinstance(features, Mapping) or not features:
        return None
    n = len(features)
    missing = sum(
        1
        for v in features.values()
        if v is None or (isinstance(v, float) and math.isnan(v))
    )
    return missing / float(n) if n else None


def _split_windows(
    items: Sequence[Any],
    *,
    get_time,
    window_days: int,
    now: Optional[datetime] = None,
) -> tuple[list[Any], list[Any]]:
    """Split into recent ``window_days`` vs prior baseline ``window_days``."""
    now_dt = now or datetime.now(timezone.utc)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)
    recent_start = now_dt - timedelta(days=window_days)
    baseline_start = recent_start - timedelta(days=window_days)

    recent: list[Any] = []
    baseline: list[Any] = []
    for item in items:
        ts = get_time(item)
        if ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if recent_start <= ts <= now_dt:
            recent.append(item)
        elif baseline_start <= ts < recent_start:
            baseline.append(item)
    return recent, baseline


class DriftDetector:
    """Detect model calibration drift and feature / data-quality drift.

    Uses :class:`~src.validation.metrics.QuantMetrics` for Brier and ECE.
    """

    def __init__(
        self,
        *,
        window_days: int = DEFAULT_WINDOW_DAYS,
        brier_increase_ratio: float = BRIER_INCREASE_RATIO,
        ece_alert_threshold: float = ECE_ALERT_THRESHOLD,
        score_drop_ratio: float = SCORE_DROP_RATIO,
        missing_rate_alert: float = MISSING_RATE_ALERT,
        metrics: Optional[type] = None,
    ) -> None:
        self.window_days = int(window_days)
        self.brier_increase_ratio = float(brier_increase_ratio)
        self.ece_alert_threshold = float(ece_alert_threshold)
        self.score_drop_ratio = float(score_drop_ratio)
        self.missing_rate_alert = float(missing_rate_alert)
        self.metrics = metrics or QuantMetrics

    def check_model_drift(
        self,
        recent_settled_bets: list[BetLike] | Sequence[BetLike],
        *,
        now: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Rolling 30d Brier + ECE vs prior baseline window.

        Alerts when:
        - recent Brier is more than ``brier_increase_ratio`` (default 15%)
          worse than the baseline window Brier, **or**
        - recent ECE exceeds ``ece_alert_threshold`` (default 0.10).

        Expected bet keys: ``settled_at``, ``y_true``/``outcome``, ``y_prob``/
        ``probability``.
        """
        bets = list(recent_settled_bets)
        recent, baseline = _split_windows(
            bets, get_time=_bet_settled_at, window_days=self.window_days, now=now
        )

        def _pairs(subset: Sequence[BetLike]) -> tuple[list[int], list[float]]:
            ys: list[int] = []
            ps: list[float] = []
            for b in subset:
                pair = _bet_label_prob(b)
                if pair is None:
                    continue
                ys.append(pair[0])
                ps.append(pair[1])
            return ys, ps

        y_r, p_r = _pairs(recent)
        y_b, p_b = _pairs(baseline)

        brier_recent = (
            self.metrics.brier_score(y_r, p_r) if y_r else float("nan")
        )
        brier_baseline = (
            self.metrics.brier_score(y_b, p_b) if y_b else float("nan")
        )
        ece_recent = (
            self.metrics.expected_calibration_error(y_r, p_r) if y_r else float("nan")
        )
        ece_baseline = (
            self.metrics.expected_calibration_error(y_b, p_b) if y_b else float("nan")
        )

        alerts: list[JsonDict] = []
        brier_ratio: Optional[float] = None
        brier_alert = False
        if (
            math.isfinite(brier_recent)
            and math.isfinite(brier_baseline)
            and brier_baseline > 0.0
        ):
            brier_ratio = (brier_recent - brier_baseline) / brier_baseline
            if brier_ratio > self.brier_increase_ratio:
                brier_alert = True
                alerts.append(
                    {
                        "code": "MODEL_DRIFT_BRIER",
                        "message": (
                            f"Brier rose {brier_ratio:.1%} vs baseline "
                            f"(threshold {self.brier_increase_ratio:.0%})"
                        ),
                        "brier_recent": brier_recent,
                        "brier_baseline": brier_baseline,
                        "increase_ratio": brier_ratio,
                    }
                )
        elif math.isfinite(brier_recent) and (
            not math.isfinite(brier_baseline) or brier_baseline == 0.0
        ):
            # No usable baseline — still flag absolute ECE below; Brier needs baseline
            pass

        ece_alert = bool(
            math.isfinite(ece_recent) and ece_recent > self.ece_alert_threshold
        )
        if ece_alert:
            alerts.append(
                {
                    "code": "MODEL_DRIFT_ECE",
                    "message": (
                        f"ECE {ece_recent:.4f} exceeds threshold "
                        f"{self.ece_alert_threshold:.2f}"
                    ),
                    "ece_recent": ece_recent,
                    "threshold": self.ece_alert_threshold,
                }
            )

        alert = bool(alerts)
        return {
            "alert": alert,
            "window_days": self.window_days,
            "n_recent": len(y_r),
            "n_baseline": len(y_b),
            "brier_recent": brier_recent,
            "brier_baseline": brier_baseline,
            "brier_increase_ratio": brier_ratio,
            "brier_alert": brier_alert,
            "ece_recent": ece_recent,
            "ece_baseline": ece_baseline,
            "ece_alert": ece_alert,
            "ece_threshold": self.ece_alert_threshold,
            "alerts": alerts,
        }

    def check_data_drift(
        self,
        recent_feature_snapshots: list[SnapshotLike] | Sequence[SnapshotLike],
        *,
        now: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Flag missing-rate spikes and sudden drops in ``aggregate_data_score``.

        Alerts when:
        - mean missing rate in the recent window exceeds ``missing_rate_alert``,
          **or**
        - recent mean ``aggregate_data_score`` drops by more than
          ``score_drop_ratio`` (default 15%) vs the prior baseline window.
        """
        snaps = list(recent_feature_snapshots)
        recent, baseline = _split_windows(
            snaps, get_time=_snapshot_time, window_days=self.window_days, now=now
        )

        def _mean(values: Sequence[float]) -> float:
            finite = [v for v in values if math.isfinite(v)]
            if not finite:
                return float("nan")
            return float(sum(finite) / len(finite))

        miss_recent = [m for s in recent if (m := _missing_rate(s)) is not None]
        miss_baseline = [m for s in baseline if (m := _missing_rate(s)) is not None]
        scores_recent = [sc for s in recent if (sc := _snapshot_score(s)) is not None]
        scores_baseline = [
            sc for s in baseline if (sc := _snapshot_score(s)) is not None
        ]

        mean_missing_recent = _mean(miss_recent)
        mean_missing_baseline = _mean(miss_baseline)
        mean_score_recent = _mean(scores_recent)
        mean_score_baseline = _mean(scores_baseline)

        alerts: list[JsonDict] = []
        missing_alert = bool(
            math.isfinite(mean_missing_recent)
            and mean_missing_recent > self.missing_rate_alert
        )
        if missing_alert:
            alerts.append(
                {
                    "code": "DATA_DRIFT_MISSING_RATE",
                    "message": (
                        f"Missing rate {mean_missing_recent:.1%} exceeds "
                        f"{self.missing_rate_alert:.0%}"
                    ),
                    "missing_rate_recent": mean_missing_recent,
                    "threshold": self.missing_rate_alert,
                }
            )

        score_drop_ratio: Optional[float] = None
        score_alert = False
        if (
            math.isfinite(mean_score_recent)
            and math.isfinite(mean_score_baseline)
            and mean_score_baseline > 0.0
        ):
            score_drop_ratio = (
                mean_score_baseline - mean_score_recent
            ) / mean_score_baseline
            if score_drop_ratio > self.score_drop_ratio:
                score_alert = True
                alerts.append(
                    {
                        "code": "DATA_DRIFT_SCORE_DROP",
                        "message": (
                            f"aggregate_data_score dropped {score_drop_ratio:.1%} "
                            f"vs baseline (threshold {self.score_drop_ratio:.0%})"
                        ),
                        "score_recent": mean_score_recent,
                        "score_baseline": mean_score_baseline,
                        "drop_ratio": score_drop_ratio,
                    }
                )

        return {
            "alert": bool(alerts),
            "window_days": self.window_days,
            "n_recent": len(recent),
            "n_baseline": len(baseline),
            "mean_missing_rate_recent": mean_missing_recent,
            "mean_missing_rate_baseline": mean_missing_baseline,
            "missing_alert": missing_alert,
            "mean_aggregate_data_score_recent": mean_score_recent,
            "mean_aggregate_data_score_baseline": mean_score_baseline,
            "score_drop_ratio": score_drop_ratio,
            "score_alert": score_alert,
            "alerts": alerts,
        }


__all__ = [
    "DriftDetector",
    "BRIER_INCREASE_RATIO",
    "ECE_ALERT_THRESHOLD",
    "DEFAULT_WINDOW_DAYS",
]
