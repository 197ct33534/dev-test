"""Ensemble routing + light calibration (Worker 5).

Combines Dixon-Coles, LightGBM, and market (vig-removed) probability dicts
into tiered fair probabilities. Hard-gates on PIT / low data quality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Optional

from src.data.quality_monitor import HARD_GATE_CODE, ModelTier

# Outcome keys used throughout the Quant Engine 1X2 path.
OUTCOME_KEYS: tuple[str, ...] = ("H", "D", "A")

MODEL_VERSION = "ensemble_v2.1"
CALIBRATOR_VERSION = "temp_mix_stub_v1"

# Worker-5 tier thresholds (intentionally distinct from DataQualityEngine tiers).
_TIER_A_MIN = 85.0
_TIER_B_MIN = 50.0

# Mix weights: full ensemble (A) vs reduced (B: drop LGBM).
_WEIGHTS_TIER_A: dict[str, float] = {"dc": 0.40, "lgbm": 0.35, "market": 0.25}
_WEIGHTS_TIER_B: dict[str, float] = {"dc": 0.60, "lgbm": 0.00, "market": 0.40}

_EPS = 1e-12
_DEFAULT_TEMPERATURE = 1.0


def _as_prob_dict(raw: Optional[Mapping[str, Any]]) -> dict[str, float]:
    """Normalize a probability mapping onto ``H/D/A``; missing → uniform."""
    if not raw:
        return {k: 1.0 / 3.0 for k in OUTCOME_KEYS}
    out: dict[str, float] = {}
    for key in OUTCOME_KEYS:
        # Accept common aliases.
        aliases = {
            "H": ("H", "home", "1", "p_home"),
            "D": ("D", "draw", "X", "p_draw"),
            "A": ("A", "away", "2", "p_away"),
        }[key]
        val: Optional[float] = None
        for alias in aliases:
            if alias in raw and raw[alias] is not None:
                try:
                    val = float(raw[alias])
                except (TypeError, ValueError):
                    val = None
                break
        out[key] = val if val is not None else 0.0
    total = sum(out.values())
    if total <= _EPS:
        return {k: 1.0 / 3.0 for k in OUTCOME_KEYS}
    return {k: out[k] / total for k in OUTCOME_KEYS}


def _renormalize(probs: Mapping[str, float]) -> dict[str, float]:
    total = sum(float(probs.get(k, 0.0)) for k in OUTCOME_KEYS)
    if total <= _EPS:
        return {k: 1.0 / 3.0 for k in OUTCOME_KEYS}
    return {k: float(probs.get(k, 0.0)) / total for k in OUTCOME_KEYS}


def _blend(
    components: Mapping[str, Mapping[str, float]],
    weights: Mapping[str, float],
) -> dict[str, float]:
    """Weighted average of probability dicts; renormalize."""
    w_sum = sum(max(0.0, float(w)) for w in weights.values())
    if w_sum <= _EPS:
        return {k: 1.0 / 3.0 for k in OUTCOME_KEYS}
    blended: dict[str, float] = {k: 0.0 for k in OUTCOME_KEYS}
    for name, w in weights.items():
        ww = max(0.0, float(w)) / w_sum
        if ww <= _EPS or name not in components:
            continue
        for key in OUTCOME_KEYS:
            blended[key] += ww * float(components[name].get(key, 0.0))
    return _renormalize(blended)


def _apply_temperature(
    probs: Mapping[str, float],
    temperature: float,
) -> dict[str, float]:
    """Softmax temperature on log-probs (stub calibrator).

    ``T > 1`` flattens toward uniform; ``T < 1`` sharpens. ``T == 1`` is a
    no-op (up to renormalization).
    """
    t = float(temperature) if temperature and temperature > 0 else 1.0
    # Use log(p) / T then softmax; clamp to avoid log(0).
    logs = [math.log(max(_EPS, float(probs.get(k, _EPS)))) / t for k in OUTCOME_KEYS]
    m = max(logs)
    exps = [math.exp(x - m) for x in logs]
    s = sum(exps)
    return {k: exps[i] / s for i, k in enumerate(OUTCOME_KEYS)}


def _fair_lines_from_probs(probs: Mapping[str, float]) -> dict[str, float]:
    """Fair decimal odds = 1 / p (capped)."""
    lines: dict[str, float] = {}
    for k in OUTCOME_KEYS:
        p = max(_EPS, float(probs.get(k, _EPS)))
        lines[k] = round(1.0 / p, 6)
    return lines


def _max_ev_percent(
    fair_probs: Mapping[str, float],
    bookmaker_odds: Optional[Mapping[str, Any]],
) -> float:
    """Best single-outcome EV% = 100 * (p * odds - 1); 0 if no odds."""
    if not bookmaker_odds:
        return 0.0
    best = 0.0
    for key in OUTCOME_KEYS:
        raw = bookmaker_odds.get(key)
        if raw is None:
            continue
        try:
            odds = float(raw)
        except (TypeError, ValueError):
            continue
        if odds <= 1.0:
            continue
        ev = float(fair_probs.get(key, 0.0)) * odds - 1.0
        best = max(best, ev)
    return round(100.0 * best, 4)


@dataclass
class EnsembleResult:
    """Output of :meth:`EnsembleEngineV2.predict`."""

    model_tier: str
    fair_probabilities: dict[str, float]
    fair_lines: dict[str, float]
    raw_probabilities: dict[str, float]
    calibrated_probabilities: dict[str, float]
    mix_weights: dict[str, float]
    temperature: float
    aggregate_data_score: float
    pit_integrity_passed: bool
    no_bet: bool
    expected_value_percent: float = 0.0
    model_version: str = MODEL_VERSION
    calibrator_version: str = CALIBRATOR_VERSION
    model_confidence_score: Optional[float] = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def hard_gate_code(self) -> Optional[str]:
        return HARD_GATE_CODE if self.no_bet else None


class EnsembleEngineV2:
    """Tier-routed ensemble of Dixon-Coles + LightGBM + market fair probs.

    Tier rules
    ----------
    - ``pit_integrity_passed`` is False **or** ``aggregate_data_score < 50``
      → ``MODEL_TIER_X`` (NO BET)
    - ``score >= 85`` → ``MODEL_TIER_A`` full ensemble
    - ``50 <= score < 85`` → ``MODEL_TIER_B`` reduced model (DC + market)

    Calibration stub
    ----------------
    Temperature-scale the blended probs toward market fair probabilities by
    choosing ``T`` that minimizes mean absolute deviation vs market (grid
    search). Returns both ``fair_probabilities`` and ``fair_lines``.
    """

    def __init__(
        self,
        *,
        temperature_grid: Optional[list[float]] = None,
        default_temperature: float = _DEFAULT_TEMPERATURE,
    ) -> None:
        self.temperature_grid = temperature_grid or [
            0.7,
            0.85,
            1.0,
            1.15,
            1.3,
            1.5,
        ]
        self.default_temperature = float(default_temperature)

    def resolve_tier(
        self,
        *,
        pit_integrity_passed: bool,
        aggregate_data_score: float,
    ) -> str:
        """Map PIT flag + aggregate score → model tier string."""
        score = float(aggregate_data_score)
        if (not pit_integrity_passed) or score < _TIER_B_MIN:
            return ModelTier.X.value
        if score >= _TIER_A_MIN:
            return ModelTier.A.value
        return ModelTier.B.value

    def predict(
        self,
        *,
        dixon_coles: Optional[Mapping[str, Any]] = None,
        lightgbm: Optional[Mapping[str, Any]] = None,
        market: Optional[Mapping[str, Any]] = None,
        pit_integrity_passed: bool = True,
        aggregate_data_score: float = 100.0,
        bookmaker_odds: Optional[Mapping[str, Any]] = None,
        quality_flags: Optional[Mapping[str, Any]] = None,
    ) -> EnsembleResult:
        """Blend model/market probs and apply the calibrator stub.

        Parameters
        ----------
        dixon_coles, lightgbm, market:
            Probability dicts with keys ``H``/``D``/``A`` (aliases accepted).
            ``market`` should already be vig-removed fair probs.
        pit_integrity_passed:
            False forces ``MODEL_TIER_X``.
        aggregate_data_score:
            0–100 quality aggregate used for tier routing.
        bookmaker_odds:
            Optional decimal odds for EV% (best single selection).
        quality_flags:
            Optional extra flags passed through to ``meta``.
        """
        dc = _as_prob_dict(dixon_coles)
        lgbm = _as_prob_dict(lightgbm)
        mkt = _as_prob_dict(market)

        tier = self.resolve_tier(
            pit_integrity_passed=pit_integrity_passed,
            aggregate_data_score=aggregate_data_score,
        )
        no_bet = tier == ModelTier.X.value

        if tier == ModelTier.A.value:
            weights = dict(_WEIGHTS_TIER_A)
        elif tier == ModelTier.B.value:
            weights = dict(_WEIGHTS_TIER_B)
        else:
            # NO BET — still emit market-anchored probs for audit, but flag.
            weights = {"dc": 0.0, "lgbm": 0.0, "market": 1.0}

        components = {"dc": dc, "lgbm": lgbm, "market": mkt}
        raw = _blend(components, weights)

        temperature, calibrated = self._calibrate_toward_market(raw, mkt)
        fair = _renormalize(calibrated)
        lines = _fair_lines_from_probs(fair)
        ev_pct = 0.0 if no_bet else _max_ev_percent(fair, bookmaker_odds)

        confidence: Optional[float]
        if no_bet:
            confidence = 0.0
        else:
            # Soft confidence from data score (scaled 0–1 within eligible band).
            score = float(aggregate_data_score)
            confidence = round(min(1.0, max(0.0, (score - _TIER_B_MIN) / 50.0)), 4)

        return EnsembleResult(
            model_tier=tier,
            fair_probabilities=fair,
            fair_lines=lines,
            raw_probabilities=raw,
            calibrated_probabilities=fair,
            mix_weights=weights,
            temperature=temperature,
            aggregate_data_score=float(aggregate_data_score),
            pit_integrity_passed=bool(pit_integrity_passed),
            no_bet=no_bet,
            expected_value_percent=ev_pct,
            model_confidence_score=confidence,
            meta={
                "quality_flags": dict(quality_flags or {}),
                "components": {"dc": dc, "lgbm": lgbm, "market": mkt},
                "hard_gate": HARD_GATE_CODE if no_bet else None,
            },
        )

    def _calibrate_toward_market(
        self,
        blended: Mapping[str, float],
        market: Mapping[str, float],
    ) -> tuple[float, dict[str, float]]:
        """Pick temperature minimizing MAE vs market fair probs (stub)."""
        best_t = self.default_temperature
        best_probs = _apply_temperature(blended, best_t)
        best_mae = self._mae(best_probs, market)

        for t in self.temperature_grid:
            cand = _apply_temperature(blended, t)
            mae = self._mae(cand, market)
            if mae < best_mae:
                best_mae = mae
                best_t = float(t)
                best_probs = cand

        # Light pull toward market: convex mix after temperature.
        mix = 0.85
        pulled: MutableMapping[str, float] = {
            k: mix * best_probs[k] + (1.0 - mix) * float(market.get(k, 0.0))
            for k in OUTCOME_KEYS
        }
        return best_t, _renormalize(pulled)

    @staticmethod
    def _mae(a: Mapping[str, float], b: Mapping[str, float]) -> float:
        return sum(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in OUTCOME_KEYS) / 3.0
