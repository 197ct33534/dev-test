"""Fractional Kelly stake sizing with data/model scaling and correlation discount.

Stake pipeline (fractions of bankroll):

1. ``f_kelly = (p * odds - 1) / (odds - 1) * fraction_multiplier``
2. ``f_scaled = f_kelly * (data_score / 100) * (model_confidence / 100)``
3. Correlated-exposure discount when other markets already open on the match
4. ``final_stake = min(f_scaled, max_bet_cap_percent)``
5. Hard gates: EV < ``min_ev_threshold`` or ``data_score < 50`` → NO BET (0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

try:
    from src.config import (
        DEFAULT_KELLY_FRACTION,
        DEFAULT_MIN_EV,
        MAX_STAKE_PCT,
    )
except ImportError:  # pragma: no cover - standalone fallback
    DEFAULT_KELLY_FRACTION = 0.10
    DEFAULT_MIN_EV = 0.05
    MAX_STAKE_PCT = 0.01

# Markets that tend to move together (e.g. Home ML + Over 2.5).
_CORRELATED_MARKET_PAIRS: frozenset[frozenset[str]] = frozenset(
    {
        frozenset({"1x2", "ou"}),
        frozenset({"ml", "ou"}),
        frozenset({"1x2", "ah"}),
        frozenset({"ml", "ah"}),
        frozenset({"ou", "ah"}),
        frozenset({"1x2", "over_under"}),
        frozenset({"ml", "over_under"}),
    }
)


@dataclass(frozen=True, slots=True)
class RiskEngineConfig:
    """Configurable Kelly / risk defaults (aligned with project ``src.config``)."""

    fraction_multiplier: float = DEFAULT_KELLY_FRACTION  # 0.10 fractional Kelly
    max_bet_cap_percent: float = MAX_STAKE_PCT  # 1% bankroll hard cap
    min_ev_threshold: float = DEFAULT_MIN_EV  # 5% EV floor
    min_data_score: float = 50.0
    correlation_discount: float = 0.30  # 30% stake cut when correlated
    # When same-match bets exist but market labels are missing, still discount.
    default_same_match_discount: float = 0.30


@dataclass
class RiskEngine:
    """Size stakes via fractional Kelly with quality gates and correlation discount."""

    config: RiskEngineConfig = field(default_factory=RiskEngineConfig)

    def calculate_stake(
        self,
        probability: float,
        odds: float,
        data_score: int | float,
        model_confidence: int | float,
        existing_match_bets: list | Sequence[Any] | None,
        *,
        market: str | None = None,
        selection: str | None = None,
    ) -> dict[str, Any]:
        """Return sized stake (bankroll fraction) and diagnostic fields.

        Parameters
        ----------
        probability:
            Model win probability ``p`` in ``(0, 1]``.
        odds:
            Decimal bookmaker odds (must be ``> 1``).
        data_score:
            Aggregate data quality score on a 0–100 scale.
        model_confidence:
            Model confidence on a 0–100 scale.
        existing_match_bets:
            Bets already open / recommended on the same match (dicts or objects
            with ``market`` / ``selection`` / ``market_type`` attributes).
        market, selection:
            Optional labels for the *candidate* bet (used for correlation).
        """
        cfg = self.config
        p = float(probability)
        o = float(odds)
        ds = float(data_score)
        mc = float(model_confidence)
        existing = list(existing_match_bets or [])

        ev = p * o - 1.0 if o > 0.0 else float("-inf")

        base: dict[str, Any] = {
            "probability": p,
            "odds": o,
            "ev": ev,
            "data_score": ds,
            "model_confidence": mc,
            "f_kelly": 0.0,
            "f_scaled": 0.0,
            "correlation_discount": 0.0,
            "correlation_multiplier": 1.0,
            "f_after_correlation": 0.0,
            "max_bet_cap_percent": cfg.max_bet_cap_percent,
            "final_stake": 0.0,
            "status": "NO_BET",
            "reason": "",
            "market": market,
            "selection": selection,
        }

        if o <= 1.0 or p <= 0.0 or p > 1.0:
            base["reason"] = "invalid_probability_or_odds"
            return base

        # Full Kelly b = odds - 1; f* = (p*b - q) / b = (p*odds - 1)/(odds - 1)
        f_full = (p * o - 1.0) / (o - 1.0)
        f_kelly = max(0.0, f_full) * float(cfg.fraction_multiplier)
        base["f_kelly"] = f_kelly

        if ds < cfg.min_data_score:
            base["reason"] = "data_score_below_threshold"
            base["status"] = "NO_BET"
            return base

        if ev < cfg.min_ev_threshold:
            base["reason"] = "ev_below_threshold"
            base["status"] = "NO_BET"
            return base

        # Negative full Kelly after fraction still means no edge in sizing terms.
        if f_kelly <= 0.0:
            base["reason"] = "non_positive_kelly"
            base["status"] = "NO_BET"
            return base

        data_w = max(0.0, min(ds, 100.0)) / 100.0
        model_w = max(0.0, min(mc, 100.0)) / 100.0
        f_scaled = f_kelly * data_w * model_w
        base["f_scaled"] = f_scaled
        base["data_weight"] = data_w
        base["model_weight"] = model_w

        discount, corr_reason = self._correlation_discount(
            existing,
            candidate_market=market,
            candidate_selection=selection,
        )
        multiplier = 1.0 - discount
        f_corr = f_scaled * multiplier
        base["correlation_discount"] = discount
        base["correlation_multiplier"] = multiplier
        base["f_after_correlation"] = f_corr
        if corr_reason:
            base["correlation_reason"] = corr_reason

        final = min(f_corr, float(cfg.max_bet_cap_percent))
        capped = final < f_corr - 1e-15
        base["final_stake"] = max(0.0, final)
        base["hard_cap_applied"] = capped

        if final <= 0.0:
            base["status"] = "NO_BET"
            base["reason"] = corr_reason or "zero_stake_after_scaling"
            return base

        base["status"] = "OK"
        reasons: list[str] = []
        if discount > 0.0:
            reasons.append(corr_reason or "correlation_discount")
        if capped:
            reasons.append("hard_cap")
        base["reason"] = "+".join(reasons) if reasons else "sized"
        return base

    def _correlation_discount(
        self,
        existing_match_bets: Sequence[Any],
        *,
        candidate_market: str | None,
        candidate_selection: str | None,
    ) -> tuple[float, str]:
        """Return ``(discount_fraction, reason)``; discount 0.30 ⇒ keep 70% stake."""
        if not existing_match_bets:
            return 0.0, ""

        cfg = self.config
        cand_mkt = _normalize_market(candidate_market, candidate_selection)
        existing_mkts: list[str] = []
        for bet in existing_match_bets:
            mkt, sel = _bet_market_selection(bet)
            existing_mkts.append(_normalize_market(mkt, sel))

        if cand_mkt:
            for em in existing_mkts:
                if not em:
                    continue
                pair = frozenset({cand_mkt, em})
                if len(pair) == 1:
                    # Same market family already open → full correlation cut.
                    return cfg.correlation_discount, "same_market_overlap"
                if pair in _CORRELATED_MARKET_PAIRS:
                    return (
                        cfg.correlation_discount,
                        f"correlated_markets:{cand_mkt}+{em}",
                    )
            # Candidate labeled but no known pair — still same-match exposure.
            return (
                cfg.default_same_match_discount,
                "same_match_exposure",
            )

        # No candidate market labels: any open bet on the match → discount.
        return (
            cfg.default_same_match_discount,
            "same_match_exposure",
        )


def _bet_market_selection(bet: Any) -> tuple[str | None, str | None]:
    if isinstance(bet, Mapping):
        mkt = bet.get("market") or bet.get("market_type")
        sel = bet.get("selection")
        return (
            str(mkt) if mkt is not None else None,
            str(sel) if sel is not None else None,
        )
    mkt = getattr(bet, "market", None) or getattr(bet, "market_type", None)
    sel = getattr(bet, "selection", None)
    return (
        str(mkt) if mkt is not None else None,
        str(sel) if sel is not None else None,
    )


def _normalize_market(market: str | None, selection: str | None = None) -> str:
    """Map free-text market/selection labels to coarse families: 1x2/ml/ou/ah."""
    raw = " ".join(
        part for part in (str(market or ""), str(selection or "")) if part
    ).strip().lower()
    if not raw:
        return ""

    if any(tok in raw for tok in ("over", "under", "ou", "o/u", "over_under")):
        return "ou"
    if any(tok in raw for tok in ("ah", "asian", "handicap", "spread")):
        return "ah"
    if any(
        tok in raw
        for tok in (
            "1x2",
            "match odds",
            "moneyline",
            " ml",
            "ml ",
            "home",
            "away",
            "draw",
            "hometeam",
            "awayteam",
        )
    ) or raw in {"h", "a", "d", "home", "away", "draw", "ml", "1", "x", "2"}:
        # Prefer ml for explicit moneyline; else 1x2.
        if "ml" in raw or "moneyline" in raw:
            return "ml"
        return "1x2"
    if "corner" in raw:
        return "corners"
    return raw.replace(" ", "_")
