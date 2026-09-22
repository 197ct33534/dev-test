r"""Bankroll / Kelly stake helpers used by recommender, scanner, and UI."""

from __future__ import annotations

from src.config import DEFAULT_KELLY_FRACTION, DEFAULT_MIN_EV, MAX_KELLY_FRACTION, MAX_STAKE_PCT


def expected_value(p_model: float, odds_bookmaker: float) -> float:
    """EV = (P_model × Odds) − 1."""
    p = float(p_model)
    o = float(odds_bookmaker)
    if p < 0.0 or p > 1.0:
        raise ValueError(f"p_model must be in [0, 1], got {p}")
    if o <= 0.0:
        raise ValueError(f"odds_bookmaker must be positive, got {o}")
    return (p * o) - 1.0


def full_kelly_fraction(p_model: float, odds_bookmaker: float) -> float:
    """Full Kelly f* = EV / (Odds − 1), clipped to [0, 1]."""
    o = float(odds_bookmaker)
    p = float(p_model)
    if o <= 1.0 or p <= 0.0:
        return 0.0
    ev = expected_value(p, o)
    if ev <= 0.0:
        return 0.0
    return float(max(0.0, min(1.0, ev / (o - 1.0))))


def calculate_kelly_stake(
    p_model: float,
    odds_bookmaker: float,
    bankroll: float,
    *,
    kelly_fraction: float = DEFAULT_KELLY_FRACTION,
    max_stake_pct: float = MAX_STAKE_PCT,
) -> tuple[float, float]:
    """Return ``(stake_pct, stake_amount)`` with risk-reduced Kelly.

    Uses fractional Kelly (default **10%** of full Kelly) then clips to
    ``max_stake_pct`` (default **1%** of bankroll).
    """
    frac = float(max(0.0, min(float(kelly_fraction), float(MAX_KELLY_FRACTION))))
    cap = float(max(0.0, min(1.0, float(max_stake_pct))))
    raw = full_kelly_fraction(float(p_model), float(odds_bookmaker)) * frac
    stake_pct = float(min(max(0.0, raw), cap))
    bank = float(max(0.0, bankroll))
    return stake_pct, stake_pct * bank


def capped_kelly_fraction(
    p_model: float,
    odds_bookmaker: float,
    *,
    kelly_fraction: float = DEFAULT_KELLY_FRACTION,
    max_stake_pct: float = MAX_STAKE_PCT,
) -> float:
    """Kelly stake as a fraction of bankroll (no cash conversion)."""
    stake_pct, _ = calculate_kelly_stake(
        p_model,
        odds_bookmaker,
        bankroll=1.0,
        kelly_fraction=kelly_fraction,
        max_stake_pct=max_stake_pct,
    )
    return stake_pct


def has_edge(
    p_model: float,
    odds_bookmaker: float,
    min_ev: float = DEFAULT_MIN_EV,
) -> bool:
    """True when EV ≥ ``min_ev``."""
    try:
        return expected_value(float(p_model), float(odds_bookmaker)) >= float(min_ev)
    except ValueError:
        return False
