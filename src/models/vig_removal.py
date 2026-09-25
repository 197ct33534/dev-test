"""Vig / overround removal for bookmaker decimal odds.

Converts raw decimal odds into fair (no-margin) probabilities that sum to 1.

Methods
-------
multiplicative
    Normalize implied probabilities: ``p_i = π_i / Σ π_j`` where ``π_i = 1/o_i``.
additive
    Subtract equal share of the overround: ``p_i = π_i − (Σπ − 1)/n``,
    then renormalize if any mass was clipped at 0.
shin
    Shin (1993) insider-trading model. Solves for ``z ∈ [0, 1)`` such that

    .. math::

        p_i(z) = \\frac{\\sqrt{z^2 + 4(1-z)\\,\\pi_i^2 / s} - z}{2(1-z)},
        \\quad s = \\sum_j \\pi_j,

    with ``Σ p_i(z) = 1``.
"""

from __future__ import annotations

from typing import Literal, Sequence

from scipy.optimize import brentq

MethodName = Literal["multiplicative", "additive", "shin"]

_EPS = 1e-12
_SHIN_Z_HI = 1.0 - 1e-9


def _validate_odds(odds: Sequence[float]) -> list[float]:
    if len(odds) < 2:
        raise ValueError(f"need at least 2 outcomes, got {len(odds)}")
    out: list[float] = []
    for i, o in enumerate(odds):
        v = float(o)
        if v <= 1.0:
            raise ValueError(f"odds[{i}] must be > 1.0, got {v}")
        out.append(v)
    return out


def _implied(odds: Sequence[float]) -> list[float]:
    return [1.0 / o for o in odds]


def _renormalize(probs: list[float]) -> list[float]:
    total = sum(probs)
    if total <= _EPS:
        raise ValueError("probability mass is non-positive after margin removal")
    return [p / total for p in probs]


class VigRemovalEngine:
    """Remove bookmaker overround (vig) from decimal odds."""

    def multiplicative_margin(self, odds: list[float]) -> list[float]:
        """Multiplicative (proportional) margin removal.

        Parameters
        ----------
        odds:
            Decimal odds for mutually exclusive outcomes.

        Returns
        -------
        list[float]
            Fair probabilities ``p_i = π_i / Σ π_j`` summing to 1.
        """
        o = _validate_odds(odds)
        pi = _implied(o)
        return _renormalize(pi)

    def additive_margin(self, odds: list[float]) -> list[float]:
        """Additive (equal) margin removal.

        Parameters
        ----------
        odds:
            Decimal odds for mutually exclusive outcomes.

        Returns
        -------
        list[float]
            Fair probabilities summing to 1. Negative interim masses are
            clipped to 0 then renormalized.
        """
        o = _validate_odds(odds)
        pi = _implied(o)
        n = len(pi)
        overround = sum(pi) - 1.0
        c = overround / float(n)
        raw = [max(0.0, p - c) for p in pi]
        return _renormalize(raw)

    def shin_method(self, odds: list[float]) -> list[float]:
        """Shin (1993) overround removal with insider-trading parameter ``z``.

        Parameters
        ----------
        odds:
            Decimal odds for mutually exclusive outcomes.

        Returns
        -------
        list[float]
            Fair probabilities ``p_i(z*)`` summing to ~1, where ``z*`` solves
            ``Σ p_i(z) = 1``.
        """
        o = _validate_odds(odds)
        pi = _implied(o)
        s = sum(pi)

        # Already fair (no overround) — multiplicative is exact.
        if abs(s - 1.0) < 1e-10:
            return _renormalize(pi)

        def _probs_at(z: float) -> list[float]:
            if abs(1.0 - z) < _EPS:
                return _renormalize(pi)
            denom = 2.0 * (1.0 - z)
            return [
                ( (z * z + 4.0 * (1.0 - z) * (p * p) / s) ** 0.5 - z ) / denom
                for p in pi
            ]

        def _objective(z: float) -> float:
            return sum(_probs_at(z)) - 1.0

        # At z→0, Σp ≈ √s > 1 when overround > 0; at z→1, Σp → 1 from above.
        f0 = _objective(0.0)
        f1 = _objective(_SHIN_Z_HI)
        if abs(f0) < 1e-12:
            return _probs_at(0.0)
        if f0 * f1 > 0:
            # Degenerate / underround edge: fall back to multiplicative.
            return self.multiplicative_margin(o)

        z_star = float(brentq(_objective, 0.0, _SHIN_Z_HI, xtol=1e-14))
        probs = _probs_at(z_star)
        # Tiny numerical drift — pin to exact unit simplex.
        return _renormalize(probs)

    def remove_vig(
        self,
        odds: list[float],
        method: MethodName = "shin",
    ) -> list[float]:
        """Dispatch to a named margin-removal method."""
        if method == "multiplicative":
            return self.multiplicative_margin(odds)
        if method == "additive":
            return self.additive_margin(odds)
        if method == "shin":
            return self.shin_method(odds)
        raise ValueError(
            f"unknown method={method!r}; expected multiplicative|additive|shin"
        )

    def convert_ah_ou_to_fair_prob(
        self,
        market_type: str,
        line: float,
        odds_1: float,
        odds_2: float,
        method: str = "shin",
    ) -> dict:
        """Convert a two-way AH or O/U book into fair probabilities.

        Parameters
        ----------
        market_type:
            ``\"AH\"`` / ``\"asian_handicap\"`` or ``\"OU\"`` / ``\"over_under\"``.
        line:
            Handicap (home perspective) or total-goals line.
        odds_1:
            Decimal odds for side 1 — AH home cover, or Over.
        odds_2:
            Decimal odds for side 2 — AH away cover, or Under.
        method:
            ``multiplicative`` | ``additive`` | ``shin`` (default).

        Returns
        -------
        dict
            Fair probs (sum ≈ 1), fair odds, and metadata. Keys depend on
            ``market_type``.
        """
        mt = str(market_type).strip().lower()
        method_key: MethodName
        m = str(method).strip().lower()
        if m in ("multiplicative", "mult", "proportional"):
            method_key = "multiplicative"
        elif m in ("additive", "add", "equal"):
            method_key = "additive"
        elif m == "shin":
            method_key = "shin"
        else:
            raise ValueError(f"unknown method={method!r}")

        fair = self.remove_vig([float(odds_1), float(odds_2)], method=method_key)
        p1, p2 = fair[0], fair[1]
        line_f = float(line)

        base = {
            "market_type": mt,
            "line": line_f,
            "method": method_key,
            "odds_1": float(odds_1),
            "odds_2": float(odds_2),
            "fair_prob_1": p1,
            "fair_prob_2": p2,
            "fair_odds_1": (1.0 / p1) if p1 > _EPS else float("inf"),
            "fair_odds_2": (1.0 / p2) if p2 > _EPS else float("inf"),
            "sum_fair_prob": p1 + p2,
        }

        if mt in ("ah", "asian_handicap", "asian handicap", "handicap"):
            base.update(
                {
                    "p_home": p1,
                    "p_away": p2,
                    "selection_1": "home",
                    "selection_2": "away",
                }
            )
        elif mt in ("ou", "o/u", "over_under", "over/under", "totals", "total"):
            base.update(
                {
                    "p_over": p1,
                    "p_under": p2,
                    "selection_1": "over",
                    "selection_2": "under",
                }
            )
        else:
            raise ValueError(
                f"unknown market_type={market_type!r}; expected AH or OU"
            )

        return base
