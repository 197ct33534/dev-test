r"""Bankroll / Kelly stake helpers used by recommender, scanner, and UI.

Also hosts model fair-line pricing (O/U + AH) from a Dixon–Coles score
matrix for line-disparity ranking and API/UI display.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_KELLY_FRACTION,
    DEFAULT_MIN_EV,
    LONGSHOT_ODDS_MIN,
    LONGSHOT_P_MAX,
    MAX_BETS_PER_DAY,
    MAX_EV,
    MAX_KELLY_FRACTION,
    MAX_STAKE_PCT,
)

# Common Asian quarter grids used when searching the 50%-cover fair line.
COMMON_OU_LINES: tuple[float, ...] = (
    1.5,
    1.75,
    2.0,
    2.25,
    2.5,
    2.75,
    3.0,
    3.25,
    3.5,
    3.75,
    4.0,
    4.25,
    4.5,
)
COMMON_AH_LINES: tuple[float, ...] = tuple(
    round(x * 0.25, 2) for x in range(-12, 13)
)  # −3.0 … +3.0

# Secondary-rank blend (does **not** change EV / Kelly). See
# :func:`line_disparity_score`.
_DISPARITY_ABS_WEIGHT = 0.25


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


# ---------------------------------------------------------------------------
# Model fair-line pricing (vectorised over score matrix)
# ---------------------------------------------------------------------------


def _split_quarter_line(line: float) -> tuple[float, float]:
    """Mirror of dixon_coles.split_handicap_line (local to avoid circular import)."""
    frac = abs(float(line)) % 1.0
    is_q = abs(frac - 0.25) < 1e-9 or abs(frac - 0.75) < 1e-9
    if not is_q:
        L = float(line)
        return (L, L)
    lower = float(np.floor(line * 2.0) / 2.0)
    upper = float(np.ceil(line * 2.0) / 2.0)
    return (lower, upper)


def _effective_cover_from_ranks(mat: np.ndarray, avg_rank: np.ndarray) -> float:
    """Map combined settlement ranks → cover mass ``win + 0.5·half_win``.

    Rank convention matches :mod:`src.dixon_coles`: win=2, half_win=1, push=0,
    half_lose=-1, lose=-2. Cover weight = ``max(0, rank/2)``.
    """
    weight = np.maximum(0.0, avg_rank / 2.0)
    return float(np.sum(mat * weight))


def over_cover_from_matrix(prob_matrix: np.ndarray, line: float) -> float:
    """Effective P(Over covers) on ``line`` from a score matrix (vectorised)."""
    mat = np.asarray(prob_matrix, dtype=float)
    g = mat.shape[0] - 1
    totals = np.add.outer(np.arange(g + 1, dtype=float), np.arange(g + 1, dtype=float))
    lo, hi = _split_quarter_line(float(line))

    def _atomic_rank(L: float) -> np.ndarray:
        return np.where(totals > L, 2.0, np.where(totals < L, -2.0, 0.0))

    if abs(lo - hi) < 1e-12:
        return _effective_cover_from_ranks(mat, _atomic_rank(lo))
    avg = (_atomic_rank(lo) + _atomic_rank(hi)) / 2.0
    return _effective_cover_from_ranks(mat, avg)


def ah_home_cover_from_matrix(prob_matrix: np.ndarray, handicap: float) -> float:
    """Effective P(AH Home covers) on home ``handicap`` from a score matrix."""
    mat = np.asarray(prob_matrix, dtype=float)
    g = mat.shape[0] - 1
    xs = np.arange(g + 1, dtype=float)
    # margin = X − Y + h  → broadcast (x, y)
    margin0 = xs[:, None] - xs[None, :]
    lo, hi = _split_quarter_line(float(handicap))

    def _atomic_rank(h: float) -> np.ndarray:
        margin = margin0 + h
        return np.where(margin > 0.0, 2.0, np.where(margin < 0.0, -2.0, 0.0))

    if abs(lo - hi) < 1e-12:
        return _effective_cover_from_ranks(mat, _atomic_rank(lo))
    avg = (_atomic_rank(lo) + _atomic_rank(hi)) / 2.0
    return _effective_cover_from_ranks(mat, avg)


def _nearest_50pct_line(
    cover_fn,
    lines: Sequence[float],
    *,
    prefer_near: float | None = None,
) -> tuple[float, float]:
    """Return ``(line, p_cover)`` where ``|p_cover − 0.5|`` is smallest.

    Ties break toward ``prefer_near`` (e.g. ``λ_h+λ_a`` for O/U), then
    toward the line closest to zero.
    """
    best_line = float(lines[0])
    best_p = float(cover_fn(best_line))
    best_gap = abs(best_p - 0.5)
    best_pref = (
        abs(best_line - float(prefer_near))
        if prefer_near is not None
        else abs(best_line)
    )

    for L in lines[1:]:
        p = float(cover_fn(float(L)))
        gap = abs(p - 0.5)
        pref = (
            abs(float(L) - float(prefer_near))
            if prefer_near is not None
            else abs(float(L))
        )
        better = gap < best_gap - 1e-12 or (
            abs(gap - best_gap) <= 1e-12
            and (
                pref < best_pref - 1e-12
                or (abs(pref - best_pref) <= 1e-12 and abs(float(L)) < abs(best_line))
            )
        )
        if better:
            best_line, best_p, best_gap, best_pref = float(L), p, gap, pref
    return best_line, best_p


def _safe_fair_odds(p: float) -> float | None:
    p = float(p)
    if p <= 1e-12 or p > 1.0:
        return None
    return float(1.0 / p)


@dataclass
class ModelFairLines:
    """No-vig fair lines derived from λ + Dixon–Coles score matrix.

    Attributes
    ----------
    fair_total_goals:
        ``λ_home + λ_away``.
    fair_ou_line:
        Common O/U line whose effective P(Over) is closest to 50%.
    fair_ah_line:
        Common home AH line whose effective P(Home covers) is closest to 50%.
    fair_odds:
        Outcome → ``1/P`` (no-vig): ``H``/``D``/``A``, ``Over``/``Under`` at
        ``fair_ou_line``, ``AH_Home``/``AH_Away`` at ``fair_ah_line``.
    p_over_at_fair / p_ah_home_at_fair:
        Cover probs at the chosen fair lines (for diagnostics).
    """

    fair_total_goals: float
    fair_ou_line: float
    fair_ah_line: float
    fair_odds: dict[str, float] = field(default_factory=dict)
    p_over_at_fair: float = 0.5
    p_ah_home_at_fair: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def calculate_model_fair_lines(
    lambda_home: float,
    lambda_away: float,
    prob_matrix: np.ndarray | None = None,
    *,
    ou_candidates: Sequence[float] = COMMON_OU_LINES,
    ah_candidates: Sequence[float] = COMMON_AH_LINES,
) -> ModelFairLines:
    """Derive fair O/U + AH lines from expected goals and a score matrix.

    Parameters
    ----------
    lambda_home, lambda_away:
        Model expected goals (λ, μ).
    prob_matrix:
        ``P(X=x, Y=y)`` of shape ``(G+1, G+1)``. When ``None``, a plain
        independent Poisson matrix (no τ) is built from λ, μ (G=10) — prefer
        passing the Dixon–Coles matrix from the caller.
    ou_candidates / ah_candidates:
        Grids searched for the 50%-cover fair line.

    Notes
    -----
    All settlement uses the same Asian quarter-line rules as
    :meth:`DixonColesModel.predict_over_under` / ``predict_asian_handicap``,
    vectorised with NumPy (target **&lt;5 ms** per match on a 11×11 matrix).
    """
    lam = float(lambda_home)
    mu = float(lambda_away)
    fair_total = lam + mu

    if prob_matrix is None:
        from scipy.stats import poisson as _pois

        g = 10
        px = _pois.pmf(np.arange(g + 1), max(lam, 1e-9))
        py = _pois.pmf(np.arange(g + 1), max(mu, 1e-9))
        mat = np.outer(px, py)
        s = float(mat.sum())
        mat = mat / s if s > 0 else mat
    else:
        mat = np.asarray(prob_matrix, dtype=float)
        s = float(mat.sum())
        if s > 0 and abs(s - 1.0) > 1e-6:
            mat = mat / s

    fair_ou, p_over = _nearest_50pct_line(
        lambda L: over_cover_from_matrix(mat, L),
        tuple(float(x) for x in ou_candidates),
        prefer_near=fair_total,
    )
    # Fair AH ≈ −(λ_h − λ_a): home favourite → negative home handicap.
    fair_ah, p_ah_h = _nearest_50pct_line(
        lambda H: ah_home_cover_from_matrix(mat, H),
        tuple(float(x) for x in ah_candidates),
        prefer_near=float(mu - lam),
    )

    p_h = float(np.tril(mat, k=-1).sum())
    p_d = float(np.trace(mat))
    p_a = float(np.triu(mat, k=1).sum())
    p_under = _under_cover_from_matrix(mat, fair_ou)
    p_ah_a = _ah_away_cover_from_matrix(mat, fair_ah)

    fair_odds: dict[str, float] = {}
    for key, p in (
        ("H", p_h),
        ("D", p_d),
        ("A", p_a),
        ("Over", float(p_over)),
        ("Under", float(p_under)),
        ("AH_Home", float(p_ah_h)),
        ("AH_Away", float(p_ah_a)),
    ):
        fo = _safe_fair_odds(p)
        if fo is not None:
            fair_odds[key] = fo

    return ModelFairLines(
        fair_total_goals=float(fair_total),
        fair_ou_line=float(fair_ou),
        fair_ah_line=float(fair_ah),
        fair_odds=fair_odds,
        p_over_at_fair=float(p_over),
        p_ah_home_at_fair=float(p_ah_h),
    )


def _under_cover_from_matrix(prob_matrix: np.ndarray, line: float) -> float:
    """Effective P(Under covers) — mirror of :func:`over_cover_from_matrix`."""
    mat = np.asarray(prob_matrix, dtype=float)
    g = mat.shape[0] - 1
    totals = np.add.outer(np.arange(g + 1, dtype=float), np.arange(g + 1, dtype=float))
    lo, hi = _split_quarter_line(float(line))

    def _atomic_rank(L: float) -> np.ndarray:
        return np.where(totals < L, 2.0, np.where(totals > L, -2.0, 0.0))

    if abs(lo - hi) < 1e-12:
        return _effective_cover_from_ranks(mat, _atomic_rank(lo))
    avg = (_atomic_rank(lo) + _atomic_rank(hi)) / 2.0
    return _effective_cover_from_ranks(mat, avg)


def _ah_away_cover_from_matrix(prob_matrix: np.ndarray, handicap: float) -> float:
    """Away AH cover = mirror of home settlement on the same home handicap."""
    mat = np.asarray(prob_matrix, dtype=float)
    g = mat.shape[0] - 1
    xs = np.arange(g + 1, dtype=float)
    margin0 = xs[:, None] - xs[None, :]
    lo, hi = _split_quarter_line(float(handicap))

    def _atomic_home_rank(h: float) -> np.ndarray:
        margin = margin0 + h
        return np.where(margin > 0.0, 2.0, np.where(margin < 0.0, -2.0, 0.0))

    # Away rank = −home rank (win↔lose, half_win↔half_lose).
    if abs(lo - hi) < 1e-12:
        return _effective_cover_from_ranks(mat, -_atomic_home_rank(lo))
    avg_home = (_atomic_home_rank(lo) + _atomic_home_rank(hi)) / 2.0
    return _effective_cover_from_ranks(mat, -avg_home)


def line_disparity_score(
    *,
    ou_line_delta: float | None,
    ah_line_delta: float | None,
    market: str | None,
    ev: float | None = None,
    abs_weight: float = _DISPARITY_ABS_WEIGHT,
) -> float:
    """Secondary ranking score from line disparity (+ optional EV blend).

    Formula
    -------
    Let ``δ`` be the market-relevant delta:

    * OU → ``ou_line_delta``
    * AH → ``ah_line_delta``
    * else → average of available absolute deltas (sign of the larger)

    Then::

        line_disparity_score = |δ| · abs_weight + max(ev, 0)

    with ``abs_weight = 0.25`` by default. **Does not** alter Kelly or the
    EV≥threshold filter — only used as a secondary sort / display rank.
    """
    mkt = str(market or "").strip().upper()
    ou_d = float(ou_line_delta) if ou_line_delta is not None else None
    ah_d = float(ah_line_delta) if ah_line_delta is not None else None

    if mkt == "OU" and ou_d is not None:
        delta = ou_d
    elif mkt == "AH" and ah_d is not None:
        delta = ah_d
    else:
        parts = [d for d in (ou_d, ah_d) if d is not None]
        if not parts:
            delta = 0.0
        elif len(parts) == 1:
            delta = parts[0]
        else:
            # Keep sign of the larger-magnitude delta.
            delta = parts[0] if abs(parts[0]) >= abs(parts[1]) else parts[1]

    ev_term = max(0.0, float(ev)) if ev is not None and ev == ev else 0.0
    return float(abs(delta) * float(abs_weight) + ev_term)


def attach_fair_line_fields(
    row: dict[str, Any],
    fair: ModelFairLines,
    *,
    bookie_ou_line: float | None = None,
    bookie_ah_line: float | None = None,
) -> dict[str, Any]:
    """Mutate/return ``row`` with fair-line + disparity + display labels.

    Adds:
    ``fair_total_goals``, ``fair_ou_line``, ``fair_ah_line``, ``fair_odds_1x2``,
    ``ou_line_delta``, ``ah_line_delta``, ``line_disparity_score``,
    ``model_fair_line``, ``bookie_market_line``, ``line_edge``.
    """
    market = str(row.get("market") or "").strip().upper()
    ou_delta = None
    ah_delta = None
    if bookie_ou_line is not None and bookie_ou_line == bookie_ou_line:
        ou_delta = float(fair.fair_ou_line) - float(bookie_ou_line)
    if bookie_ah_line is not None and bookie_ah_line == bookie_ah_line:
        ah_delta = float(fair.fair_ah_line) - float(bookie_ah_line)

    ev = None
    try:
        if row.get("ev") is not None and row.get("ev") == row.get("ev"):
            ev = float(row["ev"])
        elif row.get("ev_pct") is not None:
            ev = float(row["ev_pct"]) / 100.0
    except (TypeError, ValueError):
        ev = None

    score = line_disparity_score(
        ou_line_delta=ou_delta,
        ah_line_delta=ah_delta,
        market=market,
        ev=ev,
    )

    model_lbl, bookie_lbl, edge_lbl = _format_line_edge_labels(
        market=market,
        fair=fair,
        bookie_ou_line=bookie_ou_line,
        bookie_ah_line=bookie_ah_line,
        ou_delta=ou_delta,
        ah_delta=ah_delta,
    )

    row["fair_total_goals"] = float(fair.fair_total_goals)
    row["fair_ou_line"] = float(fair.fair_ou_line)
    row["fair_ah_line"] = float(fair.fair_ah_line)
    row["fair_odds_map"] = dict(fair.fair_odds)
    row["bookie_ou_line"] = float(bookie_ou_line) if bookie_ou_line is not None else None
    row["bookie_ah_line"] = float(bookie_ah_line) if bookie_ah_line is not None else None
    row["ou_line_delta"] = ou_delta
    row["ah_line_delta"] = ah_delta
    row["line_disparity_score"] = float(score)
    row["model_fair_line"] = model_lbl
    row["bookie_market_line"] = bookie_lbl
    row["line_edge"] = edge_lbl
    return row


def _fmt_line_num(x: float) -> str:
    """Compact line formatting: 2.5 → '2.5', 3 → '3'."""
    return f"{float(x):g}"


def _fmt_ah_num(x: float) -> str:
    return f"{float(x):+g}"


def _format_line_edge_labels(
    *,
    market: str,
    fair: ModelFairLines,
    bookie_ou_line: float | None,
    bookie_ah_line: float | None,
    ou_delta: float | None,
    ah_delta: float | None,
) -> tuple[str | None, str | None, str | None]:
    """Return ``(model_fair_line, bookie_market_line, line_edge)`` VI labels."""
    mkt = str(market or "").upper()

    def _ou_labels() -> tuple[str | None, str | None, str | None]:
        model = f"Tài Xỉu {_fmt_line_num(fair.fair_ou_line)}"
        bookie = (
            f"Tài Xỉu {_fmt_line_num(bookie_ou_line)}"
            if bookie_ou_line is not None and bookie_ou_line == bookie_ou_line
            else None
        )
        edge = f"{ou_delta:+.2f} bàn" if ou_delta is not None else None
        return model, bookie, edge

    def _ah_labels() -> tuple[str | None, str | None, str | None]:
        model = f"AH {_fmt_ah_num(fair.fair_ah_line)}"
        bookie = (
            f"AH {_fmt_ah_num(bookie_ah_line)}"
            if bookie_ah_line is not None and bookie_ah_line == bookie_ah_line
            else None
        )
        edge = f"{ah_delta:+.2f} bàn" if ah_delta is not None else None
        return model, bookie, edge

    if mkt == "OU":
        return _ou_labels()
    if mkt == "AH":
        return _ah_labels()
    # 1X2 / other: prefer OU context when bookie totals exist, else AH.
    if bookie_ou_line is not None and bookie_ou_line == bookie_ou_line:
        return _ou_labels()
    if bookie_ah_line is not None and bookie_ah_line == bookie_ah_line:
        return _ah_labels()
    return (
        f"Tài Xỉu {_fmt_line_num(fair.fair_ou_line)}",
        None,
        None,
    )


def match_group_key(row: Mapping[str, Any] | pd.Series) -> str:
    """Stable match key for correlation filtering.

    Prefer ``match_id``; else ``home|away|YYYY-MM-DD`` from kickoff / match_date.
    """
    if isinstance(row, pd.Series):
        get = row.get
    else:
        get = row.get  # type: ignore[assignment]

    mid = get("match_id")
    if mid is not None and not (isinstance(mid, float) and pd.isna(mid)):
        text = str(mid).strip()
        if text:
            return text

    home = str(get("home_team") or get("HomeTeam") or get("home") or "").strip()
    away = str(get("away_team") or get("AwayTeam") or get("away") or "").strip()
    kick = get("kickoff") or get("match_date") or get("Kickoff") or get("Date")
    day = ""
    if kick is not None and not (isinstance(kick, float) and pd.isna(kick)):
        try:
            day = pd.Timestamp(kick).strftime("%Y-%m-%d")
        except Exception:
            text = str(kick).strip()
            day = text[:10] if len(text) >= 10 else text
    return f"{home}|{away}|{day}"


def _row_ev(row: Mapping[str, Any] | pd.Series) -> float:
    """EV as a fraction (0.05 = 5%), preferring ``ev`` then ``ev_pct``."""
    if isinstance(row, pd.Series):
        get = row.get
    else:
        get = row.get  # type: ignore[assignment]
    if get("ev") is not None and not (isinstance(get("ev"), float) and pd.isna(get("ev"))):
        try:
            return float(get("ev"))
        except (TypeError, ValueError):
            pass
    if get("ev_pct") is not None and not (
        isinstance(get("ev_pct"), float) and pd.isna(get("ev_pct"))
    ):
        try:
            return float(get("ev_pct")) / 100.0
        except (TypeError, ValueError):
            pass
    return float("-inf")


def _row_disparity_sort(row: Mapping[str, Any] | pd.Series) -> float:
    """Secondary sort key: prefer precomputed ``line_disparity_score``."""
    if isinstance(row, pd.Series):
        get = row.get
    else:
        get = row.get  # type: ignore[assignment]
    raw = get("line_disparity_score")
    if raw is not None and not (isinstance(raw, float) and pd.isna(raw)):
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    return 0.0


def _row_odds(row: Mapping[str, Any] | pd.Series) -> float | None:
    if isinstance(row, pd.Series):
        get = row.get
    else:
        get = row.get  # type: ignore[assignment]
    for key in ("bookmaker_odds", "odds"):
        val = get(key)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return None


def _row_p_model(row: Mapping[str, Any] | pd.Series) -> float | None:
    if isinstance(row, pd.Series):
        get = row.get
    else:
        get = row.get  # type: ignore[assignment]
    val = get("p_model")
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _is_1x2_row(row: Mapping[str, Any] | pd.Series) -> bool:
    if isinstance(row, pd.Series):
        get = row.get
    else:
        get = row.get  # type: ignore[assignment]
    market = str(get("market") or "").strip().upper()
    if market in {"1X2", "MATCH_ODDS", "HDA"}:
        return True
    sel = str(get("selection") or "").strip().lower()
    return sel in {"home", "draw", "away", "h", "d", "a"}


def is_sane_value_bet(
    row: Mapping[str, Any] | pd.Series,
    *,
    max_ev: float = MAX_EV,
    longshot_odds: float = LONGSHOT_ODDS_MIN,
    longshot_p: float = LONGSHOT_P_MAX,
) -> bool:
    """Return False for inflated EV or overconfident 1X2 longshots.

    Rules
    -----
    - Drop when EV > ``max_ev`` (default 50%).
    - For 1X2: drop when odds > ``longshot_odds`` (15) **and** model p >
      ``longshot_p`` (10%) — overconfident longshot.
    """
    ev = _row_ev(row)
    if ev == float("-inf"):
        return True  # placeholders / missing EV — leave for other filters
    if ev > float(max_ev):
        return False
    if _is_1x2_row(row):
        odds = _row_odds(row)
        p = _row_p_model(row)
        if (
            odds is not None
            and p is not None
            and odds > float(longshot_odds)
            and p > float(longshot_p)
        ):
            return False
    return True


def apply_sanity_filters(
    bets: pd.DataFrame,
    *,
    max_ev: float = MAX_EV,
    longshot_odds: float = LONGSHOT_ODDS_MIN,
    longshot_p: float = LONGSHOT_P_MAX,
) -> pd.DataFrame:
    """Drop rows failing :func:`is_sane_value_bet` (pre Top-N / auto_scanner)."""
    if bets is None or bets.empty:
        return bets.copy() if isinstance(bets, pd.DataFrame) else pd.DataFrame()
    mask = bets.apply(
        lambda r: is_sane_value_bet(
            r, max_ev=max_ev, longshot_odds=longshot_odds, longshot_p=longshot_p
        ),
        axis=1,
    )
    return bets.loc[mask].copy()


def select_value_bets(
    bets: pd.DataFrame | None,
    *,
    max_per_day: int = MAX_BETS_PER_DAY,
    already_today: int = 0,
    allow_multi_picks_per_match: bool = True,
    one_per_match: bool | None = None,
    apply_sanity: bool = True,
) -> pd.DataFrame:
    """Risk filter: sanity → optional one-bet-per-match → Top-N by EV.

    Parameters
    ----------
    bets:
        Candidate recommendations (any markets). Empty / None → empty frame.
    max_per_day:
        Hard daily exposure / display cap (default :data:`MAX_BETS_PER_DAY` = 5).
        Callers (e.g. Top-20 UI) may pass a larger ``max_per_day``.
    already_today:
        Bets already placed / journalled on the VN calendar day. Remaining
        slots = ``max(0, max_per_day - already_today)``.
    allow_multi_picks_per_match:
        When True (default), keep multiple markets/legs on the same match and
        rank purely by EV descending. When False, keep only the single
        highest-EV selection per ``match_id`` (fallback home|away|date).
    one_per_match:
        Backward-compatible override. When set, ``True`` forces one pick per
        match (staking / auto_scanner); ``False`` allows multi. Takes
        precedence over ``allow_multi_picks_per_match`` when not ``None``.
    apply_sanity:
        When True (default), drop EV > 50% and overconfident 1X2 longshots
        before ranking so Top-20 / auto_scanner stay in a realistic EV band.

    Returns
    -------
    DataFrame
        Filtered bets sorted by EV descending (at most ``slots`` rows).
    """
    if bets is None or (isinstance(bets, pd.DataFrame) and bets.empty):
        return pd.DataFrame() if bets is None else bets.copy()

    out = bets.copy()
    if apply_sanity:
        out = apply_sanity_filters(out)
        if out.empty:
            return out

    slots = int(max(0, int(max_per_day) - int(max(0, already_today))))
    if slots <= 0:
        return out.iloc[0:0].copy()

    # Compat: explicit ``one_per_match`` overrides the multi-picks flag.
    if one_per_match is not None:
        multi = not bool(one_per_match)
    else:
        multi = bool(allow_multi_picks_per_match)

    if not multi:
        out = out.assign(
            _match_key=out.apply(match_group_key, axis=1),
            _ev_sort=out.apply(_row_ev, axis=1),
            _disp_sort=out.apply(_row_disparity_sort, axis=1),
        )
        out = (
            out.sort_values(
                ["_ev_sort", "_disp_sort"], ascending=[False, False]
            )
            .drop_duplicates(subset=["_match_key"], keep="first")
            .reset_index(drop=True)
        )
    else:
        out = out.assign(
            _ev_sort=out.apply(_row_ev, axis=1),
            _disp_sort=out.apply(_row_disparity_sort, axis=1),
        )
        out = out.sort_values(
            ["_ev_sort", "_disp_sort"], ascending=[False, False]
        ).reset_index(drop=True)

    out = out.head(slots).reset_index(drop=True)
    return out.drop(
        columns=[c for c in ("_match_key", "_ev_sort", "_disp_sort") if c in out.columns]
    )


def _insight_get(
    data: Mapping[str, Any] | pd.Series,
    *keys: str,
) -> Any:
    """First present non-null value among ``keys``."""
    if isinstance(data, pd.Series):
        for key in keys:
            if key not in data.index:
                continue
            val = data.get(key)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                continue
            if isinstance(val, str) and not val.strip():
                continue
            return val
        return None
    for key in keys:
        val = data.get(key)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        if isinstance(val, str) and not val.strip():
            continue
        return val
    return None


def _insight_float(
    data: Mapping[str, Any] | pd.Series,
    *keys: str,
) -> float | None:
    """Parse first float among ``keys``, else ``None``."""
    raw = _insight_get(data, *keys)
    if raw is None:
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if val != val:  # NaN
        return None
    return val


def _insight_truthy(val: Any) -> bool:
    """True for 1 / True / '1' style rotation flags."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    if isinstance(val, (int, float)):
        return float(val) >= 0.5
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "yes", "y"}
    return bool(val)


def _fitness_insight_line(
    home: str,
    away: str,
    data: Mapping[str, Any] | pd.Series,
) -> str | None:
    """Compare rest_days / matches_last_14d (and optional rotation risk)."""
    h_rest = _insight_float(
        data, "home_rest_days", "rest_days_h", "rest_days_home"
    )
    a_rest = _insight_float(
        data, "away_rest_days", "rest_days_a", "rest_days_away"
    )
    h_n14 = _insight_float(
        data, "home_matches_last_14d", "matches_last_14d_h", "matches_last_14d_home"
    )
    a_n14 = _insight_float(
        data, "away_matches_last_14d", "matches_last_14d_a", "matches_last_14d_away"
    )
    h_rot = _insight_truthy(
        _insight_get(
            data, "home_is_rotation_risk", "is_rotation_risk_h", "is_rotation_risk_home"
        )
    )
    a_rot = _insight_truthy(
        _insight_get(
            data, "away_is_rotation_risk", "is_rotation_risk_a", "is_rotation_risk_away"
        )
    )

    bits: list[str] = []
    if h_rest is not None:
        bits.append(f"{home} nghỉ {int(h_rest)} ngày")
    if a_rest is not None:
        bits.append(f"{away} nghỉ {int(a_rest)} ngày")
    if h_n14 is not None and h_n14 >= 3:
        bits.append(f"{home} cày {int(h_n14)} trận/14 ngày")
    if a_n14 is not None and a_n14 >= 3:
        bits.append(f"{away} cày {int(a_n14)} trận/14 ngày")

    verdict = ""
    if h_rest is not None and a_rest is not None:
        gap = h_rest - a_rest
        if gap >= 2:
            verdict = f" — {home} tươi hơn"
        elif gap <= -2:
            verdict = f" — {away} tươi hơn"
    elif h_n14 is not None and a_n14 is not None:
        if h_n14 >= 3 and a_n14 < 3:
            verdict = f" — {home} nặng lịch hơn"
        elif a_n14 >= 3 and h_n14 < 3:
            verdict = f" — {away} nặng lịch hơn"

    rot_bits: list[str] = []
    if h_rot:
        rot_bits.append(home)
    if a_rot:
        rot_bits.append(away)
    rot_note = ""
    if rot_bits:
        rot_note = f" · ⚠️ {'/'.join(rot_bits)} rủi ro xoay tua"

    if not bits and not rot_note:
        return None
    if not bits:
        return f"Thể lực:{rot_note}" if rot_note else None
    body = " · ".join(bits[:3])
    return f"Thể lực: {body}{verdict}{rot_note}"


def _form_insight_line(
    home: str,
    away: str,
    data: Mapping[str, Any] | pd.Series,
) -> str | None:
    """Compare rolling xG / goals / DC attack when available."""
    h_xg = _insight_float(
        data,
        "home_rolling_xg_5",
        "rolling_xg_home",
        "rolling_xg_h",
        "rolling_xg_home_r5",
    )
    a_xg = _insight_float(
        data,
        "away_rolling_xg_5",
        "rolling_xg_away",
        "rolling_xg_a",
        "rolling_xg_away_r5",
    )
    if h_xg is not None and a_xg is not None:
        if abs(h_xg - a_xg) < 0.15:
            return (
                f"Sức mạnh: xG5 {home} {h_xg:.1f} ≈ {away} {a_xg:.1f} — ngang phong độ"
            )
        if h_xg > a_xg:
            return (
                f"Sức mạnh: xG5 {home} {h_xg:.1f} > {away} {a_xg:.1f} — "
                f"{home} đang tạo cơ hội tốt hơn"
            )
        return (
            f"Sức mạnh: xG5 {away} {a_xg:.1f} > {home} {h_xg:.1f} — "
            f"{away} đang tạo cơ hội tốt hơn"
        )

    h_gf = _insight_float(
        data, "home_rolling_gf_5", "rolling_gf_home", "rolling_gf_h"
    )
    a_gf = _insight_float(
        data, "away_rolling_gf_5", "rolling_gf_away", "rolling_gf_a"
    )
    if h_gf is not None and a_gf is not None:
        if abs(h_gf - a_gf) < 0.2:
            return f"Phong độ: bàn/5 trận {home} {h_gf:.1f} ≈ {away} {a_gf:.1f}"
        leader = home if h_gf > a_gf else away
        return (
            f"Phong độ: bàn/5 trận {home} {h_gf:.1f} vs {away} {a_gf:.1f} — "
            f"{leader} ghi bàn tốt hơn gần đây"
        )

    h_atk = _insight_float(data, "home_attack", "attack_home", "HomeAttack")
    a_atk = _insight_float(data, "away_attack", "attack_away", "AwayAttack")
    if h_atk is not None and a_atk is not None:
        if abs(h_atk - a_atk) < 0.05:
            return f"Sức mạnh: attack DC {home} ≈ {away}"
        leader = home if h_atk > a_atk else away
        return (
            f"Sức mạnh: attack DC {home} {h_atk:.2f} vs {away} {a_atk:.2f} — "
            f"{leader} mạnh hơn trên model"
        )
    return None


def _ev_insight_line(data: Mapping[str, Any] | pd.Series) -> str | None:
    """Compare model probability vs bookmaker implied (1/odds)."""
    p = _insight_float(data, "p_model", "model_p", "prob_model")
    odds = _insight_float(data, "odds", "bookmaker_odds")
    ev = _insight_float(data, "ev")
    if ev is None:
        ev_pct = _insight_float(data, "ev_pct")
        if ev_pct is not None:
            ev = ev_pct / 100.0

    if p is not None and odds is not None and odds > 1.0:
        implied = 1.0 / odds
        edge_pp = (p - implied) * 100.0
        ev_txt = f", EV {ev:+.0%}" if ev is not None else ""
        return (
            f"EV: Model {p:.0%} vs nhà cái ~{implied:.0%} "
            f"(@{odds:.2f}) → lệch {edge_pp:+.0f}đ{ev_txt}"
        )
    if ev is not None:
        return f"EV: Model thấy cửa này lệch giá nhà cái (EV {ev:+.0%})"
    if p is not None and odds is not None:
        return f"EV: Model {p:.0%} · odds {odds:.2f} — thiếu implied sạch để so"
    return None


def _line_edge_insight_line(data: Mapping[str, Any] | pd.Series) -> str | None:
    """Surface model fair line vs bookie line when disparity is available."""
    model = _insight_get(data, "model_fair_line")
    bookie = _insight_get(data, "bookie_market_line")
    edge = _insight_get(data, "line_edge")
    if model and bookie and edge:
        return f"Line: model {model} vs nhà cái {bookie} → lệch {edge}"
    if model and edge:
        return f"Line: model {model} (lệch {edge})"
    if model and bookie:
        return f"Line: model {model} vs nhà cái {bookie}"
    return None


def generate_match_insights(
    match_data: Mapping[str, Any] | pd.Series | None,
) -> list[str]:
    """Return ≤3 short Vietnamese insight lines for Lite Mode pick cards.

    Builds fitness (rest / schedule), form (rolling xG/goals or DC attack),
    EV valuation (model p vs 1/odds), and optional fair-line disparity.
    Missing fields are skipped — never raises. Optional ``is_rotation_risk``
    flags append a soft warning.

    Parameters
    ----------
    match_data:
        Dict / Series with whatever the Lite card already has (team names,
        rest_days, matches_last_14d, rolling xG, p_model, odds, EV, …).

    Returns
    -------
    list[str]
        Zero to three shareable insight lines (Zalo/Telegram friendly).
    """
    if match_data is None:
        return []
    try:
        home = str(
            _insight_get(match_data, "home", "home_team", "HomeTeam") or "Chủ"
        ).strip() or "Chủ"
        away = str(
            _insight_get(match_data, "away", "away_team", "AwayTeam") or "Khách"
        ).strip() or "Khách"

        lines: list[str] = []
        for builder in (
            lambda: _fitness_insight_line(home, away, match_data),
            lambda: _form_insight_line(home, away, match_data),
            lambda: _ev_insight_line(match_data),
            lambda: _line_edge_insight_line(match_data),
        ):
            try:
                line = builder()
            except Exception:  # noqa: BLE001 — never crash Lite cards
                line = None
            if line:
                lines.append(line)
            if len(lines) >= 3:
                break
        return lines[:3]
    except Exception:  # noqa: BLE001
        return []
