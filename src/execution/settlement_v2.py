"""Auto-settlement engine for Quant Engine ``bet_snapshots`` (Worker 7).

Settles PENDING paper bets against finished ``canonical_matches``, computes
PnL via Asian quarter-line rules, and records Closing-Line Value (CLV) from
Pinnacle odds in ``raw_data_lake`` nearest to kickoff.

CLV definition
--------------
``clv_value = log(taken_odds / closing_odds)``
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping, Optional
from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.schema_v2 import (
    BetSnapshot,
    CanonicalMatch,
    PitFeatureSnapshot,
    PredictionSnapshot,
    RawDataLake,
)
from src.services.settlement_service import (
    payout_from_outcome,
    resolve_bet_settlement,
    settle_ah_detailed,
    settle_ou_detailed,
)
from src.validation.metrics import QuantMetrics

logger = logging.getLogger(__name__)

# Worker-7 ledger statuses (map from settlement_service WIN/LOSS/PUSH).
SettleStatusV2 = str  # WON | LOST | HALF_WIN | HALF_LOSS | VOID

_SERVICE_TO_V2: dict[str, SettleStatusV2] = {
    "WIN": "WON",
    "LOSS": "LOST",
    "HALF_WIN": "HALF_WIN",
    "HALF_LOSS": "HALF_LOSS",
    "PUSH": "VOID",
    "VOID": "VOID",
}

_FINISHED_MATCH_STATUSES = frozenset(
    {"FINISHED", "FT", "COMPLETE", "COMPLETED", "SETTLED", "CLOSED"}
)

# Window around kickoff to accept a Pinnacle closing observation.
_CLOSING_WINDOW = timedelta(minutes=30)

_SELECTION_ODDS_KEYS: dict[str, tuple[str, ...]] = {
    "home": ("H", "home", "1", "Home"),
    "draw": ("D", "draw", "X", "Draw"),
    "away": ("A", "away", "2", "Away"),
}


def _ensure_aware(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts


def map_settle_status(service_status: str) -> SettleStatusV2:
    """Map settlement_service status → Worker-7 ledger label."""
    key = str(service_status or "").strip().upper()
    if key in _SERVICE_TO_V2:
        return _SERVICE_TO_V2[key]
    if key in {"WON", "LOST", "HALF_WIN", "HALF_LOSS", "VOID"}:
        return key
    raise ValueError(f"Unknown settlement status: {service_status!r}")


def compute_clv_value(taken_odds: float, closing_odds: float) -> float:
    """CLV = ``log(taken_odds / closing_odds)`` (NaN when inputs invalid)."""
    return QuantMetrics.closing_line_value(taken_odds, closing_odds)


def settle_market(
    *,
    market_type: str,
    selection: str,
    taken_odds: float,
    stake: float,
    ft_home: int,
    ft_away: int,
    line: float | None = None,
) -> tuple[SettleStatusV2, float, float]:
    """Settle one bet → ``(status_v2, pnl, payout)``.

    Thin wrapper over :func:`resolve_bet_settlement` that remaps WIN/LOSS/PUSH
    to WON/LOST/VOID and tolerates selection strings that already include the
    line (``AH Home -0.25``) or a separate ``line`` argument.
    """
    mkt = str(market_type or "1X2").strip().upper()
    sel = str(selection or "").strip()
    if line is not None and mkt in {"AH", "ASIAN", "ASIAN HANDICAP", "OU", "O/U", "OVER/UNDER"}:
        # Ensure selection carries a numeric line for parse_selection_line.
        if not re.search(r"[-+]?\d+(?:\.\d+)?", sel):
            if mkt in {"AH", "ASIAN", "ASIAN HANDICAP"}:
                side = "Away" if "away" in sel.lower() else "Home"
                sel = f"AH {side} {float(line):+g}"
            else:
                side = "Under" if "under" in sel.lower() else "Over"
                sel = f"{side} {float(line):g}"

    fthg, ftag = int(ft_home), int(ft_away)
    if fthg > ftag:
        ftr = "H"
    elif fthg < ftag:
        ftr = "A"
    else:
        ftr = "D"

    service_status, pnl, payout = resolve_bet_settlement(
        market=mkt,
        selection=sel,
        odds=float(taken_odds),
        stake=float(stake),
        fthg=fthg,
        ftag=ftag,
        ftr=ftr,
    )
    return map_settle_status(service_status), float(pnl), float(payout)


def _extract_closing_odds_from_payload(
    payload: Mapping[str, Any] | None,
    *,
    market_type: str,
    selection: str,
) -> float | None:
    """Best-effort extract of a closing decimal price from a raw lake payload."""
    if not payload:
        return None

    # Direct scalar
    for key in ("closing_odds", "odds", "price", "decimal_odds"):
        raw = payload.get(key)
        if isinstance(raw, (int, float)) and float(raw) > 1.0:
            # Ambiguous if payload also has a market map — prefer map below.
            if not isinstance(payload.get("odds"), Mapping):
                return float(raw)

    odds_map: Any = payload.get("odds")
    if not isinstance(odds_map, Mapping):
        odds_map = payload.get("markets") or payload.get("prices") or payload

    if not isinstance(odds_map, Mapping):
        return None

    # Nested by market
    mkt = str(market_type or "").strip().upper()
    nested = odds_map.get(mkt) or odds_map.get(mkt.lower()) or odds_map.get("1X2")
    if isinstance(nested, Mapping):
        odds_map = nested

    sel_l = str(selection or "").strip().lower()
    key_group = "home"
    if "draw" in sel_l or sel_l in {"d", "x"}:
        key_group = "draw"
    elif "away" in sel_l or sel_l in {"a", "2"}:
        key_group = "away"
    elif "over" in sel_l:
        for k in ("over", "Over", "O"):
            if k in odds_map:
                try:
                    v = float(odds_map[k])
                except (TypeError, ValueError):
                    continue
                if v > 1.0:
                    return v
        return None
    elif "under" in sel_l:
        for k in ("under", "Under", "U"):
            if k in odds_map:
                try:
                    v = float(odds_map[k])
                except (TypeError, ValueError):
                    continue
                if v > 1.0:
                    return v
        return None

    for k in _SELECTION_ODDS_KEYS[key_group]:
        if k in odds_map:
            try:
                v = float(odds_map[k])
            except (TypeError, ValueError):
                continue
            if v > 1.0:
                return v

    # Selection-keyed map: {"Home": 2.1, ...}
    for cand in (selection, selection.title(), selection.upper()):
        if cand in odds_map:
            try:
                v = float(odds_map[cand])
            except (TypeError, ValueError):
                continue
            if v > 1.0:
                return v
    return None


@dataclass
class SettlementEngineV2:
    """Settle PENDING ``bet_snapshots`` against finished canonical matches."""

    session: AsyncSession
    closing_window: timedelta = field(default=_CLOSING_WINDOW)
    # Injectable for tests — ``(match_id, market, selection, kickoff) -> odds|None``.
    closing_odds_lookup: Optional[
        Callable[[UUID, str, str, datetime], float | None]
    ] = None

    async def settle_pending_bets(self, as_of_time: datetime) -> dict[str, Any]:
        """Settle all PENDING bets whose match has FT scores by ``as_of_time``.

        Returns a summary dict with counts and per-bet detail rows.
        """
        as_of = _ensure_aware(as_of_time)
        pending = await self._load_pending_with_match()
        details: list[dict[str, Any]] = []
        settled_n = skipped = 0
        status_counts: dict[str, int] = {
            "WON": 0,
            "LOST": 0,
            "HALF_WIN": 0,
            "HALF_LOSS": 0,
            "VOID": 0,
        }
        pnl_sum = 0.0

        for bet, match in pending:
            kickoff = _ensure_aware(match.kickoff_utc)
            if kickoff > as_of:
                skipped += 1
                continue

            ft_h = match.ft_home_goals
            ft_a = match.ft_away_goals
            finished = (
                str(match.status or "").strip().upper() in _FINISHED_MATCH_STATUSES
                or (ft_h is not None and ft_a is not None)
            )
            if not finished or ft_h is None or ft_a is None:
                skipped += 1
                continue

            stake = float(bet.final_stake_amount or 0.0)
            odds = float(bet.taken_odds or 0.0)
            if stake <= 0.0 or odds <= 1.0:
                skipped += 1
                continue

            try:
                status, pnl, _payout = settle_market(
                    market_type=bet.market_type,
                    selection=bet.selection,
                    taken_odds=odds,
                    stake=stake,
                    ft_home=int(ft_h),
                    ft_away=int(ft_a),
                )
            except ValueError as exc:
                logger.warning("Skip bet %s: %s", bet.bet_id, exc)
                skipped += 1
                continue

            closing = await self._resolve_closing_odds(bet, match)
            clv: float | None = None
            if closing is not None and closing > 0.0 and odds > 0.0:
                clv_raw = compute_clv_value(odds, closing)
                clv = None if (isinstance(clv_raw, float) and math.isnan(clv_raw)) else clv_raw
                bet.sharp_closing_odds = float(closing)

            bet.status = status
            bet.pnl = float(pnl)
            bet.clv_value = clv
            bet.settled_at = as_of

            settled_n += 1
            pnl_sum += float(pnl)
            status_counts[status] = status_counts.get(status, 0) + 1
            details.append(
                {
                    "bet_id": bet.bet_id,
                    "canonical_match_id": match.canonical_match_id,
                    "market_type": bet.market_type,
                    "selection": bet.selection,
                    "status": status,
                    "pnl": float(pnl),
                    "clv_value": clv,
                    "closing_odds": closing,
                    "score": f"{ft_h}-{ft_a}",
                    "ht_score": (
                        f"{match.ht_home_goals}-{match.ht_away_goals}"
                        if match.ht_home_goals is not None
                        and match.ht_away_goals is not None
                        else None
                    ),
                }
            )

        await self.session.flush()
        return {
            "settled": settled_n,
            "skipped": skipped,
            "pnl": pnl_sum,
            "by_status": status_counts,
            "details": details,
            "as_of_time": as_of,
        }

    async def _load_pending_with_match(
        self,
    ) -> list[tuple[BetSnapshot, CanonicalMatch]]:
        """PENDING bets joined through prediction → pit → canonical match."""
        stmt = (
            select(BetSnapshot, CanonicalMatch)
            .join(
                PredictionSnapshot,
                BetSnapshot.prediction_id == PredictionSnapshot.prediction_id,
            )
            .join(
                PitFeatureSnapshot,
                PredictionSnapshot.feature_snapshot_id
                == PitFeatureSnapshot.feature_snapshot_id,
            )
            .join(
                CanonicalMatch,
                PitFeatureSnapshot.canonical_match_id
                == CanonicalMatch.canonical_match_id,
            )
            .where(BetSnapshot.status == "PENDING")
        )
        rows = (await self.session.execute(stmt)).all()
        return [(bet, match) for bet, match in rows]

    async def _resolve_closing_odds(
        self,
        bet: BetSnapshot,
        match: CanonicalMatch,
    ) -> float | None:
        if self.closing_odds_lookup is not None:
            return self.closing_odds_lookup(
                match.canonical_match_id,
                str(bet.market_type),
                str(bet.selection),
                _ensure_aware(match.kickoff_utc),
            )
        return await self._pinnacle_closing_odds(
            match.canonical_match_id,
            market_type=str(bet.market_type),
            selection=str(bet.selection),
            kickoff_utc=_ensure_aware(match.kickoff_utc),
        )

    async def _pinnacle_closing_odds(
        self,
        canonical_match_id: UUID,
        *,
        market_type: str,
        selection: str,
        kickoff_utc: datetime,
    ) -> float | None:
        """Nearest Pinnacle observation at/before kickoff within ``closing_window``."""
        lo = kickoff_utc - self.closing_window
        hi = kickoff_utc + self.closing_window
        stmt = (
            select(RawDataLake)
            .where(
                RawDataLake.canonical_match_id == canonical_match_id,
                or_(
                    RawDataLake.source.ilike("%pinnacle%"),
                    RawDataLake.source == "pinnacle",
                ),
                RawDataLake.observed_at >= lo,
                RawDataLake.observed_at <= hi,
            )
            .order_by(RawDataLake.observed_at.desc())
        )
        rows = (await self.session.execute(stmt)).scalars().all()
        if not rows:
            # Fallback: any Pinnacle row for the match (closest to kickoff).
            stmt_all = (
                select(RawDataLake)
                .where(
                    RawDataLake.canonical_match_id == canonical_match_id,
                    or_(
                        RawDataLake.source.ilike("%pinnacle%"),
                        RawDataLake.source == "pinnacle",
                    ),
                )
                .order_by(RawDataLake.observed_at.desc())
            )
            rows = (await self.session.execute(stmt_all)).scalars().all()

        best: float | None = None
        best_delta: float | None = None
        for row in rows:
            obs = row.observed_at
            if obs is None:
                continue
            obs_aware = _ensure_aware(obs)
            delta = abs((obs_aware - kickoff_utc).total_seconds())
            price = _extract_closing_odds_from_payload(
                row.payload,
                market_type=market_type,
                selection=selection,
            )
            if price is None:
                continue
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best = price
        return best


# Re-export AH/OU helpers for tests that want direct access.
__all__ = [
    "SettlementEngineV2",
    "SettleStatusV2",
    "compute_clv_value",
    "map_settle_status",
    "settle_market",
    "settle_ah_detailed",
    "settle_ou_detailed",
    "payout_from_outcome",
]
