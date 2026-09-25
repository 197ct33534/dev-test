"""Daily / league exposure caps and bet_snapshot persistence helpers.

Decoupled from ``schema_v2``: if SQLAlchemy models are unavailable the manager
still builds record dicts and exposes an async save stub that no-ops / returns
the prepared payload.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, MutableMapping, Optional

try:
    from sqlalchemy.ext.asyncio import AsyncSession
except ImportError:  # pragma: no cover
    AsyncSession = Any  # type: ignore[misc, assignment]

_BetSnapshotModel: Any | None
try:
    from src.db.schema_v2 import BetSnapshot as _BetSnapshotModel
except ImportError:  # pragma: no cover
    _BetSnapshotModel = None


@dataclass(frozen=True, slots=True)
class ExposureConfig:
    """Exposure ceilings as fractions of bankroll (or absolute units if consistent)."""

    max_daily_exposure: float = 0.05  # 5% bankroll / day
    max_league_exposure: float = 0.03  # 3% bankroll / league / day


@dataclass
class ExposureManager:
    """Enforce portfolio exposure limits and prepare ``bet_snapshots`` records."""

    config: ExposureConfig = field(default_factory=ExposureConfig)
    # Running totals (caller may mutate or replace after each accepted bet).
    daily_exposure: float = 0.0
    league_exposure: MutableMapping[str, float] = field(default_factory=dict)

    def remaining_daily(self) -> float:
        return max(0.0, float(self.config.max_daily_exposure) - float(self.daily_exposure))

    def remaining_league(self, league: str) -> float:
        used = float(self.league_exposure.get(league, 0.0))
        return max(0.0, float(self.config.max_league_exposure) - used)

    def check_exposure(
        self,
        stake_fraction: float,
        *,
        league: str | None = None,
    ) -> dict[str, Any]:
        """Return whether ``stake_fraction`` fits under daily / league caps."""
        stake = max(0.0, float(stake_fraction))
        daily_ok = stake <= self.remaining_daily() + 1e-15
        league_key = (league or "").strip() or "UNKNOWN"
        league_ok = stake <= self.remaining_league(league_key) + 1e-15
        allowed = daily_ok and league_ok and stake > 0.0
        reasons: list[str] = []
        if stake <= 0.0:
            reasons.append("zero_stake")
        if not daily_ok:
            reasons.append("max_daily_exposure")
        if not league_ok:
            reasons.append("max_league_exposure")
        return {
            "allowed": allowed,
            "stake": stake,
            "league": league_key,
            "daily_exposure": self.daily_exposure,
            "daily_remaining": self.remaining_daily(),
            "league_exposure": float(self.league_exposure.get(league_key, 0.0)),
            "league_remaining": self.remaining_league(league_key),
            "max_daily_exposure": self.config.max_daily_exposure,
            "max_league_exposure": self.config.max_league_exposure,
            "reason": "ok" if allowed else "+".join(reasons) or "blocked",
        }

    def register_exposure(self, stake_fraction: float, *, league: str | None = None) -> None:
        """Accumulate exposure after a bet is accepted (caller responsibility)."""
        stake = max(0.0, float(stake_fraction))
        league_key = (league or "").strip() or "UNKNOWN"
        self.daily_exposure = float(self.daily_exposure) + stake
        self.league_exposure[league_key] = (
            float(self.league_exposure.get(league_key, 0.0)) + stake
        )

    def reset_daily(self) -> None:
        """Clear daily + per-league running totals (e.g. new calendar day)."""
        self.daily_exposure = 0.0
        self.league_exposure.clear()

    def prepare_bet_snapshot(
        self,
        *,
        prediction_id: uuid.UUID | str | None = None,
        market_type: str,
        selection: str,
        taken_odds: float | None = None,
        stake_result: Mapping[str, Any] | None = None,
        final_stake_fraction: float | None = None,
        final_stake_amount: float | None = None,
        bankroll: float | None = None,
        status: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a ``bet_snapshots``-shaped dict (no DB required).

        Maps RiskEngine output keys onto schema_v2 ``BetSnapshot`` columns when
        ``stake_result`` is provided.
        """
        sr = dict(stake_result or {})
        stake_frac = (
            float(final_stake_fraction)
            if final_stake_fraction is not None
            else float(sr.get("final_stake") or 0.0)
        )
        amount = final_stake_amount
        if amount is None and bankroll is not None:
            amount = stake_frac * float(bankroll)

        resolved_status = status
        if resolved_status is None:
            resolved_status = str(sr.get("status") or "PENDING")
            if resolved_status == "OK":
                resolved_status = "PENDING"
            elif resolved_status == "NO_BET":
                resolved_status = "NO_BET"

        pred_id: Any = prediction_id
        if isinstance(prediction_id, str):
            try:
                pred_id = uuid.UUID(prediction_id)
            except ValueError:
                pred_id = prediction_id

        record: dict[str, Any] = {
            "bet_id": uuid.uuid4(),
            "prediction_id": pred_id,
            "market_type": str(market_type),
            "selection": str(selection),
            "taken_odds": float(taken_odds) if taken_odds is not None else sr.get("odds"),
            "sharp_closing_odds": None,
            "expected_value_percent": _ev_to_percent(sr.get("ev")),
            "raw_kelly_fraction": sr.get("f_kelly"),
            "data_weight": sr.get("data_weight"),
            "model_weight": sr.get("model_weight"),
            "correlated_discount": sr.get("correlation_discount"),
            "final_stake_fraction": stake_frac,
            "final_stake_amount": amount,
            "status": resolved_status,
            "pnl": None,
            "clv_value": None,
            "settled_at": None,
            "created_at": datetime.now(timezone.utc),
            "reason": sr.get("reason"),
        }
        if extra:
            record.update(dict(extra))
        return record

    async def save_bet_snapshot(
        self,
        record: Mapping[str, Any],
        session: Optional[Any] = None,
        *,
        commit: bool = False,
    ) -> dict[str, Any]:
        """Persist ``record`` via async session when schema_v2 is available.

        If ``BetSnapshot`` / session is missing, returns ``{saved: False, record}``
        without raising — callers can still use the prepared dict.
        """
        payload = dict(record)
        if session is None or _BetSnapshotModel is None:
            return {
                "saved": False,
                "reason": "schema_or_session_unavailable",
                "record": payload,
            }

        # Strip non-column helper keys before ORM insert.
        column_keys = {
            "bet_id",
            "prediction_id",
            "market_type",
            "selection",
            "taken_odds",
            "sharp_closing_odds",
            "expected_value_percent",
            "raw_kelly_fraction",
            "data_weight",
            "model_weight",
            "correlated_discount",
            "final_stake_fraction",
            "final_stake_amount",
            "status",
            "pnl",
            "clv_value",
            "settled_at",
            "created_at",
        }
        orm_kwargs = {k: payload[k] for k in column_keys if k in payload}
        if orm_kwargs.get("prediction_id") is None:
            return {
                "saved": False,
                "reason": "prediction_id_required",
                "record": payload,
            }

        row = _BetSnapshotModel(**orm_kwargs)
        session.add(row)
        if commit:
            await session.commit()
            if hasattr(session, "refresh"):
                await session.refresh(row)
        else:
            await session.flush()

        return {
            "saved": True,
            "bet_id": getattr(row, "bet_id", orm_kwargs.get("bet_id")),
            "record": payload,
        }

    async def accept_bet(
        self,
        stake_fraction: float,
        *,
        league: str | None = None,
        market_type: str = "1X2",
        selection: str = "Home",
        taken_odds: float | None = None,
        prediction_id: uuid.UUID | str | None = None,
        stake_result: Mapping[str, Any] | None = None,
        bankroll: float | None = None,
        session: Optional[Any] = None,
        persist: bool = False,
    ) -> dict[str, Any]:
        """Check exposure, optionally register + persist a bet snapshot."""
        check = self.check_exposure(stake_fraction, league=league)
        if not check["allowed"]:
            return {
                "accepted": False,
                "exposure": check,
                "record": None,
                "persist": None,
            }

        record = self.prepare_bet_snapshot(
            prediction_id=prediction_id,
            market_type=market_type,
            selection=selection,
            taken_odds=taken_odds,
            stake_result=stake_result,
            final_stake_fraction=float(stake_fraction),
            bankroll=bankroll,
        )
        self.register_exposure(stake_fraction, league=league)

        persist_result = None
        if persist:
            persist_result = await self.save_bet_snapshot(record, session=session)

        league_key = (league or "").strip() or "UNKNOWN"
        return {
            "accepted": True,
            "exposure": {
                "allowed": True,
                "stake": float(stake_fraction),
                "league": league_key,
                "daily_exposure": self.daily_exposure,
                "daily_remaining": self.remaining_daily(),
                "league_exposure": float(self.league_exposure.get(league_key, 0.0)),
                "league_remaining": self.remaining_league(league_key),
                "max_daily_exposure": self.config.max_daily_exposure,
                "max_league_exposure": self.config.max_league_exposure,
                "reason": "registered",
            },
            "record": record,
            "persist": persist_result,
        }


def _ev_to_percent(ev: Any) -> float | None:
    if ev is None:
        return None
    try:
        v = float(ev)
    except (TypeError, ValueError):
        return None
    # Heuristic: |ev| ≤ 2 → fraction (0.08 → 8.0).
    if abs(v) <= 2.0:
        return v * 100.0
    return v
