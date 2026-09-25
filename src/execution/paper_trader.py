"""Paper trading executor — pipeline → risk sizing → ``bet_snapshots`` (Worker 7).

``PaperTrader.execute_value_bets`` is async and mock-friendly: inject
``pipeline_fn``, ``risk_engine``, and ``exposure_manager`` in tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.pipeline.run_pipeline import run_quant_pipeline
from src.risk import ExposureManager, RiskEngine

logger = logging.getLogger(__name__)

PipelineFn = Callable[..., Awaitable[Mapping[str, Any]]]

_OUTCOME_SELECTION: dict[str, str] = {
    "H": "Home",
    "D": "Draw",
    "A": "Away",
}


def _ensure_aware(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts


def _model_confidence(pipeline_out: Mapping[str, Any]) -> float:
    """Map ensemble confidence (0–1 or 0–100) onto RiskEngine's 0–100 scale."""
    ens = pipeline_out.get("ensemble")
    raw = None
    if ens is not None:
        raw = getattr(ens, "model_confidence_score", None)
    if raw is None:
        snap = pipeline_out.get("prediction_snapshot") or {}
        raw = snap.get("model_confidence_score")
    if raw is None:
        # Soft fallback from aggregate data score.
        raw = float(pipeline_out.get("aggregate_data_score") or 0.0) / 100.0
    val = float(raw)
    if 0.0 <= val <= 1.0:
        return val * 100.0
    return max(0.0, min(100.0, val))


def _fair_probabilities(pipeline_out: Mapping[str, Any]) -> dict[str, float]:
    probs = pipeline_out.get("fair_probabilities")
    if isinstance(probs, Mapping) and probs:
        return {str(k): float(v) for k, v in probs.items()}
    ens = pipeline_out.get("ensemble")
    if ens is not None:
        cal = getattr(ens, "calibrated_probabilities", None) or getattr(
            ens, "fair_probabilities", None
        )
        if isinstance(cal, Mapping) and cal:
            return {str(k): float(v) for k, v in cal.items()}
    # Invert fair_lines when probs missing.
    lines = pipeline_out.get("fair_lines") or {}
    if isinstance(lines, Mapping) and lines:
        out: dict[str, float] = {}
        for k, odd in lines.items():
            try:
                o = float(odd)
            except (TypeError, ValueError):
                continue
            if o > 0:
                out[str(k)] = 1.0 / o
        return out
    return {}


def build_1x2_candidates(
    pipeline_out: Mapping[str, Any],
    bookmaker_odds: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Build 1X2 candidate dicts from fair probs × bookmaker decimal odds."""
    probs = _fair_probabilities(pipeline_out)
    odds_map = dict(bookmaker_odds or {})
    candidates: list[dict[str, Any]] = []
    for key, selection in _OUTCOME_SELECTION.items():
        if key not in odds_map or key not in probs:
            continue
        try:
            o = float(odds_map[key])
            p = float(probs[key])
        except (TypeError, ValueError):
            continue
        if o <= 1.0 or p <= 0.0:
            continue
        candidates.append(
            {
                "market_type": "1X2",
                "selection": selection,
                "probability": p,
                "odds": o,
                "ev": p * o - 1.0,
            }
        )
    return candidates


@dataclass
class PaperTrader:
    """Execute value bets: quant pipeline → Kelly/risk caps → PENDING snapshots."""

    session: Optional[AsyncSession] = None
    risk_engine: RiskEngine = field(default_factory=RiskEngine)
    exposure_manager: ExposureManager = field(default_factory=ExposureManager)
    bankroll: float = 10_000.0
    league: str | None = None
    pipeline_fn: PipelineFn = field(default=run_quant_pipeline)  # type: ignore[assignment]
    persist: bool = True
    # Optional pre-built candidates (AH/OU etc.) — skips 1X2 auto-discovery.
    candidates: Optional[Sequence[Mapping[str, Any]]] = None

    async def execute_value_bets(
        self,
        canonical_match_id: UUID,
        as_of_time: datetime,
        *,
        bookmaker_odds: Optional[Mapping[str, Any]] = None,
        existing_match_bets: Optional[Sequence[Any]] = None,
        candidates: Optional[Sequence[Mapping[str, Any]]] = None,
        **pipeline_kwargs: Any,
    ) -> list[UUID]:
        """Run pipeline + risk sizing; insert PENDING ``bet_snapshots`` when stake > 0.

        Parameters
        ----------
        canonical_match_id, as_of_time:
            Match / PIT cutoff forwarded to ``run_quant_pipeline``.
        bookmaker_odds:
            Decimal H/D/A (or market) prices used for EV and taken odds.
        existing_match_bets:
            Open bets on the same match (correlation discount).
        candidates:
            Optional explicit bet candidates (overrides instance + 1X2 discovery).
            Each mapping needs ``market_type``, ``selection``, ``probability``,
            ``odds``.
        pipeline_kwargs:
            Extra kwargs for the pipeline (raw_records, dixon_coles_probs, …).

        Returns
        -------
        list[UUID]
            ``bet_id`` values successfully written (status PENDING).
        """
        as_of = _ensure_aware(as_of_time)
        pipe_kwargs = dict(pipeline_kwargs)
        pipe_kwargs.setdefault("session", self.session)
        pipe_kwargs.setdefault("persist", self.persist and self.session is not None)
        if bookmaker_odds is not None:
            pipe_kwargs.setdefault("bookmaker_odds", bookmaker_odds)

        pipeline_out = dict(
            await self.pipeline_fn(canonical_match_id, as_of, **pipe_kwargs)
        )

        if pipeline_out.get("no_bet"):
            logger.info(
                "PaperTrader: pipeline NO_BET for match=%s (tier=%s)",
                canonical_match_id,
                pipeline_out.get("model_tier"),
            )
            return []

        prediction_id = pipeline_out.get("prediction_id")
        if prediction_id is None:
            logger.warning("PaperTrader: missing prediction_id; aborting")
            return []

        data_score = float(pipeline_out.get("aggregate_data_score") or 0.0)
        model_conf = _model_confidence(pipeline_out)
        odds_for_candidates = bookmaker_odds or pipe_kwargs.get("bookmaker_odds")

        cand_list: list[Mapping[str, Any]]
        explicit = candidates if candidates is not None else self.candidates
        if explicit is not None:
            cand_list = list(explicit)
        else:
            cand_list = build_1x2_candidates(pipeline_out, odds_for_candidates)

        placed: list[UUID] = []
        open_bets: list[Any] = list(existing_match_bets or [])

        for cand in cand_list:
            bet_id = await self._try_place_candidate(
                cand,
                prediction_id=prediction_id,
                data_score=data_score,
                model_confidence=model_conf,
                existing_match_bets=open_bets,
            )
            if bet_id is not None:
                placed.append(bet_id)
                open_bets.append(
                    {
                        "market_type": cand.get("market_type") or cand.get("market"),
                        "selection": cand.get("selection"),
                        "bet_id": bet_id,
                    }
                )

        return placed

    async def _try_place_candidate(
        self,
        cand: Mapping[str, Any],
        *,
        prediction_id: UUID,
        data_score: float,
        model_confidence: float,
        existing_match_bets: Sequence[Any],
    ) -> UUID | None:
        market = str(cand.get("market_type") or cand.get("market") or "1X2")
        selection = str(cand.get("selection") or "")
        try:
            probability = float(cand["probability"])
            odds = float(cand["odds"])
        except (KeyError, TypeError, ValueError):
            logger.debug("Skip candidate missing probability/odds: %s", cand)
            return None

        stake_result = self.risk_engine.calculate_stake(
            probability=probability,
            odds=odds,
            data_score=data_score,
            model_confidence=model_confidence,
            existing_match_bets=existing_match_bets,
            market=market,
            selection=selection,
        )
        final_stake = float(stake_result.get("final_stake") or 0.0)
        if final_stake <= 0.0:
            return None

        # Exposure / risk caps (daily + league).
        exposure = self.exposure_manager.check_exposure(
            final_stake, league=self.league
        )
        if not exposure["allowed"]:
            logger.info(
                "PaperTrader: exposure blocked %s %s (%s)",
                market,
                selection,
                exposure.get("reason"),
            )
            return None

        accepted = await self.exposure_manager.accept_bet(
            final_stake,
            league=self.league,
            market_type=market,
            selection=selection,
            taken_odds=odds,
            prediction_id=prediction_id,
            stake_result=stake_result,
            bankroll=self.bankroll,
            session=self.session,
            persist=bool(self.session is not None and self.persist),
        )
        if not accepted.get("accepted"):
            return None

        record = accepted.get("record") or {}
        bet_id = record.get("bet_id")
        persist_info = accepted.get("persist") or {}
        if persist_info.get("saved") and persist_info.get("bet_id") is not None:
            bet_id = persist_info["bet_id"]

        if bet_id is None:
            return None

        # Ensure PENDING status on the ORM row when persisted.
        if (
            self.session is not None
            and self.persist
            and persist_info.get("saved")
        ):
            try:
                from src.db.schema_v2 import BetSnapshot

                row = await self.session.get(BetSnapshot, bet_id)
                if row is not None and row.status != "PENDING":
                    row.status = "PENDING"
                    await self.session.flush()
            except Exception:  # noqa: BLE001
                logger.exception("Failed to force PENDING on bet %s", bet_id)

        return bet_id if isinstance(bet_id, UUID) else UUID(str(bet_id))


__all__ = [
    "PaperTrader",
    "build_1x2_candidates",
]
