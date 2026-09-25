"""API v2 routers — Schema / Pipeline V2 (Quant Engine).

Endpoints call ``run_quant_pipeline``, ``PaperTrader``, and ``AuditEngine``
directly. Telegram Mini App auth is applied at app mount (same as v1).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Annotated, Any, Mapping
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.api.schemas_v2 import (
    AuditPredictionOut,
    PlaceBetRequest,
    PlaceBetResponse,
    ValueBetV2Out,
    ValueBetsV2Response,
)
from src.data.quality_monitor import HARD_GATE_CODE
from src.db.schema_v2 import CanonicalCompetition, CanonicalMatch, RawDataLake
from src.db.session import get_session_factory
from src.execution.paper_trader import PaperTrader
from src.monitoring.audit_engine import AuditEngine
from src.pipeline.run_pipeline import run_quant_pipeline

logger = logging.getLogger(__name__)

api_v2 = APIRouter(prefix="/api/v2")

# Concurrent pipeline pricing for the scan endpoint.
_DEFAULT_CONCURRENCY = 8
_MODEL_TIER_X = HARD_GATE_CODE  # "MODEL_TIER_X"


# ---------------------------------------------------------------------------
# Upcoming-match helpers (v2 schema)
# ---------------------------------------------------------------------------


def _ensure_aware(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts


def _iso(ts: datetime | None) -> str | None:
    if ts is None:
        return None
    return _ensure_aware(ts).isoformat()


def _team_name(match: CanonicalMatch, side: str) -> str | None:
    rel = getattr(match, f"{side}_team", None)
    if rel is not None and getattr(rel, "name", None):
        return str(rel.name)
    return None


def _league_label(match: CanonicalMatch) -> str | None:
    comp = getattr(match, "competition", None)
    if comp is None:
        return None
    code = getattr(comp, "code", None)
    if code:
        return str(code)
    name = getattr(comp, "name", None)
    return str(name) if name else None


async def _odds_from_lake(
    session: AsyncSession, match_id: UUID
) -> dict[str, float] | None:
    stmt = (
        select(RawDataLake)
        .where(RawDataLake.canonical_match_id == match_id)
        .order_by(RawDataLake.observed_at.desc())
        .limit(8)
    )
    rows = (await session.execute(stmt)).scalars().all()
    for row in rows:
        payload = row.payload or {}
        if not isinstance(payload, Mapping):
            continue
        odds = payload.get("odds") or payload.get("bookmaker_odds")
        if isinstance(odds, Mapping) and {"H", "D", "A"} <= set(str(k) for k in odds):
            try:
                return {k: float(odds[k]) for k in ("H", "D", "A")}
            except (TypeError, ValueError, KeyError):
                continue
        # Flat H/D/A on payload
        if {"H", "D", "A"} <= set(str(k) for k in payload):
            try:
                return {k: float(payload[k]) for k in ("H", "D", "A")}
            except (TypeError, ValueError, KeyError):
                continue
    return None


async def _raw_records_from_lake(
    session: AsyncSession, match_id: UUID
) -> list[dict[str, Any]]:
    stmt = (
        select(RawDataLake)
        .where(RawDataLake.canonical_match_id == match_id)
        .order_by(RawDataLake.observed_at.asc())
    )
    rows = (await session.execute(stmt)).scalars().all()
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "id": str(row.raw_id),
                "raw_id": str(row.raw_id),
                "source_type": row.source,
                "source": row.source,
                "observed_at": row.observed_at,
                "payload": row.payload,
            }
        )
    return out


async def fetch_upcoming_matches(
    session: AsyncSession,
    *,
    league: str | None = None,
    limit: int = 50,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Load SCHEDULED (or future) canonical matches with odds + raw records.

    Returns an empty list when the table is empty or the query fails at the
    call site — callers should catch DB errors separately.
    """
    as_of = _ensure_aware(now or datetime.now(timezone.utc))
    stmt = (
        select(CanonicalMatch)
        .options(
            selectinload(CanonicalMatch.home_team),
            selectinload(CanonicalMatch.away_team),
            selectinload(CanonicalMatch.competition),
        )
        .where(CanonicalMatch.kickoff_utc >= as_of)
        .where(CanonicalMatch.status.in_(("SCHEDULED", "NS", "TIMED", "PENDING")))
        .order_by(CanonicalMatch.kickoff_utc.asc())
        .limit(max(1, int(limit)))
    )
    if league and str(league).strip():
        code = str(league).strip().upper()
        stmt = stmt.join(
            CanonicalCompetition,
            CanonicalMatch.competition_id == CanonicalCompetition.id,
        ).where(
            (CanonicalCompetition.code == code)
            | (CanonicalCompetition.name.ilike(f"%{league.strip()}%"))
        )

    rows = (await session.execute(stmt)).scalars().unique().all()
    fixtures: list[dict[str, Any]] = []
    for m in rows:
        odds = await _odds_from_lake(session, m.canonical_match_id)
        raw = await _raw_records_from_lake(session, m.canonical_match_id)
        fixtures.append(
            {
                "canonical_match_id": m.canonical_match_id,
                "kickoff_utc": m.kickoff_utc,
                "home": _team_name(m, "home"),
                "away": _team_name(m, "away"),
                "league": _league_label(m),
                "status": m.status,
                "bookmaker_odds": odds,
                "raw_records": raw,
            }
        )
    return fixtures


# ---------------------------------------------------------------------------
# Pipeline scan helpers
# ---------------------------------------------------------------------------


def _model_confidence(pipe: Mapping[str, Any]) -> float | None:
    snap = pipe.get("prediction_snapshot") or {}
    raw = snap.get("model_confidence_score")
    if raw is None:
        ens = pipe.get("ensemble")
        if ens is not None:
            raw = getattr(ens, "model_confidence_score", None)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _passes_filters(
    pipe: Mapping[str, Any],
    *,
    min_ev_pct: float,
) -> bool:
    tier = str(pipe.get("model_tier") or _MODEL_TIER_X)
    if tier == _MODEL_TIER_X or pipe.get("no_bet"):
        return False
    try:
        ev = float(pipe.get("expected_value_percent") or 0.0)
    except (TypeError, ValueError):
        return False
    return ev >= float(min_ev_pct)


async def _price_one(
    fixture: Mapping[str, Any],
    *,
    sem: asyncio.Semaphore,
    as_of: datetime,
    pipeline_fn: Any | None = None,
) -> dict[str, Any] | None:
    """Run quant pipeline for one fixture; return None on error."""
    match_id = fixture["canonical_match_id"]
    fn = pipeline_fn if pipeline_fn is not None else run_quant_pipeline
    async with sem:
        try:
            out = await fn(
                match_id,
                as_of,
                raw_records=list(fixture.get("raw_records") or []),
                bookmaker_odds=fixture.get("bookmaker_odds"),
                session=None,
                persist=False,
            )
            return {
                **dict(out),
                "_fixture": fixture,
            }
        except Exception:  # noqa: BLE001
            logger.exception("Pipeline failed for match=%s", match_id)
            return None


def _to_value_bet_out(priced: Mapping[str, Any]) -> ValueBetV2Out:
    fx = priced.get("_fixture") or {}
    fair_lines = priced.get("fair_lines") or {}
    if not isinstance(fair_lines, Mapping):
        fair_lines = {}
    fair_probs = priced.get("fair_probabilities") or {}
    if not isinstance(fair_probs, Mapping):
        fair_probs = {}
    pred_id = priced.get("prediction_id")
    return ValueBetV2Out(
        match_id=str(fx.get("canonical_match_id") or priced.get("canonical_match_id")),
        prediction_id=str(pred_id) if pred_id is not None else None,
        home=fx.get("home"),
        away=fx.get("away"),
        league=fx.get("league"),
        kickoff=_iso(fx.get("kickoff_utc")),
        fair_lines={str(k): float(v) for k, v in fair_lines.items()},
        fair_probabilities={str(k): float(v) for k, v in fair_probs.items()},
        bookmaker_odds=fx.get("bookmaker_odds"),
        data_score=float(priced.get("aggregate_data_score") or 0.0),
        model_tier=str(priced.get("model_tier") or _MODEL_TIER_X),
        model_confidence=_model_confidence(priced),
        expected_value_percent=float(priced.get("expected_value_percent") or 0.0),
        no_bet=bool(priced.get("no_bet")),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


async def _run_place_with_session(
    body: PlaceBetRequest,
    as_of: datetime,
    session: AsyncSession | None,
) -> PlaceBetResponse:
    trader = PaperTrader(
        session=session,
        league=body.league,
        bankroll=float(body.bankroll) if body.bankroll is not None else 10_000.0,
        persist=bool(body.persist and session is not None),
        candidates=(
            [c.model_dump() for c in body.candidates]
            if body.candidates is not None
            else None
        ),
    )
    pipe_kwargs: dict[str, Any] = {}
    if body.dixon_coles_probs is not None:
        pipe_kwargs["dixon_coles_probs"] = body.dixon_coles_probs
    if body.lightgbm_probs is not None:
        pipe_kwargs["lightgbm_probs"] = body.lightgbm_probs
    if body.raw_records is not None:
        pipe_kwargs["raw_records"] = body.raw_records

    bet_ids = await trader.execute_value_bets(
        body.canonical_match_id,
        as_of,
        bookmaker_odds=body.bookmaker_odds,
        **pipe_kwargs,
    )
    if session is not None:
        await session.commit()

    ids = [str(b) for b in bet_ids]
    return PlaceBetResponse(
        ok=True,
        bet_ids=ids,
        count=len(ids),
        canonical_match_id=str(body.canonical_match_id),
        message=None if ids else "No bets placed (NO_BET / zero stake / exposure)",
    )


@api_v2.get(
    "/value-bets",
    response_model=ValueBetsV2Response,
    tags=["value-bets-v2"],
    summary="Upcoming value bets via Quant Pipeline V2",
)
async def get_value_bets_v2(
    min_ev: Annotated[
        float,
        Query(description="Minimum EV in percent (default 3.0 = 3%)"),
    ] = 3.0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    league: Annotated[str | None, Query()] = None,
) -> ValueBetsV2Response:
    """Price upcoming canonical matches and filter by tier + EV.

    Filters: ``model_tier != MODEL_TIER_X`` and ``expected_value_percent >= min_ev``.
    Empty DB / connection errors return an empty list with ``notes`` (HTTP 200).
    """
    as_of = datetime.now(timezone.utc)
    fixtures: list[dict[str, Any]] = []
    notes: str | None = None

    try:
        factory = get_session_factory()
        async with factory() as session:
            fixtures = await fetch_upcoming_matches(
                session,
                league=league,
                limit=limit,
                now=as_of,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("v2 value-bets: DB unavailable: %s", exc)
        return ValueBetsV2Response(
            bets=[],
            count=0,
            min_ev_pct=float(min_ev),
            limit=int(limit),
            league=league,
            scanned=0,
            notes=f"Database unavailable: {exc}",
        )

    if not fixtures:
        return ValueBetsV2Response(
            bets=[],
            count=0,
            min_ev_pct=float(min_ev),
            limit=int(limit),
            league=league,
            scanned=0,
            notes="No upcoming matches in v2 schema (empty or filtered)",
        )

    sem = asyncio.Semaphore(max(1, _DEFAULT_CONCURRENCY))
    priced = await asyncio.gather(
        *[_price_one(fx, sem=sem, as_of=as_of) for fx in fixtures]
    )

    bets: list[ValueBetV2Out] = []
    for item in priced:
        if item is None:
            continue
        if not _passes_filters(item, min_ev_pct=float(min_ev)):
            continue
        bets.append(_to_value_bet_out(item))

    bets.sort(key=lambda b: b.expected_value_percent, reverse=True)
    bets = bets[: int(limit)]

    return ValueBetsV2Response(
        bets=bets,
        count=len(bets),
        min_ev_pct=float(min_ev),
        limit=int(limit),
        league=league,
        scanned=len(fixtures),
        notes=notes,
    )


@api_v2.post(
    "/bets/place",
    response_model=PlaceBetResponse,
    tags=["bets-v2"],
    summary="Place paper value bets via PaperTrader",
)
async def place_bets_v2(body: PlaceBetRequest) -> PlaceBetResponse:
    """Forward WebApp payload to ``PaperTrader.execute_value_bets``."""
    as_of = _ensure_aware(body.as_of_time or datetime.now(timezone.utc))
    try:
        factory = None
        if body.persist:
            try:
                factory = get_session_factory()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "v2 place: session factory failed (%s); running without persist",
                    exc,
                )
                factory = None

        if factory is not None:
            async with factory() as session:
                return await _run_place_with_session(body, as_of, session)
        return await _run_place_with_session(body, as_of, None)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("v2 place failed")
        raise HTTPException(status_code=500, detail=f"Place bet failed: {exc}") from exc


@api_v2.get(
    "/audit/{prediction_id}",
    response_model=AuditPredictionOut,
    tags=["audit-v2"],
    summary="Audit trail / lineage for a prediction",
)
async def audit_prediction_v2(prediction_id: UUID) -> AuditPredictionOut:
    """Return ``AuditEngine.audit_prediction`` lineage JSON."""
    try:
        factory = get_session_factory()
    except Exception as exc:  # noqa: BLE001
        logger.warning("v2 audit: DB unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Database unavailable: {exc}",
        ) from exc

    try:
        async with factory() as session:
            engine = AuditEngine(session=session)
            report = await engine.audit_prediction(prediction_id)
            if not report.get("found"):
                raise HTTPException(
                    status_code=404,
                    detail=f"Prediction not found: {prediction_id}",
                )
            return AuditPredictionOut(**report)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("v2 audit failed for %s", prediction_id)
        raise HTTPException(status_code=500, detail=f"Audit failed: {exc}") from exc


__all__ = ["api_v2", "fetch_upcoming_matches"]
