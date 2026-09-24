"""Performance analytics routes."""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Query

from src.api.schemas import PerformanceAnalyticsOut
from src.services.settlement_service import performance_analytics_sync

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get(
    "/performance",
    response_model=PerformanceAnalyticsOut,
    summary="Paper-journal performance (win rate, PnL, ROI, Brier)",
)
async def get_performance(
    days: Annotated[
        int | None,
        Query(ge=1, le=3650, description="Optional lookback window in days"),
    ] = None,
    league: Annotated[
        str | None,
        Query(description="Optional league filter, e.g. EPL / UWCL"),
    ] = None,
) -> PerformanceAnalyticsOut:
    """Aggregate settled ``live_bets`` / ``user_bets`` performance metrics.

    ``ev_vs_realized_gap`` = mean expected EV% − realized ROI%.
    ``brier_score`` is null when too few settled rows have ``p_model``.
    """
    payload = await asyncio.to_thread(
        performance_analytics_sync,
        days=days,
        league=league,
    )
    return PerformanceAnalyticsOut(**payload)
