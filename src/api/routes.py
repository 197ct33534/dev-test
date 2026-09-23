"""API v1 routers."""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import (
    HealthOut,
    TeamImportRequest,
    TeamImportResponse,
    TeamProfileOut,
    ValueBetsResponse,
)
from src.api.services import teams as teams_svc
from src.api.services import value_bets as vb_svc

router = APIRouter()
api_v1 = APIRouter(prefix="/api/v1")


@router.get("/health", response_model=HealthOut, tags=["health"])
async def health() -> HealthOut:
    return HealthOut(status="ok")


@api_v1.get(
    "/value-bets",
    response_model=ValueBetsResponse,
    tags=["value-bets"],
    summary="Top value bets (VN today/tomorrow window)",
)
async def get_value_bets(
    min_ev: Annotated[
        float,
        Query(description="Minimum EV in percent (default 5.0 = 5%)"),
    ] = 5.0,
    markets: Annotated[
        str | None,
        Query(description="Comma-separated: 1X2,AH,OU,Corners"),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    league: Annotated[str | None, Query()] = None,
    comp_id: Annotated[str | None, Query()] = None,
) -> ValueBetsResponse:
    """Prefer cached upcoming fixtures + model pickles for a fast response.

    Cold path (empty DB cache / missing ``models/*.pkl``) may fit Dixon–Coles
    and take several seconds on first call.
    """
    payload = await asyncio.to_thread(
        vb_svc.scan_value_bets_api,
        min_ev_pct=float(min_ev),
        markets=markets,
        limit=int(limit),
        league=league,
        comp_id=comp_id,
    )
    return ValueBetsResponse(**payload)


@api_v1.get(
    "/teams/{team_id}/profile",
    response_model=TeamProfileOut,
    tags=["teams"],
    summary="Team dossier: last 5 + next 5 from global_matches.db",
)
async def get_team_profile(team_id: str) -> TeamProfileOut:
    data = await asyncio.to_thread(teams_svc.fetch_team_profile, team_id, limit=5)
    if not teams_svc.team_profile_found(data):
        raise HTTPException(status_code=404, detail=f"Team not found: {team_id}")
    return TeamProfileOut(**data)


@api_v1.post(
    "/teams/import-url",
    response_model=TeamImportResponse,
    tags=["teams"],
    summary="Import team history from a Flashscore team URL",
)
async def import_team(body: TeamImportRequest) -> TeamImportResponse:
    result = await asyncio.to_thread(
        teams_svc.import_team_url,
        body.team_id,
        body.flashscore_url,
    )
    if not result.get("ok"):
        err = str(result.get("error") or "Import failed")
        # Validation / bad URL → 400; other failures still returned as JSON body.
        low = err.casefold()
        if "url" in low or "hợp lệ" in low or "thiếu" in low:
            raise HTTPException(status_code=400, detail=err)
        return TeamImportResponse(**{**result, "ok": False, "error": err})
    return TeamImportResponse(**result)
