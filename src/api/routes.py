"""API v1 routers."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.auth import require_admin
from src.api.schemas import (
    ApiConfigOut,
    FixturesResponse,
    HealthOut,
    LeagueOut,
    LeaguesResponse,
    StakePolicyOut,
    TeamImportRequest,
    TeamImportResponse,
    TeamProfileOut,
    ValueBetsResponse,
)
from src.api.services import teams as teams_svc
from src.api.services import value_bets as vb_svc
from src.config import (
    DEFAULT_KELLY_FRACTION,
    DEFAULT_MIN_EV,
    MAX_BETS_PER_DAY,
    MAX_KELLY_FRACTION,
    MAX_STAKE_PCT,
    NEW_TEAM_MAX_STAKE_PCT,
)
from src.league_registry import get_available_leagues, is_cup_competition

logger = logging.getLogger(__name__)

# Project rule: EV >= 5 %. API EV thresholds are always in **percent**.
DEFAULT_MIN_EV_PCT = float(DEFAULT_MIN_EV) * 100.0

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
        Query(description="Minimum EV in percent (default 5.0 = 5%; 1.0 = 1%)"),
    ] = DEFAULT_MIN_EV_PCT,
    markets: Annotated[
        str | None,
        Query(description="Comma-separated: 1X2,AH,OU,Corners"),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    league: Annotated[str | None, Query()] = None,
    comp_id: Annotated[str | None, Query()] = None,
    leagues: Annotated[
        str | None,
        Query(
            description=(
                "Comma-separated league keys from /api/v1/leagues "
                "(overrides league/comp_id; default EPL,UWCL)"
            ),
        ),
    ] = None,
    kelly_fraction: Annotated[
        float | None,
        Query(
            gt=0.0,
            le=MAX_KELLY_FRACTION,
            description="Fractional Kelly multiplier (policy default / ceiling in /api/v1/config)",
        ),
    ] = None,
    one_per_match: Annotated[
        bool,
        Query(
            description=(
                "Keep only the best-EV leg per match (correlated legs such as "
                "1X2 Home and AH Home -0.5). false = research view"
            ),
        ),
    ] = True,
) -> ValueBetsResponse:
    """Prefer cached upcoming fixtures + model pickles for a fast response.

    Cold path (empty DB cache / missing ``models/*.pkl``) may fit Dixon–Coles
    and take several seconds on first call.
    """
    try:
        payload = await asyncio.to_thread(
            vb_svc.scan_value_bets_api,
            min_ev_pct=float(min_ev),
            markets=markets,
            limit=int(limit),
            league=league,
            comp_id=comp_id,
            leagues=leagues,
            kelly_fraction=kelly_fraction,
            one_per_match=bool(one_per_match),
        )
    except ValueError as exc:
        # Unknown league key / too many leagues.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ValueBetsResponse(**payload)


@api_v1.get(
    "/fixtures",
    response_model=FixturesResponse,
    tags=["value-bets"],
    summary="Every upcoming match (VN today/tomorrow) with its pricing status",
)
async def get_fixtures(
    leagues: Annotated[
        str | None,
        Query(description="Comma-separated league keys (default EPL,UWCL)"),
    ] = None,
    min_ev: Annotated[float, Query(description="EV floor in percent")] = DEFAULT_MIN_EV_PCT,
    markets: Annotated[str | None, Query(description="Comma-separated: 1X2,AH,OU")] = None,
    kelly_fraction: Annotated[
        float | None, Query(gt=0.0, le=MAX_KELLY_FRACTION)
    ] = None,
) -> FixturesResponse:
    try:
        payload = await asyncio.to_thread(
            vb_svc.list_fixtures_api,
            leagues=leagues,
            min_ev_pct=float(min_ev),
            markets=markets,
            kelly_fraction=kelly_fraction,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FixturesResponse(**payload)


@api_v1.get(
    "/leagues",
    response_model=LeaguesResponse,
    tags=["meta"],
    summary="Leagues from config/leagues.json (dashboard filter source)",
)
async def get_leagues() -> LeaguesResponse:
    default_scan = set(vb_svc.DEFAULT_SCAN_LEAGUES)
    rows = [
        LeagueOut(
            code=str(r["code"]),
            name=str(r.get("name") or r["code"]),
            country=r.get("country"),
            time_zone=r.get("time_zone"),
            league_weight=float(r.get("league_weight") or 1.0),
            is_cup=is_cup_competition(str(r["code"])),
            default_scan=str(r["code"]) in default_scan,
        )
        for r in get_available_leagues()
    ]
    return LeaguesResponse(leagues=rows, count=len(rows))


@api_v1.get(
    "/config",
    response_model=ApiConfigOut,
    tags=["meta"],
    summary="Stake / EV policy and data-freshness settings",
)
async def get_api_config() -> ApiConfigOut:
    return ApiConfigOut(
        policy=StakePolicyOut(
            min_ev_pct=DEFAULT_MIN_EV_PCT,
            kelly_fraction=float(DEFAULT_KELLY_FRACTION),
            max_kelly_fraction=float(MAX_KELLY_FRACTION),
            max_stake_pct=float(MAX_STAKE_PCT),
            new_team_max_stake_pct=float(NEW_TEAM_MAX_STAKE_PCT),
            max_bets_per_day=int(MAX_BETS_PER_DAY),
        ),
        markets=sorted(vb_svc.VALID_MARKETS),
        fixtures_max_age_minutes=float(vb_svc.fixtures_max_age_minutes()),
        server_time=datetime.now(timezone.utc).isoformat(),
    )


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
    summary="Import team history from a Flashscore team URL (admin only)",
    dependencies=[Depends(require_admin)],
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
