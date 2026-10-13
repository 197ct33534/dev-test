"""Pydantic request / response models for the public API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PastMatchOut(BaseModel):
    date: str | None = None
    competition: str | None = None
    comp_id: str | None = None
    home: str | None = None
    away: str | None = None
    home_score: int | None = None
    away_score: int | None = None
    score: str | None = None
    result: str | None = None


class UpcomingMatchOut(BaseModel):
    kickoff: str | None = None
    competition: str | None = None
    comp_id: str | None = None
    home: str | None = None
    away: str | None = None
    opponent: str | None = None
    is_home: bool | None = None


class TeamProfileOut(BaseModel):
    team_id: str | int
    db_team_id: int | None = None
    team_key: str | None = None
    canonical_name: str | None = None
    display_name: str | None = None
    past_matches: list[PastMatchOut] = Field(default_factory=list)
    upcoming_matches: list[UpcomingMatchOut] = Field(default_factory=list)
    as_of: str | None = None


class TeamImportRequest(BaseModel):
    team_id: str = Field(..., min_length=1, description="Club code or display name")
    flashscore_url: str = Field(..., min_length=1, description="Flashscore team page URL")


class TeamImportResponse(BaseModel):
    ok: bool
    error: str | None = None
    slug: str | None = None
    hash: str | None = None
    db_team_id: int | None = None
    matches_fetched: int | None = None
    last_match_date: str | None = None
    persist: dict[str, Any] | None = None
    rest: dict[str, Any] | None = None
    status: dict[str, Any] | None = None


class ValueBetOut(BaseModel):
    pick: str | None = None
    selection: str | None = None
    market: str | None = None
    odds: float | None = None
    bookmaker_odds: float | None = None
    ev: float | None = None
    ev_pct: float | None = None
    p_model: float | None = None
    fair_odds: float | None = None
    kelly_fraction: float | None = None
    kelly_pct: float | None = None
    stake: float | None = None
    home: str | None = None
    away: str | None = None
    home_team: str | None = None
    away_team: str | None = None
    home_display: str | None = Field(
        default=None, description="Friendly name (home/away stay stable team ids)"
    )
    away_display: str | None = None
    match_id: str | None = None
    league: str | None = None
    competition: str | None = None
    kickoff: str | None = None
    kickoff_vn: str | None = None
    home_rest_days: float | None = None
    away_rest_days: float | None = None
    fatigue_label: str | None = None
    ai_reasons: list[str] = Field(default_factory=list)
    recommended: bool | None = None
    home_thin: bool | None = None
    away_thin: bool | None = None
    thin_teams: list[str] = Field(default_factory=list)
    p_source: str | None = None
    new_team_fallback: bool = Field(
        default=False,
        description="Model priced a side with the new-team prior (stake capped)",
    )
    ev_method: str | None = Field(
        default=None, description='"settlement" for push-aware AH / O/U EV'
    )
    # Model fair line vs bookie (line disparity)
    fair_total_goals: float | None = None
    fair_ou_line: float | None = None
    fair_ah_line: float | None = None
    bookie_ou_line: float | None = None
    bookie_ah_line: float | None = None
    ou_line_delta: float | None = None
    ah_line_delta: float | None = None
    line_disparity_score: float | None = None
    model_fair_line: str | None = Field(
        default=None,
        description='e.g. "Tài Xỉu 2.75" or "AH -0.5"',
    )
    bookie_market_line: str | None = Field(
        default=None,
        description='e.g. "Tài Xỉu 2.25"',
    )
    line_edge: str | None = Field(
        default=None,
        description='e.g. "+0.50 bàn"',
    )


class LeagueScanMetaOut(BaseModel):
    """Per-league data-confidence for one scan (odds freshness, model)."""

    league: str
    label: str | None = None
    fixtures_source: str = Field(
        "empty", description="db_cache | network | empty"
    )
    stale: bool = False
    odds_age_minutes: float | None = None
    fixtures: int = 0
    model_note: str | None = None
    ml_enabled: bool = False
    model_source: str | None = Field(
        default=None, description="pickle | cold_fit | null (no model)"
    )
    excluded_prior: int = Field(
        0, description="Legs dropped: EV rested on new-team priors"
    )


class ValueBetsResponse(BaseModel):
    """Top value bets payload.

    Cold path (no cached fixtures / no ``models/*.pkl``) may fit Dixon–Coles
    and optionally LightGBM — first response can take seconds. Warm path reads
    ``upcoming_fixtures`` from ``global_matches.db`` and loads pickles when
    present.
    """

    bets: list[ValueBetOut]
    count: int
    min_ev_pct: float
    markets: list[str]
    limit: int
    league: str | None = None
    comp_id: str | None = None
    below_threshold: bool = False
    odds_missing: bool = False
    stale: bool = Field(
        default=False,
        description=(
            "True when odds came from an upcoming-fixtures cache older than "
            "FIXTURES_MAX_AGE_MINUTES because the network refresh failed"
        ),
    )
    stale_leagues: list[str] = Field(default_factory=list)
    source: str = "scan"
    generated_at: str | None = Field(
        default=None, description="Server UTC time the scan was computed (ISO)"
    )
    kelly_fraction: float | None = Field(
        default=None, description="Fractional-Kelly multiplier used for stakes"
    )
    leagues: list[str] = Field(default_factory=list)
    league_meta: list[LeagueScanMetaOut] = Field(default_factory=list)
    excluded_prior: int = Field(
        0,
        description=(
            "Legs dropped because both sides (or any side in a cup) were priced "
            "on new-team priors"
        ),
    )
    notes: str | None = None


class HealthOut(BaseModel):
    status: str = "ok"


class PerformanceAnalyticsOut(BaseModel):
    """Post-match paper-trading performance summary."""

    total_bets_placed: int = 0
    total_bets_settled: int = 0
    win_rate_percent: float = 0.0
    net_pnl: float = 0.0
    realized_roi_percent: float = 0.0
    ev_vs_realized_gap: float | None = None
    expected_ev_percent: float | None = None
    brier_score: float | None = None
    brier_note: str | None = Field(
        default=None,
        description="Explains missing Brier when insufficient p_model data",
    )
    days: int | None = None
    league: str | None = None
    notes: str | None = None



class LeagueOut(BaseModel):
    """One league from ``config/leagues.json`` (∪ hardcoded fallback)."""

    code: str
    name: str
    country: str | None = None
    time_zone: str | None = None
    league_weight: float = 1.0
    is_cup: bool = False
    default_scan: bool = Field(
        False, description="Scanned when /value-bets is called without leagues"
    )


class LeaguesResponse(BaseModel):
    leagues: list[LeagueOut]
    count: int


class StakePolicyOut(BaseModel):
    """Stake / EV policy from ``src.config`` — every UI renders these values."""

    min_ev_pct: float
    kelly_fraction: float
    max_kelly_fraction: float
    max_stake_pct: float
    new_team_max_stake_pct: float
    max_bets_per_day: int


class ApiConfigOut(BaseModel):
    policy: StakePolicyOut
    markets: list[str]
    fixtures_max_age_minutes: float
    server_time: str
