/**
 * Wire types for the FastAPI backend (`src/api/schemas.py`).
 * Keep in sync with the Pydantic models — every field the dashboard reads.
 */

export interface ValueBetOut {
  pick: string | null;
  selection: string | null;
  market: string | null;
  odds: number | null;
  bookmaker_odds: number | null;
  ev: number | null;
  ev_pct: number | null;
  p_model: number | null;
  fair_odds: number | null;
  kelly_fraction: number | null;
  kelly_pct: number | null;
  stake: number | null;
  home: string | null;
  away: string | null;
  /** Friendly names; ``home`` / ``away`` stay stable team ids (JP_*). */
  home_display?: string | null;
  away_display?: string | null;
  match_id: string | null;
  league: string | null;
  competition: string | null;
  kickoff: string | null;
  kickoff_vn: string | null;
  home_rest_days: number | null;
  away_rest_days: number | null;
  fatigue_label: string | null;
  ai_reasons: string[];
  recommended: boolean | null;
  home_thin: boolean | null;
  away_thin: boolean | null;
  thin_teams: string[];
  p_source: string | null;
  /** Dixon–Coles priced a side with the new-team prior (stake capped). */
  new_team_fallback?: boolean;
  ev_method?: string | null;
  fair_total_goals: number | null;
  fair_ou_line: number | null;
  fair_ah_line: number | null;
  bookie_ou_line: number | null;
  bookie_ah_line: number | null;
  ou_line_delta: number | null;
  ah_line_delta: number | null;
  line_disparity_score: number | null;
  model_fair_line: string | null;
  bookie_market_line: string | null;
  line_edge: string | null;
}

export interface LeagueScanMetaOut {
  league: string;
  label: string | null;
  fixtures_source: string;
  stale: boolean;
  odds_age_minutes: number | null;
  fixtures: number;
  model_note: string | null;
  ml_enabled: boolean;
  model_source: string | null;
  /** Legs dropped because EV rested on new-team priors. */
  excluded_prior?: number;
}

export interface ValueBetsResponse {
  bets: ValueBetOut[];
  count: number;
  min_ev_pct: number;
  markets: string[];
  limit: number;
  league: string | null;
  below_threshold: boolean;
  odds_missing: boolean;
  stale: boolean;
  stale_leagues: string[];
  source: string;
  generated_at: string | null;
  kelly_fraction: number | null;
  leagues: string[];
  league_meta: LeagueScanMetaOut[];
  excluded_prior?: number;
  notes: string | null;
}

export interface LeagueOut {
  code: string;
  name: string;
  country: string | null;
  time_zone: string | null;
  league_weight: number;
  is_cup: boolean;
  default_scan: boolean;
}

export interface LeaguesResponse {
  leagues: LeagueOut[];
  count: number;
}

export interface StakePolicyOut {
  min_ev_pct: number;
  kelly_fraction: number;
  max_kelly_fraction: number;
  max_stake_pct: number;
  new_team_max_stake_pct: number;
  max_bets_per_day: number;
}

export interface ApiConfigOut {
  policy: StakePolicyOut;
  markets: string[];
  fixtures_max_age_minutes: number;
  server_time: string;
}

export interface PastMatchOut {
  date: string | null;
  competition: string | null;
  home: string | null;
  away: string | null;
  home_score: number | null;
  away_score: number | null;
  score: string | null;
  result: string | null;
}

export interface TeamProfileOut {
  team_id: string | number;
  canonical_name: string | null;
  display_name: string | null;
  past_matches: PastMatchOut[];
}

export type FixtureStatus = "value" | "below" | "gated" | "no_odds" | "no_model";

export interface FixtureOut {
  match_id: string;
  league: string;
  league_label: string | null;
  /** UTC ISO with offset. */
  kickoff: string | null;
  home: string;
  away: string;
  home_display: string | null;
  away_display: string | null;
  odds_home: number | null;
  odds_draw: number | null;
  odds_away: number | null;
  bookmaker: string | null;
  ou_line: number | null;
  ah_line: number | null;
  status: FixtureStatus;
  gate_reason: string | null;
  best_pick: ValueBetOut | null;
}

export interface FixturesResponse {
  fixtures: FixtureOut[];
  count: number;
  min_ev_pct: number;
  leagues: string[];
  league_meta: LeagueScanMetaOut[];
  generated_at: string | null;
}

export interface ValueBetsParams {
  leagues: string[];
  markets: string[];
  minEvPct: number;
  kellyFraction: number;
  limit?: number;
}
