export type BetMarket = "1X2" | "AH" | "OU" | "Corners";
export type ViewMode = "lite" | "pro";
export type FormResult = "W" | "D" | "L";

export type NavId =
  | "top"
  | "detail"
  | "corners"
  | "compare"
  | "budget"
  | "journal"
  | "history"
  | "backtest";

export interface H2HRow {
  date: string;
  home: string;
  away: string;
  score: string;
}

/** Data-confidence flags shown on each card (plan Giai đoạn 4.4). */
export interface MatchConfidence {
  /** db_cache | network | mock */
  oddsSource: string | null;
  oddsAgeMinutes: number | null;
  /** Odds served from a cache older than FIXTURES_MAX_AGE_MINUTES. */
  stale: boolean;
  /** Fallback row shown although EV is below the backend threshold. */
  belowThreshold: boolean;
  thinTeams: string[];
  /** Model used the new-team prior (EV least reliable, stake capped at half). */
  newTeamPrior: boolean;
  /** LightGBM blended in (false → Dixon–Coles only). null = unknown. */
  mlEnabled: boolean | null;
  pSource: string | null;
  /** Backend model note, e.g. dc_pickle+lgbm_pickle / dc_pooled_fitted_cold. */
  modelNote?: string | null;
}

export interface MatchCardData {
  id: string;
  matchTime: string; // ISO
  league: string;
  leagueLabel: string;
  /** Display names. */
  homeTeam: string;
  awayTeam: string;
  /** Stable backend team ids (profile lookups); equal to names for mock data. */
  homeId?: string;
  awayId?: string;
  homeAbbr: string;
  awayAbbr: string;
  betType: BetMarket;
  recommendation: string;
  odd: number;
  /** Recommended stake, % of bankroll (backend fractional Kelly, capped). */
  kellyPct: number;
  /** Backend EV (exact settlement EV for AH / O/U incl. push & half outcomes). */
  evPct: number;
  modelProb: number;
  impliedProb: number;
  fairOdds: number | null;
  /** Model fair O/U line vs bookmaker main line (null when unknown). */
  modelOU: number | null;
  bookieOU: number | null;
  difference: number | null;
  aiReasons: string[];
  confidence: MatchConfidence;
  /** Only set for mock data — live data loads team profiles on demand. */
  formHome?: FormResult[];
  formAway?: FormResult[];
  h2h?: H2HRow[];
  tracked?: boolean;
}

export interface FilterState {
  viewMode: ViewMode;
  leagues: string[];
  minEv: number;
  /** Fractional Kelly in % (backend default 10 = 0.10); stake capped at 1% bankroll. */
  kellyFractionPct: number;
  markets: BetMarket[];
  query: string;
}
