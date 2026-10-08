export type BetMarket = "1X2" | "AH" | "OU" | "Corners";
export type ViewMode = "lite" | "pro";

export type NavId =
  | "top"
  | "detail"
  | "corners"
  | "compare"
  | "budget"
  | "journal"
  | "history"
  | "backtest";

export interface MatchCardData {
  id: string;
  matchTime: string; // ISO
  league: string;
  leagueLabel: string;
  homeTeam: string;
  awayTeam: string;
  homeAbbr: string;
  awayAbbr: string;
  betType: BetMarket;
  recommendation: string;
  odd: number;
  /** Recommended stake, % of bankroll (fractional Kelly, capped at 1%). */
  kellyPct: number;
  evPct: number;
  modelProb: number;
  impliedProb: number;
  modelOU: number;
  bookieOU: number;
  difference: number;
  formHome: ("W" | "D" | "L")[];
  formAway: ("W" | "D" | "L")[];
  h2h: { date: string; home: string; away: string; score: string }[];
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
