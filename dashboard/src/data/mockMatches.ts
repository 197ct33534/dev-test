import { cappedKellyPct, evPct } from "../lib/utils";
import type { MatchCardData } from "../types";

/**
 * MOCK DATA — used only when ``VITE_USE_MOCK=true`` (offline UI work).
 *
 * Kickoffs are generated relative to page load (always inside the next 48h)
 * and EV / stake are derived from ``modelProb`` × ``odd`` with the backend
 * stake policy (Kelly fraction 0.10, max 1% bankroll) so numbers stay
 * internally consistent.
 */
type RawMock = Omit<
  MatchCardData,
  "matchTime" | "kellyPct" | "evPct" | "fairOdds" | "aiReasons" | "confidence"
> & {
  kickoffInHours: number;
};

const LOADED_AT = Date.now();

/** When the mock snapshot was "generated" (a few minutes before page load). */
export const MOCK_DATA_UPDATED_AT = new Date(LOADED_AT - 4 * 60_000).toISOString();

const RAW_MOCKS: RawMock[] = [
  {
    id: "jlc-2509-urawa-kashima",
    kickoffInHours: 3,
    league: "J_LEAGUE_CUP",
    leagueLabel: "J.League Cup",
    homeTeam: "Urawa Red Diamonds",
    awayTeam: "Kashima Antlers",
    homeAbbr: "URA",
    awayAbbr: "KAS",
    betType: "1X2",
    recommendation: "Khách",
    odd: 4.5,
    modelProb: 0.25,
    impliedProb: 0.222,
    modelOU: 2.45,
    bookieOU: 2.5,
    difference: 0.05,
    formHome: ["W", "D", "L", "W", "W"],
    formAway: ["W", "W", "D", "W", "L"],
    h2h: [
      { date: "2025-08-12", home: "Urawa", away: "Kashima", score: "1-2" },
      { date: "2025-03-01", home: "Kashima", away: "Urawa", score: "2-1" },
      { date: "2024-10-19", home: "Urawa", away: "Kashima", score: "0-0" },
    ],
  },
  {
    id: "jlc-2509-yokohama-fc-tokyo",
    kickoffInHours: 5,
    league: "J_LEAGUE_CUP",
    leagueLabel: "J.League Cup",
    homeTeam: "Yokohama F. Marinos",
    awayTeam: "FC Tokyo",
    homeAbbr: "YFM",
    awayAbbr: "FCT",
    betType: "OU",
    recommendation: "Tài 2.5",
    odd: 1.85,
    modelProb: 0.594,
    impliedProb: 0.541,
    modelOU: 2.85,
    bookieOU: 2.5,
    difference: 0.35,
    formHome: ["W", "W", "W", "D", "L"],
    formAway: ["L", "D", "W", "L", "D"],
    h2h: [
      { date: "2025-07-20", home: "Marinos", away: "FC Tokyo", score: "3-1" },
      { date: "2025-02-15", home: "FC Tokyo", away: "Marinos", score: "1-1" },
    ],
  },
  {
    id: "jlc-2609-cerezo-gamba",
    kickoffInHours: 20,
    league: "J_LEAGUE_CUP",
    leagueLabel: "J.League Cup",
    homeTeam: "Cerezo Osaka",
    awayTeam: "Gamba Osaka",
    homeAbbr: "CER",
    awayAbbr: "GAM",
    betType: "AH",
    recommendation: "Chấp chủ -0.5",
    odd: 1.93,
    modelProb: 0.745,
    impliedProb: 0.518,
    modelOU: 2.6,
    bookieOU: 2.25,
    difference: 0.35,
    formHome: ["W", "W", "D", "W", "W"],
    formAway: ["L", "L", "D", "W", "L"],
    h2h: [
      { date: "2025-09-01", home: "Cerezo", away: "Gamba", score: "2-0" },
      { date: "2025-04-12", home: "Gamba", away: "Cerezo", score: "1-2" },
      { date: "2024-11-03", home: "Cerezo", away: "Gamba", score: "1-1" },
    ],
  },
  {
    id: "jlc-2609-vissel-sanfre",
    kickoffInHours: 20.5,
    league: "J_LEAGUE_CUP",
    leagueLabel: "J.League Cup",
    homeTeam: "Vissel Kobe",
    awayTeam: "Sanfrecce Hiroshima",
    homeAbbr: "VIS",
    awayAbbr: "SAN",
    betType: "1X2",
    recommendation: "Chủ",
    odd: 2.05,
    modelProb: 0.52,
    impliedProb: 0.488,
    modelOU: 2.15,
    bookieOU: 2.25,
    difference: -0.1,
    formHome: ["D", "W", "W", "D", "W"],
    formAway: ["W", "D", "L", "W", "D"],
    h2h: [
      { date: "2025-06-08", home: "Vissel", away: "Sanfrecce", score: "1-0" },
      { date: "2025-01-22", home: "Sanfrecce", away: "Vissel", score: "2-2" },
    ],
  },
  {
    id: "jlc-2709-kawasaki-nagoya",
    kickoffInHours: 29,
    league: "J_LEAGUE_CUP",
    leagueLabel: "J.League Cup",
    homeTeam: "Kawasaki Frontale",
    awayTeam: "Nagoya Grampus",
    homeAbbr: "KAW",
    awayAbbr: "NAG",
    betType: "OU",
    recommendation: "Xỉu 2.5",
    odd: 1.95,
    modelProb: 0.54,
    impliedProb: 0.513,
    modelOU: 2.1,
    bookieOU: 2.75,
    difference: -0.65,
    formHome: ["L", "W", "D", "W", "L"],
    formAway: ["D", "D", "L", "W", "D"],
    h2h: [
      { date: "2025-05-18", home: "Kawasaki", away: "Nagoya", score: "0-1" },
      { date: "2024-12-01", home: "Nagoya", away: "Kawasaki", score: "1-1" },
    ],
  },
  {
    id: "jlc-2709-avispa-shonan",
    kickoffInHours: 31,
    league: "J_LEAGUE_CUP",
    leagueLabel: "J.League Cup",
    homeTeam: "Avispa Fukuoka",
    awayTeam: "Shonan Bellmare",
    homeAbbr: "AVI",
    awayAbbr: "SHO",
    betType: "AH",
    recommendation: "Chấp khách +0.25",
    odd: 1.88,
    modelProb: 0.607,
    impliedProb: 0.532,
    modelOU: 2.35,
    bookieOU: 2.25,
    difference: 0.1,
    formHome: ["D", "L", "W", "D", "W"],
    formAway: ["W", "L", "L", "D", "W"],
    h2h: [
      { date: "2025-08-30", home: "Avispa", away: "Shonan", score: "1-1" },
      { date: "2025-03-22", home: "Shonan", away: "Avispa", score: "0-2" },
    ],
  },
  {
    id: "epl-2809-arsenal-liverpool",
    kickoffInHours: 40,
    league: "EPL",
    leagueLabel: "Premier League",
    homeTeam: "Arsenal",
    awayTeam: "Liverpool",
    homeAbbr: "ARS",
    awayAbbr: "LIV",
    betType: "1X2",
    recommendation: "Hòa",
    odd: 3.6,
    modelProb: 0.284,
    impliedProb: 0.278,
    modelOU: 2.9,
    bookieOU: 2.75,
    difference: 0.15,
    formHome: ["W", "W", "D", "W", "W"],
    formAway: ["W", "W", "W", "D", "W"],
    h2h: [
      { date: "2025-01-10", home: "Arsenal", away: "Liverpool", score: "2-2" },
      { date: "2024-10-27", home: "Liverpool", away: "Arsenal", score: "2-1" },
    ],
  },
];

export const MOCK_MATCHES: MatchCardData[] = RAW_MOCKS.map(
  ({ kickoffInHours, ...m }) => ({
    ...m,
    matchTime: new Date(LOADED_AT + kickoffInHours * 3_600_000).toISOString(),
    evPct: Number(evPct(m.modelProb, m.odd).toFixed(1)),
    kellyPct: Number(cappedKellyPct(m.modelProb, m.odd).toFixed(2)),
    fairOdds: 1 / m.modelProb,
    aiReasons: [],
    confidence: {
      oddsSource: "mock",
      oddsAgeMinutes: 4,
      stale: false,
      belowThreshold: false,
      thinTeams: [],
      newTeamPrior: false,
      mlEnabled: null,
      pSource: null,
    },
  }),
);

/** Keys mirror config/leagues.json (offline fallback; live list: /api/v1/leagues). */
export const LEAGUE_OPTIONS = [
  { id: "EPL", label: "Premier League" },
  { id: "UWCL", label: "UWCL (Women's Champions League)" },
  { id: "LALIGA", label: "LaLiga" },
  { id: "J_LEAGUE_CUP", label: "J.League Cup" },
  { id: "EMPERORS_CUP", label: "Emperor's Cup" },
] as const;

export const NAV_ITEMS = [
  { id: "top" as const, label: "Top Kèo Hời (Daily Cap)", icon: "flame" },
  { id: "detail" as const, label: "Chi Tiết Trận Đấu & Mô Hình", icon: "chart" },
  { id: "corners" as const, label: "Phân Tích Phạt Góc", icon: "corner" },
  { id: "compare" as const, label: "So Sánh Đội", icon: "swords" },
  { id: "budget" as const, label: "Quản Lý Ngân Sách & Bankroll", icon: "wallet" },
  { id: "journal" as const, label: "Nhật Ký Cược", icon: "scroll" },
  { id: "history" as const, label: "Dự Đoán & Lịch Sử", icon: "crystal" },
  { id: "backtest" as const, label: "Backtest Algorithm", icon: "flask" },
];
