import type {
  BetMarket,
  FormResult,
  H2HRow,
  MatchCardData,
} from "../types";
import type {
  LeagueScanMetaOut,
  PastMatchOut,
  TeamProfileOut,
  ValueBetOut,
} from "./types";

const MARKETS: readonly BetMarket[] = ["1X2", "AH", "OU", "Corners"];

export function toBetMarket(raw: string | null): BetMarket | null {
  const key = (raw ?? "").trim().toUpperCase();
  if (key === "CORNERS") return "Corners";
  return (MARKETS as readonly string[]).includes(key) ? (key as BetMarket) : null;
}

/** Short badge text: "Manchester United" → "MU", "Arsenal" → "ARS". */
export function teamAbbr(name: string): string {
  const words = name
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .split(/\s+/)
    .filter(Boolean)
    .filter((w) => !/^(fc|cf|afc|sc|the)$/i.test(w));
  if (words.length === 0) return "?";
  if (words.length === 1) return words[0].slice(0, 3).toUpperCase();
  return words
    .slice(0, 3)
    .map((w) => w[0])
    .join("")
    .toUpperCase();
}

/** Backend selection labels (src/recommender.py) → Vietnamese UI text. */
export function viSelection(selection: string | null): string {
  const s = (selection ?? "").trim();
  if (!s) return "—";
  const simple: Record<string, string> = {
    Home: "Chủ nhà",
    Draw: "Hòa",
    Away: "Khách",
  };
  if (simple[s]) return simple[s];
  const ou = /^(Over|Under)\s+(.+)$/i.exec(s);
  if (ou) return `${ou[1].toLowerCase() === "over" ? "Tài" : "Xỉu"} ${ou[2]}`;
  const ah = /^AH\s+(Home|Away)\s+(.+)$/i.exec(s);
  if (ah) return `Chấp ${ah[1].toLowerCase() === "home" ? "Chủ" : "Khách"} ${ah[2]}`;
  return s;
}

function num(v: number | null | undefined, fallback = 0): number {
  return typeof v === "number" && Number.isFinite(v) ? v : fallback;
}

function maybeNum(v: number | null | undefined): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/**
 * Map one API value bet to the card model. Returns ``null`` for rows the
 * dashboard cannot render (no teams / odds / unknown market).
 */
export function toMatchCard(
  bet: ValueBetOut,
  metaByLeague: Map<string, LeagueScanMetaOut>,
): MatchCardData | null {
  const market = toBetMarket(bet.market);
  const odd = maybeNum(bet.odds ?? bet.bookmaker_odds);
  if (!market || !bet.home || !bet.away || odd === null || odd <= 1) return null;

  const league = bet.league ?? "";
  const meta = metaByLeague.get(league);
  const modelProb = num(bet.p_model);
  const evPct = maybeNum(bet.ev_pct) ?? (bet.ev != null ? bet.ev * 100 : modelProb * odd * 100 - 100);
  const modelOU = maybeNum(bet.fair_ou_line);
  const bookieOU = maybeNum(bet.bookie_ou_line);
  const difference =
    maybeNum(bet.ou_line_delta) ??
    (modelOU !== null && bookieOU !== null ? modelOU - bookieOU : null);

  const homeName = bet.home_display || bet.home;
  const awayName = bet.away_display || bet.away;
  const display = (t: string) =>
    t === bet.home ? homeName : t === bet.away ? awayName : t;
  const thinTeams = (bet.thin_teams?.length
    ? bet.thin_teams
    : [bet.home_thin ? bet.home : null, bet.away_thin ? bet.away : null].filter(
        (t): t is string => Boolean(t),
      )
  ).map(display);

  return {
    id: [bet.match_id ?? `${bet.home}-${bet.away}-${bet.kickoff}`, market, bet.selection ?? ""].join("|"),
    matchTime: bet.kickoff ?? "",
    league,
    leagueLabel: bet.competition ?? meta?.label ?? league,
    homeTeam: homeName,
    awayTeam: awayName,
    homeId: bet.home,
    awayId: bet.away,
    homeAbbr: teamAbbr(homeName),
    awayAbbr: teamAbbr(awayName),
    betType: market,
    recommendation: viSelection(bet.selection),
    odd,
    kellyPct: num(bet.kelly_pct),
    evPct,
    modelProb,
    impliedProb: 1 / odd,
    fairOdds: maybeNum(bet.fair_odds),
    modelOU,
    bookieOU,
    difference,
    aiReasons: bet.ai_reasons ?? [],
    confidence: {
      oddsSource: meta?.fixtures_source ?? null,
      oddsAgeMinutes: maybeNum(meta?.odds_age_minutes),
      stale: Boolean(meta?.stale),
      belowThreshold: bet.recommended === false,
      thinTeams,
      newTeamPrior: Boolean(bet.new_team_fallback),
      mlEnabled: meta ? meta.ml_enabled : null,
      pSource: bet.p_source,
      modelNote: meta?.model_note ?? null,
    },
  };
}

export function toMatchCards(
  bets: ValueBetOut[],
  meta: LeagueScanMetaOut[],
): MatchCardData[] {
  const byLeague = new Map(meta.map((m) => [m.league, m]));
  const out: MatchCardData[] = [];
  const seen = new Set<string>();
  for (const bet of bets) {
    const card = toMatchCard(bet, byLeague);
    if (!card || seen.has(card.id)) continue;
    seen.add(card.id);
    out.push(card);
  }
  return out;
}

function sameTeam(a: string | null, b: string | null): boolean {
  return Boolean(a && b) && a!.trim().toLowerCase() === b!.trim().toLowerCase();
}

function teamNames(profile: TeamProfileOut, fallback: string): string[] {
  return [profile.display_name, profile.canonical_name, String(profile.team_id), fallback].filter(
    (n): n is string => Boolean(n),
  );
}

/** W/D/L for ``team`` in one past match (uses ``result`` when it is W/D/L). */
export function matchResultFor(m: PastMatchOut, names: string[]): FormResult | null {
  const r = (m.result ?? "").trim().toUpperCase();
  if (r === "W" || r === "D" || r === "L") return r;
  if (m.home_score == null || m.away_score == null) return null;
  const isHome = names.some((n) => sameTeam(m.home, n));
  const isAway = names.some((n) => sameTeam(m.away, n));
  if (!isHome && !isAway) return null;
  if (m.home_score === m.away_score) return "D";
  const homeWon = m.home_score > m.away_score;
  return homeWon === isHome ? "W" : "L";
}

/** Last-5 form, most recent first. */
export function formFromProfile(profile: TeamProfileOut | undefined, team: string): FormResult[] {
  if (!profile) return [];
  const names = teamNames(profile, team);
  return profile.past_matches
    .map((m) => matchResultFor(m, names))
    .filter((r): r is FormResult => r !== null)
    .slice(0, 5);
}

/** Head-to-head rows found in either team's recent matches (deduped). */
export function h2hFromProfiles(
  home: TeamProfileOut | undefined,
  away: TeamProfileOut | undefined,
  homeTeam: string,
  awayTeam: string,
): H2HRow[] {
  const homeNames = home ? teamNames(home, homeTeam) : [homeTeam];
  const awayNames = away ? teamNames(away, awayTeam) : [awayTeam];
  const involves = (m: PastMatchOut, names: string[]) =>
    names.some((n) => sameTeam(m.home, n) || sameTeam(m.away, n));
  const rows = new Map<string, H2HRow>();
  for (const m of [...(home?.past_matches ?? []), ...(away?.past_matches ?? [])]) {
    if (!involves(m, homeNames) || !involves(m, awayNames)) continue;
    const score =
      m.score ?? (m.home_score != null && m.away_score != null ? `${m.home_score}-${m.away_score}` : "");
    const date = (m.date ?? "").slice(0, 10);
    rows.set(`${date}|${m.home}|${m.away}`, {
      date,
      home: m.home ?? "",
      away: m.away ?? "",
      score,
    });
  }
  return [...rows.values()].sort((a, b) => b.date.localeCompare(a.date));
}
