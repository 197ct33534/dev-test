import { describe, expect, it } from "vitest";
import {
  formFromProfile,
  h2hFromProfiles,
  teamAbbr,
  toBetMarket,
  toMatchCard,
  toMatchCards,
  viSelection,
} from "./adapters";
import { buildValueBetsQuery } from "./client";
import type { LeagueScanMetaOut, TeamProfileOut, ValueBetOut } from "./types";

function bet(over: Partial<ValueBetOut> = {}): ValueBetOut {
  return {
    pick: "OU · Over 2.5 @ 1.95",
    selection: "Over 2.5",
    market: "OU",
    odds: 1.95,
    bookmaker_odds: 1.95,
    ev: 0.08,
    ev_pct: 8,
    p_model: 0.554,
    fair_odds: 1.805,
    kelly_fraction: 0.0084,
    kelly_pct: 0.84,
    stake: 8.4,
    home: "Arsenal",
    away: "Manchester United",
    match_id: "m1",
    league: "EPL",
    competition: "EPL - Premier League",
    kickoff: "2026-10-14T19:00:00+00:00",
    kickoff_vn: null,
    home_rest_days: null,
    away_rest_days: null,
    fatigue_label: null,
    ai_reasons: ["Model total 2.9 vs line 2.5"],
    recommended: true,
    home_thin: false,
    away_thin: true,
    thin_teams: [],
    p_source: "Dixon-Coles",
    fair_total_goals: 2.9,
    fair_ou_line: 2.75,
    fair_ah_line: -0.5,
    bookie_ou_line: 2.5,
    bookie_ah_line: -0.25,
    ou_line_delta: 0.25,
    ah_line_delta: -0.25,
    line_disparity_score: 0.3,
    model_fair_line: null,
    bookie_market_line: null,
    line_edge: null,
    ...over,
  };
}

const META: LeagueScanMetaOut = {
  league: "EPL",
  label: "EPL - Premier League",
  fixtures_source: "db_cache",
  stale: false,
  odds_age_minutes: 12,
  fixtures: 10,
  model_note: "dc_pickle",
  ml_enabled: false,
  model_source: "pickle",
};

describe("labels", () => {
  it("maps markets", () => {
    expect(toBetMarket("ou")).toBe("OU");
    expect(toBetMarket("CORNERS")).toBe("Corners");
    expect(toBetMarket("BTTS")).toBeNull();
  });

  it("translates selections", () => {
    expect(viSelection("Home")).toBe("Chủ nhà");
    expect(viSelection("Over 2.5")).toBe("Tài 2.5");
    expect(viSelection("Under 3")).toBe("Xỉu 3");
    expect(viSelection("AH Away +0.25")).toBe("Chấp Khách +0.25");
    expect(viSelection("Something")).toBe("Something");
    expect(viSelection(null)).toBe("—");
  });

  it("abbreviates team names", () => {
    expect(teamAbbr("Arsenal")).toBe("ARS");
    expect(teamAbbr("Manchester United")).toBe("MU");
    expect(teamAbbr("FC Tokyo")).toBe("TOK");
  });
});

describe("toMatchCard", () => {
  const meta = new Map([["EPL", META]]);

  it("uses backend EV / stake and league confidence", () => {
    const card = toMatchCard(bet(), meta)!;
    expect(card.betType).toBe("OU");
    expect(card.recommendation).toBe("Tài 2.5");
    expect(card.evPct).toBe(8);
    expect(card.kellyPct).toBe(0.84);
    expect(card.impliedProb).toBeCloseTo(1 / 1.95);
    expect(card.modelOU).toBe(2.75);
    expect(card.bookieOU).toBe(2.5);
    expect(card.difference).toBe(0.25);
    expect(card.confidence).toMatchObject({
      oddsSource: "db_cache",
      oddsAgeMinutes: 12,
      stale: false,
      belowThreshold: false,
      mlEnabled: false,
      thinTeams: ["Manchester United"],
    });
  });

  it("shows display names but keeps stable ids for lookups", () => {
    const card = toMatchCard(
      bet({
        home: "JP_NAGOYA",
        away: "JP_URAWA",
        home_display: "Nagoya Grampus",
        away_display: "Urawa Reds",
        away_thin: true,
      }),
      meta,
    )!;
    expect(card.homeTeam).toBe("Nagoya Grampus");
    expect(card.homeId).toBe("JP_NAGOYA");
    expect(card.homeAbbr).toBe("NG");
    expect(card.confidence.thinTeams).toEqual(["Urawa Reds"]);
  });

  it("flags the new-team prior", () => {
    expect(toMatchCard(bet(), meta)!.confidence.newTeamPrior).toBe(false);
    expect(toMatchCard(bet({ new_team_fallback: true }), meta)!.confidence.newTeamPrior).toBe(true);
  });

  it("flags fallback rows below the EV threshold", () => {
    expect(toMatchCard(bet({ recommended: false }), meta)!.confidence.belowThreshold).toBe(true);
  });

  it("keeps OU comparison null when lines are missing", () => {
    const card = toMatchCard(
      bet({ fair_ou_line: null, bookie_ou_line: null, ou_line_delta: null }),
      meta,
    )!;
    expect(card.modelOU).toBeNull();
    expect(card.difference).toBeNull();
  });

  it("drops unrenderable rows and duplicates", () => {
    expect(toMatchCard(bet({ odds: null, bookmaker_odds: null }), meta)).toBeNull();
    expect(toMatchCard(bet({ market: "BTTS" }), meta)).toBeNull();
    expect(toMatchCard(bet({ home: null }), meta)).toBeNull();
    expect(toMatchCards([bet(), bet()], [META])).toHaveLength(1);
  });

  it("works without league meta", () => {
    const card = toMatchCard(bet({ league: "UWCL" }), meta)!;
    expect(card.confidence.mlEnabled).toBeNull();
    expect(card.confidence.oddsAgeMinutes).toBeNull();
  });
});

describe("team profiles", () => {
  const arsenal: TeamProfileOut = {
    team_id: "Arsenal",
    canonical_name: "Arsenal",
    display_name: "Arsenal",
    past_matches: [
      { date: "2026-10-05", competition: "EPL", home: "Arsenal", away: "Chelsea", home_score: 2, away_score: 0, score: "2-0", result: null },
      { date: "2026-09-28", competition: "EPL", home: "Manchester United", away: "Arsenal", home_score: 1, away_score: 1, score: "1-1", result: null },
      { date: "2026-09-20", competition: "EPL", home: "Arsenal", away: "Spurs", home_score: 0, away_score: 1, score: "0-1", result: "L" },
      { date: "2026-09-13", competition: "EPL", home: "Arsenal", away: "Leeds", home_score: null, away_score: null, score: null, result: null },
    ],
  };

  it("derives W/D/L from scores when result is missing", () => {
    expect(formFromProfile(arsenal, "Arsenal")).toEqual(["W", "D", "L"]);
    expect(formFromProfile(undefined, "Arsenal")).toEqual([]);
  });

  it("finds head-to-head rows", () => {
    const rows = h2hFromProfiles(arsenal, undefined, "Arsenal", "Manchester United");
    expect(rows).toEqual([
      { date: "2026-09-28", home: "Manchester United", away: "Arsenal", score: "1-1" },
    ]);
  });
});

describe("buildValueBetsQuery", () => {
  it("serialises dashboard filters", () => {
    const q = new URLSearchParams(
      buildValueBetsQuery({
        leagues: ["EPL", "J_LEAGUE_CUP"],
        markets: ["1X2", "OU"],
        minEvPct: 5,
        kellyFraction: 0.08,
      }),
    );
    expect(q.get("leagues")).toBe("EPL,J_LEAGUE_CUP");
    expect(q.get("markets")).toBe("1X2,OU");
    expect(q.get("min_ev")).toBe("5");
    expect(q.get("kelly_fraction")).toBe("0.08");
    expect(q.get("limit")).toBe("50");
  });
});
