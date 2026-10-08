import { useEffect, useMemo, useState } from "react";
import { Filter, Search } from "lucide-react";
import { Header } from "./components/Header";
import { MatchCard } from "./components/MatchCard";
import { MatchDetailDrawer } from "./components/MatchDetailDrawer";
import { Sidebar } from "./components/Sidebar";
import { MOCK_DATA_UPDATED_AT, MOCK_MATCHES } from "./data/mockMatches";
import { cappedKellyPct, formatAgo } from "./lib/utils";
import type { FilterState, MatchCardData, NavId } from "./types";

const DEFAULT_FILTERS: FilterState = {
  viewMode: "pro",
  leagues: ["J_LEAGUE_CUP", "EPL"],
  minEv: 5,
  kellyFractionPct: 10,
  markets: ["1X2", "AH", "OU"],
  query: "",
};

function filterMatches(rows: MatchCardData[], f: FilterState): MatchCardData[] {
  const q = f.query.trim().toLowerCase();
  return rows
    .map((m) => ({
      ...m,
      // Stake follows the chosen Kelly fraction, always capped at 1% bankroll.
      kellyPct: cappedKellyPct(m.modelProb, m.odd, f.kellyFractionPct / 100),
    }))
    .filter((m) => {
      if (f.leagues.length && !f.leagues.includes(m.league)) return false;
      if (f.markets.length && !f.markets.includes(m.betType)) return false;
      if (m.evPct < f.minEv) return false;
      if (!q) return true;
      const blob = `${m.homeTeam} ${m.awayTeam} ${m.league} ${m.leagueLabel} ${m.recommendation}`.toLowerCase();
      return blob.includes(q);
    })
    .sort((a, b) => b.evPct - a.evPct);
}

export default function App() {
  const [draft, setDraft] = useState<FilterState>(DEFAULT_FILTERS);
  const [applied, setApplied] = useState<FilterState>(DEFAULT_FILTERS);
  const [activeNav, setActiveNav] = useState<NavId>("top");
  const [mobileFilters, setMobileFilters] = useState(false);
  const [selected, setSelected] = useState<MatchCardData | null>(null);
  const [tracked, setTracked] = useState<Record<string, boolean>>({});

  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 60_000);
    return () => window.clearInterval(id);
  }, []);

  // Live filter: EV / Kelly / search / markets update the feed immediately;
  // leagues apply on "Áp Dụng". Memoized so useMemo below is not defeated.
  const liveFilters = useMemo<FilterState>(
    () => ({
      ...applied,
      minEv: draft.minEv,
      kellyFractionPct: draft.kellyFractionPct,
      markets: draft.markets,
      query: draft.query,
      viewMode: draft.viewMode,
    }),
    [
      applied,
      draft.minEv,
      draft.kellyFractionPct,
      draft.markets,
      draft.query,
      draft.viewMode,
    ],
  );

  const matches = useMemo(() => {
    const rows = MOCK_MATCHES.map((m) => ({
      ...m,
      tracked: Boolean(tracked[m.id]),
    }));
    return filterMatches(rows, liveFilters);
  }, [liveFilters, tracked]);

  const updatedAgo = formatAgo(MOCK_DATA_UPDATED_AT, now);

  const apply = () => {
    setApplied({ ...draft });
    setMobileFilters(false);
  };

  const reset = () => {
    setDraft(DEFAULT_FILTERS);
    setApplied(DEFAULT_FILTERS);
  };

  return (
    <div className="flex min-h-screen flex-col">
      <Header
        activeNav={activeNav}
        onNavChange={setActiveNav}
        viewMode={draft.viewMode}
        onViewModeChange={(viewMode) => setDraft((d) => ({ ...d, viewMode }))}
        updatedAgo={updatedAgo}
        isMock
      />

      <div className="mx-auto flex w-full max-w-[1600px] flex-1">
        <Sidebar
          filters={applied}
          draft={draft}
          onDraftChange={setDraft}
          onApply={apply}
          onReset={reset}
          mobileOpen={mobileFilters}
          onCloseMobile={() => setMobileFilters(false)}
        />

        <main className="min-w-0 flex-1 px-4 py-5 lg:px-6">
          {activeNav !== "top" ? (
            <div className="vb-card flex min-h-[40vh] flex-col items-center justify-center gap-2 p-10 text-center">
              <p className="font-display text-lg font-semibold text-ink">
                Module đang kết nối pipeline Streamlit / API
              </p>
              <p className="max-w-md text-sm text-ink-muted">
                Tab này giữ chỗ điều hướng Pro. Feed Value Betting chính nằm ở{" "}
                <button
                  type="button"
                  className="text-accent underline-offset-2 hover:underline"
                  onClick={() => setActiveNav("top")}
                >
                  Top Kèo Hời
                </button>
                .
              </p>
            </div>
          ) : (
            <>
              <div className="mb-5 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
                <div>
                  <h1 className="font-display text-2xl font-semibold tracking-tight text-ink">
                    Top Kèo Hời{" "}
                    <span className="text-ink-faint">(Daily Cap)</span>
                  </h1>
                  <p className="mt-1 text-sm text-ink-muted">
                    Tìm thấy{" "}
                    <span className="font-semibold text-accent">
                      {matches.length} kèo hời
                    </span>{" "}
                    (+EV ≥ {liveFilters.minEv.toFixed(0)}%) trong 48 giờ tới ·
                    Kelly {liveFilters.kellyFractionPct.toFixed(0)}% (stake ≤ 1%)
                    {draft.viewMode === "pro" ? " · Pro filters" : " · Lite"}
                    <span className="ml-1 text-accent-warn">· dữ liệu mẫu</span>
                  </p>
                </div>
                <button
                  type="button"
                  className="vb-btn-ghost lg:hidden"
                  onClick={() => setMobileFilters(true)}
                >
                  <Filter className="h-4 w-4" />
                  Bộ lọc
                </button>
              </div>

              <div className="relative mb-5">
                <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-faint" />
                <input
                  className="vb-input pl-10"
                  placeholder="Tìm CLB hoặc giải đấu (vd: Cerezo, J_LEAGUE_CUP)…"
                  value={draft.query}
                  onChange={(e) =>
                    setDraft((d) => ({ ...d, query: e.target.value }))
                  }
                />
              </div>

              <div className="space-y-3">
                {matches.length === 0 ? (
                  <div className="vb-card p-8 text-center text-sm text-ink-muted">
                    Không có kèo thỏa ngưỡng hiện tại. Hạ EV% hoặc mở thêm thị
                    trường / giải đấu.
                  </div>
                ) : (
                  matches.map((m) => (
                    <MatchCard
                      key={m.id}
                      match={m}
                      onOpenDetail={setSelected}
                      onToggleTrack={(id) =>
                        setTracked((t) => ({ ...t, [id]: !t[id] }))
                      }
                    />
                  ))
                )}
              </div>
            </>
          )}
        </main>
      </div>

      <MatchDetailDrawer match={selected} onClose={() => setSelected(null)} />
    </div>
  );
}
