import { useEffect, useMemo, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, Filter, Loader2, Search } from "lucide-react";
import { Header, type FeedStatus } from "./components/Header";
import { MatchCard } from "./components/MatchCard";
import { MatchDetailDrawer } from "./components/MatchDetailDrawer";
import { Sidebar, type LeagueOption } from "./components/Sidebar";
import { TokenGate } from "./components/TokenGate";
import { toMatchCards } from "./api/adapters";
import { ApiError, getStoredToken, setStoredToken } from "./api/client";
import { useApiConfig, useLeagues, useValueBets } from "./api/hooks";
import type { LeagueScanMetaOut } from "./api/types";
import { LEAGUE_OPTIONS, MOCK_DATA_UPDATED_AT, MOCK_MATCHES } from "./data/mockMatches";
import { cappedKellyPct, formatAgo } from "./lib/utils";
import type { FilterState, MatchCardData, NavId } from "./types";

const USE_MOCK = import.meta.env.VITE_USE_MOCK === "true";

/** Backend default scan set (``DEFAULT_SCAN_LEAGUES`` in value_bets.py). */
const DEFAULT_LEAGUES = USE_MOCK ? ["J_LEAGUE_CUP", "EPL"] : ["EPL", "UWCL"];

const DEFAULT_FILTERS: FilterState = {
  viewMode: "pro",
  leagues: DEFAULT_LEAGUES,
  minEv: 5,
  kellyFractionPct: 10,
  markets: ["1X2", "AH", "OU"],
  query: "",
};

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const id = window.setTimeout(() => setV(value), ms);
    return () => window.clearTimeout(id);
  }, [value, ms]);
  return v;
}

function filterMatches(rows: MatchCardData[], f: FilterState): MatchCardData[] {
  const q = f.query.trim().toLowerCase();
  return rows
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

function isUnauthorized(err: unknown): boolean {
  return err instanceof ApiError && err.status === 401;
}

export default function App() {
  const queryClient = useQueryClient();
  const [draft, setDraft] = useState<FilterState>(DEFAULT_FILTERS);
  const [applied, setApplied] = useState<FilterState>(DEFAULT_FILTERS);
  const [activeNav, setActiveNav] = useState<NavId>("top");
  const [mobileFilters, setMobileFilters] = useState(false);
  const [selected, setSelected] = useState<MatchCardData | null>(null);
  const [tracked, setTracked] = useState<Record<string, boolean>>({});
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 30_000);
    return () => window.clearInterval(id);
  }, []);

  // --- Server data -------------------------------------------------------
  const configQ = useApiConfig(!USE_MOCK);
  const leaguesQ = useLeagues(!USE_MOCK);
  const policy = configQ.data?.policy;

  // Kelly is priced server-side (exact settlement Kelly + new-team caps), so
  // slider moves refetch — debounced to avoid one request per tick.
  const kellyPct = useDebounced(draft.kellyFractionPct, 400);
  const betsQ = useValueBets(
    {
      leagues: applied.leagues,
      markets: applied.markets,
      minEvPct: applied.minEv,
      kellyFraction: kellyPct / 100,
    },
    !USE_MOCK,
  );

  const unauthorized =
    !USE_MOCK && [betsQ.error, leaguesQ.error, configQ.error].some(isUnauthorized);

  const leagueOptions: LeagueOption[] = useMemo(
    () =>
      leaguesQ.data?.leagues.map((l) => ({ id: l.code, label: l.name })) ??
      LEAGUE_OPTIONS.map((l) => ({ id: l.id, label: l.label })),
    [leaguesQ.data],
  );

  // Live filter: EV / search / markets narrow the fetched feed immediately;
  // leagues + server-side EV floor apply on "Áp Dụng".
  const liveFilters = useMemo<FilterState>(
    () => ({
      ...applied,
      minEv: draft.minEv,
      kellyFractionPct: draft.kellyFractionPct,
      markets: draft.markets,
      query: draft.query,
      viewMode: draft.viewMode,
    }),
    [applied, draft.minEv, draft.kellyFractionPct, draft.markets, draft.query, draft.viewMode],
  );

  const rows = useMemo<MatchCardData[]>(() => {
    if (USE_MOCK) {
      return MOCK_MATCHES.map((m) => ({
        ...m,
        kellyPct: cappedKellyPct(m.modelProb, m.odd, liveFilters.kellyFractionPct / 100),
      }));
    }
    const data = betsQ.data;
    return data ? toMatchCards(data.bets, data.league_meta) : [];
  }, [betsQ.data, liveFilters.kellyFractionPct]);

  const matches = useMemo(
    () =>
      filterMatches(
        rows.map((m) => ({ ...m, tracked: Boolean(tracked[m.id]) })),
        liveFilters,
      ),
    [rows, liveFilters, tracked],
  );

  const updatedAt = USE_MOCK ? MOCK_DATA_UPDATED_AT : betsQ.data?.generated_at ?? null;
  const updatedAgo = updatedAt ? formatAgo(updatedAt, now) : "—";

  let status: FeedStatus;
  if (USE_MOCK) status = "mock";
  else if (betsQ.isError && !betsQ.data) status = "error";
  else if (!betsQ.data) status = "loading";
  else if (betsQ.data.stale) status = "stale";
  else status = "live";

  const apply = () => {
    setApplied({ ...draft });
    setMobileFilters(false);
  };

  const reset = () => {
    setDraft(DEFAULT_FILTERS);
    setApplied(DEFAULT_FILTERS);
  };

  const saveToken = (token: string) => {
    setStoredToken(token);
    void queryClient.invalidateQueries();
  };

  const data = betsQ.data;
  const errorMsg =
    betsQ.error instanceof Error && !isUnauthorized(betsQ.error) ? betsQ.error.message : null;

  return (
    <div className="flex min-h-screen flex-col">
      <Header
        activeNav={activeNav}
        onNavChange={setActiveNav}
        viewMode={draft.viewMode}
        onViewModeChange={(viewMode) => setDraft((d) => ({ ...d, viewMode }))}
        updatedAgo={updatedAgo}
        status={status}
        refreshing={betsQ.isFetching}
        onRefresh={USE_MOCK ? undefined : () => void betsQ.refetch()}
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
          leagueOptions={leagueOptions}
          maxKellyPct={policy ? policy.max_kelly_fraction * 100 : 10}
          maxStakePct={policy ? policy.max_stake_pct * 100 : 1}
        />

        <main className="min-w-0 flex-1 px-4 py-5 lg:px-6">
          {unauthorized ? (
            <TokenGate rejected={Boolean(getStoredToken())} onSubmit={saveToken} />
          ) : activeNav !== "top" ? (
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
                    Top Kèo Hời <span className="text-ink-faint">(Daily Cap)</span>
                  </h1>
                  <p className="mt-1 text-sm text-ink-muted">
                    Tìm thấy{" "}
                    <span className="font-semibold text-accent">{matches.length} kèo hời</span>{" "}
                    (+EV ≥ {liveFilters.minEv.toFixed(1)}%) hôm nay & ngày mai (giờ VN) · Kelly{" "}
                    {liveFilters.kellyFractionPct.toFixed(0)}% (stake ≤{" "}
                    {policy ? (policy.max_stake_pct * 100).toFixed(0) : 1}%)
                    {draft.viewMode === "pro" ? " · Pro filters" : " · Lite"}
                    {USE_MOCK ? <span className="ml-1 text-accent-warn">· dữ liệu mẫu</span> : null}
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

              {data?.stale ? (
                <Banner tone="warn">
                  Odds cũ (cache quá {configQ.data?.fixtures_max_age_minutes ?? 60} phút, làm mới
                  thất bại) cho: {data.stale_leagues.join(", ")}. Không đặt cược theo các kèo này.
                </Banner>
              ) : null}
              {data?.below_threshold && data.count > 0 ? (
                <Banner tone="info">
                  Không có kèo đạt ngưỡng EV của backend — đang hiển thị kèo gần ngưỡng (gắn nhãn
                  “Dưới ngưỡng”).
                </Banner>
              ) : null}
              {data?.excluded_prior ? (
                <Banner tone="info">
                  Đã loại {data.excluded_prior} kèo vì mô hình thiếu dữ liệu đội
                  {excludedLeagues(data.league_meta)} — EV dựa trên prior đội mới không đáng tin.
                </Banner>
              ) : null}
              {errorMsg && data ? (
                <Banner tone="warn">Làm mới thất bại: {errorMsg}. Đang hiển thị dữ liệu trước đó.</Banner>
              ) : null}

              <div className="relative mb-5">
                <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-faint" />
                <input
                  className="vb-input pl-10"
                  placeholder="Tìm CLB hoặc giải đấu (vd: Arsenal, EPL)…"
                  value={draft.query}
                  onChange={(e) => setDraft((d) => ({ ...d, query: e.target.value }))}
                />
              </div>

              <div className="space-y-3">
                {status === "loading" ? (
                  <div className="vb-card flex items-center justify-center gap-2 p-8 text-sm text-ink-muted">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Đang quét kèo… (lần đầu có thể mất vài giây khi server fit mô hình)
                  </div>
                ) : status === "error" ? (
                  <div className="vb-card space-y-3 p-8 text-center text-sm text-ink-muted">
                    <p>Không tải được feed: {errorMsg ?? "lỗi không xác định"}.</p>
                    <button type="button" className="vb-btn-ghost" onClick={() => void betsQ.refetch()}>
                      Thử lại
                    </button>
                  </div>
                ) : matches.length === 0 ? (
                  <div className="vb-card p-8 text-center text-sm text-ink-muted">
                    Không có kèo thỏa ngưỡng hiện tại. Hạ EV% hoặc mở thêm thị trường / giải đấu.
                    {data?.notes && draft.viewMode === "pro" ? (
                      <p className="mt-2 text-[11px] text-ink-faint">{data.notes}</p>
                    ) : null}
                  </div>
                ) : (
                  matches.map((m) => (
                    <MatchCard
                      key={m.id}
                      match={m}
                      viewMode={draft.viewMode}
                      onOpenDetail={setSelected}
                      onToggleTrack={(id) => setTracked((t) => ({ ...t, [id]: !t[id] }))}
                    />
                  ))
                )}
              </div>
            </>
          )}
        </main>
      </div>

      <MatchDetailDrawer
        match={selected}
        onClose={() => setSelected(null)}
        loadProfiles={!USE_MOCK}
      />
    </div>
  );
}

function excludedLeagues(meta: LeagueScanMetaOut[]): string {
  const hit = meta.filter((m) => (m.excluded_prior ?? 0) > 0).map((m) => m.label ?? m.league);
  return hit.length ? ` (${hit.join(", ")})` : "";
}

function Banner({ tone, children }: { tone: "warn" | "info"; children: React.ReactNode }) {
  return (
    <div
      className={
        tone === "warn"
          ? "mb-4 flex items-start gap-2 rounded-xl border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-accent-warn"
          : "mb-4 flex items-start gap-2 rounded-xl border border-cyan-900/60 bg-cyan-950/30 px-3 py-2 text-xs text-accent-cyan"
      }
    >
      <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
      <span>{children}</span>
    </div>
  );
}
