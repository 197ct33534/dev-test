import { Loader2, X } from "lucide-react";
import { formFromProfile, h2hFromProfiles } from "../api/adapters";
import { useTeamProfile } from "../api/hooks";
import { cn, formatKickoff, formatPct } from "../lib/utils";
import type { FormResult, H2HRow, MatchCardData } from "../types";

interface MatchDetailDrawerProps {
  match: MatchCardData | null;
  onClose: () => void;
  /** Fetch form / H2H from /api/v1/teams/{team}/profile (false in mock mode). */
  loadProfiles: boolean;
}

function FormDots({ form }: { form: FormResult[] }) {
  if (form.length === 0) return <span className="text-[11px] text-ink-faint">Chưa có dữ liệu</span>;
  const color = { W: "bg-accent", D: "bg-slate-500", L: "bg-accent-danger" };
  return (
    <div className="flex gap-1">
      {form.map((r, i) => (
        <span
          key={`${r}-${i}`}
          className={cn(
            "flex h-7 w-7 items-center justify-center rounded-md text-[10px] font-bold text-white",
            color[r],
          )}
        >
          {r}
        </span>
      ))}
    </div>
  );
}

function ProbBar({
  label,
  model,
  bookie,
}: {
  label: string;
  model: number;
  bookie: number;
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between text-xs text-ink-muted">
        <span>{label}</span>
        <span className="tabular-nums">
          Model {(model * 100).toFixed(1)}% · NC {(bookie * 100).toFixed(1)}%
        </span>
      </div>
      <div className="relative h-2 rounded-full bg-slate-800">
        <div
          className="absolute inset-y-0 left-0 rounded-full bg-slate-600"
          style={{ width: `${Math.min(100, bookie * 100)}%` }}
        />
        <div
          className="absolute inset-y-0 left-0 rounded-full bg-accent"
          style={{ width: `${Math.min(100, model * 100)}%` }}
        />
      </div>
    </div>
  );
}

export function MatchDetailDrawer({ match, onClose, loadProfiles }: MatchDetailDrawerProps) {
  const homeId = match ? match.homeId ?? match.homeTeam : null;
  const awayId = match ? match.awayId ?? match.awayTeam : null;
  const homeQ = useTeamProfile(loadProfiles ? homeId : null);
  const awayQ = useTeamProfile(loadProfiles ? awayId : null);
  if (!match) return null;

  const profilesLoading = loadProfiles && (homeQ.isLoading || awayQ.isLoading);
  const formHome = match.formHome ?? formFromProfile(homeQ.data, homeId ?? match.homeTeam);
  const formAway = match.formAway ?? formFromProfile(awayQ.data, awayId ?? match.awayTeam);
  const h2h: H2HRow[] =
    match.h2h ??
    h2hFromProfiles(homeQ.data, awayQ.data, homeId ?? match.homeTeam, awayId ?? match.awayTeam);
  // AH / O/U EV comes from the backend's exact settlement table (push / half
  // outcomes), so the simple P×O−1 identity only holds for 1X2.
  const simpleEv = match.betType === "1X2";
  const fairOdds = match.fairOdds ?? (match.modelProb > 0 ? 1 / match.modelProb : null);

  return (
    <>
      <button
        type="button"
        className="fixed inset-0 z-40 bg-black/65 backdrop-blur-[2px]"
        aria-label="Đóng chi tiết"
        onClick={onClose}
      />
      <aside className="fixed inset-y-0 right-0 z-50 flex w-full max-w-md flex-col border-l border-slate-800 bg-canvas-panel shadow-card animate-slide-up sm:animate-fade-in">
        <div className="flex items-start justify-between gap-3 border-b border-slate-800 px-5 py-4">
          <div>
            <p className="text-[11px] uppercase tracking-wider text-ink-faint">
              Chi tiết mô hình · {formatKickoff(match.matchTime)}
            </p>
            <h3 className="mt-1 font-display text-lg font-semibold text-ink">
              {match.homeTeam} vs {match.awayTeam}
            </h3>
            <p className="mt-1 text-xs text-accent-cyan">{match.leagueLabel}</p>
          </div>
          <button
            type="button"
            className="vb-btn-ghost h-9 w-9 !px-0"
            onClick={onClose}
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="vb-scrollbar flex-1 space-y-5 overflow-y-auto px-5 py-4">
          <section className="rounded-xl border border-slate-800 bg-slate-950/50 p-4">
            <p className="text-xs font-medium text-ink-muted">AI Pick</p>
            <p className="mt-1 text-base font-semibold text-ink">
              [{match.betType}] {match.recommendation} @ {match.odd.toFixed(2)}
            </p>
            <p className="mt-2 text-sm text-accent">
              EV {formatPct(match.evPct)} · Kelly {match.kellyPct.toFixed(1)}%
            </p>
          </section>

          <section className="space-y-3">
            <h4 className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-ink-faint">
              Phong độ 5 trận gần nhất
              {profilesLoading ? <Loader2 className="h-3 w-3 animate-spin" /> : null}
            </h4>
            <div className="space-y-3 rounded-xl border border-slate-800 p-3">
              <div className="flex items-center justify-between gap-2">
                <span className="truncate text-xs text-ink-muted">
                  {match.homeTeam}
                </span>
                <FormDots form={formHome} />
              </div>
              <div className="flex items-center justify-between gap-2">
                <span className="truncate text-xs text-ink-muted">
                  {match.awayTeam}
                </span>
                <FormDots form={formAway} />
              </div>
            </div>
          </section>

          <section className="space-y-3">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-ink-faint">
              Xác suất & công thức EV
            </h4>
            <div className="space-y-3 rounded-xl border border-slate-800 bg-slate-950/40 p-3">
              <ProbBar
                label="P(selection)"
                model={match.modelProb}
                bookie={match.impliedProb}
              />
              <div className="rounded-lg bg-slate-900/80 p-3 font-mono text-[11px] leading-relaxed text-ink-muted">
                {simpleEv ? (
                  <>
                    <p>EV = P_model × Odds − 1</p>
                    <p className="mt-1 text-ink">
                      = {match.modelProb.toFixed(3)} × {match.odd.toFixed(2)} − 1 ={" "}
                      <span className="text-accent">{formatPct(match.evPct)}</span>
                    </p>
                  </>
                ) : (
                  <>
                    <p>EV = Σ P(kết quả) × lãi/lỗ (thắng, thắng nửa, hoà tiền, thua nửa, thua)</p>
                    <p className="mt-1 text-ink">
                      = <span className="text-accent">{formatPct(match.evPct)}</span> @{" "}
                      {match.odd.toFixed(2)}
                    </p>
                  </>
                )}
                {fairOdds !== null ? (
                  <p className="mt-2">Fair odds = {fairOdds.toFixed(2)}</p>
                ) : null}
                {match.confidence.pSource ? (
                  <p className="mt-1 text-ink-faint">Xác suất: {match.confidence.pSource}</p>
                ) : null}
              </div>
            </div>
          </section>

          {match.aiReasons.length ? (
            <section className="space-y-2">
              <h4 className="text-xs font-semibold uppercase tracking-wider text-ink-faint">
                Lý do mô hình
              </h4>
              <ul className="space-y-1.5 text-xs text-ink-muted">
                {match.aiReasons.map((r) => (
                  <li key={r} className="flex gap-1.5">
                    <span className="text-accent-cyan">›</span>
                    <span>{r}</span>
                  </li>
                ))}
              </ul>
            </section>
          ) : null}

          <section className="space-y-3">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-ink-faint">
              Lịch sử đối đầu (H2H)
            </h4>
            {h2h.length === 0 ? (
              <p className="text-xs text-ink-faint">
                {profilesLoading ? "Đang tải…" : "Chưa có trận đối đầu gần đây trong DB."}
              </p>
            ) : null}
            <ul className="space-y-2">
              {h2h.map((row) => (
                <li
                  key={`${row.date}-${row.score}`}
                  className="flex items-center justify-between rounded-xl border border-slate-800 px-3 py-2 text-xs"
                >
                  <span className="text-ink-faint">{row.date}</span>
                  <span className="text-ink-muted">
                    {row.home} vs {row.away}
                  </span>
                  <span className="font-semibold tabular-nums text-ink">
                    {row.score}
                  </span>
                </li>
              ))}
            </ul>
          </section>
        </div>
      </aside>
    </>
  );
}
