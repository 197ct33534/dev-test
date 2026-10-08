import { X } from "lucide-react";
import { cn, formatKickoff, formatPct } from "../lib/utils";
import type { MatchCardData } from "../types";

interface MatchDetailDrawerProps {
  match: MatchCardData | null;
  onClose: () => void;
}

function FormDots({ form }: { form: ("W" | "D" | "L")[] }) {
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

export function MatchDetailDrawer({ match, onClose }: MatchDetailDrawerProps) {
  if (!match) return null;

  const ev = match.modelProb * match.odd - 1;

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
            <h4 className="text-xs font-semibold uppercase tracking-wider text-ink-faint">
              Phong độ 5 trận gần nhất
            </h4>
            <div className="space-y-3 rounded-xl border border-slate-800 p-3">
              <div className="flex items-center justify-between gap-2">
                <span className="truncate text-xs text-ink-muted">
                  {match.homeTeam}
                </span>
                <FormDots form={match.formHome} />
              </div>
              <div className="flex items-center justify-between gap-2">
                <span className="truncate text-xs text-ink-muted">
                  {match.awayTeam}
                </span>
                <FormDots form={match.formAway} />
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
                <p>EV = P_model × Odds − 1</p>
                <p className="mt-1 text-ink">
                  = {match.modelProb.toFixed(3)} × {match.odd.toFixed(2)} − 1 ={" "}
                  <span className="text-accent">{formatPct(ev * 100)}</span>
                </p>
                <p className="mt-2">
                  Fair odds = 1 / P_model ={" "}
                  {(1 / match.modelProb).toFixed(2)}
                </p>
              </div>
            </div>
          </section>

          <section className="space-y-3">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-ink-faint">
              Lịch sử đối đầu (H2H)
            </h4>
            <ul className="space-y-2">
              {match.h2h.map((row) => (
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
