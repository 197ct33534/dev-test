import { viSelection } from "../api/adapters";
import type { FixtureOut, FixtureStatus } from "../api/types";
import { cn, formatKickoff, formatPct } from "../lib/utils";

const STATUS: Record<FixtureStatus, { label: string; cls: string; hint: string }> = {
  value: {
    label: "Có kèo",
    cls: "border-emerald-500/30 bg-emerald-500/10 text-accent",
    hint: "Kèo tốt nhất của trận đạt ngưỡng EV",
  },
  below: {
    label: "Dưới ngưỡng",
    cls: "border-slate-700 bg-slate-900 text-ink-muted",
    hint: "Đã định giá, nhưng không kèo nào đạt ngưỡng EV",
  },
  gated: {
    label: "Không định giá",
    cls: "border-amber-500/30 bg-amber-500/10 text-accent-warn",
    hint: "Mô hình thiếu dữ liệu đội — EV chỉ dựa trên prior đội mới nên bị bỏ",
  },
  no_odds: {
    label: "Chưa có odds",
    cls: "border-slate-700 bg-slate-900 text-ink-faint",
    hint: "Nhà cái chưa mở giá cho trận này",
  },
  no_model: {
    label: "Lỗi mô hình",
    cls: "border-red-500/30 bg-red-500/10 text-accent-danger",
    hint: "Không tải / fit được mô hình cho giải này",
  },
};

const GATE_TEXT: Record<string, string> = {
  both_teams_prior: "cả hai đội thiếu dữ liệu",
  cup_new_team_prior: "giải cúp, có đội thiếu dữ liệu",
};

function odd(v: number | null): string {
  return v == null ? "—" : v.toFixed(2);
}

function line(v: number | null): string {
  if (v == null) return "—";
  return `${v > 0 ? "+" : ""}${v}`;
}

interface FixtureListProps {
  fixtures: FixtureOut[];
}

/** Every match in the VN window grouped by league, with its pricing status. */
export function FixtureList({ fixtures }: FixtureListProps) {
  const groups = new Map<string, FixtureOut[]>();
  for (const f of fixtures) {
    const key = f.league_label ?? f.league;
    groups.set(key, [...(groups.get(key) ?? []), f]);
  }

  return (
    <div className="space-y-4">
      {[...groups.entries()].map(([label, rows]) => (
        <section key={label} className="vb-card overflow-hidden">
          <header className="flex items-center justify-between border-b border-slate-800 px-4 py-2.5">
            <h2 className="text-sm font-semibold text-ink">{label}</h2>
            <span className="text-[11px] text-ink-faint">{rows.length} trận</span>
          </header>
          <ul className="divide-y divide-slate-800/80">
            {rows.map((f) => {
              const st = STATUS[f.status] ?? STATUS.below;
              const pick = f.best_pick;
              return (
                <li
                  key={f.match_id}
                  className="grid gap-2 px-4 py-3 text-sm md:grid-cols-[4.5rem_minmax(0,1fr)_10rem_7rem_minmax(0,12rem)] md:items-center"
                >
                  <span className="tabular-nums text-xs text-ink-muted">
                    {f.kickoff ? formatKickoff(f.kickoff) : "—"}
                  </span>
                  <span className="min-w-0 truncate font-medium text-ink">
                    {f.home_display ?? f.home}{" "}
                    <span className="text-ink-faint">vs</span> {f.away_display ?? f.away}
                  </span>
                  <span
                    className="tabular-nums text-xs text-ink-muted"
                    title={f.bookmaker ? `1 / X / 2 — ${f.bookmaker}` : "1 / X / 2"}
                  >
                    {odd(f.odds_home)} · {odd(f.odds_draw)} · {odd(f.odds_away)}
                  </span>
                  <span className="text-xs text-ink-faint" title="Chấp chủ nhà · Tài xỉu">
                    AH {line(f.ah_line)} · O/U {f.ou_line ?? "—"}
                  </span>
                  <span className="flex min-w-0 flex-wrap items-center gap-1.5">
                    <span
                      className={cn("rounded-lg border px-2 py-0.5 text-[11px] font-medium", st.cls)}
                      title={st.hint}
                    >
                      {st.label}
                    </span>
                    {pick && pick.selection ? (
                      <span
                        className={cn(
                          "truncate text-xs tabular-nums",
                          f.status === "value" ? "text-accent" : "text-ink-faint",
                        )}
                      >
                        {viSelection(pick.selection)} @ {odd(pick.odds)}
                        {pick.ev_pct != null ? ` (${formatPct(pick.ev_pct)})` : ""}
                      </span>
                    ) : null}
                    {f.status === "gated" && f.gate_reason ? (
                      <span className="text-[11px] text-ink-faint">
                        {GATE_TEXT[f.gate_reason] ?? f.gate_reason}
                      </span>
                    ) : null}
                  </span>
                </li>
              );
            })}
          </ul>
        </section>
      ))}
    </div>
  );
}
