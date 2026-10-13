import { RotateCcw, SlidersHorizontal, X } from "lucide-react";
import { cn } from "../lib/utils";
import type { BetMarket, FilterState, ViewMode } from "../types";

const MARKET_OPTS: { id: BetMarket; label: string }[] = [
  { id: "1X2", label: "1X2 (Châu Âu)" },
  { id: "AH", label: "Kèo Chấp (AH)" },
  { id: "OU", label: "Tài Xỉu (O/U)" },
  { id: "Corners", label: "Phạt Góc" },
];

export interface LeagueOption {
  id: string;
  label: string;
}

interface SidebarProps {
  filters: FilterState;
  draft: FilterState;
  onDraftChange: (next: FilterState) => void;
  onApply: () => void;
  onReset: () => void;
  mobileOpen: boolean;
  onCloseMobile: () => void;
  /** From /api/v1/leagues (config/leagues.json). */
  leagueOptions: LeagueOption[];
  /** Policy ceilings from /api/v1/config, in percent. */
  maxKellyPct: number;
  maxStakePct: number;
}

function Toggle({
  value,
  onChange,
}: {
  value: ViewMode;
  onChange: (v: ViewMode) => void;
}) {
  return (
    <div className="grid grid-cols-2 gap-1 rounded-xl border border-slate-800 bg-slate-950/70 p-1">
      {(
        [
          ["lite", "Thẩm Định (Thường)"],
          ["pro", "Chuyên Gia (Pro)"],
        ] as const
      ).map(([id, label]) => (
        <button
          key={id}
          type="button"
          onClick={() => onChange(id)}
          className={cn(
            "rounded-lg px-2 py-2 text-[11px] font-semibold leading-tight transition",
            value === id
              ? "bg-accent/20 text-accent"
              : "text-ink-muted hover:text-ink",
          )}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

function SliderField({
  label,
  value,
  min,
  max,
  step,
  suffix,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  suffix: string;
  onChange: (v: number) => void;
}) {
  return (
    <label className="block space-y-2">
      <div className="flex items-center justify-between text-xs">
        <span className="text-ink-muted">{label}</span>
        <span className="font-semibold tabular-nums text-accent">
          {value.toFixed(2)}
          {suffix}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-slate-800 accent-emerald-500"
      />
      <div className="flex justify-between text-[10px] text-ink-faint">
        <span>
          {min}
          {suffix}
        </span>
        <span>
          {max}
          {suffix}
        </span>
      </div>
    </label>
  );
}

export function Sidebar({
  draft,
  onDraftChange,
  onApply,
  onReset,
  mobileOpen,
  onCloseMobile,
  leagueOptions,
  maxKellyPct,
  maxStakePct,
}: SidebarProps) {
  const patch = (partial: Partial<FilterState>) =>
    onDraftChange({ ...draft, ...partial });

  const toggleLeague = (id: string) => {
    const set = new Set(draft.leagues);
    if (set.has(id)) set.delete(id);
    else set.add(id);
    patch({ leagues: [...set] });
  };

  const toggleMarket = (id: BetMarket) => {
    const set = new Set(draft.markets);
    if (set.has(id)) set.delete(id);
    else set.add(id);
    patch({ markets: [...set] });
  };

  const body = (
    <div className="flex h-full flex-col gap-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <SlidersHorizontal className="h-4 w-4 text-accent-cyan" />
          <h2 className="font-display text-sm font-semibold">Control Panel</h2>
        </div>
        <button
          type="button"
          className="vb-btn-ghost h-8 w-8 !px-0 lg:hidden"
          onClick={onCloseMobile}
          aria-label="Đóng bộ lọc"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      <section className="space-y-2">
        <p className="text-[11px] font-medium uppercase tracking-wider text-ink-faint">
          Chế độ hiển thị
        </p>
        <Toggle
          value={draft.viewMode}
          onChange={(viewMode) => patch({ viewMode })}
        />
      </section>

      <section className="space-y-2">
        <p className="text-[11px] font-medium uppercase tracking-wider text-ink-faint">
          Giải đấu
        </p>
        <div className="max-h-36 space-y-1.5 overflow-y-auto vb-scrollbar rounded-xl border border-slate-800 bg-slate-950/40 p-2">
          {leagueOptions.map((lg) => {
            const checked = draft.leagues.includes(lg.id);
            return (
              <label
                key={lg.id}
                className="flex cursor-pointer items-center gap-2 rounded-lg px-2 py-1.5 text-xs hover:bg-slate-900"
              >
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={() => toggleLeague(lg.id)}
                  className="rounded border-slate-700 accent-emerald-500"
                />
                <span className={checked ? "text-ink" : "text-ink-muted"}>
                  {lg.label}
                </span>
              </label>
            );
          })}
        </div>
      </section>

      <section className="space-y-4">
        <p className="text-[11px] font-medium uppercase tracking-wider text-ink-faint">
          Tham số mô hình
        </p>
        <SliderField
          label="Ngưỡng EV tối thiểu"
          value={draft.minEv}
          min={0}
          max={20}
          step={0.5}
          suffix="%"
          onChange={(minEv) => patch({ minEv })}
        />
        <SliderField
          label="Tỷ lệ Kelly (fraction)"
          value={draft.kellyFractionPct}
          min={Math.min(5, maxKellyPct)}
          max={maxKellyPct}
          step={1}
          suffix="%"
          onChange={(kellyFractionPct) => patch({ kellyFractionPct })}
        />
        <p className="text-[10px] leading-snug text-ink-faint">
          Stake do backend tính: Kelly đúng theo bảng tất toán (push / thua
          nửa) × fraction, tối đa {maxStakePct.toFixed(maxStakePct < 1 ? 1 : 0)}% bankroll
          (đội ít dữ liệu: một nửa). Mô hình: Dixon–Coles (+ LightGBM khi có).
        </p>
      </section>

      <section className="space-y-2">
        <p className="text-[11px] font-medium uppercase tracking-wider text-ink-faint">
          Thị trường quét
        </p>
        <div className="space-y-1.5">
          {MARKET_OPTS.map((m) => (
            <label
              key={m.id}
              className="flex cursor-pointer items-center gap-2 rounded-lg px-1 py-1.5 text-xs"
            >
              <input
                type="checkbox"
                checked={draft.markets.includes(m.id)}
                onChange={() => toggleMarket(m.id)}
                className="rounded border-slate-700 accent-emerald-500"
              />
              <span className="text-ink-muted">{m.label}</span>
            </label>
          ))}
        </div>
      </section>

      <div className="mt-auto grid gap-2 pt-2">
        <button type="button" className="vb-btn-primary w-full" onClick={onApply}>
          Áp Dụng Bộ Lọc
        </button>
        <button type="button" className="vb-btn-ghost w-full" onClick={onReset}>
          <RotateCcw className="h-3.5 w-3.5" />
          Đặt Về Mặc Định
        </button>
      </div>
    </div>
  );

  return (
    <>
      <aside className="hidden w-72 shrink-0 border-r border-slate-800 bg-canvas-panel/60 p-4 lg:block">
        {body}
      </aside>
      {mobileOpen ? (
        <>
          <button
            type="button"
            className="fixed inset-0 z-40 bg-black/60 lg:hidden"
            aria-label="Đóng overlay"
            onClick={onCloseMobile}
          />
          <aside className="sidebar-sheet lg:hidden">{body}</aside>
        </>
      ) : null}
    </>
  );
}
