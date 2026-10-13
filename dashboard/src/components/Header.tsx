import {
  Bell,
  ChevronDown,
  CircleUserRound,
  FlaskConical,
  Flame,
  Gem,
  LayoutGrid,
  ScrollText,
  Sparkles,
  Swords,
  Wallet,
  CornerDownRight,
  LineChart,
  RefreshCw,
} from "lucide-react";
import { cn } from "../lib/utils";
import { NAV_ITEMS } from "../data/mockMatches";
import type { NavId, ViewMode } from "../types";

const ICONS = {
  flame: Flame,
  chart: LineChart,
  corner: CornerDownRight,
  swords: Swords,
  wallet: Wallet,
  scroll: ScrollText,
  crystal: Gem,
  flask: FlaskConical,
} as const;

/** Feed state shown in the "Cập nhật" badge. */
export type FeedStatus = "live" | "stale" | "loading" | "error" | "mock";

const STATUS_STYLE: Record<FeedStatus, { dot: string; label: string | null; text: string }> = {
  live: { dot: "bg-accent", label: null, text: "" },
  stale: { dot: "bg-accent-warn", label: "Odds cũ", text: "text-accent-warn" },
  loading: { dot: "bg-slate-500 animate-pulse", label: "Đang tải", text: "text-ink-muted" },
  error: { dot: "bg-accent-danger", label: "Mất kết nối API", text: "text-accent-danger" },
  mock: { dot: "bg-accent-warn", label: "Dữ liệu mẫu (mock)", text: "text-accent-warn" },
};

interface HeaderProps {
  activeNav: NavId;
  onNavChange: (id: NavId) => void;
  viewMode: ViewMode;
  onViewModeChange: (mode: ViewMode) => void;
  /** "x phút trước" from the API ``generated_at``. */
  updatedAgo: string;
  status: FeedStatus;
  refreshing?: boolean;
  onRefresh?: () => void;
}

export function Header({
  activeNav,
  onNavChange,
  viewMode,
  onViewModeChange,
  updatedAgo,
  status,
  refreshing = false,
  onRefresh,
}: HeaderProps) {
  const st = STATUS_STYLE[status];
  return (
    <header className="sticky top-0 z-30 border-b border-slate-800/90 bg-canvas/85 backdrop-blur-xl">
      <div className="mx-auto flex max-w-[1600px] items-center gap-4 px-4 py-3 lg:px-6">
        <div className="flex min-w-0 items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-accent to-accent-cyan shadow-[0_0_24px_rgba(16,185,129,0.35)]">
            <Sparkles className="h-4 w-4 text-slate-950" strokeWidth={2.4} />
          </div>
          <div className="min-w-0">
            <p className="font-display text-sm font-semibold tracking-tight text-ink sm:text-base">
              VALUE BETTING
            </p>
            <p className="truncate text-[11px] text-ink-faint">
              Pro Sports Analytics
            </p>
          </div>
        </div>

        <nav className="vb-scrollbar hidden min-w-0 flex-1 items-center gap-1 overflow-x-auto lg:flex">
          {NAV_ITEMS.map((item) => {
            const Icon = ICONS[item.icon as keyof typeof ICONS] ?? LayoutGrid;
            const active = activeNav === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onNavChange(item.id)}
                className={cn(
                  "inline-flex shrink-0 items-center gap-1.5 rounded-xl px-3 py-2 text-xs font-medium transition",
                  active
                    ? "bg-emerald-500/15 text-accent"
                    : "text-ink-muted hover:bg-slate-900 hover:text-ink",
                )}
              >
                <Icon className="h-3.5 w-3.5" />
                <span className="max-w-[11rem] truncate">{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="ml-auto flex items-center gap-2">
          <div className="hidden items-center gap-2 rounded-xl border border-slate-800 bg-slate-950/60 px-3 py-1.5 md:flex">
            <span className={cn("inline-flex h-2 w-2 rounded-full", st.dot)} />
            <p className="text-[11px] text-ink-muted">
              Cập nhật: <span className="text-ink">{updatedAgo}</span>
              {st.label ? <span className={cn("ml-1", st.text)}>{st.label}</span> : null}
            </p>
            {onRefresh ? (
              <button
                type="button"
                onClick={onRefresh}
                disabled={refreshing}
                className="text-ink-faint transition hover:text-ink disabled:opacity-60"
                aria-label="Làm mới feed"
                title="Làm mới feed"
              >
                <RefreshCw className={cn("h-3.5 w-3.5", refreshing && "animate-spin")} />
              </button>
            ) : null}
          </div>

          <div className="flex rounded-xl border border-slate-800 bg-slate-950/50 p-0.5 text-[11px]">
            <button
              type="button"
              onClick={() => onViewModeChange("lite")}
              className={cn(
                "rounded-lg px-2.5 py-1.5 font-medium transition",
                viewMode === "lite"
                  ? "bg-slate-800 text-ink"
                  : "text-ink-faint hover:text-ink",
              )}
            >
              Nhanh
            </button>
            <button
              type="button"
              onClick={() => onViewModeChange("pro")}
              className={cn(
                "rounded-lg px-2.5 py-1.5 font-medium transition",
                viewMode === "pro"
                  ? "bg-emerald-500/20 text-accent"
                  : "text-ink-faint hover:text-ink",
              )}
            >
              Nâng cao
            </button>
          </div>

          <button
            type="button"
            className="vb-btn-ghost relative h-9 w-9 !px-0"
            aria-label="Thông báo"
          >
            <Bell className="h-4 w-4" />
            <span className="absolute right-1.5 top-1.5 h-1.5 w-1.5 rounded-full bg-accent-warn" />
          </button>
          <button
            type="button"
            className="vb-btn-ghost hidden h-9 items-center gap-1.5 sm:inline-flex"
          >
            <CircleUserRound className="h-4 w-4" />
            <span className="text-xs">Pro User</span>
            <ChevronDown className="h-3.5 w-3.5 text-ink-faint" />
          </button>
        </div>
      </div>

      {/* Mobile nav strip */}
      <div className="vb-scrollbar flex gap-1 overflow-x-auto border-t border-slate-800/80 px-3 py-2 lg:hidden">
        {NAV_ITEMS.map((item) => {
          const active = activeNav === item.id;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => onNavChange(item.id)}
              className={cn(
                "shrink-0 rounded-lg px-2.5 py-1.5 text-[11px] font-medium",
                active
                  ? "bg-emerald-500/15 text-accent"
                  : "bg-slate-900/60 text-ink-muted",
              )}
            >
              {item.label.split(" ")[0]} {item.label.includes("Kèo") ? "Kèo" : ""}
            </button>
          );
        })}
      </div>
    </header>
  );
}
