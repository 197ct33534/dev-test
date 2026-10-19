import {
  Bookmark,
  BookmarkCheck,
  ChevronRight,
  TrendingUp,
} from "lucide-react";
import { cn, formatKickoff, formatPct, teamHue } from "../lib/utils";
import type { MatchCardData, MatchConfidence, ViewMode } from "../types";

interface MatchCardProps {
  match: MatchCardData;
  viewMode: ViewMode;
  onOpenDetail: (match: MatchCardData) => void;
  onToggleTrack: (id: string) => void;
}

function TeamBadge({ abbr, name }: { abbr: string; name: string }) {
  return (
    <div className="flex items-center gap-2">
      <div
        className="flex h-8 w-8 items-center justify-center rounded-lg text-[10px] font-bold text-white"
        style={{ background: teamHue(name) }}
        title={name}
      >
        {abbr}
      </div>
      <span className="truncate text-sm font-semibold text-ink">{name}</span>
    </div>
  );
}

function formatAge(min: number): string {
  if (min < 1) return "vừa xong";
  if (min < 60) return `${Math.round(min)} phút`;
  return `${(min / 60).toFixed(1)} giờ`;
}

/** Data-confidence chips: odds freshness, fallback, thin data, ML status. */
function ConfidenceChips({ c, pro }: { c: MatchConfidence; pro: boolean }) {
  const warn = "vb-chip border-amber-500/30 bg-amber-500/10 text-accent-warn";
  const muted = "vb-chip text-ink-faint";
  return (
    <>
      {c.belowThreshold ? <span className={warn}>Dưới ngưỡng EV</span> : null}
      {c.stale ? <span className={warn}>Odds cũ</span> : null}
      {c.thinTeams.length ? (
        <span className={warn} title={c.thinTeams.join(", ")}>
          Ít dữ liệu: {c.thinTeams.length === 1 ? c.thinTeams[0] : `${c.thinTeams.length} đội`}
        </span>
      ) : null}
      {c.modelNote?.includes("pooled") ? (
        <span
          className={warn}
          title="Định giá bằng model gộp J1/J2/J3 + cúp, chưa backtest — EV cao chủ yếu là chênh lệch model vs nhà cái, chưa phải edge đã kiểm chứng"
        >
          Model gộp JP · chưa kiểm chứng
        </span>
      ) : null}
      {c.newTeamPrior ? (
        <span
          className={warn}
          title="Mô hình dùng prior đội mới cho ít nhất một đội — EV kém tin cậy, stake bị giới hạn một nửa"
        >
          Prior đội mới
        </span>
      ) : null}
      {pro && c.oddsAgeMinutes !== null && !c.stale ? (
        <span className={muted} title={`Nguồn odds: ${c.oddsSource ?? "?"}`}>
          Odds {formatAge(c.oddsAgeMinutes)}
        </span>
      ) : null}
      {pro && c.mlEnabled === false ? (
        <span className={muted} title="Chỉ Dixon–Coles (LightGBM chưa có cho giải này)">
          ML tắt
        </span>
      ) : null}
    </>
  );
}

export function MatchCard({ match, viewMode, onOpenDetail, onToggleTrack }: MatchCardProps) {
  const ouWarn = match.difference !== null && Math.abs(match.difference) >= 0.3;
  const evPositive = match.evPct >= 5;
  const pro = viewMode === "pro";

  return (
    <article className="vb-card animate-fade-in p-4 sm:p-5">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-stretch">
        <div className="min-w-0 flex-1 space-y-4">
          {/* Header */}
          <div className="flex flex-wrap items-center gap-2">
            <span className="vb-chip tabular-nums text-ink">
              {formatKickoff(match.matchTime)}
            </span>
            <span className="vb-chip border-cyan-900/60 bg-cyan-950/40 text-accent-cyan">
              {match.league}
            </span>
            <span className="vb-chip">{match.leagueLabel}</span>
            <ConfidenceChips c={match.confidence} pro={pro} />
          </div>

          <div className="grid gap-2 sm:grid-cols-[1fr_auto_1fr] sm:items-center">
            <TeamBadge abbr={match.homeAbbr} name={match.homeTeam} />
            <span className="px-2 text-center text-xs font-semibold text-ink-faint">
              VS
            </span>
            <TeamBadge abbr={match.awayAbbr} name={match.awayTeam} />
          </div>

          {/* 1X2 / primary pick block */}
          <div className="grid gap-3 rounded-xl border border-slate-800 bg-slate-950/50 p-3 sm:grid-cols-3">
            <div>
              <p className="text-[10px] uppercase tracking-wider text-ink-faint">
                Gợi ý cược [{match.betType}]
              </p>
              <p className="mt-1 text-sm font-semibold text-ink">
                {match.recommendation}{" "}
                <span className="text-accent-cyan">@ {match.odd.toFixed(2)}</span>
              </p>
            </div>
            <div>
              <p className="text-[10px] uppercase tracking-wider text-ink-faint">
                Kelly khuyến nghị
              </p>
              <p className="mt-1 text-sm font-semibold tabular-nums text-ink">
                {match.kellyPct.toFixed(2)}% bankroll
              </p>
            </div>
            <div>
              <p className="text-[10px] uppercase tracking-wider text-ink-faint">
                Kỳ vọng +EV
              </p>
              <p
                className={cn(
                  "mt-1 inline-flex items-center gap-1 text-sm font-bold tabular-nums",
                  evPositive ? "text-accent" : "text-accent-danger",
                )}
              >
                <TrendingUp className="h-3.5 w-3.5" />
                {formatPct(match.evPct)}
              </p>
            </div>
          </div>

          {/* OU comparison */}
          {match.modelOU !== null && match.bookieOU !== null ? (
            <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-3">
              <p className="text-[10px] uppercase tracking-wider text-ink-faint">
                So sánh Tài Xỉu (Model vs Nhà cái)
              </p>
              <div className="mt-2 flex flex-wrap items-center gap-3 text-sm">
                <span className="text-ink-muted">
                  Model <strong className="text-ink">{match.modelOU.toFixed(2)}</strong>
                </span>
                <span className="text-slate-700">·</span>
                <span className="text-ink-muted">
                  NC <strong className="text-ink">{match.bookieOU.toFixed(2)}</strong>
                </span>
                {match.difference !== null ? (
                  <span
                    className={cn(
                      "rounded-lg px-2 py-0.5 text-xs font-semibold tabular-nums",
                      ouWarn
                        ? "bg-amber-500/15 text-accent-warn"
                        : "bg-emerald-500/10 text-accent",
                    )}
                  >
                    Độ lệch {match.difference > 0 ? "+" : ""}
                    {match.difference.toFixed(2)} bàn
                  </span>
                ) : null}
              </div>
            </div>
          ) : null}

          {pro && match.aiReasons.length ? (
            <ul className="space-y-1 text-xs text-ink-muted">
              {match.aiReasons.slice(0, 2).map((r) => (
                <li key={r} className="flex gap-1.5">
                  <span className="text-accent-cyan">›</span>
                  <span>{r}</span>
                </li>
              ))}
            </ul>
          ) : null}
        </div>

        {/* Actions */}
        <div className="flex shrink-0 flex-row gap-2 lg:w-40 lg:flex-col lg:justify-center">
          <button
            type="button"
            className="vb-btn-primary flex-1"
            onClick={() => onOpenDetail(match)}
          >
            Xem Chi Tiết
            <ChevronRight className="h-4 w-4" />
          </button>
          <button
            type="button"
            className="vb-btn-ghost flex-1"
            onClick={() => onToggleTrack(match.id)}
          >
            {match.tracked ? (
              <>
                <BookmarkCheck className="h-4 w-4 text-accent" />
                Đang theo dõi
              </>
            ) : (
              <>
                <Bookmark className="h-4 w-4" />
                Thêm Sổ Cược
              </>
            )}
          </button>
        </div>
      </div>
    </article>
  );
}
