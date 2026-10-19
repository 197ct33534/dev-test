import { clsx, type ClassValue } from "clsx";

export function cn(...inputs: ClassValue[]) {
  return clsx(inputs);
}

/** Product time zone: every kickoff is shown in Vietnam time. */
export const DISPLAY_TZ = "Asia/Ho_Chi_Minh";

const KICKOFF_FMT = new Intl.DateTimeFormat("en-GB", {
  timeZone: DISPLAY_TZ,
  day: "2-digit",
  month: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
});

/** "dd/mm HH:MM" in Vietnam time (API kickoffs are UTC ISO with offset). */
export function formatKickoff(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  const parts = Object.fromEntries(KICKOFF_FMT.formatToParts(d).map((p) => [p.type, p.value]));
  return `${parts.day}/${parts.month} ${parts.hour}:${parts.minute}`;
}

export function formatPct(n: number, digits = 1): string {
  const sign = n > 0 ? "+" : "";
  return `${sign}${n.toFixed(digits)}%`;
}

export function teamHue(name: string): string {
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) % 360;
  return `hsl(${h} 42% 38%)`;
}

/** Backend stake policy (src/config.py): fractional Kelly 0.10, stake cap 1% bankroll. */
export const DEFAULT_KELLY_FRACTION = 0.1;
export const MAX_STAKE_PCT = 0.01;

/** EV % = (P_model × odds − 1) × 100. */
export function evPct(modelProb: number, odd: number): number {
  return (modelProb * odd - 1) * 100;
}

/**
 * Recommended stake as % of bankroll: fractional Kelly capped at
 * {@link MAX_STAKE_PCT} (mirrors ``capped_kelly_fraction`` in src/strategy.py).
 */
export function cappedKellyPct(
  modelProb: number,
  odd: number,
  kellyFraction: number = DEFAULT_KELLY_FRACTION,
  maxStakePct: number = MAX_STAKE_PCT,
): number {
  if (odd <= 1 || modelProb <= 0) return 0;
  const fullKelly = (modelProb * odd - 1) / (odd - 1);
  if (fullKelly <= 0) return 0;
  return Math.min(fullKelly * kellyFraction, maxStakePct) * 100;
}

/** Human "x phút trước" label for an ISO timestamp. */
export function formatAgo(iso: string, now: number = Date.now()): string {
  const diffMin = Math.max(0, Math.round((now - new Date(iso).getTime()) / 60000));
  if (diffMin < 1) return "vừa xong";
  if (diffMin < 60) return `${diffMin} phút trước`;
  const h = Math.floor(diffMin / 60);
  if (h < 24) return `${h} giờ trước`;
  return `${Math.floor(h / 24)} ngày trước`;
}
