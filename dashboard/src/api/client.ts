import type {
  ApiConfigOut,
  FixturesResponse,
  LeaguesResponse,
  TeamProfileOut,
  ValueBetsParams,
  ValueBetsResponse,
} from "./types";

/**
 * API origin. Empty → same origin (served by FastAPI at /dashboard, or the
 * Vite dev proxy forwarding /api to the backend).
 */
export const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/+$/, "");

const TOKEN_KEY = "vb.dashboardToken";

interface TelegramWindow {
  Telegram?: { WebApp?: { initData?: string } };
}

export function getStoredToken(): string | null {
  try {
    return window.localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setStoredToken(token: string | null): void {
  try {
    if (token) window.localStorage.setItem(TOKEN_KEY, token);
    else window.localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* private mode / blocked storage: token lives for this tab only */
  }
}

/** Inside Telegram → Mini App initData; otherwise the dashboard bearer token. */
export function authHeaders(): Record<string, string> {
  const initData = (window as unknown as TelegramWindow).Telegram?.WebApp?.initData;
  if (initData) return { Authorization: `tma ${initData}` };
  const token = getStoredToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      headers: { Accept: "application/json", ...authHeaders() },
      signal,
    });
  } catch (err) {
    if ((err as Error).name === "AbortError") throw err;
    throw new ApiError(0, "Không kết nối được API");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail || `HTTP ${res.status}`);
  }
  return (await res.json()) as T;
}

export function buildValueBetsQuery(p: ValueBetsParams, withLimit = true): string {
  const q = new URLSearchParams();
  if (p.leagues.length) q.set("leagues", p.leagues.join(","));
  if (p.markets.length) q.set("markets", p.markets.join(","));
  q.set("min_ev", String(p.minEvPct));
  q.set("kelly_fraction", String(p.kellyFraction));
  if (withLimit) q.set("limit", String(p.limit ?? 50));
  return q.toString();
}

export const api = {
  valueBets: (p: ValueBetsParams, signal?: AbortSignal) =>
    getJson<ValueBetsResponse>(`/api/v1/value-bets?${buildValueBetsQuery(p)}`, signal),
  fixtures: (p: ValueBetsParams, signal?: AbortSignal) =>
    getJson<FixturesResponse>(`/api/v1/fixtures?${buildValueBetsQuery(p, false)}`, signal),
  leagues: (signal?: AbortSignal) => getJson<LeaguesResponse>("/api/v1/leagues", signal),
  config: (signal?: AbortSignal) => getJson<ApiConfigOut>("/api/v1/config", signal),
  teamProfile: (team: string, signal?: AbortSignal) =>
    getJson<TeamProfileOut>(`/api/v1/teams/${encodeURIComponent(team)}/profile`, signal),
};
