import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { api } from "./client";
import type { ValueBetsParams } from "./types";

/** Feed refresh cadence (odds cache is refreshed every ~20 min server-side). */
export const VALUE_BETS_POLL_MS = 60_000;

export function useValueBets(params: ValueBetsParams, enabled = true) {
  return useQuery({
    queryKey: ["value-bets", params],
    queryFn: ({ signal }) => api.valueBets(params, signal),
    enabled,
    refetchInterval: VALUE_BETS_POLL_MS,
    staleTime: 30_000,
    placeholderData: keepPreviousData,
  });
}

export function useLeagues(enabled = true) {
  return useQuery({
    queryKey: ["leagues"],
    queryFn: ({ signal }) => api.leagues(signal),
    enabled,
    staleTime: 60 * 60_000,
  });
}

export function useApiConfig(enabled = true) {
  return useQuery({
    queryKey: ["config"],
    queryFn: ({ signal }) => api.config(signal),
    enabled,
    staleTime: 60 * 60_000,
  });
}

export function useTeamProfile(team: string | null) {
  return useQuery({
    queryKey: ["team-profile", team],
    queryFn: ({ signal }) => api.teamProfile(team as string, signal),
    enabled: Boolean(team),
    staleTime: 10 * 60_000,
    // 404 = team not in DB yet; not worth retrying.
    retry: false,
  });
}
