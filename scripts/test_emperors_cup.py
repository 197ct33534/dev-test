"""Quick verification for Emperor's Cup (EMPERORS_CUP) multi-league integration.

Examples
--------
    python scripts/test_emperors_cup.py
    python scripts/test_emperors_cup.py --dry-run
    python scripts/test_emperors_cup.py --hours 168 --seasons 2 --fit-models
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LEAGUE = "EMPERORS_CUP"


def _soft_network_fail(exc: BaseException) -> bool:
    """True when failure looks like network / remote timeout (exit 0 with message)."""
    name = type(exc).__name__
    msg = str(exc).lower()
    needles = (
        "timeout",
        "timed out",
        "urlerror",
        "httperror",
        "connection",
        "temporarily",
        "403",
        "429",
        "502",
        "503",
        "failed to fetch",
        "name or service not known",
        "getaddrinfo",
        "network",
    )
    if name in {"URLError", "HTTPError", "TimeoutError", "OSError"}:
        return True
    return any(n in msg for n in needles)


def _sample_rest_days(upcoming: Any, history: Any) -> list[dict[str, Any]]:
    """Compute rest_days for a few upcoming JP_* sides from history in-memory."""
    if upcoming is None or getattr(upcoming, "empty", True):
        return []
    if history is None or getattr(history, "empty", True):
        return [{"note": "no history in DB/frame — rest_days N/A (J1 not configured yet)"}]

    import pandas as pd

    from src.features import calculate_multi_comp_features

    hist = history.copy()
    if "home_team_id" not in hist.columns:
        # Synthetic ids from JP_* (or display) names so multi-comp helper works.
        names = sorted(
            set(hist["HomeTeam"].astype(str)) | set(hist["AwayTeam"].astype(str))
        )
        id_map = {n: i + 1 for i, n in enumerate(names)}
        hist["home_team_id"] = hist["HomeTeam"].astype(str).map(id_map)
        hist["away_team_id"] = hist["AwayTeam"].astype(str).map(id_map)
    else:
        id_map = {}
        for _, row in hist.iterrows():
            id_map[str(row["HomeTeam"])] = int(row["home_team_id"])
            id_map[str(row["AwayTeam"])] = int(row["away_team_id"])

    samples: list[dict[str, Any]] = []
    for _, row in upcoming.head(5).iterrows():
        home = str(row.get("HomeTeam") or "")
        away = str(row.get("AwayTeam") or "")
        kick = row.get("Kickoff") or row.get("Date")
        entry: dict[str, Any] = {"home": home, "away": away}
        for side, name in (("home", home), ("away", away)):
            tid = id_map.get(name)
            if tid is None:
                entry[f"{side}_rest_days"] = None
                continue
            feats = calculate_multi_comp_features(tid, kick, hist)
            entry[f"{side}_rest_days"] = feats.get("rest_days")
            entry[f"{side}_matches_last_14d"] = feats.get("matches_last_14d")
        samples.append(entry)
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify EMPERORS_CUP fetch → global_matches.db"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch only; do not write SQLite",
    )
    parser.add_argument("--seasons", type=int, default=2, help="History seasons")
    parser.add_argument(
        "--hours",
        type=float,
        default=168.0,
        help="Upcoming horizon in hours (default 7d)",
    )
    parser.add_argument(
        "--fit-models",
        action="store_true",
        help="Fit Dixon-Coles / LightGBM on fetched history",
    )
    parser.add_argument(
        "--no-odds",
        action="store_true",
        help="Skip bookmaker odds fetch (faster)",
    )
    args = parser.parse_args()

    # Windows consoles often default to cp1252 — keep JSON printable.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    from src.league_registry import clear_leagues_cache, get_league_entry, resolve_league_config

    clear_leagues_cache()
    entry = get_league_entry(LEAGUE)
    if entry is None:
        print(f"HARD FAIL: {LEAGUE} missing from config/leagues.json", file=sys.stderr)
        return 2

    try:
        code, cfg = resolve_league_config(LEAGUE)
    except ValueError as exc:
        print(f"HARD FAIL: resolve_league_config: {exc}", file=sys.stderr)
        return 2

    print("=== EMPERORS_CUP registry ===")
    print(
        json.dumps(
            {
                "code": code,
                "name": cfg.get("label"),
                "flashscore_id": cfg.get("flashscore_id"),
                "flashscore_path": cfg.get("flashscore_path"),
                "time_zone": cfg.get("time_zone"),
                "league_weight": cfg.get("league_weight"),
                "history_source": cfg.get("history_source"),
                "seasons": cfg.get("seasons"),
                "n_aliases": len(cfg.get("team_aliases") or {}),
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )

    from src.fetchers.flashscore_league import (
        fetch_league_history,
        fetch_league_upcoming,
        persist_league_matches,
        persist_league_upcoming,
        resolve_league_team_name,
    )

    # Alias smoke check (no network).
    alias_checks = {
        "Gamba Osaka": resolve_league_team_name("Gamba Osaka", LEAGUE),
        "Vissel Kobe": resolve_league_team_name("Vissel Kobe", LEAGUE),
        "Kashima Antlers": resolve_league_team_name("Kashima Antlers", LEAGUE),
    }
    print("=== JP_* aliases ===")
    print(json.dumps(alias_checks, indent=2, ensure_ascii=False))
    if alias_checks.get("Gamba Osaka") != "JP_G_OSAKA":
        print("HARD FAIL: Gamba Osaka alias", file=sys.stderr)
        return 2
    if alias_checks.get("Vissel Kobe") != "JP_VISSEL_KOBE":
        print("HARD FAIL: Vissel Kobe alias", file=sys.stderr)
        return 2

    history = None
    upcoming = None
    try:
        print(f"=== Fetching history (n_seasons={args.seasons}) ===")
        history = fetch_league_history(LEAGUE, n_seasons=int(args.seasons))
        print(f"=== Fetching upcoming (within_hours={args.hours}) ===")
        upcoming = fetch_league_upcoming(
            LEAGUE,
            within_hours=float(args.hours),
            include_odds=not bool(args.no_odds),
        )
    except Exception as exc:  # noqa: BLE001
        if _soft_network_fail(exc):
            print(
                f"SOFT FAIL (network): {type(exc).__name__}: {exc}\n"
                "Registry + JP_* aliases OK; retry when Flashscore is reachable.",
                file=sys.stderr,
            )
            return 0
        print(f"HARD FAIL during fetch: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    hist_warn = list(getattr(history, "attrs", {}).get("download_warnings") or [])
    up_warn = list(getattr(upcoming, "attrs", {}).get("download_warnings") or [])

    # Soft-fail when both empty and warnings look like network.
    if (history is None or history.empty) and (upcoming is None or upcoming.empty):
        joined = " ".join(hist_warn + up_warn).lower()
        if any(
            n in joined
            for n in ("failed to fetch", "timeout", "403", "429", "connection")
        ):
            print(
                "SOFT FAIL (network): empty history+upcoming with fetch warnings:\n"
                + "\n".join(hist_warn + up_warn),
                file=sys.stderr,
            )
            return 0

    teams: set[str] = set()
    if history is not None and not history.empty:
        teams |= set(history["HomeTeam"].astype(str)) | set(
            history["AwayTeam"].astype(str)
        )
    if upcoming is not None and not upcoming.empty:
        teams |= set(upcoming["HomeTeam"].astype(str)) | set(
            upcoming["AwayTeam"].astype(str)
        )
    jp_teams = sorted(t for t in teams if str(t).startswith("JP_"))

    odds_cols = [c for c in ("B365H", "B365D", "B365A") if upcoming is not None and c in upcoming.columns]
    odds_present = 0
    if upcoming is not None and not upcoming.empty and odds_cols:
        odds_present = int(upcoming[odds_cols].notna().any(axis=1).sum())

    rest_sample = _sample_rest_days(upcoming, history)

    summary: dict[str, Any] = {
        "history_rows": int(len(history)) if history is not None else 0,
        "upcoming_rows": int(len(upcoming)) if upcoming is not None else 0,
        "jp_team_sample": jp_teams[:12],
        "n_jp_teams": len(jp_teams),
        "odds_rows_with_any_1x2": odds_present,
        "rest_days_sample": rest_sample,
        "history_warnings": hist_warn,
        "upcoming_warnings": up_warn,
        "dry_run": bool(args.dry_run),
    }

    if upcoming is not None and not upcoming.empty:
        cols = [
            c
            for c in (
                "Kickoff",
                "HomeTeam",
                "AwayTeam",
                "B365H",
                "B365D",
                "B365A",
                "FlashscoreEventId",
            )
            if c in upcoming.columns
        ]
        summary["upcoming_sample"] = (
            upcoming.head(8)[cols]
            .assign(
                Kickoff=lambda d: d["Kickoff"].astype(str) if "Kickoff" in d.columns else None
            )
            .to_dict(orient="records")
        )

    if not args.dry_run:
        try:
            if history is not None and not history.empty:
                summary["persist_history"] = persist_league_matches(LEAGUE, history)
            if upcoming is not None:
                summary["persist_upcoming"] = persist_league_upcoming(LEAGUE, upcoming)
        except Exception as exc:  # noqa: BLE001
            print(f"HARD FAIL during persist: {exc}", file=sys.stderr)
            traceback.print_exc()
            return 1

    if args.fit_models:
        if history is None or history.empty:
            summary["models"] = {"skipped": True, "reason": "empty history"}
        else:
            try:
                from src.models import fit_league_models

                fitted = fit_league_models(history, league=LEAGUE)
                summary["models"] = {
                    "n_train": int(fitted.n_train),
                    "w_ml": float(fitted.w_ml),
                    "ml_ok": fitted.ml is not None,
                    "corner_ok": fitted.corner is not None,
                    "warnings": list(fitted.warnings),
                    "ml_error": fitted.ml_error,
                }
            except Exception as exc:  # noqa: BLE001
                summary["models"] = {"ok": False, "error": str(exc)}

    print("=== Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))

    # Hard failure: config resolved but zero JP_* after a successful-looking fetch
    # with rows that somehow skipped aliasing.
    n_hist = summary["history_rows"]
    n_up = summary["upcoming_rows"]
    if (n_hist + n_up) > 0 and summary["n_jp_teams"] == 0:
        print(
            "HARD FAIL: matches present but no JP_* team ids — aliasing broken",
            file=sys.stderr,
        )
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
