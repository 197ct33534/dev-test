"""Verify Flashscore team-results feeds → multi-comp rest_days (Kobe / Tosu).

Examples
--------
    python scripts/test_team_rest_days.py
    python scripts/test_team_rest_days.py --kickoff 2026-09-23 --n-matches 15
    python scripts/test_team_rest_days.py --dry-run
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

TEAMS = ("JP_VISSEL_KOBE", "JP_SAGAN_TOSU", "JP_MACHIDA", "JP_TOCHIGI_CITY")
# Expected rest_days for 2026-09-23 when Team Feed has recent J1 (Machida vs Kashiwa ~20/09).
ASSERT_REST: dict[str, int] = {
    "JP_MACHIDA": 3,
}


def _soft_network_fail(exc: BaseException) -> bool:
    name = type(exc).__name__
    msg = str(exc).lower()
    needles = (
        "timeout",
        "timed out",
        "urlerror",
        "httperror",
        "connection",
        "403",
        "429",
        "502",
        "503",
        "failed to fetch",
        "getaddrinfo",
        "network",
    )
    if name in {"URLError", "HTTPError", "TimeoutError", "OSError"}:
        return True
    return any(n in msg for n in needles)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch team feeds and print multi-comp rest_days"
    )
    parser.add_argument(
        "--kickoff",
        default="2026-09-23",
        help="Upcoming fixture date (YYYY-MM-DD), default Emperor's Cup day",
    )
    parser.add_argument("--n-matches", type=int, default=15)
    parser.add_argument(
        "--teams",
        default=",".join(TEAMS),
        help="Comma-separated JP_* codes (default: Kobe,Tosu,Machida,Tochigi)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch + print only; do not write global_matches.db",
    )
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="Skip live fetch; compute rest_days from existing DB only",
    )
    parser.add_argument(
        "--assert-machida",
        action="store_true",
        help="Exit non-zero if JP_MACHIDA rest_days != 3 for --kickoff",
    )
    args = parser.parse_args()

    teams = tuple(
        t.strip().upper() for t in str(args.teams).split(",") if t.strip()
    ) or TEAMS

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    from src.fetchers.flashscore_team import (
        ensure_jp_aliases_registered,
        fetch_teams_recent_matches_parallel,
        hash_for_team_id,
        persist_team_matches_to_global_db,
        resolve_team_hash,
        team_hash_registry,
        team_results_url,
    )
    from src.features import (
        calculate_multi_comp_features,
        format_fatigue_label,
        format_team_fatigue_phrase,
    )
    from src.global_db import (
        GLOBAL_DB_PATH,
        connect_global_db,
        read_matches_as_legacy,
        resolve_team_id,
    )

    ensure_jp_aliases_registered()
    registry = team_hash_registry()
    print("=== Team hash registry ===")
    print(
        json.dumps(
            {t: resolve_team_hash(t) for t in teams},
            indent=2,
            ensure_ascii=False,
        )
    )
    for tid in teams:
        meta = resolve_team_hash(tid)
        if not meta:
            print(f"HARD FAIL: no hash for {tid}", file=sys.stderr)
            return 2
        h = meta["hash"]
        print(
            f"URL {tid}: {team_results_url(h, slug=meta.get('slug'), team_id=tid)}"
        )

    feeds: dict[str, list[dict[str, Any]]] = {}
    if not args.no_network:
        try:
            print(f"=== Fetching team feeds (n={args.n_matches}, parallel) ===")
            feeds = fetch_teams_recent_matches_parallel(
                list(teams), n_matches=int(args.n_matches), max_workers=4
            )
        except Exception as exc:  # noqa: BLE001
            if _soft_network_fail(exc):
                print(
                    f"SOFT FAIL (network): {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                return 0
            print(f"HARD FAIL during fetch: {exc}", file=sys.stderr)
            traceback.print_exc()
            return 1

    sample: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for tid in teams:
        rows = feeds.get(tid) or []
        sample[tid] = {
            "n": len(rows),
            "recent": [
                {
                    "match_date": str(r.get("match_date"))[:10],
                    "home_team": r.get("home_team"),
                    "away_team": r.get("away_team"),
                    "score": r.get("score"),
                    "competition_name": r.get("competition_name"),
                    "comp_id": r.get("comp_id"),
                    "flashscore_event_id": r.get("flashscore_event_id"),
                }
                for r in rows[:5]
            ],
        }
        all_rows.extend(rows)

    print("=== Feed sample ===")
    print(json.dumps(sample, indent=2, ensure_ascii=False, default=str))

    if not args.dry_run and all_rows:
        try:
            persist_stats = persist_team_matches_to_global_db(all_rows)
            print("=== Persist ===")
            print(json.dumps(persist_stats, indent=2, ensure_ascii=False, default=str))
        except Exception as exc:  # noqa: BLE001
            print(f"HARD FAIL during persist: {exc}", file=sys.stderr)
            traceback.print_exc()
            return 1
    elif args.dry_run:
        print("=== Persist skipped (--dry-run) ===")

    # Rest days from global DB (or in-memory synthetic if dry-run with feeds).
    import pandas as pd

    kick = args.kickoff
    hist = None
    if not args.dry_run:
        try:
            hist = read_matches_as_legacy(GLOBAL_DB_PATH)
        except Exception as exc:  # noqa: BLE001
            print(f"warn: could not read global DB: {exc}", file=sys.stderr)

    if hist is None or getattr(hist, "empty", True):
        # Build synthetic frame from feeds for dry-run rest_days.
        from src.fetchers.flashscore_team import matches_to_legacy_frames

        frames = matches_to_legacy_frames(all_rows)
        if frames:
            hist = pd.concat(frames.values(), ignore_index=True)
            names = sorted(
                set(hist["HomeTeam"].astype(str)) | set(hist["AwayTeam"].astype(str))
            )
            id_map = {n: i + 1 for i, n in enumerate(names)}
            hist["home_team_id"] = hist["HomeTeam"].astype(str).map(id_map)
            hist["away_team_id"] = hist["AwayTeam"].astype(str).map(id_map)
            hist["comp_id"] = hist.get("league_id", "FLASH_TEAM")
        else:
            hist = pd.DataFrame()

    rest_out: dict[str, Any] = {"kickoff": kick, "teams": {}}
    feats_by: dict[str, dict[str, Any]] = {}
    if hist is not None and not hist.empty:
        conn = None
        id_map: dict[str, int] = {}
        if not args.dry_run:
            conn = connect_global_db(GLOBAL_DB_PATH, init=True)
            try:
                for tid in teams:
                    rid = resolve_team_id(conn, tid, comp_id="EMPERORS_CUP", create=False)
                    if rid is None:
                        rid = resolve_team_id(conn, tid, gender="M", create=False)
                    if rid is not None:
                        id_map[tid] = int(rid)
            finally:
                conn.close()
        if not id_map and "home_team_id" in hist.columns:
            for _, row in hist.iterrows():
                id_map[str(row["HomeTeam"])] = int(row["home_team_id"])
                id_map[str(row["AwayTeam"])] = int(row["away_team_id"])

        for tid in teams:
            tid_num = id_map.get(tid)
            if tid_num is None:
                rest_out["teams"][tid] = {"error": "team_id not in DB/frame"}
                continue
            feats = calculate_multi_comp_features(
                tid_num, kick, hist, upcoming_comp_id="EMPERORS_CUP"
            )
            feats_by[tid] = feats
            sub = hist.loc[
                (hist["home_team_id"] == tid_num) | (hist["away_team_id"] == tid_num)
            ].copy()
            sub = sub.loc[pd.to_datetime(sub["Date"]) < pd.Timestamp(kick)]
            last = None
            if not sub.empty:
                last = str(pd.to_datetime(sub["Date"]).max().date())
            display = (resolve_team_hash(tid) or {}).get("name") or tid
            rest_out["teams"][tid] = {
                "rest_days": feats.get("rest_days"),
                "matches_last_14d": feats.get("matches_last_14d"),
                "last_match_before_kickoff": last,
                "last_comp_id": feats.get("last_comp_id"),
                "fatigue_phrase": format_team_fatigue_phrase(display, feats),
            }

        if "JP_MACHIDA" in feats_by and "JP_TOCHIGI_CITY" in feats_by:
            rest_out["fatigue_label_machida_tochigi"] = format_fatigue_label(
                "Machida Zelvia",
                feats_by["JP_MACHIDA"],
                "Tochigi City",
                feats_by["JP_TOCHIGI_CITY"],
            )
        if "JP_VISSEL_KOBE" in feats_by and "JP_SAGAN_TOSU" in feats_by:
            rest_out["fatigue_label_kobe_tosu"] = format_fatigue_label(
                "Vissel Kobe",
                feats_by["JP_VISSEL_KOBE"],
                "Sagan Tosu",
                feats_by["JP_SAGAN_TOSU"],
            )

    print("=== Rest days ===")
    print(json.dumps(rest_out, indent=2, ensure_ascii=False, default=str))
    for key in ("fatigue_label_machida_tochigi", "fatigue_label_kobe_tosu"):
        if rest_out.get(key):
            print(f"\nLite label ({key}): {rest_out[key]}")

    # Soft fail if all feeds empty after network attempt.
    if not args.no_network and not any(feeds.get(t) for t in teams):
        print(
            "SOFT FAIL: empty feeds for all teams (network / parse).",
            file=sys.stderr,
        )
        return 0

    # Optional hard assert for Machida rest_days.
    if args.assert_machida or "JP_MACHIDA" in teams:
        expected = ASSERT_REST.get("JP_MACHIDA", 3)
        got = (rest_out.get("teams") or {}).get("JP_MACHIDA", {}).get("rest_days")
        try:
            got_f = float(got) if got is not None else float("nan")
        except (TypeError, ValueError):
            got_f = float("nan")
        if got_f != float(expected):
            msg = (
                f"ASSERT FAIL: JP_MACHIDA rest_days={got!r} expected {expected} "
                f"for kickoff={kick}"
            )
            if args.assert_machida:
                print(msg, file=sys.stderr)
                return 3
            print(f"WARN: {msg}", file=sys.stderr)
        else:
            print(f"ASSERT OK: JP_MACHIDA rest_days == {expected}")

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
