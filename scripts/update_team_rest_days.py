"""Lazy Team Feed sync + rest_days cache helpers.

Default (fast):
    python scripts/update_team_rest_days.py
    python scripts/update_team_rest_days.py --leagues EMPERORS_CUP,LALIGA --max-teams 10

Full batch (optional, slow):
    python scripts/update_team_rest_days.py --full --days 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAMPLE_TEAMS = (
    "JP_MACHIDA",
    "JP_VISSEL_KOBE",
    "ES_BARCELONA",
    "ES_BETIS",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fast/lazy Flashscore team-feed sync (or --full batch)"
    )
    parser.add_argument(
        "--leagues",
        default="EMPERORS_CUP,LALIGA,EPL",
        help="Comma-separated league keys for upcoming cache filter",
    )
    parser.add_argument("--days", type=int, default=3, help="Upcoming horizon (full mode)")
    parser.add_argument("--n-matches", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--max-teams", type=int, default=10, help="Fast mode: max UI sides")
    parser.add_argument(
        "--timeout",
        type=float,
        default=3.0,
        help="Per-request HTTP timeout seconds (fast mode)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run heavy update_all_upcoming_teams_rest_days (slow; optional)",
    )
    parser.add_argument("--json", action="store_true", help="Print full result JSON")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    leagues = [
        x.strip().upper() for x in str(args.leagues).split(",") if x.strip()
    ]

    from src.fetchers.flashscore_team import (
        resolve_team_hash,
        sync_upcoming_teams_fast,
        update_all_upcoming_teams_rest_days,
        _list_upcoming_within_days,
    )
    from src.features import get_rest_days
    from src.global_db import GLOBAL_DB_PATH, connect_global_db, get_team_feed_meta

    conn = connect_global_db(GLOBAL_DB_PATH, init=True)
    cols = {
        str(r[1]) for r in conn.execute("PRAGMA table_info(teams)").fetchall()
    }
    print(
        "schema:",
        f"flashscore_hash={('flashscore_hash' in cols)}",
        f"last_match_date={('last_match_date' in cols)}",
        f"feed_updated_at={('feed_updated_at' in cols)}",
    )
    conn.close()

    t0 = time.perf_counter()
    if args.full:
        result = update_all_upcoming_teams_rest_days(
            days_ahead=int(args.days),
            league_keys=leagues,
            n_matches=int(args.n_matches),
            max_workers=max(int(args.max_workers), 6),
        )
        print(
            f"[full] leagues={leagues} upcoming={result.get('upcoming_rows')} "
            f"outliers={len(result.get('outliers') or [])} "
            f"elapsed={time.perf_counter() - t0:.2f}s"
        )
    else:
        upcoming = _list_upcoming_within_days(
            days_ahead=max(int(args.days), 3),
            league_keys=leagues,
        )
        result = sync_upcoming_teams_fast(
            upcoming,
            max_teams=int(args.max_teams),
            n_matches=int(args.n_matches),
            max_workers=int(args.max_workers),
            timeout=float(args.timeout),
        )
        print(
            f"[fast] leagues={leagues} upcoming_rows={len(upcoming)} "
            f"to_fetch={len(result.get('to_fetch') or [])} "
            f"skipped_fresh={len(result.get('skipped_fresh') or [])} "
            f"fetched={sum((result.get('fetched') or {}).values())} "
            f"elapsed={time.perf_counter() - t0:.2f}s"
        )

    print("\nsample resolve / get_rest_days (DB-only):")
    kick = "2026-09-23"
    for code in SAMPLE_TEAMS:
        meta = resolve_team_hash(code)
        rd = get_rest_days(code, kick)
        feed = None
        conn = connect_global_db(GLOBAL_DB_PATH, init=True)
        try:
            feed = get_team_feed_meta(conn, code)
        finally:
            conn.close()
        print(
            f"  {code}: hash={(meta or {}).get('hash')} "
            f"last={(feed or {}).get('last_match_date')} "
            f"feed_at={(feed or {}).get('feed_updated_at')} "
            f"rest_days={rd}"
        )

    if args.json:
        def _default(o: Any) -> Any:
            if hasattr(o, "item"):
                try:
                    return o.item()
                except Exception:  # noqa: BLE001
                    return str(o)
            return str(o)

        print(json.dumps(result, ensure_ascii=False, indent=2, default=_default))

    if args.full and result.get("outliers"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
