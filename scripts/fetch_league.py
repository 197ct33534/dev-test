"""CLI: fetch a league from config/leagues.json (Flashscore / football-data).

Examples
--------
    python scripts/fetch_league.py --league LALIGA --dry-run
    python scripts/fetch_league.py --league LALIGA --seasons 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fetchers.flashscore_league import fetch_and_persist_league  # noqa: E402
from src.league_registry import get_available_leagues, get_league_entry  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch & persist a configured league")
    parser.add_argument(
        "--league",
        required=True,
        help="League key from config/leagues.json (e.g. LALIGA)",
    )
    parser.add_argument("--seasons", type=int, default=2, help="Number of seasons")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch only; do not write SQLite",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_leagues",
        help="Print available leagues and exit",
    )
    parser.add_argument(
        "--fit-models",
        action="store_true",
        help="Also fit Dixon-Coles / LightGBM after persist",
    )
    args = parser.parse_args()

    if args.list_leagues:
        print(json.dumps(get_available_leagues(), indent=2, ensure_ascii=False))
        return 0

    entry = get_league_entry(args.league)
    if entry is None:
        print(f"Unknown league {args.league!r}. Use --list to see keys.", file=sys.stderr)
        return 2

    result = fetch_and_persist_league(
        args.league,
        n_seasons=int(args.seasons),
        dry_run=bool(args.dry_run),
        fit_models=bool(args.fit_models),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
