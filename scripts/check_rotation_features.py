"""Smoke-check rotation / density features (Kobe vs Tosu if data present).

Examples
--------
    python scripts/test_rotation_features.py
    python scripts/test_rotation_features.py --kickoff 2026-09-23 --no-network
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEAMS = ("JP_VISSEL_KOBE", "JP_SAGAN_TOSU")


def main() -> int:
    parser = argparse.ArgumentParser(description="Print rotation / rest features")
    parser.add_argument("--kickoff", default="2026-09-23")
    parser.add_argument("--no-network", action="store_true")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    from src.features import calculate_rest_days_for_fixture
    from src.league_registry import is_cup_competition

    print("=== Cup registry ===")
    print(
        json.dumps(
            {
                "EMPERORS_CUP": is_cup_competition("EMPERORS_CUP"),
                "EPL": is_cup_competition("EPL"),
                "J1": is_cup_competition("J1"),
            },
            indent=2,
        )
    )

    result: dict[str, Any] = calculate_rest_days_for_fixture(
        "Vissel Kobe",
        "Sagan Tosu",
        args.kickoff,
        refresh_team_feeds=not bool(args.no_network),
        n_matches=15,
    )
    payload = {
        "match_date": result.get("match_date"),
        "home": result.get("home"),
        "away": result.get("away"),
        "home_team_id": result.get("home_team_id"),
        "away_team_id": result.get("away_team_id"),
        "home_feats": result.get("home_feats"),
        "away_feats": result.get("away_feats"),
        "fatigue_label": result.get("fatigue_label"),
    }
    print("=== Kobe vs Tosu (EMPERORS_CUP context) ===")
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
