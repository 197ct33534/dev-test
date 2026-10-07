"""One-shot: refresh Japan cups and print picks for today's fixtures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fetchers.flashscore_league import fetch_and_persist_league, fetch_league_upcoming
from src.japan_schedule import (
    JP_COMP_IDS,
    attach_picks_to_records,
    japan_match_records,
    load_japan_upcoming,
)
from src.data_loader import load_league_data
from src.models import fit_league_models, maybe_wrap_with_league_weights


def main() -> int:
    for key, fit in (("EMPERORS_CUP", True), ("J_LEAGUE_CUP", False)):
        try:
            r = fetch_and_persist_league(key, n_seasons=1, dry_run=False, fit_models=fit)
            print(
                f"{key}: history={r.get('history_rows')} upcoming={r.get('upcoming_rows')}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{key} persist failed: {exc}", flush=True)

    live = fetch_league_upcoming("EMPERORS_CUP", within_hours=72, include_odds=True, max_events=30)
    print(f"LIVE EMPERORS upcoming={len(live)}", flush=True)
    if not live.empty:
        cols = [
            c
            for c in (
                "Kickoff",
                "HomeTeam",
                "AwayTeam",
                "FlashscoreEventId",
                "B365H",
                "B365D",
                "B365A",
                "OU_Line",
                "AHh",
            )
            if c in live.columns
        ]
        print(live[cols].to_string(), flush=True)

    fx = load_japan_upcoming()
    recs = japan_match_records()
    print(f"DB fixtures={0 if fx is None else len(fx)} records={len(recs)}", flush=True)

    models_by_comp: dict = {}
    for code in JP_COMP_IDS:
        try:
            hist = load_league_data(str(code), n_seasons=2)
        except Exception as exc:  # noqa: BLE001
            print(f"history {code}: {exc}", flush=True)
            continue
        if hist is None or getattr(hist, "empty", True):
            print(f"model {code}: no history", flush=True)
            continue
        try:
            fitted = fit_league_models(hist, league=str(code), use_ml=False, fit_corners=False)
        except Exception as exc:  # noqa: BLE001
            print(f"model {code}: {exc}", flush=True)
            continue
        dc = getattr(fitted, "dixon_coles", None)
        if dc is None and isinstance(fitted, dict):
            dc = fitted.get("dixon_coles") or fitted.get("dc")
        if dc is None:
            print(f"model {code}: no dixon_coles", flush=True)
            continue
        models_by_comp[str(code)] = maybe_wrap_with_league_weights(dc, str(code))
        print(f"model {code}: ok n={len(hist)}", flush=True)

    picks_recs = attach_picks_to_records(
        recs,
        fx if fx is not None else live,
        models_by_comp,
        min_ev=0.05,
        include_ou=True,
        include_ah=True,
    )

    out = []
    for r in picks_recs:
        mid = r.get("match_id") or r.get("FlashscoreEventId")
        out.append(
            {
                "comp": r.get("comp_id") or r.get("competition") or r.get("league"),
                "kickoff_vn": r.get("kickoff_vn") or r.get("time_vn") or r.get("kickoff"),
                "home": r.get("home") or r.get("HomeTeam"),
                "away": r.get("away") or r.get("AwayTeam"),
                "match_id": mid,
                "picks": r.get("picks_text") or r.get("picks"),
                "flashscore": r.get("flashscore_url"),
                "detail": r.get("detail_url")
                or (f"http://localhost:8501/?match_id={mid}" if mid else None),
            }
        )

    out_path = ROOT / "data" / "japan_briefing_today.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"WROTE {out_path} n={len(out)}", flush=True)
    for row in out:
        print("---", flush=True)
        print(
            f"{row.get('comp')} | {row.get('kickoff_vn')} | {row.get('home')} vs {row.get('away')}",
            flush=True,
        )
        print(f"picks: {row.get('picks')}", flush=True)
        print(f"app: {row.get('detail')}", flush=True)
        print(f"fs: {row.get('flashscore')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
