#!/usr/bin/env python
"""CLI: walk-forward calibration + optional DC vs Ensemble ablation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/run_calibration_ablation.py` from repo root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtester import BacktestConfig, format_backtest_report, run_ablation, run_backtest
from src.data_loader import load_league_data


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--league", default="EPL", choices=("EPL", "UWCL"))
    p.add_argument("--seasons", type=int, default=2)
    p.add_argument("--min-train", type=int, default=100)
    p.add_argument("--commission", type=float, default=0.0, help="e.g. 0.02 = 2%%")
    p.add_argument("--ablation", action="store_true", help="Run w_ML grid")
    p.add_argument("--use-ml", action="store_true", help="Single run with Ensemble")
    p.add_argument("--w-ml", type=float, default=0.2)
    args = p.parse_args()

    print(f"=== Load {args.league} ({args.seasons} seasons) ===")
    data = load_league_data(league=args.league, n_seasons=args.seasons)
    print(f"Matches: {len(data)}")

    cfg = BacktestConfig(
        min_train_matches=int(args.min_train),
        use_ml=bool(args.use_ml),
        ensemble_w_ml=float(args.w_ml),
        allowed_markets=["1X2"],
        persist=False,
        calibration="isotonic",
        commission_pct=float(args.commission),
        closing_ref_family="Avg",
        odds_family="B365",
        adapt_w_ml=True,
        refit_every=7,
        min_odds=1.40,
        max_odds=3.50,
    )

    def _progress(step: int, total: int, msg: str) -> None:
        print(f"  [{step}/{total}] {msg}")

    result = run_backtest(data, config=cfg, progress=_progress)
    print()
    print(format_backtest_report(result.summary))

    if args.ablation:
        print("\n=== Ablation DC vs Ensemble ===")
        table = run_ablation(
            data,
            base_config=cfg,
            w_ml_grid=(0.0, 0.2, 0.4, 0.6),
            progress=_progress,
        )
        print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
