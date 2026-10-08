"""Smoke run: safer walk-forward backtest — 1X2 market only (EPL).

Not a pytest test (formerly ``test_backtest.py`` at repo root).

Usage::

    python scripts/run_backtest_smoke.py            # bets -> temporary SQLite
    python scripts/run_backtest_smoke.py --db PATH  # bets -> PATH (replaces bet_history!)

By default simulated bets are written to a throw-away temp DB so the real
``bet_history`` in ``DEFAULT_DB_PATH`` is never wiped by ``replace_history``.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.backtester import (  # noqa: E402
    BacktestConfig,
    format_backtest_report,
    get_backtest_summary,
    run_backtest,
)
from src.config import DEFAULT_KELLY_FRACTION, DEFAULT_W_ML, MAX_STAKE_PCT  # noqa: E402
from src.data_loader import DEFAULT_DB_PATH, load_epl_data  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="SQLite file for bet_history (REPLACED). Default: temp DB.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.db is not None:
        bets_db = Path(args.db)
        print(f"!! bet_history in {bets_db} will be REPLACED by this run")
    else:
        bets_db = Path(tempfile.mkdtemp(prefix="bt_smoke_")) / "backtest_smoke.db"
    print("=== Load EPL data (2 seasons -> SQLite) ===")
    data = load_epl_data(n_seasons=2, force_refresh=False)
    print(
        f"Loaded {len(data)} matches · "
        f"{data['Date'].min().date()} -> {data['Date'].max().date()} · "
        f"data DB={DEFAULT_DB_PATH} · bets DB={bets_db}"
    )
    if "Season" in data.columns:
        print("Seasons:", ", ".join(sorted(data["Season"].astype(str).unique())))

    min_train = 80 if len(data) < 250 else 120
    cfg = BacktestConfig(
        min_ev=0.05,
        kelly_fraction=DEFAULT_KELLY_FRACTION,
        initial_bankroll=1000.0,
        min_train_matches=min_train,
        xi=0.0018,
        odds_family="B365",
        use_ml=False,
        ensemble_w_ml=DEFAULT_W_ML,
        # Only 1X2 — OU previously dragged ROI to -27%
        allowed_markets=["1X2"],
        min_ev_1x2=0.05,
        min_ev_ou=0.10,
        min_ev_ah=0.05,
        persist=True,
        replace_history=True,
        db_path=bets_db,
        refit_every=7,
        calibration="isotonic",
        min_cal_samples=80,
        min_odds=1.40,
        max_odds=3.50,
        max_stake_pct=MAX_STAKE_PCT,
    )
    print(
        f"\n=== 1X2-only walk-forward ===\n"
        f"allowed_markets={list(cfg.normalized_markets())} · "
        f"min_ev_1x2={cfg.min_ev_1x2:.0%} · "
        f"calib={cfg.calibration} · odds=[{cfg.min_odds:.2f}, {cfg.max_odds:.2f}] · "
        f"max_stake={cfg.max_stake_pct:.1%}"
    )

    def _progress(step: int, total: int, msg: str) -> None:
        print(f"  [{step}/{total}] {msg}")

    try:
        result = run_backtest(data, config=cfg, progress=_progress)
    except ValueError as exc:
        print(f"Backtest aborted: {exc}", file=sys.stderr)
        return 1

    print()
    print("=== Bao cao Backtest RIENG thi truong 1X2 ===")
    print(format_backtest_report(result.summary))

    start = cfg.initial_bankroll
    end = float(result.summary["final_bankroll"])
    growth = end - start
    print(
        f"\nBankroll check: {start:.0f} -> {end:.2f} "
        f"({growth:+.2f}, {result.summary['bankroll_return_pct']:+.2f}%)"
    )
    if end > start:
        print("PASS: Bankroll tang truong duong tren 1X2.")
    else:
        print("NOTE: Bankroll chua tang (van co the tot hon chay full 3 markets).")

    from_db = get_backtest_summary(
        db_path=bets_db, initial_bankroll=cfg.initial_bankroll
    )
    print(f"\nPersisted bets in SQLite `bet_history`: {from_db['total_bets']}")

    if not result.bets.empty:
        print("\nSample 1X2 bets (last 5):")
        sample = result.bets.tail(5).copy()
        sample["match_date"] = pd.to_datetime(sample["match_date"]).dt.date
        cols = [
            "match_date",
            "home_team",
            "away_team",
            "market",
            "selection",
            "p_model",
            "odds",
            "ev",
            "stake_pct",
            "result",
            "pnl",
        ]
        print(sample[cols].to_string(index=False))
        assert set(result.bets["market"].unique()) == {"1X2"}, result.bets["market"].unique()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
