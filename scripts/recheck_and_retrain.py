#!/usr/bin/env python3
"""Post-match automation: settle paper bets, refresh CLV, retrain models.

Pipeline
--------
1. ``fetch_and_update_results`` — refresh league history (football-data / Fotmob
   → SQLite); sync closing odds into ``live_bets``. No SCHEDULED fixtures
   table is required: we settle PENDING journal rows whose kickoff has passed.
2. ``evaluate_paper_bets`` — settle vs FTR / OU / AH / Corners; report PnL,
   ROI, CLV, overnight Brier.
3. ``retrain_models`` — fit Dixon–Coles + calibrated LightGBM; save under
   ``models/``.
4. Telegram short summary via ``src.notifier`` (skipped with ``--dry-run``).

Example
-------
python scripts/recheck_and_retrain.py --league EPL --force-refresh
python scripts/recheck_and_retrain.py --league UWCL --skip-retrain --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Allow `python scripts/recheck_and_retrain.py` from repo root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
load_dotenv()

import pandas as pd

from src.data_loader import (
    league_db_path,
    league_label,
    league_telegram_tag,
    load_league_data,
    normalize_league,
)
from src.journal import (
    ensure_live_bets_table,
    evaluate_pending_against_results,
    journal_bankroll_summary,
    summarise_evaluation_session,
    sync_closing_odds_from_results,
)
from src.notifier import format_recheck_summary, send_telegram_message

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("recheck_and_retrain")

MODELS_DIR = ROOT / "models"
DC_MODEL_NAME = "dixon_coles_latest.pkl"
LGBM_MODEL_NAME = "lgbm_latest.pkl"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    return int(raw)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Post-match recheck: settle paper bets + retrain models"
    )
    p.add_argument(
        "--league",
        type=str,
        default=os.environ.get("LEAGUE", "EPL"),
        help="Competition: EPL or UWCL (default EPL)",
    )
    p.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-download league CSVs / Fotmob history into SQLite",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print Telegram summary, do not send",
    )
    p.add_argument(
        "--skip-retrain",
        action="store_true",
        help="Skip Dixon–Coles / LightGBM refit + save",
    )
    p.add_argument("--n-seasons", type=int, default=_env_int("N_SEASONS", 3))
    p.add_argument("--n-train", type=int, default=_env_int("N_TRAIN", 800))
    p.add_argument("--xi", type=float, default=_env_float("XI", 0.0018))
    p.add_argument(
        "--bankroll",
        type=float,
        default=_env_float("BANKROLL", 1000.0),
        help="Starting paper bankroll for ROI / bankroll summary",
    )
    p.add_argument(
        "--token",
        type=str,
        default=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        help="Telegram bot token (or env TELEGRAM_BOT_TOKEN)",
    )
    p.add_argument(
        "--chat-id",
        type=str,
        default=os.environ.get("TELEGRAM_CHAT_ID", ""),
        help="Telegram chat id (or env TELEGRAM_CHAT_ID)",
    )
    p.add_argument(
        "--models-dir",
        type=str,
        default=str(MODELS_DIR),
        help="Directory for dixon_coles_latest.pkl / lgbm_latest.pkl",
    )
    return p


def fetch_and_update_results(
    *,
    league: str = "EPL",
    force_refresh: bool = False,
    n_seasons: int = 3,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Refresh finished match history and backfill journal closing odds / CLV.

    Prefers ``load_league_data(..., force_refresh=…)`` (football-data.co.uk for
    EPL, Fotmob for UWCL) over fragile Flashscore score scraping. Corners
    ``HC``/``AC`` are updated when present in the source; UWCL often lacks
    them and that is skipped gracefully later at settle time.

    There is no SCHEDULED fixtures table — settlement targets PENDING
    ``live_bets`` whose kickoff has passed (see ``evaluate_paper_bets``).
    """
    code = normalize_league(league)
    path = Path(db_path) if db_path is not None else league_db_path(code)
    ensure_live_bets_table(path)

    print(
        f"[1/4] Fetching {league_label(code)} history "
        f"(n_seasons={n_seasons}, force_refresh={force_refresh})…"
    )
    history = load_league_data(
        code, n_seasons=n_seasons, force_refresh=force_refresh, db_path=path
    )
    if history.empty:
        raise RuntimeError(f"No historical matches loaded for {code}")

    has_corners = {"HC", "AC"}.issubset(set(history.columns))
    corner_cov = 0
    if has_corners:
        corner_cov = int(history[["HC", "AC"]].notna().all(axis=1).sum())

    print(
        f"      matches={len(history)} · DB={path} · "
        f"corners={'yes' if has_corners else 'no'}"
        + (f" ({corner_cov} rows)" if has_corners else "")
    )

    clv = sync_closing_odds_from_results(history, path)
    print(f"      CLV sync: updated={clv['updated']} skipped={clv['skipped']}")

    return {
        "league": code,
        "db_path": path,
        "history": history,
        "n_matches": int(len(history)),
        "has_corners": bool(has_corners),
        "corner_rows": corner_cov,
        "clv_sync": clv,
        "source": history.attrs.get("data_source"),
    }


def evaluate_paper_bets(
    history: pd.DataFrame,
    *,
    db_path: Path | str,
    bankroll: float = 1000.0,
) -> dict[str, Any]:
    """Settle PENDING past-kickoff paper bets; print PnL / ROI / CLV / Brier."""
    print("[2/4] Evaluating paper bets (PENDING past kickoff)…")
    session = evaluate_pending_against_results(
        history, db_path, sync_clv=True
    )
    summary = summarise_evaluation_session(
        session, db_path=db_path, initial_bankroll=float(bankroll)
    )

    print(
        f"      settled={session['settled']} "
        f"({session['wins']}W/{session['losses']}L/{session['pushes']}P) · "
        f"session PnL={session['pnl']:+.2f} · skipped={session['skipped']}"
    )
    journal = summary["journal"]
    brier_s = (
        f"{journal['brier']:.3f}"
        if journal["brier"] == journal["brier"]
        else "n/a"
    )
    clv_s = (
        f"{journal['avg_clv_pct']:+.2f}%"
        if journal["avg_clv"] == journal["avg_clv"]
        else "n/a"
    )
    print(
        f"      journal: n_settled={journal['n_settled']} · "
        f"PnL={journal['realised_pnl']:+.2f} · ROI={journal['roi_pct']:+.1f}% · "
        f"hit={journal['hit_rate']*100:.0f}% · CLV={clv_s} · Brier={brier_s}"
    )
    for reason in session.get("skipped_reasons", [])[:8]:
        print(f"      skip: {reason}")
    if session.get("details"):
        for d in session["details"][:15]:
            print(
                f"      #{d['id']} {d['home_team']} vs {d['away_team']} "
                f"[{d['market']}] {d['selection']} → {d['status']} "
                f"({d['score']}) PnL={d['pnl']:+.2f}"
            )
    return summary


def retrain_models(
    history: pd.DataFrame,
    *,
    league: str = "EPL",
    xi: float = 0.0018,
    n_train: int = 800,
    models_dir: Path | str = MODELS_DIR,
) -> dict[str, Any]:
    """Fit Dixon–Coles + calibrated LightGBM; save pickles under ``models/``."""
    print("[3/4] Retraining models…")
    code = normalize_league(league)
    out_dir = Path(models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ".gitkeep").touch(exist_ok=True)

    data = history.sort_values("Date").tail(int(n_train)).reset_index(drop=True)
    print(f"      train rows={len(data)} (tail n_train={n_train})")

    try:
        import joblib
    except ImportError:  # pragma: no cover
        import pickle as joblib  # type: ignore

    from src.dixon_coles import DixonColesModel

    dc_path = out_dir / DC_MODEL_NAME
    lgbm_path = out_dir / LGBM_MODEL_NAME
    detail_parts: list[str] = []
    errors: list[str] = []

    try:
        dc = DixonColesModel(xi=float(xi)).fit(data)
        joblib.dump(dc, dc_path)
        detail_parts.append("DC")
        print(f"      saved {dc_path.name}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"DixonColes: {exc}")
        logger.exception("Dixon–Coles retrain failed")

    try:
        from src.ml_model import EPLMachineLearningModel

        ml = EPLMachineLearningModel(calibrate=True).fit(data)
        joblib.dump(ml, lgbm_path)
        cal = "calibrated" if getattr(ml, "calibrated_", False) else "raw"
        detail_parts.append(f"LGBM({cal})")
        print(f"      saved {lgbm_path.name} ({cal})")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"LightGBM: {exc}")
        logger.warning("LightGBM retrain failed: %s", exc)
        print(f"      LightGBM skipped: {exc}")

    # Success if Dixon–Coles saved (LightGBM may fail on thin samples).
    ok = any(p.startswith("DC") for p in detail_parts)
    return {
        "ok": ok,
        "league": code,
        "dc_path": str(dc_path) if dc_path.exists() else None,
        "lgbm_path": str(lgbm_path) if lgbm_path.exists() else None,
        "detail": " + ".join(detail_parts) if detail_parts else "none",
        "errors": errors,
        "n_train": int(len(data)),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    league = normalize_league(args.league)
    db_path = league_db_path(league)
    tag = league_telegram_tag(league)

    n_seasons = int(args.n_seasons)
    n_train = int(args.n_train)
    if league == "UWCL":
        if n_seasons < 3:
            n_seasons = 5
        if n_train > 400 and os.environ.get("N_TRAIN") is None and args.n_train == 800:
            n_train = 250

    print(f"League={league} ({league_label(league)}) · DB={db_path}")

    retrain_info: dict[str, Any] = {
        "ok": None,
        "detail": "skipped",
        "errors": [],
    }

    try:
        fetch = fetch_and_update_results(
            league=league,
            force_refresh=bool(args.force_refresh),
            n_seasons=n_seasons,
            db_path=db_path,
        )
        history = fetch["history"]
        eval_summary = evaluate_paper_bets(
            history, db_path=db_path, bankroll=float(args.bankroll)
        )

        if not args.skip_retrain:
            retrain_info = retrain_models(
                history,
                league=league,
                xi=float(args.xi),
                n_train=n_train,
                models_dir=Path(args.models_dir),
            )
            if not retrain_info["ok"]:
                print(f"      retrain issues: {retrain_info.get('errors')}")
        else:
            print("[3/4] Retrain skipped (--skip-retrain)")
            retrain_info = {"ok": None, "detail": "skipped", "errors": []}

    except Exception as exc:  # noqa: BLE001
        print(f"RECHECK FAILED: {exc}", file=sys.stderr)
        token, chat = args.token.strip(), str(args.chat_id).strip()
        if token and chat and not args.dry_run:
            try:
                send_telegram_message(
                    token,
                    chat,
                    f"⚠️ <b>[{tag} Recheck lỗi]</b>\n<code>{exc}</code>",
                )
            except Exception:
                pass
        return 1

    journal = eval_summary["journal"]
    text = format_recheck_summary(
        league=tag,
        settled=int(eval_summary.get("settled", 0)),
        wins=int(eval_summary.get("wins", 0)),
        losses=int(eval_summary.get("losses", 0)),
        pushes=int(eval_summary.get("pushes", 0)),
        session_pnl=float(eval_summary.get("pnl", 0.0)),
        hit_rate=float(journal.get("hit_rate", 0.0)),
        realised_pnl=float(journal.get("realised_pnl", 0.0)),
        roi=float(journal.get("roi", 0.0)) if journal.get("roi") == journal.get("roi") else None,
        avg_clv=journal.get("avg_clv"),
        brier=journal.get("brier"),
        retrain_ok=retrain_info.get("ok"),
        retrain_detail=str(retrain_info.get("detail") or ""),
        skipped=int(eval_summary.get("skipped", 0)),
    )

    print("[4/4] Telegram summary:")
    print(text.replace("<b>", "").replace("</b>", "").replace("<code>", "").replace("</code>", ""))

    token, chat = args.token.strip(), str(args.chat_id).strip()
    if args.dry_run:
        print("      dry-run: not sent")
    elif not token or not chat:
        print("      Telegram skipped (missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID)")
    else:
        try:
            send_telegram_message(token, chat, text)
            print("      sent OK")
        except Exception as exc:  # noqa: BLE001
            print(f"      Telegram send failed: {exc}", file=sys.stderr)
            return 1

    # Always refresh bankroll line for operators
    bank = journal_bankroll_summary(float(args.bankroll), db_path)
    print(
        f"\nDone. Bankroll {bank['current_bankroll']:.2f} "
        f"(pending stake {bank['pending_stake']:.2f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
