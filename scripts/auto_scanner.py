#!/usr/bin/env python3
"""Headless EPL Value Bet scanner for cron / GitHub Actions.

Pipeline
--------
1. Refresh historical matches + upcoming fixtures/odds into SQLite
2. Fit Dixon–Coles (+ optional LightGBM Ensemble)
3. Scan 1X2 Value Bets (EV ≥ threshold, odds band, markets filter)
4. Push Telegram alerts (no Streamlit UI required)

Environment / CLI
-----------------
TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID  (required to send)
BANKROLL, MIN_EV, KELLY_FRACTION, N_SEASONS, N_TRAIN, XI, USE_ML, W_ML
MIN_ODDS, MAX_ODDS, ALLOWED_MARKETS (comma list, default 1X2)

Example
-------
python scripts/auto_scanner.py --dry-run
python scripts/auto_scanner.py --force-refresh
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Allow `python scripts/auto_scanner.py` from repo root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load .env ASAP so TELEGRAM_* / BANKROLL are available before argparse defaults
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
load_dotenv()

import pandas as pd

from src.data_loader import (
    league_db_path,
    league_label,
    league_telegram_tag,
    load_league_data,
    load_upcoming_fixtures,
    normalize_league,
)
from src.config import (
    DEFAULT_KELLY_FRACTION,
    DEFAULT_MIN_EV,
    DEFAULT_W_ML,
    MAX_STAKE_PCT,
)
from src.dixon_coles import DixonColesModel
from src.journal import (
    add_recommendations_to_journal,
    sync_closing_odds_from_results,
)
from src.models import ensemble_weight_for_sample
from src.notifier import send_telegram_message, send_telegram_value_bets
from src.recommender import recommend_upcoming

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("auto_scanner")


def _mask_secret(value: str, *, keep: int = 4) -> str:
    text = str(value or "").strip()
    if not text:
        return "<EMPTY>"
    if len(text) <= keep * 2:
        return text[:1] + "***" + text[-1:]
    return f"{text[:keep]}…{text[-keep:]} (len={len(text)})"


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


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _parse_markets(raw: str | None) -> tuple[str, ...]:
    text = (raw or os.environ.get("ALLOWED_MARKETS") or "1X2").strip()
    alias = {
        "1X2": "1X2",
        "OU": "OU",
        "OVER/UNDER": "OU",
        "AH": "AH",
        "ASIAN HANDICAP": "AH",
        "CORNERS": "Corners",
        "CORNER": "Corners",
        "PHAT GOC": "Corners",
    }
    out: list[str] = []
    for part in text.split(","):
        key = alias.get(part.strip().upper())
        if key and key not in out:
            out.append(key)
    return tuple(out) or ("1X2",)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EPL/UWCL Value Bet auto scanner → Telegram")
    p.add_argument(
        "--league",
        type=str,
        default=os.environ.get("LEAGUE", "EPL"),
        help="Competition: EPL or UWCL (default EPL)",
    )
    p.add_argument("--force-refresh", action="store_true", help="Re-download CSVs/odds")
    p.add_argument("--dry-run", action="store_true", help="Print alerts, do not send")
    p.add_argument(
        "--no-journal",
        action="store_true",
        help="Do not write PENDING bets into live_bets journal",
    )
    p.add_argument("--n-seasons", type=int, default=_env_int("N_SEASONS", 3))
    p.add_argument("--n-train", type=int, default=_env_int("N_TRAIN", 800))
    p.add_argument("--xi", type=float, default=_env_float("XI", 0.0018))
    p.add_argument("--min-ev", type=float, default=_env_float("MIN_EV", DEFAULT_MIN_EV))
    p.add_argument(
        "--kelly",
        type=float,
        default=_env_float("KELLY_FRACTION", DEFAULT_KELLY_FRACTION),
    )
    p.add_argument("--bankroll", type=float, default=_env_float("BANKROLL", 1000.0))
    p.add_argument("--min-odds", type=float, default=_env_float("MIN_ODDS", 1.40))
    p.add_argument("--max-odds", type=float, default=_env_float("MAX_ODDS", 3.50))
    p.add_argument("--use-ml", action="store_true", default=_env_bool("USE_ML", False))
    p.add_argument("--w-ml", type=float, default=_env_float("W_ML", DEFAULT_W_ML))
    p.add_argument(
        "--markets",
        type=str,
        default=os.environ.get("ALLOWED_MARKETS", "1X2"),
        help="Comma list: 1X2,OU,AH,Corners (default 1X2)",
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
    return p


def _blend(p_dc: dict[str, float], p_ml: dict[str, float], w: float) -> dict[str, float]:
    w = max(0.0, min(1.0, float(w)))
    out = {k: (1 - w) * p_dc[k] + w * p_ml[k] for k in ("H", "D", "A")}
    s = sum(out.values())
    return {k: v / s for k, v in out.items()} if s > 0 else p_dc


def scan_value_bets(
    *,
    league: str,
    force_refresh: bool,
    n_seasons: int,
    n_train: int,
    xi: float,
    min_ev: float,
    kelly: float,
    bankroll: float,
    min_odds: float,
    max_odds: float,
    use_ml: bool,
    w_ml: float,
    markets: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load data, fit models, return (recommendations, full_history_for_clv)."""
    code = normalize_league(league)
    db_path = league_db_path(code)
    print(
        f"[1/4] Loading {league_label(code)} "
        f"(n_seasons={n_seasons}, force_refresh={force_refresh})…"
    )
    full_history = load_league_data(
        code, n_seasons=n_seasons, force_refresh=force_refresh, db_path=db_path
    )
    if full_history.empty:
        raise RuntimeError(f"No historical matches loaded for {code}")
    data = full_history.sort_values("Date").tail(int(n_train)).reset_index(drop=True)
    print(f"      Train rows={len(data)} · full={len(full_history)} · DB={db_path}")

    w_eff, w_warn = ensemble_weight_for_sample(len(full_history), float(w_ml))
    if w_warn:
        print(f"      {w_warn}")
        w_ml = float(w_eff)

    print("[2/4] Loading upcoming fixtures + odds…")
    fixtures = load_upcoming_fixtures(
        league=code,
        results=data,
        only_future=True,
        exclude_played=True,
    )
    print(f"      Fixtures={len(fixtures)}")

    print(f"[3/4] Fitting Dixon–Coles (xi={xi})…")
    dc = DixonColesModel(xi=float(xi)).fit(data)
    ml = None
    if use_ml and w_ml > 0:
        try:
            from src.ml_model import EPLMachineLearningModel

            print(f"      Fitting LightGBM (w_ML={w_ml:.0%})…")
            ml = EPLMachineLearningModel().fit(data)
        except Exception as exc:  # noqa: BLE001
            print(f"      LightGBM skipped: {exc}")
            w_ml = 0.0

    print(
        f"[4/4] Scanning markets={list(markets)} · min_ev={min_ev:.0%} · "
        f"odds=[{min_odds:.2f},{max_odds:.2f}]…"
    )
    include_ou = "OU" in markets
    include_ah = "AH" in markets
    goal_markets = tuple(m for m in markets if m != "Corners")
    recs = recommend_upcoming(
        dc,
        fixtures,
        odds_family="B365",
        min_ev=min_ev,
        kelly_fraction=kelly,
        only_value=False,
        include_ou=include_ou,
        include_ah=include_ah,
    )
    if recs.empty and "Corners" not in markets:
        return recs, full_history

    # Optional Ensemble override for 1X2 rows
    if (not recs.empty) and ml is not None and "1X2" in markets:
        rows = []
        for _, r in recs.iterrows():
            if r["market"] != "1X2":
                rows.append(r.to_dict())
                continue
            home, away = str(r["home_team"]), str(r["away_team"])
            try:
                p_dc = dc.predict_match_probs(home, away)
                p_ml = ml.predict_proba(home, away)
                p = _blend(p_dc, p_ml, w_ml)
                key = {"Home": "H", "Draw": "D", "Away": "A"}.get(
                    str(r["selection"]), ""
                )
                if key and key in p:
                    from src.recommender import expected_value, fractional_kelly

                    odds = float(r["bookmaker_odds"])
                    p_use = float(p[key])
                    ev = expected_value(p_use, odds)
                    kelly_f = fractional_kelly(p_use, odds, kelly)
                    d = r.to_dict()
                    d["p_model"] = p_use
                    d["ev"] = ev
                    d["ev_pct"] = ev * 100.0
                    d["kelly_fraction"] = kelly_f
                    d["kelly_pct"] = kelly_f * 100.0
                    d["recommended"] = ev >= min_ev
                    rows.append(d)
                    continue
            except Exception:
                pass
            rows.append(r.to_dict())
        recs = pd.DataFrame(rows)

    # CornerPredictor legs (default book odds 1.90 when feed has no corner line)
    if "Corners" in markets and not fixtures.empty:
        try:
            from src.corner_model import CornerPredictor
            from src.recommender import fractional_kelly

            print("      Fitting CornerPredictor…")
            cp_backend = "poisson" if code == "UWCL" else "auto"
            cp = CornerPredictor(backend=cp_backend).fit(data)
            corner_line = float(os.environ.get("CORNER_OU_LINE", "10.5"))
            c_rows: list[dict] = []
            for _, fx in fixtures.iterrows():
                home, away = str(fx["HomeTeam"]), str(fx["AwayTeam"])
                odds_over = (
                    float(fx["OddsCornerOver"])
                    if "OddsCornerOver" in fx.index and pd.notna(fx.get("OddsCornerOver"))
                    else 1.90
                )
                odds_under = (
                    float(fx["OddsCornerUnder"])
                    if "OddsCornerUnder" in fx.index
                    and pd.notna(fx.get("OddsCornerUnder"))
                    else 1.90
                )
                try:
                    legs = cp.predict_corner_ev(
                        home,
                        away,
                        {"over": odds_over, "under": odds_under},
                        line=corner_line,
                        min_ev=-1.0,
                    )
                except Exception:
                    continue
                kick = fx["Kickoff"] if "Kickoff" in fx.index else None
                for leg in legs:
                    kf = fractional_kelly(float(leg["p_model"]), float(leg["odds"]), kelly)
                    c_rows.append(
                        {
                            "kickoff": kick,
                            "home_team": home,
                            "away_team": away,
                            "market": "Corners",
                            "selection": leg["selection"],
                            "bookmaker_odds": leg["odds"],
                            "p_model": leg["p_model"],
                            "ev": leg["ev"],
                            "ev_pct": leg["ev_pct"],
                            "kelly_fraction": kf,
                            "kelly_pct": kf * 100.0,
                            "recommended": leg["ev"] >= min_ev,
                        }
                    )
            if c_rows:
                cdf = pd.DataFrame(c_rows)
                recs = (
                    pd.concat([recs, cdf], ignore_index=True)
                    if not recs.empty
                    else cdf
                )
                print(f"      Corner legs scanned: {len(c_rows)}")
        except Exception as exc:  # noqa: BLE001
            print(f"      Corners scan skipped: {exc}")

    if recs.empty:
        return recs, full_history

    # Market + EV + odds band filters
    allowed = list(markets) if markets else list(goal_markets)
    mask = recs["market"].isin(allowed)
    mask &= recs["ev"] >= float(min_ev)
    mask &= recs["bookmaker_odds"].between(float(min_odds), float(max_odds))
    out = recs.loc[mask].copy()
    if out.empty:
        return out, full_history
    out["stake"] = out["kelly_fraction"] * float(bankroll)
    # Cap stake at MAX_STAKE_PCT bankroll (same safety as strategy / backtester)
    max_stake = float(bankroll) * float(MAX_STAKE_PCT)
    out["stake"] = out["stake"].clip(upper=max_stake)
    out["kelly_fraction"] = out["stake"] / float(bankroll)
    out["kelly_pct"] = out["kelly_fraction"] * 100.0
    return out.sort_values("ev", ascending=False).reset_index(drop=True), full_history


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    markets = _parse_markets(args.markets)
    league = normalize_league(args.league)
    db_path = league_db_path(league)
    tag = league_telegram_tag(league)
    # UWCL seasons are shorter — default more seasons / fewer train rows if unset via CLI defaults
    n_seasons = int(args.n_seasons)
    n_train = int(args.n_train)
    if league == "UWCL":
        if n_seasons < 3:
            n_seasons = 5
        if n_train > 400 and os.environ.get("N_TRAIN") is None and args.n_train == 800:
            n_train = 250

    print(f"League={league} ({league_label(league)}) · DB={db_path}")

    try:
        recs, history = scan_value_bets(
            league=league,
            force_refresh=bool(args.force_refresh),
            n_seasons=n_seasons,
            n_train=n_train,
            xi=float(args.xi),
            min_ev=float(args.min_ev),
            kelly=float(args.kelly),
            bankroll=float(args.bankroll),
            min_odds=float(args.min_odds),
            max_odds=float(args.max_odds),
            use_ml=bool(args.use_ml),
            w_ml=float(args.w_ml),
            markets=markets,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"SCAN FAILED: {exc}", file=sys.stderr)
        token, chat = args.token.strip(), str(args.chat_id).strip()
        if token and chat and not args.dry_run:
            try:
                send_telegram_message(
                    token,
                    chat,
                    f"⚠️ <b>[{tag} Scanner lỗi]</b>\n<code>{exc}</code>",
                )
            except Exception:
                pass
        return 1

    # Backfill CLV for journal rows whose matches now have closing odds in DB
    try:
        clv_sync = sync_closing_odds_from_results(history, db_path)
        print(
            f"CLV sync: updated={clv_sync['updated']} skipped={clv_sync['skipped']}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"CLV sync skipped: {exc}")

    n = len(recs)
    print(f"\nFound {n} Value Bet(s) after filters.")
    if n:
        cols = [
            c
            for c in (
                "kickoff",
                "home_team",
                "away_team",
                "market",
                "selection",
                "bookmaker_odds",
                "ev_pct",
                "kelly_pct",
                "stake",
            )
            if c in recs.columns
        ]
        show = recs[cols].copy()
        if "kickoff" in show.columns:
            show["kickoff"] = pd.to_datetime(show["kickoff"], errors="coerce").dt.strftime(
                "%Y-%m-%d %H:%M"
            )
        print(show.to_string(index=False))

    # Paper-trading journal (PENDING + dedup) — runs even without Telegram
    if n and not args.no_journal and not args.dry_run:
        journal_out = add_recommendations_to_journal(
            recs,
            db_path=db_path,
            bankroll=float(args.bankroll),
            markets=markets,
        )
        print(
            f"Journal: created={journal_out['created']} "
            f"duplicates={journal_out['duplicates']} "
            f"ids={journal_out['ids'][:10]}"
            f"{'…' if len(journal_out['ids']) > 10 else ''}"
        )
    elif n and args.dry_run:
        print("[dry-run] Skip journal write + Telegram send.")
    elif args.no_journal:
        print("Journal write disabled (--no-journal).")

    token, chat = args.token.strip(), str(args.chat_id).strip()
    print(
        f"Telegram creds: token={_mask_secret(token)} · chat_id={_mask_secret(chat)}"
    )

    if args.dry_run:
        return 0

    if not token or not chat:
        print(
            "WARNING: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing — Telegram skipped.\n"
            "  → Điền vào .env rồi chạy: python scripts/test_telegram.py",
            file=sys.stderr,
        )
        return 0

    min_ev_pct = float(args.min_ev) * 100.0
    if n == 0:
        mk = ",".join(markets) if markets else "1X2"
        msg = (
            f"ℹ️ Hôm nay không có kèo hời ({mk}) nào đạt EV >= {min_ev_pct:.0f}%"
        )
        print(msg)
        try:
            send_telegram_message(
                token,
                chat,
                f"ℹ️ <b>[{tag} daily scan]</b>\n"
                f"Hôm nay không có kèo hời ({mk}) nào đạt EV >= {min_ev_pct:.0f}%.",
            )
            print("Sent empty-scan notice to Telegram.")
        except Exception as exc:  # noqa: BLE001
            logger.error("Telegram empty-scan send failed: %s", exc)
            print(f"FAIL sending empty-scan notice: {exc}", file=sys.stderr)
            return 3
        return 0

    try:
        result = send_telegram_value_bets(
            recs,
            token,
            chat,
            bankroll=float(args.bankroll),
            min_ev=float(args.min_ev),
            markets=markets,
            league=tag,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Telegram batch send crashed")
        print(f"FAIL Telegram batch: {exc}", file=sys.stderr)
        return 3

    print(
        f"Telegram: sent={result['sent']} skipped={result['skipped']} "
        f"errors={len(result['errors'])}"
    )
    for err in result["errors"][:5]:
        print(f"  ! {err}", file=sys.stderr)
    if result["errors"] and result["sent"] == 0:
        return 3
    return 0 if result["sent"] or n == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
