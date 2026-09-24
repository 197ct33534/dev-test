r"""Post-match settlement against ``live_bets`` (journal) + ``global_matches.db``.

Source of truth
---------------
Paper bets live in SQLite ``live_bets`` (see :mod:`src.journal`). Compatibility
views ``user_bets`` / ``paper_trades`` alias the same table — do **not** invent a
parallel ledger.

Settlement rules
----------------
* Only ``PENDING`` rows whose kickoff + **120 minutes** have passed are settled.
* Full-time scores come from ``global_matches.db`` (legacy columns FTHG/FTAG/FTR).
* **1X2**: Home / Draw / Away vs ``FTR`` ∈ {H, D, A}.
* **OU**: total goals vs line; Asian quarter lines (``.25`` / ``.75``) split stake
  → HALF_WIN / HALF_LOSS when one half pushes and the other wins/loses.
* **AH**: goal difference ± handicap (home-line convention); same quarter rules.
* Cash:
  - WIN: PnL = stake×(odds−1), payout = stake×odds
  - HALF_WIN: PnL = ½·stake×(odds−1), payout = ½·stake×odds + ½·stake
  - PUSH / VOID: PnL = 0, payout = stake (stake returned)
  - HALF_LOSS: PnL = −½·stake, payout = ½·stake
  - LOSS: PnL = −stake, payout = 0

Call from FastAPI via :func:`settle_completed_matches_async` (``asyncio.to_thread``).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from src.backtester import settle_1x2
from src.dixon_coles import (
    _combine_settlements,
    _mirror_settlement,
    _settle_ah_home,
    _settle_over,
    _settle_under,
)
from src.global_db import GLOBAL_DB_PATH, read_matches_as_legacy
from src.journal import (
    DEFAULT_SETTLE_GRACE,
    ensure_live_bets_table,
    find_match_result,
    kickoff_has_passed,
    list_pending_past_kickoff,
    load_live_bets,
    load_live_bets_all_leagues,
    normalize_selection,
    parse_selection_line,
    settle_live_bet,
)

logger = logging.getLogger(__name__)

SettleStatus = Literal["WIN", "LOSS", "HALF_WIN", "HALF_LOSS", "PUSH", "VOID"]
AtomicLabel = str  # win | half_win | push | half_lose | lose

# Grace after kickoff before a match is considered finished (FT + buffer).
SETTLE_GRACE_MINUTES = 120
DEFAULT_GRACE = timedelta(minutes=SETTLE_GRACE_MINUTES)

_ATOMIC_TO_STATUS: dict[str, SettleStatus] = {
    "win": "WIN",
    "half_win": "HALF_WIN",
    "push": "PUSH",
    "half_lose": "HALF_LOSS",
    "lose": "LOSS",
}


def payout_from_outcome(
    status: str,
    odds: float,
    stake: float,
) -> tuple[float, float]:
    """Return ``(pnl, payout)`` for a settled outcome label.

    Parameters
    ----------
    status:
        WIN | LOSS | HALF_WIN | HALF_LOSS | PUSH | VOID.
    odds:
        Decimal bookmaker odds.
    stake:
        Stake amount in currency units.
    """
    o = float(odds)
    s = float(stake)
    key = str(status or "").strip().upper()
    if key == "VOID":
        key = "PUSH"
    if key == "WIN":
        return s * (o - 1.0), s * o
    if key == "HALF_WIN":
        return 0.5 * s * (o - 1.0), 0.5 * s * o + 0.5 * s
    if key in {"PUSH", "VOID"}:
        return 0.0, s
    if key == "HALF_LOSS":
        return -0.5 * s, 0.5 * s
    if key == "LOSS":
        return -s, 0.0
    raise ValueError(f"Unknown settlement status: {status!r}")


def atomic_label_to_status(label: str) -> SettleStatus:
    """Map Dixon–Coles atomic/combined label → journal status."""
    key = str(label or "").strip().lower()
    if key not in _ATOMIC_TO_STATUS:
        raise ValueError(f"Unknown atomic settlement label: {label!r}")
    return _ATOMIC_TO_STATUS[key]


def settle_1x2_detailed(selection: str, ftr: str, odds: float, stake: float) -> tuple[SettleStatus, float, float]:
    """Settle 1X2 → (status, pnl, payout)."""
    raw = settle_1x2(normalize_selection(selection), str(ftr))
    status: SettleStatus = "WIN" if raw == "WIN" else ("LOSS" if raw == "LOSS" else "PUSH")
    pnl, payout = payout_from_outcome(status, odds, stake)
    return status, pnl, payout


def settle_ou_detailed(
    selection: str,
    home_goals: int,
    away_goals: int,
    line: float,
    odds: float,
    stake: float,
) -> tuple[SettleStatus, float, float]:
    """Settle Over/Under including Asian quarter lines."""
    total = int(home_goals) + int(away_goals)
    if str(selection).lower().startswith("over"):
        combined = _combine_settlements(_settle_over(total, float(line)))
    else:
        combined = _combine_settlements(_settle_under(total, float(line)))
    status = atomic_label_to_status(combined)
    pnl, payout = payout_from_outcome(status, odds, stake)
    return status, pnl, payout


def settle_ah_detailed(
    selection: str,
    home_goals: int,
    away_goals: int,
    handicap: float,
    odds: float,
    stake: float,
) -> tuple[SettleStatus, float, float]:
    """Settle Asian Handicap (home-line convention; Away mirrors)."""
    home_s = _combine_settlements(
        _settle_ah_home(int(home_goals), int(away_goals), float(handicap))
    )
    if "away" in str(selection).lower():
        home_s = _mirror_settlement(home_s)
    status = atomic_label_to_status(home_s)
    pnl, payout = payout_from_outcome(status, odds, stake)
    return status, pnl, payout


def resolve_bet_settlement(
    *,
    market: str,
    selection: str,
    odds: float,
    stake: float,
    fthg: int,
    ftag: int,
    ftr: str,
    hc: float | int | None = None,
    ac: float | int | None = None,
) -> tuple[SettleStatus, float, float]:
    """Settle one bet → ``(status, pnl, payout)`` with half-win/half-loss support."""
    mkt = str(market or "1X2").strip().upper()
    sel = str(selection or "").strip()
    label, line = parse_selection_line(sel)
    o, s = float(odds), float(stake)

    if mkt in {"1X2", "MATCH", "H2H"}:
        return settle_1x2_detailed(sel, str(ftr), o, s)

    if mkt in {"OU", "O/U", "OVER/UNDER"}:
        if line is None:
            raise ValueError(f"OU selection missing line: {sel!r}")
        return settle_ou_detailed(label, int(fthg), int(ftag), float(line), o, s)

    if mkt in {"AH", "ASIAN", "ASIAN HANDICAP"}:
        if line is None:
            raise ValueError(f"AH selection missing handicap: {sel!r}")
        # Recommender stores away as ``AH Away {-home_hand:+g}``; convert
        # back to home-convention before settle_ah_detailed (which mirrors Away).
        if "away" in label.lower():
            return settle_ah_detailed("AH Away", int(fthg), int(ftag), -float(line), o, s)
        return settle_ah_detailed("AH Home", int(fthg), int(ftag), float(line), o, s)

    if mkt in {"CORNERS", "CORNER"}:
        if hc is None or ac is None or (isinstance(hc, float) and np.isnan(hc)):
            raise ValueError("Corners settlement requires HC/AC")
        hc_i, ac_i = int(hc), int(ac)
        if label in {"Over", "Under"}:
            if line is None:
                raise ValueError(f"Corners OU missing line: {sel!r}")
            return settle_ou_detailed(label, hc_i, ac_i, float(line), o, s)
        if "AH" in label.upper() or label in {"Home", "Away"}:
            if line is None:
                raise ValueError(f"Corners AH missing line: {sel!r}")
            if "away" in label.lower():
                return settle_ah_detailed("AH Away", hc_i, ac_i, -float(line), o, s)
            return settle_ah_detailed("AH Home", hc_i, ac_i, float(line), o, s)
        raise ValueError(f"Unsupported Corners selection: {sel!r}")

    return settle_1x2_detailed(sel, str(ftr), o, s)


def _journal_db_paths(
    leagues: Sequence[str] | None = None,
    *,
    db_path: Path | str | None = None,
) -> list[tuple[str | None, Path]]:
    """Resolve ``(league_code, path)`` pairs for journal settlement."""
    if db_path is not None:
        return [(None, Path(db_path))]
    from src.data_loader import get_available_leagues, league_db_path, normalize_league

    if leagues:
        codes = [normalize_league(x) for x in leagues]
    else:
        try:
            codes = [str(x["key"]) for x in get_available_leagues() if x.get("key")]
        except Exception:  # noqa: BLE001
            codes = ["EPL", "UWCL"]
        if not codes:
            codes = ["EPL", "UWCL"]
    return [(c, league_db_path(c)) for c in codes]


def _load_ft_results(
    global_db: Path | str = GLOBAL_DB_PATH,
    *,
    comp_id: str | None = None,
) -> pd.DataFrame:
    """Finished matches from ``global_matches.db`` (FTHG/FTAG/FTR)."""
    df = read_matches_as_legacy(global_db, comp_id=comp_id)
    if df.empty:
        return df
    # Keep only rows with a decisive FT board.
    if "FTR" in df.columns:
        scored = df.loc[df["FTR"].notna() & (df["FTR"].astype(str).str.strip() != "")]
        return scored.copy()
    return df


def settle_pending_against_frame(
    results: pd.DataFrame,
    db_path: Path | str,
    *,
    now: datetime | None = None,
    grace: timedelta = DEFAULT_GRACE,
) -> dict[str, Any]:
    """Settle PENDING past-grace bets in one journal DB against a results frame."""
    ensure_live_bets_table(db_path)
    pending = list_pending_past_kickoff(db_path, now=now, grace=grace)

    details: list[dict[str, Any]] = []
    settled_n = wins = losses = pushes = half_wins = half_losses = 0
    pnl_sum = 0.0
    skipped = 0
    skipped_reasons: list[str] = []

    if pending.empty:
        return {
            "settled": 0,
            "wins": 0,
            "losses": 0,
            "half_wins": 0,
            "half_losses": 0,
            "pushes": 0,
            "pnl": 0.0,
            "skipped": 0,
            "skipped_reasons": [],
            "details": [],
            "pending_past_kickoff": 0,
        }

    res = results.copy() if results is not None and not results.empty else pd.DataFrame()
    if not res.empty and "Date" in res.columns and "_day" not in res.columns:
        res["_day"] = pd.to_datetime(res["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    for _, bet in pending.iterrows():
        bet_id = int(bet["id"])
        home = str(bet["home_team"])
        away = str(bet["away_team"])
        match = find_match_result(
            res, home_team=home, away_team=away, match_date=bet.get("match_date")
        )
        if match is None:
            skipped += 1
            skipped_reasons.append(f"#{bet_id} no result yet ({home} vs {away})")
            continue

        try:
            fthg = int(match["FTHG"])
            ftag = int(match["FTAG"])
            ftr = str(match["FTR"]).strip().upper()
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            skipped_reasons.append(f"#{bet_id} bad scoreboard: {exc}")
            continue

        hc = match["HC"] if "HC" in match.index and pd.notna(match.get("HC")) else None
        ac = match["AC"] if "AC" in match.index and pd.notna(match.get("AC")) else None
        market = str(bet.get("market") or "1X2")

        try:
            status, pnl, payout = resolve_bet_settlement(
                market=market,
                selection=str(bet["selection"]),
                odds=float(bet["odds"]),
                stake=float(bet["stake_amount"]),
                fthg=fthg,
                ftag=ftag,
                ftr=ftr,
                hc=hc,
                ac=ac,
            )
        except ValueError as exc:
            skipped += 1
            skipped_reasons.append(f"#{bet_id} {exc}")
            continue

        try:
            settle_live_bet(
                bet_id,
                status,  # type: ignore[arg-type]
                db_path,
                pnl=pnl,
                payout=payout,
            )
        except RuntimeError:
            skipped += 1
            skipped_reasons.append(f"#{bet_id} already settled")
            continue

        settled_n += 1
        pnl_sum += float(pnl)
        if status == "WIN":
            wins += 1
        elif status == "HALF_WIN":
            half_wins += 1
        elif status == "LOSS":
            losses += 1
        elif status == "HALF_LOSS":
            half_losses += 1
        else:
            pushes += 1
        details.append(
            {
                "id": bet_id,
                "home_team": home,
                "away_team": away,
                "market": market,
                "selection": str(bet["selection"]),
                "status": status,
                "pnl": float(pnl),
                "payout": float(payout),
                "score": f"{fthg}-{ftag}",
            }
        )

    return {
        "settled": settled_n,
        "wins": wins,
        "losses": losses,
        "half_wins": half_wins,
        "half_losses": half_losses,
        "pushes": pushes,
        "pnl": pnl_sum,
        "skipped": skipped,
        "skipped_reasons": skipped_reasons,
        "details": details,
        "pending_past_kickoff": int(len(pending)),
    }


def settle_completed_matches(
    *,
    global_db: Path | str = GLOBAL_DB_PATH,
    leagues: Sequence[str] | None = None,
    db_path: Path | str | None = None,
    now: datetime | None = None,
    grace_minutes: int = SETTLE_GRACE_MINUTES,
    comp_id: str | None = None,
) -> dict[str, Any]:
    """Scan PENDING journal bets and settle those finished > grace after kickoff.

    Pulls FT scores from ``global_matches.db``. When ``db_path`` is set, only that
    journal DB is processed; otherwise each league SQLite journal is scanned.
    """
    grace = timedelta(minutes=int(grace_minutes))
    # Align with journal helper default when caller passes the project constant.
    if grace_minutes == SETTLE_GRACE_MINUTES and DEFAULT_SETTLE_GRACE != grace:
        pass  # explicit 120 min wins

    results = _load_ft_results(global_db, comp_id=comp_id)
    paths = _journal_db_paths(leagues, db_path=db_path)

    aggregate: dict[str, Any] = {
        "settled": 0,
        "wins": 0,
        "losses": 0,
        "half_wins": 0,
        "half_losses": 0,
        "pushes": 0,
        "pnl": 0.0,
        "skipped": 0,
        "skipped_reasons": [],
        "details": [],
        "by_league": {},
        "grace_minutes": int(grace_minutes),
        "results_rows": int(len(results)),
    }

    for league, path in paths:
        session = settle_pending_against_frame(
            results, path, now=now, grace=grace
        )
        key = league or path.name
        aggregate["by_league"][key] = {
            "settled": session["settled"],
            "pnl": session["pnl"],
            "skipped": session["skipped"],
        }
        for field in (
            "settled",
            "wins",
            "losses",
            "half_wins",
            "half_losses",
            "pushes",
            "skipped",
        ):
            aggregate[field] = int(aggregate[field]) + int(session[field])
        aggregate["pnl"] = float(aggregate["pnl"]) + float(session["pnl"])
        aggregate["skipped_reasons"].extend(session.get("skipped_reasons") or [])
        for d in session.get("details") or []:
            row = dict(d)
            if league:
                row["league"] = league
            aggregate["details"].append(row)

    logger.info(
        "settle_completed_matches: settled=%s pnl=%.2f skipped=%s (grace=%sm)",
        aggregate["settled"],
        aggregate["pnl"],
        aggregate["skipped"],
        grace_minutes,
    )
    return aggregate


async def settle_completed_matches_async(**kwargs: Any) -> dict[str, Any]:
    """Async wrapper — runs sync SQLite settlement off the event loop."""
    return await asyncio.to_thread(settle_completed_matches, **kwargs)


# ---------------------------------------------------------------------------
# Performance analytics
# ---------------------------------------------------------------------------

_SETTLED = frozenset({"WIN", "LOSS", "PUSH", "HALF_WIN", "HALF_LOSS", "VOID"})
_WIN_LIKE = frozenset({"WIN", "HALF_WIN"})
_LOSS_LIKE = frozenset({"LOSS", "HALF_LOSS"})


def _ev_to_percent(ev: float) -> float:
    """Normalise stored EV (fraction or already-percent) → percent points."""
    x = float(ev)
    # Heuristic: |ev| ≤ 2 → treat as fraction (0.08 → 8%).
    if abs(x) <= 2.0:
        return x * 100.0
    return x


def compute_performance_analytics(
    bets: pd.DataFrame | None = None,
    *,
    days: int | None = None,
    league: str | None = None,
    db_path: Path | str | None = None,
    leagues: Sequence[str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Aggregate journal performance metrics for the analytics API.

    Parameters
    ----------
    days:
        Optional lookback window on ``created_at`` / ``match_date`` (inclusive).
    league:
        Filter to one league when loading multi-DB journals.
    """
    if bets is None:
        if db_path is not None:
            bets = load_live_bets(db_path)
            if not bets.empty:
                bets = bets.copy()
                bets["league"] = league
        else:
            codes = [league] if league else list(leagues or ("EPL", "UWCL"))
            bets = load_live_bets_all_leagues(codes)

    df = bets.copy() if bets is not None else pd.DataFrame()
    note_parts: list[str] = []

    if not df.empty and league and "league" in df.columns:
        df = df.loc[df["league"].astype(str).str.upper() == str(league).strip().upper()]

    if not df.empty and days is not None and int(days) > 0:
        ref = now or datetime.now(timezone.utc)
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)
        cutoff = ref - timedelta(days=int(days))
        ts_col = None
        for cand in ("created_at", "match_date", "settled_at"):
            if cand in df.columns:
                ts_col = cand
                break
        if ts_col:
            ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
            df = df.loc[ts.notna() & (ts >= cutoff)].copy()
        else:
            note_parts.append("days filter ignored (no date column)")

    total_placed = int(len(df))
    settled = (
        df.loc[df["status"].astype(str).str.upper().isin(_SETTLED)].copy()
        if not df.empty and "status" in df.columns
        else pd.DataFrame()
    )

    status_u = (
        settled["status"].astype(str).str.upper()
        if not settled.empty
        else pd.Series(dtype=str)
    )
    n_win = int(status_u.isin(_WIN_LIKE).sum()) if not settled.empty else 0
    n_loss = int(status_u.isin(_LOSS_LIKE).sum()) if not settled.empty else 0
    decisive = n_win + n_loss
    win_rate = (100.0 * n_win / decisive) if decisive else 0.0

    net_pnl = (
        float(pd.to_numeric(settled["pnl"], errors="coerce").fillna(0).sum())
        if not settled.empty and "pnl" in settled.columns
        else 0.0
    )
    stake_settled = (
        float(pd.to_numeric(settled["stake_amount"], errors="coerce").fillna(0).sum())
        if not settled.empty and "stake_amount" in settled.columns
        else 0.0
    )
    realized_roi = (100.0 * net_pnl / stake_settled) if stake_settled > 0 else 0.0

    expected_ev_pct = float("nan")
    if not settled.empty and "ev" in settled.columns:
        evs = pd.to_numeric(settled["ev"], errors="coerce").dropna()
        if len(evs):
            expected_ev_pct = float(np.mean([_ev_to_percent(x) for x in evs]))
        else:
            note_parts.append("ev_vs_realized_gap: no EV on settled bets")
    else:
        note_parts.append("ev_vs_realized_gap: no EV column / no settled bets")

    if expected_ev_pct == expected_ev_pct:  # not NaN
        gap = float(expected_ev_pct - realized_roi)
    else:
        gap = None

    brier: float | None
    brier_note: str | None
    try:
        from src.journal import compute_settled_brier

        raw_brier = compute_settled_brier(settled)
        if raw_brier != raw_brier:  # NaN
            brier = None
            brier_note = (
                "Insufficient data for Brier score "
                "(need settled WIN/LOSS/HALF_* rows with p_model)."
            )
        else:
            brier = float(raw_brier)
            brier_note = None
    except Exception as exc:  # noqa: BLE001
        brier = None
        brier_note = f"Brier unavailable: {exc}"

    return {
        "total_bets_placed": total_placed,
        "total_bets_settled": int(len(settled)),
        "win_rate_percent": round(win_rate, 2),
        "net_pnl": round(net_pnl, 2),
        "realized_roi_percent": round(realized_roi, 2),
        "ev_vs_realized_gap": None if gap is None else round(gap, 2),
        "expected_ev_percent": (
            None if expected_ev_pct != expected_ev_pct else round(float(expected_ev_pct), 2)
        ),
        "brier_score": brier,
        "brier_note": brier_note,
        "days": days,
        "league": league,
        "notes": "; ".join(note_parts) if note_parts else None,
    }


def performance_analytics_sync(
    *,
    days: int | None = None,
    league: str | None = None,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Blocking analytics entry used by API ``asyncio.to_thread``."""
    return compute_performance_analytics(days=days, league=league, db_path=db_path)


async def performance_analytics_async(
    *,
    days: int | None = None,
    league: str | None = None,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    return await asyncio.to_thread(
        performance_analytics_sync,
        days=days,
        league=league,
        db_path=db_path,
    )
