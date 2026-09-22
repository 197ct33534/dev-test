"""Batch Top-N value-bet scanner for the VN landing dashboard.

Reuses :func:`src.recommender.recommend_upcoming` (+ optional Ensemble blend)
rather than duplicating market math from ``scripts/auto_scanner.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from src.config import (
    DEFAULT_KELLY_FRACTION,
    DEFAULT_MIN_EV,
    DEFAULT_W_ML,
    MAX_STAKE_PCT,
)
from src.dixon_coles import DixonColesModel
from src.recommender import expected_value, recommend_upcoming
from src.strategy import capped_kelly_fraction
from src.timezone_utils import filter_matches_today_tomorrow, format_kickoff_vn, kickoff_to_vn

DEFAULT_ALLOWED_MARKETS: tuple[str, ...] = ("1X2",)


def _blend_1x2(
    p_dc: Mapping[str, float],
    p_ml: Mapping[str, float],
    w_ml: float,
) -> dict[str, float]:
    """P = (1 − w)·P_DC + w·P_ML, renormalised."""
    w = float(max(0.0, min(1.0, w_ml)))
    out = {
        k: (1.0 - w) * float(p_dc[k]) + w * float(p_ml[k]) for k in ("H", "D", "A")
    }
    total = sum(out.values())
    if total <= 0:
        return dict(p_dc)
    return {k: v / total for k, v in out.items()}


def _match_id_from_row(row: Mapping[str, Any] | pd.Series, home: str, away: str) -> str:
    if isinstance(row, pd.Series):
        if "FlashscoreEventId" in row.index and pd.notna(row.get("FlashscoreEventId")):
            return str(row["FlashscoreEventId"])
    elif isinstance(row, Mapping):
        mid = row.get("FlashscoreEventId")
        if mid is not None and not (isinstance(mid, float) and pd.isna(mid)):
            return str(mid)
    return f"{home}__{away}".replace(" ", "_").replace("'", "")


def _selection_key(selection: str) -> str | None:
    return {"Home": "H", "Draw": "D", "Away": "A"}.get(str(selection))


def _apply_ensemble_1x2(
    recs: pd.DataFrame,
    dc_model: DixonColesModel,
    ml_model: Any,
    *,
    w_ml: float,
    min_ev: float,
    kelly_fraction: float,
    max_stake_pct: float,
) -> pd.DataFrame:
    """Recompute 1X2 P/EV/Kelly with Ensemble blend when LightGBM is available."""
    if recs.empty or ml_model is None or w_ml <= 0:
        return recs

    rows: list[dict[str, Any]] = []
    for _, r in recs.iterrows():
        d = r.to_dict()
        if str(r.get("market", "")) != "1X2":
            rows.append(d)
            continue
        home, away = str(r["home_team"]), str(r["away_team"])
        key = _selection_key(str(r.get("selection", "")))
        if not key:
            rows.append(d)
            continue
        try:
            p_dc = dc_model.predict_match_probs(home, away)
            p_ml = ml_model.predict_proba(home, away)
            p = _blend_1x2(p_dc, p_ml, w_ml)
            odds = float(r["bookmaker_odds"])
            p_use = float(p[key])
            ev = expected_value(p_use, odds)
            kelly_f = capped_kelly_fraction(
                p_use,
                odds,
                kelly_fraction=kelly_fraction,
                max_stake_pct=max_stake_pct,
            )
            d["p_model"] = p_use
            d["fair_odds"] = (1.0 / p_use) if p_use > 0 else d.get("fair_odds")
            d["ev"] = ev
            d["ev_pct"] = ev * 100.0
            d["kelly_fraction"] = kelly_f
            d["kelly_pct"] = kelly_f * 100.0
            d["recommended"] = ev >= float(min_ev)
            d["p_source"] = f"Ensemble (w_ML={w_ml:.0%})"
        except Exception:  # noqa: BLE001
            pass
        rows.append(d)
    return pd.DataFrame(rows)


def _enrich_display_rows(
    recs: pd.DataFrame,
    fixtures: pd.DataFrame,
    *,
    league: str,
    bankroll: float,
    max_stake_pct: float,
) -> pd.DataFrame:
    """Add kickoff_vn, match_id, stake, league columns for the landing UI."""
    if recs.empty:
        return recs

    fx_index: dict[tuple[str, str], pd.Series] = {}
    if fixtures is not None and not fixtures.empty:
        for _, fx in fixtures.iterrows():
            fx_index[(str(fx["HomeTeam"]), str(fx["AwayTeam"]))] = fx

    rows: list[dict[str, Any]] = []
    for _, r in recs.iterrows():
        d = r.to_dict()
        home = str(d.get("home_team", ""))
        away = str(d.get("away_team", ""))
        fx = fx_index.get((home, away))
        kick = d.get("kickoff") or d.get("match_date")
        if fx is not None and "Kickoff" in fx.index and pd.notna(fx["Kickoff"]):
            kick = fx["Kickoff"]
        d["kickoff"] = kick
        d["kickoff_vn"] = format_kickoff_vn(kick)
        vn_ts = kickoff_to_vn(kick)
        d["kickoff_vn_ts"] = vn_ts if not pd.isna(vn_ts) else None
        d["league"] = league
        d["home"] = home
        d["away"] = away
        d["odds"] = float(d.get("bookmaker_odds") or 0.0)
        d["match_id"] = _match_id_from_row(fx if fx is not None else d, home, away)
        kelly_f = float(d.get("kelly_fraction") or 0.0)
        stake = kelly_f * float(bankroll)
        cap = float(bankroll) * float(max_stake_pct)
        if stake > cap:
            stake = cap
            kelly_f = stake / float(bankroll) if bankroll else 0.0
            d["kelly_fraction"] = kelly_f
            d["kelly_pct"] = kelly_f * 100.0
        d["stake"] = stake
        d["ev_pct"] = float(d.get("ev_pct") if d.get("ev_pct") is not None else float(d.get("ev", 0.0)) * 100.0)
        d["p_model"] = float(d.get("p_model") or 0.0)
        d["market"] = str(d.get("market", ""))
        d["selection"] = str(d.get("selection", ""))
        rows.append(d)

    out = pd.DataFrame(rows)
    if "ev" in out.columns:
        out = out.sort_values("ev", ascending=False).reset_index(drop=True)
    return out


@dataclass
class TopScanResult:
    """Payload for the Streamlit Top-20 landing tab."""

    value_bets: pd.DataFrame
    fallback_bets: pd.DataFrame
    display_bets: pd.DataFrame
    below_threshold: bool
    n_value: int
    window_start_utc: pd.Timestamp | None = None
    window_end_utc: pd.Timestamp | None = None
    odds_missing: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "value_bets": self.value_bets,
            "fallback_bets": self.fallback_bets,
            "display_bets": self.display_bets,
            "below_threshold": self.below_threshold,
            "n_value": self.n_value,
            "window_start_utc": self.window_start_utc,
            "window_end_utc": self.window_end_utc,
            "odds_missing": self.odds_missing,
        }


def _fixture_has_1x2_odds(row: Mapping[str, Any] | pd.Series) -> bool:
    cols = ("B365H", "B365D", "B365A")
    if isinstance(row, pd.Series):
        return all(c in row.index and pd.notna(row[c]) for c in cols)
    return all(c in row and row.get(c) is not None and not pd.isna(row.get(c)) for c in cols)


def _fixtures_no_odds_fallback(
    fixtures: pd.DataFrame,
    *,
    league: str,
    top_n: int,
) -> pd.DataFrame:
    """Build display rows when fixtures exist but book odds / EV are unavailable."""
    if fixtures is None or fixtures.empty:
        return pd.DataFrame()

    ordered = fixtures.sort_values("Kickoff") if "Kickoff" in fixtures.columns else fixtures
    rows: list[dict[str, Any]] = []
    for _, fx in ordered.head(int(top_n)).iterrows():
        home = str(fx.get("HomeTeam", ""))
        away = str(fx.get("AwayTeam", ""))
        kick = fx.get("Kickoff") if "Kickoff" in fx.index else fx.get("Date")
        vn_ts = kickoff_to_vn(kick)
        has_odds = _fixture_has_1x2_odds(fx)
        odds_h = float(fx["B365H"]) if has_odds else float("nan")
        rows.append(
            {
                "home_team": home,
                "away_team": away,
                "home": home,
                "away": away,
                "market": "1X2",
                "selection": "—",
                "p_model": float("nan"),
                "fair_odds": float("nan"),
                "bookmaker_odds": odds_h,
                "odds": odds_h,
                "ev": float("nan"),
                "ev_pct": float("nan"),
                "kelly_fraction": 0.0,
                "kelly_pct": 0.0,
                "stake": 0.0,
                "recommended": False,
                "kickoff": kick,
                "kickoff_vn": format_kickoff_vn(kick),
                "kickoff_vn_ts": vn_ts if not pd.isna(vn_ts) else None,
                "league": league,
                "match_id": _match_id_from_row(fx, home, away),
                "odds_missing": not has_odds,
                "p_source": "n/a",
            }
        )
    return pd.DataFrame(rows)


def scan_top_value_bets(
    fixtures: pd.DataFrame,
    dc_model: DixonColesModel,
    ml_model: Any | None = None,
    *,
    w_ml: float = DEFAULT_W_ML,
    min_ev: float = DEFAULT_MIN_EV,
    kelly_fraction: float = DEFAULT_KELLY_FRACTION,
    max_stake_pct: float = MAX_STAKE_PCT,
    allowed_markets: Sequence[str] = DEFAULT_ALLOWED_MARKETS,
    top_n: int = 20,
    bankroll: float = 1000.0,
    league: str = "EPL",
    odds_family: str = "B365",
    filter_vn_window: bool = True,
    now: Any | None = None,
) -> TopScanResult:
    """Scan fixtures for Top-N value bets (EV descending).

    Blends Ensemble ``w_ML`` on 1X2 when ``ml_model`` is provided. Filters to
    the VN today+tomorrow window by default. When fewer than ``top_n`` bets
    meet ``min_ev``, ``fallback_bets`` holds the best-EV ranking (may be below
    threshold) and ``below_threshold`` is True when no EV≥threshold bets exist.
    """
    from src.timezone_utils import get_vn_today_tomorrow_window

    win_start, win_end = get_vn_today_tomorrow_window(now=now, as_naive_utc=True)

    if fixtures is None or fixtures.empty:
        empty = pd.DataFrame()
        return TopScanResult(
            value_bets=empty,
            fallback_bets=empty,
            display_bets=empty,
            below_threshold=True,
            n_value=0,
            window_start_utc=win_start,
            window_end_utc=win_end,
            odds_missing=False,
        )

    fx = fixtures.copy()
    if filter_vn_window and "Kickoff" in fx.columns:
        fx = filter_matches_today_tomorrow(fx, kickoff_col="Kickoff", now=now)

    markets = tuple(allowed_markets) or DEFAULT_ALLOWED_MARKETS
    include_ou = "OU" in markets
    include_ah = "AH" in markets
    goal_markets = [m for m in markets if m != "Corners"]

    if fx.empty or not goal_markets:
        empty = pd.DataFrame()
        return TopScanResult(
            value_bets=empty,
            fallback_bets=empty,
            display_bets=empty,
            below_threshold=True,
            n_value=0,
            window_start_utc=win_start,
            window_end_utc=win_end,
            odds_missing=False,
        )

    # Scan all legs (only_value=False) so we can build a below-threshold fallback.
    recs = recommend_upcoming(
        dc_model,
        fx,
        odds_family=odds_family,
        min_ev=min_ev,
        kelly_fraction=kelly_fraction,
        only_value=False,
        include_ou=include_ou,
        include_ah=include_ah,
    )

    if not recs.empty:
        recs = recs.loc[recs["market"].isin(goal_markets)].copy()

    use_w = float(w_ml) if ml_model is not None else 0.0
    if not recs.empty and ml_model is not None and use_w > 0 and "1X2" in markets:
        recs = _apply_ensemble_1x2(
            recs,
            dc_model,
            ml_model,
            w_ml=use_w,
            min_ev=min_ev,
            kelly_fraction=kelly_fraction,
            max_stake_pct=max_stake_pct,
        )

    # Optional Corners via CornerPredictor is left to the UI / auto_scanner;
    # landing Top-20 focuses on goal markets from recommend_upcoming.

    if recs.empty:
        # Fixtures in window but no scannable odds (or all legs skipped) —
        # still surface the match list instead of a dead-end empty state.
        fallback = _fixtures_no_odds_fallback(fx, league=league, top_n=top_n)
        return TopScanResult(
            value_bets=pd.DataFrame(),
            fallback_bets=fallback,
            display_bets=fallback,
            below_threshold=True,
            n_value=0,
            window_start_utc=win_start,
            window_end_utc=win_end,
            odds_missing=True,
        )

    # Recompute capped Kelly for all rows (recommend_upcoming uses uncapped fractional).
    kelly_vals = []
    for _, r in recs.iterrows():
        kelly_vals.append(
            capped_kelly_fraction(
                float(r["p_model"]),
                float(r["bookmaker_odds"]),
                kelly_fraction=kelly_fraction,
                max_stake_pct=max_stake_pct,
            )
        )
    recs = recs.copy()
    recs["kelly_fraction"] = kelly_vals
    recs["kelly_pct"] = recs["kelly_fraction"] * 100.0
    recs["recommended"] = recs["ev"] >= float(min_ev)

    enriched = _enrich_display_rows(
        recs,
        fx,
        league=league,
        bankroll=bankroll,
        max_stake_pct=max_stake_pct,
    )

    value = enriched.loc[enriched["ev"] >= float(min_ev)].copy()
    value = value.sort_values("ev", ascending=False).head(int(top_n)).reset_index(drop=True)

    fallback = (
        enriched.sort_values("ev", ascending=False)
        .head(int(top_n))
        .reset_index(drop=True)
    )

    below = value.empty
    display = value if not below else fallback

    return TopScanResult(
        value_bets=value,
        fallback_bets=fallback,
        display_bets=display,
        below_threshold=below,
        n_value=int(len(value)),
        window_start_utc=win_start,
        window_end_utc=win_end,
        odds_missing=False,
    )
