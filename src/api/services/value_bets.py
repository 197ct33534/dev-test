"""Value-bet scan service — DB-first fixtures + model pickles when available."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.config import DEFAULT_KELLY_FRACTION, DEFAULT_MIN_EV, DEFAULT_W_ML, MAX_STAKE_PCT
from src.data_loader import PROJECT_ROOT, league_label, load_league_data, normalize_league
from src.dixon_coles import (
    DEFAULT_XI,
    WEAK_TIER_NEW_TEAM_ATTACK,
    WEAK_TIER_NEW_TEAM_DEFENCE,
    DixonColesModel,
)
from src.features import attach_rest_days_from_db
from src.global_db import load_upcoming_from_db
from src.models import maybe_wrap_with_league_weights
from src.scanner import merge_top_scan_results, scan_top_value_bets
from src.strategy import generate_match_insights

logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "models"
DC_PICKLE = "dixon_coles_latest.pkl"
LGBM_PICKLE = "lgbm_latest.pkl"

_LEAGUE_DEFAULTS: dict[str, dict[str, int]] = {
    "EPL": {"n_seasons": 3, "n_train": 800},
    "UWCL": {"n_seasons": 5, "n_train": 200},
    "LALIGA": {"n_seasons": 3, "n_train": 800},
}

# Shared ``models/*_latest.pkl`` files are EPL-trained. Reusing them for UWCL /
# LaLiga / cups marks every side as thin and inflates EV — only use for EPL.
_PICKLE_COMPATIBLE_LEAGUES = frozenset({"EPL"})

# In-process cache for cold-fitted DC models (league → resolve tuple).
_MODEL_CACHE: dict[str, tuple[Any, Any | None, float, str]] = {}

VALID_MARKETS = frozenset({"1X2", "AH", "OU", "Corners"})


def parse_markets(raw: str | Sequence[str] | None) -> list[str]:
    """Parse comma-separated or multi markets into a normalised list."""
    if raw is None:
        return ["1X2", "AH", "OU"]
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
    else:
        parts = []
        for item in raw:
            parts.extend(p.strip() for p in str(item).split(",") if p.strip())
    out: list[str] = []
    for p in parts:
        key = p.upper()
        aliases = {
            "CHẤP": "AH",
            "CHAP": "AH",
            "HANDICAP": "AH",
            "TAI XIU": "OU",
            "TÀI XỈU": "OU",
            "OVER/UNDER": "OU",
            "CORNER": "Corners",
            "CORNERS": "Corners",
        }
        mapped = aliases.get(key, key if key != "CORNERS" else "Corners")
        if mapped == "CORNERS":
            mapped = "Corners"
        if mapped in VALID_MARKETS and mapped not in out:
            out.append(mapped)
    return out or ["1X2", "AH", "OU"]


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        if f != f:  # NaN
            return None
        return f
    except (TypeError, ValueError):
        return None


def _safe_str(val: Any) -> str | None:
    if val is None:
        return None
    try:
        if isinstance(val, float) and val != val:
            return None
    except TypeError:
        pass
    s = str(val).strip()
    return s or None


def _load_joblib(path: Path) -> Any | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        try:
            import joblib
        except ImportError:  # pragma: no cover
            import pickle as joblib  # type: ignore

        return joblib.load(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load model %s: %s", path, exc)
        return None


def _fit_dc(league: str, n_seasons: int, n_train: int, xi: float = DEFAULT_XI) -> DixonColesModel:
    code = normalize_league(league)
    history = load_league_data(code, n_seasons=n_seasons, force_refresh=False)
    train = history.sort_values("Date").tail(int(n_train)).reset_index(drop=True)
    use_weak = code == "UWCL" or int(len(train)) < 100
    return DixonColesModel(
        xi=float(xi),
        use_weak_tier_priors=use_weak,
        new_team_attack=WEAK_TIER_NEW_TEAM_ATTACK if use_weak else None,
        new_team_defence=WEAK_TIER_NEW_TEAM_DEFENCE if use_weak else None,
    ).fit(train)


def clear_model_cache() -> None:
    """Drop in-process cold-fit cache (tests / after retrain)."""
    _MODEL_CACHE.clear()


def _resolve_models(league: str) -> tuple[Any, Any | None, float, str]:
    """Return (dc_model, ml_model, w_ml, note).

    Prefer ``models/*_latest.pkl`` **only** for EPL (pickle training target).
    Other leagues cold-fit Dixon–Coles on that league's history (cached
    in-process) so WebApp/API EV is not driven by EPL thin-team priors.
    """
    code = normalize_league(league)
    cached = _MODEL_CACHE.get(code)
    if cached is not None:
        return cached

    if code in _PICKLE_COMPATIBLE_LEAGUES:
        dc = _load_joblib(MODELS_DIR / DC_PICKLE)
        ml = _load_joblib(MODELS_DIR / LGBM_PICKLE)
        if dc is not None:
            notes: list[str] = ["dc_pickle"]
            if ml is not None:
                notes.append("lgbm_pickle")
                w_ml = float(DEFAULT_W_ML)
            else:
                w_ml = 0.0
            result = (
                maybe_wrap_with_league_weights(dc, code),
                ml,
                w_ml,
                "+".join(notes),
            )
            _MODEL_CACHE[code] = result
            return result

    defaults = _LEAGUE_DEFAULTS.get(code, {"n_seasons": 3, "n_train": 800})
    dc = _fit_dc(code, int(defaults["n_seasons"]), int(defaults["n_train"]))
    result = (
        maybe_wrap_with_league_weights(dc, code),
        None,
        0.0,
        "dc_fitted_cold",
    )
    _MODEL_CACHE[code] = result
    return result


def _load_fixtures(comp_id: str) -> tuple[pd.DataFrame, str]:
    """Prefer ``upcoming_fixtures`` cache; network only on empty cache."""
    code = normalize_league(comp_id)
    db_df, _last = load_upcoming_from_db(code)
    if db_df is not None and not db_df.empty:
        return db_df, "db_cache"
    try:
        from src.data_loader import load_upcoming_fixtures

        fx = load_upcoming_fixtures(league=code, only_future=True, include_odds=True)
        if fx is not None and not fx.empty:
            return fx, "network"
    except Exception as exc:  # noqa: BLE001
        logger.warning("network fixtures failed for %s: %s", code, exc)
    return pd.DataFrame(), "empty"


def _row_to_bet(row: dict[str, Any] | pd.Series) -> dict[str, Any]:
    d = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    home = _safe_str(d.get("home") or d.get("home_team"))
    away = _safe_str(d.get("away") or d.get("away_team"))
    market = _safe_str(d.get("market"))
    selection = _safe_str(d.get("selection"))
    odds = _safe_float(d.get("odds") if d.get("odds") is not None else d.get("bookmaker_odds"))
    ev = _safe_float(d.get("ev"))
    ev_pct = _safe_float(d.get("ev_pct"))
    if ev_pct is None and ev is not None:
        ev_pct = ev * 100.0
    pick = None
    if market and selection:
        pick = f"{market} · {selection}"
        if odds is not None:
            pick = f"{pick} @ {odds:.2f}"

    insights = generate_match_insights(d)
    thin = d.get("thin_teams") or []
    if isinstance(thin, str):
        thin = [thin] if thin else []
    elif not isinstance(thin, list):
        thin = list(thin) if thin is not None else []

    lg = _safe_str(d.get("league") or d.get("comp_id"))
    competition = _safe_str(d.get("competition"))
    if not competition and lg:
        try:
            competition = league_label(lg)
        except Exception:  # noqa: BLE001
            competition = lg

    kick = d.get("kickoff")
    kick_s = None
    if kick is not None:
        try:
            if hasattr(kick, "isoformat"):
                kick_s = kick.isoformat()
            else:
                kick_s = str(kick)
        except Exception:  # noqa: BLE001
            kick_s = str(kick)

    return {
        "pick": pick,
        "selection": selection,
        "market": market,
        "odds": odds,
        "bookmaker_odds": _safe_float(d.get("bookmaker_odds")) or odds,
        "ev": ev,
        "ev_pct": ev_pct,
        "p_model": _safe_float(d.get("p_model")),
        "fair_odds": _safe_float(d.get("fair_odds")),
        "kelly_fraction": _safe_float(d.get("kelly_fraction")),
        "kelly_pct": _safe_float(d.get("kelly_pct")),
        "stake": _safe_float(d.get("stake")),
        "home": home,
        "away": away,
        "home_team": home,
        "away_team": away,
        "match_id": _safe_str(d.get("match_id")),
        "league": lg,
        "competition": competition,
        "kickoff": kick_s,
        "kickoff_vn": _safe_str(d.get("kickoff_vn")),
        "home_rest_days": _safe_float(d.get("home_rest_days")),
        "away_rest_days": _safe_float(d.get("away_rest_days")),
        "fatigue_label": _safe_str(d.get("fatigue_label")),
        "ai_reasons": list(insights or []),
        "recommended": bool(d.get("recommended")) if d.get("recommended") is not None else None,
        "home_thin": bool(d.get("home_thin")) if d.get("home_thin") is not None else None,
        "away_thin": bool(d.get("away_thin")) if d.get("away_thin") is not None else None,
        "thin_teams": [str(t) for t in thin],
        "p_source": _safe_str(d.get("p_source")),
        "fair_total_goals": _safe_float(d.get("fair_total_goals")),
        "fair_ou_line": _safe_float(d.get("fair_ou_line")),
        "fair_ah_line": _safe_float(d.get("fair_ah_line")),
        "bookie_ou_line": _safe_float(d.get("bookie_ou_line")),
        "bookie_ah_line": _safe_float(d.get("bookie_ah_line")),
        "ou_line_delta": _safe_float(d.get("ou_line_delta")),
        "ah_line_delta": _safe_float(d.get("ah_line_delta")),
        "line_disparity_score": _safe_float(d.get("line_disparity_score")),
        "model_fair_line": _safe_str(d.get("model_fair_line")),
        "bookie_market_line": _safe_str(d.get("bookie_market_line")),
        "line_edge": _safe_str(d.get("line_edge")),
    }


def scan_value_bets_api(
    *,
    min_ev_pct: float = 5.0,
    markets: str | Sequence[str] | None = None,
    limit: int = 20,
    league: str | None = None,
    comp_id: str | None = None,
    bankroll: float = 1000.0,
) -> dict[str, Any]:
    """Scan Top-N value bets (mirrors Streamlit Top-20 path).

    Warm path: cached upcoming fixtures + ``models/*.pkl``.
    Cold path: may fit Dixon–Coles (slow) and/or fetch fixtures over the network.
    """
    mkt_list = parse_markets(markets)
    # Scanner ignores Corners today (goal markets only); keep in response meta.
    goal_markets = tuple(m for m in mkt_list if m != "Corners") or ("1X2",)
    min_ev = float(min_ev_pct) / 100.0 if float(min_ev_pct) > 1.0 else float(min_ev_pct)
    if min_ev <= 0:
        min_ev = float(DEFAULT_MIN_EV)

    scope = normalize_league(comp_id or league) if (comp_id or league) else None
    leagues = [scope] if scope else ["EPL", "UWCL"]

    results = []
    source_bits: list[str] = []
    notes: list[str] = []

    for lg in leagues:
        fx, fx_src = _load_fixtures(lg)
        source_bits.append(f"{lg}:{fx_src}")
        if fx is None or fx.empty:
            continue
        try:
            dc, ml, w_ml, model_note = _resolve_models(lg)
            notes.append(f"{lg}:{model_note}")
        except Exception as exc:  # noqa: BLE001
            logger.exception("model resolve failed for %s", lg)
            notes.append(f"{lg}:model_error:{exc}")
            continue
        results.append(
            scan_top_value_bets(
                fx,
                dc,
                ml,
                w_ml=float(w_ml) if ml is not None else 0.0,
                min_ev=float(min_ev),
                kelly_fraction=float(DEFAULT_KELLY_FRACTION),
                max_stake_pct=MAX_STAKE_PCT,
                allowed_markets=goal_markets,
                top_n=int(max(1, limit)),
                bankroll=float(bankroll),
                league=str(lg),
                filter_vn_window=True,
            )
        )

    if not results:
        return {
            "bets": [],
            "count": 0,
            "min_ev_pct": float(min_ev_pct),
            "markets": mkt_list,
            "limit": int(limit),
            "league": scope,
            "comp_id": scope,
            "below_threshold": True,
            "odds_missing": False,
            "source": ",".join(source_bits) or "empty",
            "notes": (
                "No cached fixtures and cold network/model path failed or empty. "
                "Refresh fixtures via Streamlit or auto_scanner first."
            ),
        }

    if len(results) == 1:
        merged = results[0]
    else:
        merged = merge_top_scan_results(
            results, top_n=int(max(1, limit)), min_ev=float(min_ev)
        )

    display = merged.display_bets
    if display is None or getattr(display, "empty", True):
        display = pd.DataFrame()
    else:
        try:
            display = attach_rest_days_from_db(display)
        except Exception:  # noqa: BLE001
            pass
        if "competition" not in display.columns and "league" in display.columns:
            display = display.copy()
            display["competition"] = display["league"].map(
                lambda c: league_label(str(c)) if pd.notna(c) and str(c) else ""
            )

    bets = [_row_to_bet(r) for _, r in display.head(int(limit)).iterrows()]
    cold = any("fitted_cold" in n or ":network" in s for n, s in zip(notes, source_bits))
    note_txt = "; ".join(notes) if notes else None
    if cold:
        note_txt = (
            (note_txt + "; " if note_txt else "")
            + "Cold path may load/fit models — subsequent calls are faster with DB cache + pickles."
        )

    return {
        "bets": bets,
        "count": len(bets),
        "min_ev_pct": float(min_ev_pct),
        "markets": mkt_list,
        "limit": int(limit),
        "league": scope,
        "comp_id": scope,
        "below_threshold": bool(merged.below_threshold),
        "odds_missing": bool(merged.odds_missing),
        "source": ",".join(source_bits),
        "notes": note_txt,
    }
