"""Value-bet scan service — DB-first fixtures + model pickles when available."""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.config import (
    DEFAULT_KELLY_FRACTION,
    DEFAULT_MIN_EV,
    DEFAULT_W_ML,
    MAX_KELLY_FRACTION,
    MAX_STAKE_PCT,
)
from src.data_loader import PROJECT_ROOT, league_label, load_league_data, normalize_league
from src.dixon_coles import DEFAULT_XI, DixonColesModel
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

# In-process cache for resolved models (league → resolve tuple).
_MODEL_CACHE: dict[str, tuple[Any, Any | None, float, str]] = {}
# league → (pickle signature | None, monotonic load time) for invalidation.
_MODEL_CACHE_META: dict[str, tuple[Any, float]] = {}
# Per-league locks so concurrent cold requests fit Dixon–Coles only once.
_MODEL_LOCKS: dict[str, threading.Lock] = {}
_MODEL_LOCKS_GUARD = threading.Lock()

# Cold-fitted models are refreshed after this many minutes (new results land).
DEFAULT_COLD_FIT_TTL_MINUTES = 720.0

# Upcoming-fixtures cache freshness (odds move — never serve hours-old prices).
FIXTURES_MAX_AGE_ENV = "FIXTURES_MAX_AGE_MINUTES"
DEFAULT_FIXTURES_MAX_AGE_MINUTES = 60.0

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


def _fit_dc_frame(
    train: pd.DataFrame, *, xi: float = DEFAULT_XI, force_weak: bool = False
) -> DixonColesModel:
    use_weak = force_weak or int(len(train)) < 100
    # New-team α/δ priors are relative to the fitted league means
    # (``model.new_team_priors()``) — no absolute WEAK_TIER_* constants.
    return DixonColesModel(
        xi=float(xi),
        use_weak_tier_priors=use_weak,
    ).fit(train)


def _fit_dc(league: str, n_seasons: int, n_train: int, xi: float = DEFAULT_XI) -> DixonColesModel:
    code = normalize_league(league)
    history = load_league_data(code, n_seasons=n_seasons, force_refresh=False)
    train = history.sort_values("Date").tail(int(n_train)).reset_index(drop=True)
    return _fit_dc_frame(train, xi=xi, force_weak=code == "UWCL")


# Japanese cups mix J1 / J2 / J3 clubs that have only a handful of *cup*
# matches each: fitted on cup history alone, 51 of 59 clubs were thin and every
# fixture was priced on new-team priors. Pool every Japanese men's competition
# so each club is rated from its league form; cup ties between tiers link the
# tiers, and the league-weight proxy (J1 0.9 / J2 0.75 / J3 0.65) scales the
# remaining gap.
JAPAN_POOL: tuple[str, ...] = ("J1", "J2", "J3", "J_LEAGUE_CUP", "EMPERORS_CUP", "ACL")
POOLED_FIT_COMPETITIONS: dict[str, tuple[str, ...]] = {
    "J_LEAGUE_CUP": JAPAN_POOL,
    "EMPERORS_CUP": JAPAN_POOL,
}
MIN_POOLED_MATCHES = 150


def _read_global_matches(*, comp_id: str | None = None) -> pd.DataFrame:
    from src.global_db import read_matches_as_legacy

    return read_matches_as_legacy(comp_id=comp_id)


def _pooled_history(comps: Sequence[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for comp in comps:
        try:
            df = _read_global_matches(comp_id=comp)
        except Exception as exc:  # noqa: BLE001
            logger.warning("pooled history: %s unavailable (%s)", comp, exc)
            continue
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    # A match can be filed under two comp ids (team feed + league feed).
    out = out.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"])
    return out.sort_values("Date").reset_index(drop=True)


def clear_model_cache() -> None:
    """Drop in-process cold-fit cache (tests / after retrain)."""
    _MODEL_CACHE.clear()
    _MODEL_CACHE_META.clear()


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _pickle_signature() -> tuple[tuple[int, int] | None, ...]:
    """(mtime_ns, size) of the shared pickles — changes after a retrain."""
    sig: list[tuple[int, int] | None] = []
    for name in (DC_PICKLE, LGBM_PICKLE):
        try:
            st = (MODELS_DIR / name).stat()
            sig.append((int(st.st_mtime_ns), int(st.st_size)))
        except OSError:
            sig.append(None)
    return tuple(sig)


def _league_lock(code: str) -> threading.Lock:
    with _MODEL_LOCKS_GUARD:
        lock = _MODEL_LOCKS.get(code)
        if lock is None:
            lock = threading.Lock()
            _MODEL_LOCKS[code] = lock
        return lock


def _cached_models(code: str) -> tuple[Any, Any | None, float, str] | None:
    """Return a still-valid cache entry, else ``None``.

    EPL (pickle) entries are invalidated when ``models/*.pkl`` mtime/size
    change; cold-fitted entries expire after ``MODEL_COLD_FIT_TTL_MINUTES``.
    """
    cached = _MODEL_CACHE.get(code)
    if cached is None:
        return None
    meta = _MODEL_CACHE_META.get(code)
    if meta is None:
        # Entry injected directly (tests / legacy) — trust it.
        return cached
    sig, loaded_at = meta
    if sig is not None:
        if sig != _pickle_signature():
            return None
    else:
        ttl_min = _env_float("MODEL_COLD_FIT_TTL_MINUTES", DEFAULT_COLD_FIT_TTL_MINUTES)
        if ttl_min > 0 and (time.monotonic() - loaded_at) > ttl_min * 60.0:
            return None
    return cached


def _resolve_models(league: str) -> tuple[Any, Any | None, float, str]:
    """Return (dc_model, ml_model, w_ml, note).

    Prefer ``models/*_latest.pkl`` **only** for EPL (pickle training target).
    Other leagues cold-fit Dixon–Coles on that league's history (cached
    in-process) so WebApp/API EV is not driven by EPL thin-team priors.

    A per-league lock ensures concurrent cold requests fit only once; the
    cache reloads when the pickle files change on disk.
    """
    code = normalize_league(league)
    cached = _cached_models(code)
    if cached is not None:
        return cached
    with _league_lock(code):
        cached = _cached_models(code)
        if cached is not None:
            return cached
        return _resolve_models_uncached(code)


def _resolve_models_uncached(code: str) -> tuple[Any, Any | None, float, str]:
    if code in _PICKLE_COMPATIBLE_LEAGUES:
        sig = _pickle_signature()
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
            _MODEL_CACHE_META[code] = (sig, time.monotonic())
            return result

    pool = POOLED_FIT_COMPETITIONS.get(code)
    if pool:
        history = _pooled_history(pool)
        if len(history) >= MIN_POOLED_MATCHES:
            dc = _fit_dc_frame(history)
            result = (
                maybe_wrap_with_league_weights(dc, code),
                None,
                0.0,
                "dc_pooled_fitted_cold",
            )
            _MODEL_CACHE[code] = result
            _MODEL_CACHE_META[code] = (None, time.monotonic())
            return result
        logger.info(
            "pooled fit for %s skipped: %d matches < %d", code, len(history), MIN_POOLED_MATCHES
        )

    defaults = _LEAGUE_DEFAULTS.get(code, {"n_seasons": 3, "n_train": 800})
    dc = _fit_dc(code, int(defaults["n_seasons"]), int(defaults["n_train"]))
    result = (
        maybe_wrap_with_league_weights(dc, code),
        None,
        0.0,
        "dc_fitted_cold",
    )
    _MODEL_CACHE[code] = result
    _MODEL_CACHE_META[code] = (None, time.monotonic())
    return result


# ---------------------------------------------------------------------------
# Upcoming fixtures (freshness-aware)
# ---------------------------------------------------------------------------


def fixtures_max_age_minutes() -> float:
    """Max acceptable age of the ``upcoming_fixtures`` cache (env, minutes)."""
    return _env_float(FIXTURES_MAX_AGE_ENV, DEFAULT_FIXTURES_MAX_AGE_MINUTES)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def filter_future_fixtures(
    df: pd.DataFrame | None, now: datetime | None = None
) -> pd.DataFrame:
    """Keep rows whose kickoff is strictly after ``now`` (UTC).

    ``Kickoff`` / ``kickoff`` are naive-UTC in this project; tz-aware values
    are converted. Rows with an unparseable kickoff are dropped (not
    actionable). Frames without any kickoff column are returned unchanged.
    """
    if df is None:
        return pd.DataFrame()
    if df.empty:
        return df
    col = next((c for c in ("Kickoff", "kickoff", "kickoff_utc") if c in df.columns), None)
    if col is None:
        return df
    ref = now or _utc_now()
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    kick = pd.to_datetime(df[col], errors="coerce", utc=True)
    mask = (kick > pd.Timestamp(ref).tz_convert("UTC")).fillna(False)
    out = df.loc[mask.to_numpy(dtype=bool)].reset_index(drop=True)
    out.attrs.update(dict(getattr(df, "attrs", {}) or {}))
    return out


def refresh_upcoming_fixtures(comp_id: str, *, persist: bool = True) -> pd.DataFrame:
    """Fetch upcoming fixtures + odds over the network and refresh the DB cache.

    Shared by the API fallback path and ``src.workers.fixtures_worker``. Only a
    non-empty result is persisted (an empty write would wipe the cache).
    """
    from src.data_loader import load_upcoming_fixtures

    code = normalize_league(comp_id)
    results = None
    try:
        results = load_league_data(code, n_seasons=1, force_refresh=False)
    except Exception as exc:  # noqa: BLE001
        logger.info("history load for %s failed (fixtures still fetched): %s", code, exc)
    fx = load_upcoming_fixtures(
        league=code,
        results=results,
        only_future=True,
        exclude_played=True,
        include_odds=True,
    )
    if fx is None:
        return pd.DataFrame()
    if persist and not fx.empty:
        try:
            from src.global_db import save_upcoming_to_db

            save_upcoming_to_db(code, fx)
        except Exception as exc:  # noqa: BLE001
            logger.warning("persist upcoming fixtures for %s failed: %s", code, exc)
    return fx


_REFRESH_LOCKS: dict[str, threading.Lock] = {}
# league → (monotonic time of last network attempt, succeeded?)
_LAST_REFRESH: dict[str, tuple[float, bool]] = {}
DEFAULT_REFRESH_BACKOFF_SECONDS = 300.0


def _guarded_refresh(code: str, ref: datetime) -> pd.DataFrame | None:
    """Network refresh serialised per league, with backoff after attempts.

    Concurrent stale requests wait for one fetch instead of all hitting
    Flashscore; within the backoff window a recent success is re-read from
    the DB and a recent failure is not retried.
    """
    with _MODEL_LOCKS_GUARD:
        lock = _REFRESH_LOCKS.setdefault(code, threading.Lock())
    backoff = _env_float("FIXTURES_REFRESH_BACKOFF_SECONDS", DEFAULT_REFRESH_BACKOFF_SECONDS)
    with lock:
        last = _LAST_REFRESH.get(code)
        if last is not None and (time.monotonic() - last[0]) < backoff:
            if not last[1]:
                return None
            db_df, _ts = load_upcoming_from_db(code)
            return filter_future_fixtures(db_df, ref)
        ok = False
        try:
            fx = filter_future_fixtures(refresh_upcoming_fixtures(code), ref)
            ok = not fx.empty
            return fx
        except Exception as exc:  # noqa: BLE001
            logger.warning("network fixtures failed for %s: %s", code, exc)
            return None
        finally:
            _LAST_REFRESH[code] = (time.monotonic(), ok)


def load_fixtures_with_meta(
    comp_id: str,
    *,
    max_age_minutes: float | None = None,
    now: datetime | None = None,
    allow_network: bool = True,
) -> dict[str, Any]:
    """Load future fixtures, refreshing over the network when the cache is stale.

    Returns ``{"df", "source", "stale", "age_minutes"}``:

    * fresh cache (age ≤ max age) with future rows → ``source="db_cache"``
    * otherwise try :func:`refresh_upcoming_fixtures` → ``source="network"``
    * network failed/empty but cache has future rows → ``source="db_cache"``,
      ``stale=True`` (callers must not push alerts built on these prices)
    """
    code = normalize_league(comp_id)
    ref = now or _utc_now()
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    limit = fixtures_max_age_minutes() if max_age_minutes is None else float(max_age_minutes)

    db_df, last_ts = load_upcoming_from_db(code)
    age_min: float | None = None
    if last_ts is not None:
        try:
            age_min = max(0.0, (ref.timestamp() - float(last_ts)) / 60.0)
        except (TypeError, ValueError):
            age_min = None
    cached_future = filter_future_fixtures(db_df, ref)
    # Unknown age counts as stale; ``limit <= 0`` disables the age check.
    fresh = limit <= 0 or (age_min is not None and age_min <= limit)

    if fresh and not cached_future.empty:
        return {
            "df": cached_future,
            "source": "db_cache",
            "stale": False,
            "age_minutes": age_min,
        }

    if allow_network:
        fx = _guarded_refresh(code, ref)
        if fx is not None and not fx.empty:
            return {"df": fx, "source": "network", "stale": False, "age_minutes": 0.0}

    if not cached_future.empty:
        logger.warning(
            "serving STALE upcoming fixtures for %s (age=%s min, max=%s)",
            code,
            None if age_min is None else round(age_min, 1),
            limit,
        )
        return {
            "df": cached_future,
            "source": "db_cache",
            "stale": True,
            "age_minutes": age_min,
        }
    return {"df": pd.DataFrame(), "source": "empty", "stale": False, "age_minutes": age_min}


def _load_fixtures(comp_id: str) -> tuple[pd.DataFrame, str]:
    """Back-compat wrapper: ``(df, source)``; see :func:`load_fixtures_with_meta`."""
    meta = load_fixtures_with_meta(comp_id)
    src = str(meta["source"]) + (":stale" if meta.get("stale") else "")
    return meta["df"], src


def kickoff_iso_utc(kick: Any) -> str | None:
    """Kickoff as an explicit-UTC ISO string (``…+00:00``).

    Fixture kickoffs are naive UTC by project convention; emitting them without
    an offset makes browsers parse them as *local* time (a 17:00 VN match showed
    as 10:00).
    """
    if kick is None:
        return None
    try:
        ts = pd.Timestamp(kick)
    except (TypeError, ValueError):
        return _safe_str(kick)
    if pd.isna(ts):
        return None
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.isoformat()


def _display_name(raw: str | None) -> str | None:
    """Friendly club name for stable codes (``JP_NAGOYA`` → ``Nagoya Grampus``)."""
    if not raw:
        return raw
    try:
        from src.japan_schedule import display_team_name

        return display_team_name(raw) or raw
    except Exception:  # noqa: BLE001
        return raw


def _league_label_safe(code: str) -> str:
    try:
        return str(league_label(code))
    except Exception:  # noqa: BLE001
        return str(code)


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

    detail = d.get("detail") if isinstance(d.get("detail"), dict) else {}
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

    kick_s = kickoff_iso_utc(d.get("kickoff"))

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
        "home_display": _display_name(home),
        "away_display": _display_name(away),
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
        # Dixon–Coles used a new-team prior → stake capped (NEW_TEAM_KELLY_CAP);
        # EV on these rows is the least reliable.
        "new_team_fallback": bool(detail.get("new_team_fallback")),
        "ev_method": _safe_str(detail.get("ev_method")),
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


DEFAULT_SCAN_LEAGUES: tuple[str, ...] = ("EPL", "UWCL")
MAX_SCAN_LEAGUES = 8


def parse_leagues(raw: str | Sequence[str] | None) -> list[str]:
    """Comma-separated / list of league keys → canonical codes (deduped).

    Raises ``ValueError`` for an unknown league (routes map it to HTTP 400).
    """
    if raw is None:
        return []
    items = [raw] if isinstance(raw, str) else list(raw)
    out: list[str] = []
    for item in items:
        for part in str(item).split(","):
            part = part.strip()
            if not part:
                continue
            code = normalize_league(part)
            if code not in out:
                out.append(code)
    if len(out) > MAX_SCAN_LEAGUES:
        raise ValueError(f"At most {MAX_SCAN_LEAGUES} leagues per scan")
    return out


def _model_flags(model_note: str | None) -> dict[str, Any]:
    note = str(model_note or "")
    return {
        "ml_enabled": "lgbm" in note,
        "model_source": "pickle" if "pickle" in note else ("cold_fit" if note else None),
    }


def scan_value_bets_api(
    *,
    min_ev_pct: float = 5.0,
    markets: str | Sequence[str] | None = None,
    limit: int = 20,
    league: str | None = None,
    comp_id: str | None = None,
    leagues: str | Sequence[str] | None = None,
    kelly_fraction: float | None = None,
    one_per_match: bool = True,
    bankroll: float = 1000.0,
) -> dict[str, Any]:
    """Scan Top-N value bets (mirrors Streamlit Top-20 path).

    Units: ``min_ev_pct`` is **always percent** (``5.0`` = 5 %, ``1.0`` = 1 %).
    Values ≤ 0 fall back to ``src.config.DEFAULT_MIN_EV`` (fraction × 100).

    Warm path: fresh cached upcoming fixtures + ``models/*.pkl``.
    Cold path: may fit Dixon–Coles (slow) and/or fetch fixtures over the network.
    The response carries ``stale=True`` when any league was served from a cache
    older than ``FIXTURES_MAX_AGE_MINUTES`` because the network refresh failed.

    ``one_per_match`` (default) keeps only the best-EV leg per fixture so the
    feed never lists correlated legs with separate stakes. Legs priced on
    new-team priors are dropped by the scanner (``excluded_prior``).
    """
    mkt_list = parse_markets(markets)
    # Scanner ignores Corners today (goal markets only); keep in response meta.
    goal_markets = tuple(m for m in mkt_list if m != "Corners") or ("1X2",)
    min_ev = float(min_ev_pct) / 100.0
    if min_ev <= 0:
        min_ev = float(DEFAULT_MIN_EV)

    generated_at = _utc_now().isoformat()
    # Policy ceiling (src.config.MAX_KELLY_FRACTION) still applies downstream.
    kf = float(DEFAULT_KELLY_FRACTION) if kelly_fraction is None else float(kelly_fraction)
    kf = max(0.0, min(kf, float(MAX_KELLY_FRACTION)))

    scope = normalize_league(comp_id or league) if (comp_id or league) else None
    multi = parse_leagues(leagues)
    if multi:
        scan_leagues = multi
        if len(multi) == 1:
            scope = multi[0]
    else:
        scan_leagues = [scope] if scope else list(DEFAULT_SCAN_LEAGUES)

    results = []
    source_bits: list[str] = []
    notes: list[str] = []
    stale_leagues: list[str] = []
    league_meta: list[dict[str, Any]] = []

    for lg in scan_leagues:
        fx_meta = load_fixtures_with_meta(lg)
        fx = fx_meta["df"]
        fx_src = str(fx_meta["source"])
        if fx_meta.get("stale"):
            fx_src += ":stale"
        source_bits.append(f"{lg}:{fx_src}")
        meta: dict[str, Any] = {
            "league": str(lg),
            "label": _league_label_safe(lg),
            "fixtures_source": str(fx_meta["source"]),
            "stale": bool(fx_meta.get("stale")),
            "odds_age_minutes": _safe_float(fx_meta.get("age_minutes")),
            "fixtures": 0 if fx is None else int(len(fx)),
            "model_note": None,
            "ml_enabled": False,
            "model_source": None,
            "excluded_prior": 0,
        }
        league_meta.append(meta)
        if fx is None or fx.empty:
            continue
        if fx_meta.get("stale"):
            stale_leagues.append(str(lg))
        try:
            dc, ml, w_ml, model_note = _resolve_models(lg)
            notes.append(f"{lg}:{model_note}")
            meta["model_note"] = model_note
            meta.update(_model_flags(model_note))
        except Exception:  # noqa: BLE001
            logger.exception("model resolve failed for %s", lg)
            notes.append(f"{lg}:model_error")
            meta["model_note"] = "model_error"
            continue
        scan = scan_top_value_bets(
            fx,
            dc,
            ml,
            w_ml=float(w_ml) if ml is not None else 0.0,
            min_ev=float(min_ev),
            kelly_fraction=kf,
            max_stake_pct=MAX_STAKE_PCT,
            allowed_markets=goal_markets,
            top_n=int(max(1, limit)),
            bankroll=float(bankroll),
            league=str(lg),
            filter_vn_window=True,
            one_per_match=bool(one_per_match),
        )
        meta["excluded_prior"] = int(scan.excluded_prior)
        results.append(scan)

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
            "stale": bool(stale_leagues),
            "stale_leagues": stale_leagues,
            "source": ",".join(source_bits) or "empty",
            "generated_at": generated_at,
            "kelly_fraction": kf,
            "leagues": list(scan_leagues),
            "league_meta": league_meta,
            "excluded_prior": sum(int(m["excluded_prior"]) for m in league_meta),
            "notes": (
                "No cached fixtures and cold network/model path failed or empty. "
                "Refresh fixtures via Streamlit or auto_scanner first."
            ),
        }

    if len(results) == 1:
        merged = results[0]
    else:
        merged = merge_top_scan_results(
            results,
            top_n=int(max(1, limit)),
            min_ev=float(min_ev),
            one_per_match=bool(one_per_match),
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
    excluded_total = sum(int(m["excluded_prior"]) for m in league_meta)
    if excluded_total:
        note_txt = (
            (note_txt + "; " if note_txt else "")
            + f"Excluded {excluded_total} legs priced on new-team priors"
        )
    if stale_leagues:
        note_txt = (
            (note_txt + "; " if note_txt else "")
            + "STALE odds (cache older than FIXTURES_MAX_AGE_MINUTES, refresh failed): "
            + ",".join(stale_leagues)
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
        "stale": bool(stale_leagues),
        "stale_leagues": stale_leagues,
        "source": ",".join(source_bits),
        "generated_at": generated_at,
        "kelly_fraction": kf,
        "leagues": list(scan_leagues),
        "league_meta": league_meta,
        "excluded_prior": excluded_total,
        "notes": note_txt,
    }


# ---------------------------------------------------------------------------
# Fixture schedule (every match in the VN window, with or without a pick)
# ---------------------------------------------------------------------------

FIXTURE_STATUS_VALUE = "value"  # best leg EV >= threshold
FIXTURE_STATUS_BELOW = "below"  # priced, best leg below threshold
FIXTURE_STATUS_GATED = "gated"  # new-team prior gate — not priced
FIXTURE_STATUS_NO_ODDS = "no_odds"
FIXTURE_STATUS_NO_MODEL = "no_model"


def _fx_float(row: pd.Series, *cols: str) -> float | None:
    for c in cols:
        if c in row.index:
            v = _safe_float(row.get(c))
            if v is not None:
                return v
    return None


def list_fixtures_api(
    *,
    leagues: str | Sequence[str] | None = None,
    min_ev_pct: float = 5.0,
    markets: str | Sequence[str] | None = None,
    kelly_fraction: float | None = None,
) -> dict[str, Any]:
    """Every upcoming fixture in the VN today/tomorrow window, with a status.

    Unlike :func:`scan_value_bets_api` (picks only), this lists matches that
    have no pick too, so the UI can show *why*: ``value`` / ``below`` (best
    leg under the EV floor) / ``gated`` (new-team prior, see
    :func:`src.scanner.new_team_prior_gate`) / ``no_odds`` / ``no_model``.
    """
    from src.scanner import _match_id_from_row, new_team_prior_gate
    from src.timezone_utils import filter_matches_today_tomorrow

    generated_at = _utc_now().isoformat()
    mkt_list = parse_markets(markets)
    goal_markets = tuple(m for m in mkt_list if m != "Corners") or ("1X2",)
    min_ev = float(min_ev_pct) / 100.0
    if min_ev <= 0:
        min_ev = float(DEFAULT_MIN_EV)
    kf = float(DEFAULT_KELLY_FRACTION) if kelly_fraction is None else float(kelly_fraction)
    kf = max(0.0, min(kf, float(MAX_KELLY_FRACTION)))
    scan_leagues = parse_leagues(leagues) or list(DEFAULT_SCAN_LEAGUES)

    fixtures_out: list[dict[str, Any]] = []
    league_meta: list[dict[str, Any]] = []

    for lg in scan_leagues:
        fx_meta = load_fixtures_with_meta(lg)
        fx = fx_meta["df"]
        if fx is not None and not fx.empty and "Kickoff" in fx.columns:
            fx = filter_matches_today_tomorrow(fx, kickoff_col="Kickoff", exclude_started=True)
        meta: dict[str, Any] = {
            "league": str(lg),
            "label": _league_label_safe(lg),
            "fixtures_source": str(fx_meta["source"]),
            "stale": bool(fx_meta.get("stale")),
            "odds_age_minutes": _safe_float(fx_meta.get("age_minutes")),
            "fixtures": 0 if fx is None else int(len(fx)),
            "model_note": None,
            "ml_enabled": False,
            "model_source": None,
            "excluded_prior": 0,
        }
        league_meta.append(meta)
        if fx is None or fx.empty:
            continue

        dc = ml = None
        w_ml = 0.0
        try:
            dc, ml, w_ml, model_note = _resolve_models(lg)
            meta["model_note"] = model_note
            meta.update(_model_flags(model_note))
        except Exception:  # noqa: BLE001
            logger.exception("model resolve failed for %s", lg)
            meta["model_note"] = "model_error"

        best: dict[str, dict[str, Any]] = {}
        if dc is not None:
            try:
                scan = scan_top_value_bets(
                    fx,
                    dc,
                    ml,
                    w_ml=float(w_ml) if ml is not None else 0.0,
                    min_ev=float(min_ev),
                    kelly_fraction=kf,
                    max_stake_pct=MAX_STAKE_PCT,
                    allowed_markets=goal_markets,
                    top_n=max(1, len(fx)),
                    league=str(lg),
                    filter_vn_window=True,
                    one_per_match=True,
                )
                meta["excluded_prior"] = int(scan.excluded_prior)
                pool = scan.fallback_bets
                if pool is not None and not pool.empty and "match_id" in pool.columns:
                    for _, r in pool.iterrows():
                        if r.get("ev") is None or pd.isna(r.get("ev")):
                            continue
                        best.setdefault(str(r["match_id"]), _row_to_bet(r))
            except Exception:  # noqa: BLE001
                logger.exception("fixture scan failed for %s", lg)

        for _, row in fx.iterrows():
            home = str(row.get("HomeTeam") or "")
            away = str(row.get("AwayTeam") or "")
            mid = _match_id_from_row(row, home, away)
            odds_h = _fx_float(row, "OddsH", "B365H")
            odds_d = _fx_float(row, "OddsD", "B365D")
            odds_a = _fx_float(row, "OddsA", "B365A")
            pick = best.get(mid)
            gate = new_team_prior_gate(dc, home, away, league=lg) if dc is not None else None
            if dc is None:
                status = FIXTURE_STATUS_NO_MODEL
            elif gate is not None:
                status = FIXTURE_STATUS_GATED
            elif pick is not None:
                ev = pick.get("ev")
                status = (
                    FIXTURE_STATUS_VALUE
                    if ev is not None and ev >= min_ev
                    else FIXTURE_STATUS_BELOW
                )
            elif odds_h is None and _fx_float(row, "OddsOver", "B365AHH") is None:
                status = FIXTURE_STATUS_NO_ODDS
            else:
                status = FIXTURE_STATUS_BELOW
            fixtures_out.append(
                {
                    "match_id": mid,
                    "league": str(lg),
                    "league_label": meta["label"],
                    "kickoff": kickoff_iso_utc(row.get("Kickoff")),
                    "home": home,
                    "away": away,
                    "home_display": _display_name(home),
                    "away_display": _display_name(away),
                    "odds_home": odds_h,
                    "odds_draw": odds_d,
                    "odds_away": odds_a,
                    "bookmaker": _safe_str(row.get("OddsBookmaker")),
                    "ou_line": _fx_float(row, "OU_Line"),
                    "ah_line": _fx_float(row, "AHh"),
                    "status": status,
                    "gate_reason": gate,
                    "best_pick": pick if status in (FIXTURE_STATUS_VALUE, FIXTURE_STATUS_BELOW) else None,
                }
            )

    fixtures_out.sort(key=lambda f: (f["kickoff"] or "", f["league"], f["home"]))
    return {
        "fixtures": fixtures_out,
        "count": len(fixtures_out),
        "min_ev_pct": float(min_ev_pct),
        "leagues": list(scan_leagues),
        "league_meta": league_meta,
        "generated_at": generated_at,
    }
