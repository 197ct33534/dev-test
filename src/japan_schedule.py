"""Japan cup schedule helpers (J.League Cup + Emperor's Cup).

Loads upcoming fixtures from ``global_matches.db``, builds App detail /
Flashscore links, and groups rows by Vietnam calendar date for UI views.

Also evaluates per-match value picks (Dixon–Coles 1X2 / OU / AH) for the
tổng hợp ``Kèo gợi ý`` column — all legs with EV ≥ threshold, else the
single best model pick with EV noted.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Mapping, Sequence

import pandas as pd

from src.config import DEFAULT_KELLY_FRACTION, DEFAULT_MIN_EV
from src.global_db import GLOBAL_DB_PATH, load_upcoming_from_db
from src.share import match_id_for_fixture
from src.timezone_utils import kickoff_to_vn

JP_COMP_IDS: tuple[str, ...] = ("J_LEAGUE_CUP", "EMPERORS_CUP")

_COMP_SHORT: dict[str, str] = {
    "J_LEAGUE_CUP": "J.League Cup",
    "EMPERORS_CUP": "Emperor's Cup",
}

FLASHSCORE_BASE = "https://www.flashscore.com"

_MARKET_BADGE: dict[str, str] = {
    "1X2": "[1X2]",
    "OU": "[Tài/Xỉu]",
    "AH": "[Kèo Chấp AH]",
    "Corners": "[Phạt Góc]",
}


def _odds_cell(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f <= 1.0:
        return None
    return f


def format_1x2_odds(row: Mapping[str, Any] | pd.Series) -> str:
    """Format ``H / D / A`` odds, or empty when any leg is missing."""
    h = _odds_cell(row.get("B365H") if hasattr(row, "get") else None)
    d = _odds_cell(row.get("B365D") if hasattr(row, "get") else None)
    a = _odds_cell(row.get("B365A") if hasattr(row, "get") else None)
    if h is None or d is None or a is None:
        return ""
    return f"{h:.2f} / {d:.2f} / {a:.2f}"


def _team_display_map() -> dict[str, str]:
    """JP_* / ES_* → human name (best-effort, cached via registry)."""
    try:
        from src.global_db import _registry_display_map

        return dict(_registry_display_map())
    except Exception:  # noqa: BLE001
        return {}


def display_team_name(raw: str, *, name_map: Mapping[str, str] | None = None) -> str:
    """Prefer a friendly club name over stable ``JP_*`` codes."""
    text = str(raw or "").strip()
    if not text:
        return text
    mapping = name_map if name_map is not None else _team_display_map()
    hit = mapping.get(text.upper()) or mapping.get(text)
    if hit:
        return str(hit)
    if "_" in text and text[:3].isupper():
        parts = text.split("_", 1)
        body = parts[1] if len(parts) == 2 and len(parts[0]) <= 3 else text
        return " ".join(p.capitalize() for p in body.split("_") if p)
    return text


def build_flashscore_match_url(row: Mapping[str, Any] | pd.Series) -> str:
    """Flashscore match URL from mid + optional home/away slug-hash tokens.

    Prefers ``…/match/football/{home}-{hash}/{away}-{hash}/?mid=…`` when
    slugs/hashes are stored; otherwise ``…/match/{mid}/``.
    """
    mid = ""
    if hasattr(row, "get"):
        mid = str(row.get("FlashscoreEventId") or "").strip()
    if not mid or mid.lower() in {"nan", "none"}:
        return FLASHSCORE_BASE

    h_slug = str(row.get("HomeFlashscoreSlug") or "").strip().strip("/")
    a_slug = str(row.get("AwayFlashscoreSlug") or "").strip().strip("/")
    h_hash = str(row.get("HomeFlashscoreHash") or "").strip()
    a_hash = str(row.get("AwayFlashscoreHash") or "").strip()

    if h_slug and a_slug and h_hash and a_hash:
        return (
            f"{FLASHSCORE_BASE}/match/football/"
            f"{h_slug}-{h_hash}/{a_slug}-{a_hash}/?mid={mid}"
        )
    return f"{FLASHSCORE_BASE}/match/{mid}/"


def build_app_detail_url(match_id: str, *, base_url: str | None = None) -> str:
    """Streamlit deep-link ``/?match_id=…`` (absolute when ``base_url`` given)."""
    from urllib.parse import quote

    q = f"match_id={quote(str(match_id), safe='')}"
    base = (base_url or "").strip().rstrip("/")
    if not base:
        return f"/?{q}"
    return f"{base}/?{q}"

def load_japan_upcoming(
    *,
    db_path: Any = GLOBAL_DB_PATH,
    comp_ids: Sequence[str] = JP_COMP_IDS,
) -> pd.DataFrame:
    """Combine cached upcoming fixtures for Japan cup competitions.

    Adds ``comp_id`` when missing and sorts by kickoff ascending.
    """
    frames: list[pd.DataFrame] = []
    for code in comp_ids:
        df, _ts = load_upcoming_from_db(str(code), db_path=db_path)
        if df is None or df.empty:
            continue
        out = df.copy()
        if "comp_id" not in out.columns and "league_id" in out.columns:
            out["comp_id"] = out["league_id"].map(
                lambda x: str(x or code).strip().upper()
            )
        elif "comp_id" not in out.columns:
            out["comp_id"] = str(code).upper()
        else:
            out["comp_id"] = out["comp_id"].fillna(code).map(
                lambda x: str(x or code).strip().upper()
            )
        frames.append(out)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    if "Kickoff" in combined.columns:
        combined = combined.sort_values("Kickoff", kind="mergesort").reset_index(
            drop=True
        )
    return combined


def load_japan_upcoming_frames(
    *,
    db_path: Any = GLOBAL_DB_PATH,
    comp_ids: Sequence[str] = JP_COMP_IDS,
) -> list[tuple[str, pd.DataFrame]]:
    """Per-competition frames for deep-link search: ``[(comp_id, df), …]``."""
    out: list[tuple[str, pd.DataFrame]] = []
    for code in comp_ids:
        df, _ts = load_upcoming_from_db(str(code), db_path=db_path)
        if df is None or df.empty:
            continue
        out.append((str(code).upper(), df))
    return out


def _vn_date(kickoff: Any) -> date | None:
    vn = kickoff_to_vn(kickoff)
    if pd.isna(vn):
        return None
    return vn.date()


def date_group_label(d: date, *, today: date | None = None) -> str:
    """Human VN date bucket label (tonight / weekday + day month)."""
    if today is not None:
        ref = today
    else:
        from datetime import datetime, timezone

        ref = kickoff_to_vn(datetime.now(timezone.utc)).date()
    if d == ref:
        return f"Đêm nay ({d.day}/{d.month})"
    weekdays = (
        "Thứ Hai",
        "Thứ Ba",
        "Thứ Tư",
        "Thứ Năm",
        "Thứ Sáu",
        "Thứ Bảy",
        "Chủ Nhật",
    )
    return f"{d.day}/{d.month} · {weekdays[d.weekday()]}"


def japan_match_records(
    df: pd.DataFrame | None = None,
    *,
    db_path: Any = GLOBAL_DB_PATH,
    base_url: str | None = None,
) -> list[dict[str, Any]]:
    """Normalize upcoming Japan rows for Streamlit / Canvas tables."""
    if df is None:
        df = load_japan_upcoming(db_path=db_path)
    if df is None or df.empty:
        return []

    name_map = _team_display_map()
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        home_raw = str(row.get("HomeTeam") or "")
        away_raw = str(row.get("AwayTeam") or "")
        mid = match_id_for_fixture(row, home_raw, away_raw)
        ko = row.get("Kickoff")
        vn_d = _vn_date(ko)
        comp = str(row.get("comp_id") or row.get("league_id") or "").upper()
        records.append(
            {
                "match_id": mid,
                "home": display_team_name(home_raw, name_map=name_map),
                "away": display_team_name(away_raw, name_map=name_map),
                "home_raw": home_raw,
                "away_raw": away_raw,
                "match": (
                    f"{display_team_name(home_raw, name_map=name_map)} vs "
                    f"{display_team_name(away_raw, name_map=name_map)}"
                ),
                "kickoff": ko,
                "kickoff_vn": (
                    kickoff_to_vn(ko).strftime("%H:%M")
                    if not pd.isna(kickoff_to_vn(ko))
                    else ""
                ),
                "kickoff_vn_full": (
                    kickoff_to_vn(ko).strftime("%d/%m %H:%M")
                    if not pd.isna(kickoff_to_vn(ko))
                    else ""
                ),
                "vn_date": vn_d.isoformat() if vn_d else "",
                "vn_date_label": date_group_label(vn_d) if vn_d else "Khác",
                "comp_id": comp,
                "comp_label": _COMP_SHORT.get(comp, comp or "Japan"),
                "odds": format_1x2_odds(row),
                "detail_url": build_app_detail_url(mid, base_url=base_url),
                "flashscore_url": build_flashscore_match_url(row),
            }
        )
    return records


def group_japan_matches(
    records: Sequence[Mapping[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Group normalized records by ``vn_date_label``, preserving kickoff order."""
    groups: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for rec in records:
        label = str(rec.get("vn_date_label") or "Khác")
        if label not in groups:
            groups[label] = []
            order.append(label)
        groups[label].append(dict(rec))
    return [(label, groups[label]) for label in order]


# ---------------------------------------------------------------------------
# Per-match value picks (Kèo gợi ý)
# ---------------------------------------------------------------------------


def _selection_label(text: str) -> str:
    """Compact selection label (keep Over/Under / AH English tokens)."""
    raw = str(text or "").strip()
    if raw == "Home":
        return "Home"
    if raw == "Draw":
        return "Draw"
    if raw == "Away":
        return "Away"
    return raw


def format_pick_chip(pick: Mapping[str, Any]) -> str:
    """One-line chip: ``[1X2] Home @1.85 EV+8%``."""
    mkt = str(pick.get("market") or "").strip().upper()
    badge = _MARKET_BADGE.get(mkt, f"[{mkt or '?'}]")
    sel = _selection_label(str(pick.get("selection") or ""))
    try:
        odds = float(pick.get("bookmaker_odds") or pick.get("odds") or 0.0)
    except (TypeError, ValueError):
        odds = 0.0
    try:
        ev_pct = float(pick.get("ev_pct"))
        if pick.get("ev_pct") is None and pick.get("ev") is not None:
            ev_pct = float(pick["ev"]) * 100.0
    except (TypeError, ValueError):
        ev_pct = float("nan")
    odds_s = f"@{odds:.2f}" if odds > 1.0 else "@—"
    if ev_pct == ev_pct:  # not NaN
        return f"{badge} {sel} {odds_s} EV{ev_pct:+.0f}%"
    return f"{badge} {sel} {odds_s}"


def format_picks_column(
    picks: Sequence[Mapping[str, Any]],
    *,
    below_threshold: bool = False,
    odds_missing: bool = False,
) -> str:
    """Multiline ``Kèo gợi ý`` cell text."""
    if odds_missing:
        return "— (chưa có odds)"
    if not picks:
        return "—"
    lines = [format_pick_chip(p) for p in picks]
    if below_threshold and lines:
        lines[0] = lines[0] + " · best"
    return "\n".join(lines)


def extract_fixture_markets(
    row: Mapping[str, Any] | pd.Series,
    *,
    odds_family: str = "B365",
) -> tuple[
    dict[str, float] | None,
    dict[str, float] | None,
    dict[str, float] | None,
]:
    """Pull 1X2 / OU / AH odds dicts from an upcoming fixture row."""
    from src.recommender import _fixture_side_markets, _resolve_odds_columns

    series = row if isinstance(row, pd.Series) else pd.Series(dict(row))
    col_map = _resolve_odds_columns(odds_family)
    odds_1x2: dict[str, float] | None = None
    if all(c in series.index and pd.notna(series[c]) for c in col_map.values()):
        try:
            odds_1x2 = {k: float(series[col_map[k]]) for k in ("H", "D", "A")}
            if any(v <= 1.0 for v in odds_1x2.values()):
                odds_1x2 = None
        except (TypeError, ValueError):
            odds_1x2 = None
    ou, ah = _fixture_side_markets(series, odds_family)
    ou_out = dict(ou) if ou is not None else None
    ah_out = dict(ah) if ah is not None else None
    return odds_1x2, ou_out, ah_out


def select_picks_for_match(
    bets: pd.DataFrame | None,
    *,
    min_ev: float = DEFAULT_MIN_EV,
) -> tuple[list[dict[str, Any]], bool]:
    """Keep all value legs (multi-market); else fall back to best EV pick.

    Returns
    -------
    picks, below_threshold
        ``below_threshold`` is True when no leg met ``min_ev`` (after sanity)
        and the best model pick is shown instead.
    """
    from src.strategy import apply_sanity_filters, select_value_bets

    if bets is None or (isinstance(bets, pd.DataFrame) and bets.empty):
        return [], False

    frame = bets.copy()
    if "ev" not in frame.columns and "ev_pct" in frame.columns:
        frame["ev"] = frame["ev_pct"].astype(float) / 100.0
    if "ev_pct" not in frame.columns and "ev" in frame.columns:
        frame["ev_pct"] = frame["ev"].astype(float) * 100.0

    value = frame.loc[frame["ev"].astype(float) >= float(min_ev)].copy()
    if not value.empty:
        filtered = select_value_bets(
            value,
            max_per_day=max(20, len(value)),
            already_today=0,
            allow_multi_picks_per_match=True,
            apply_sanity=True,
        )
        if not filtered.empty:
            return filtered.to_dict(orient="records"), False

    # No sane EV≥threshold: show single best sane model pick (may be < min_ev).
    sane = apply_sanity_filters(frame)
    pool = sane if not sane.empty else frame
    best = pool.sort_values("ev", ascending=False).head(1)
    below = True
    if not best.empty and float(best.iloc[0]["ev"]) >= float(min_ev):
        below = False
    return best.to_dict(orient="records"), below


def evaluate_japan_fixture_picks(
    model: Any,
    row: Mapping[str, Any] | pd.Series,
    *,
    min_ev: float = DEFAULT_MIN_EV,
    kelly_fraction: float = DEFAULT_KELLY_FRACTION,
    include_ou: bool = True,
    include_ah: bool = True,
    odds_family: str = "B365",
) -> dict[str, Any]:
    """Evaluate all scannable markets for one Japan cup fixture.

    Returns a dict with ``picks``, ``text``, ``below_threshold``,
    ``odds_missing``, and ``n_value``.
    """
    from src.recommender import ValueBetRecommender

    series = row if isinstance(row, pd.Series) else pd.Series(dict(row))
    home = str(series.get("HomeTeam") or "")
    away = str(series.get("AwayTeam") or "")
    odds_1x2, ou, ah = extract_fixture_markets(series, odds_family=odds_family)

    empty = {
        "picks": [],
        "text": "— (chưa có odds)",
        "below_threshold": False,
        "odds_missing": True,
        "n_value": 0,
        "home_team": home,
        "away_team": away,
    }
    if odds_1x2 is None and ou is None and ah is None:
        return empty

    rec = ValueBetRecommender(
        model,
        min_ev=float(min_ev),
        kelly_fraction=float(kelly_fraction),
        allowed_markets=("1X2", "OU", "AH"),
    )
    match_date = series.get("Kickoff") if "Kickoff" in series.index else series.get("Date")
    try:
        bets = rec.evaluate_match(
            home,
            away,
            odds_1x2=odds_1x2,
            over_under=ou if include_ou else None,
            asian_handicap=ah if include_ah else None,
            only_value=False,
            odds_source=odds_family,
            match_date=match_date,
        )
    except Exception:  # noqa: BLE001
        return {
            **empty,
            "text": "— (lỗi model)",
            "odds_missing": False,
        }

    if not bets:
        return {
            **empty,
            "text": "—",
            "odds_missing": False,
        }

    bets_df = pd.DataFrame([b.to_dict() for b in bets])
    picks, below = select_picks_for_match(bets_df, min_ev=float(min_ev))
    n_value = sum(1 for p in picks if float(p.get("ev") or 0.0) >= float(min_ev))
    text = format_picks_column(
        picks, below_threshold=below, odds_missing=False
    )
    return {
        "picks": picks,
        "text": text,
        "below_threshold": below,
        "odds_missing": False,
        "n_value": n_value,
        "home_team": home,
        "away_team": away,
    }


def fixture_row_by_match_id(
    df: pd.DataFrame,
) -> dict[str, pd.Series]:
    """Index upcoming Japan rows by ``match_id`` (Flashscore mid / slug)."""
    out: dict[str, pd.Series] = {}
    if df is None or df.empty:
        return out
    for _, row in df.iterrows():
        home = str(row.get("HomeTeam") or "")
        away = str(row.get("AwayTeam") or "")
        mid = match_id_for_fixture(row, home, away)
        if mid:
            out[str(mid)] = row
    return out


def attach_picks_to_records(
    records: Sequence[Mapping[str, Any]],
    fixtures_df: pd.DataFrame,
    models_by_comp: Mapping[str, Any],
    *,
    min_ev: float = DEFAULT_MIN_EV,
    kelly_fraction: float = DEFAULT_KELLY_FRACTION,
    cache: dict[str, dict[str, Any]] | None = None,
    include_ou: bool = True,
    include_ah: bool = True,
) -> list[dict[str, Any]]:
    """Copy records and fill ``picks_text`` / ``picks`` from evaluate + cache.

    ``cache`` is keyed by ``match_id`` (mutated in place when provided) so the
    Streamlit UI can store results in ``st.session_state`` across reruns.
    """
    index = fixture_row_by_match_id(fixtures_df)
    store = cache if cache is not None else {}
    out: list[dict[str, Any]] = []
    for rec in records:
        item = dict(rec)
        mid = str(item.get("match_id") or "")
        if mid and mid in store:
            cached = store[mid]
            item["picks"] = list(cached.get("picks") or [])
            item["picks_text"] = str(cached.get("text") or "—")
            item["picks_below_threshold"] = bool(cached.get("below_threshold"))
            item["picks_odds_missing"] = bool(cached.get("odds_missing"))
            out.append(item)
            continue

        row = index.get(mid)
        comp = str(item.get("comp_id") or "").upper()
        model = models_by_comp.get(comp) or models_by_comp.get("EMPERORS_CUP")
        if row is None or model is None:
            payload = {
                "picks": [],
                "text": "— (chưa model)" if model is None else "—",
                "below_threshold": False,
                "odds_missing": row is None,
                "n_value": 0,
            }
        else:
            payload = evaluate_japan_fixture_picks(
                model,
                row,
                min_ev=float(min_ev),
                kelly_fraction=float(kelly_fraction),
                include_ou=include_ou,
                include_ah=include_ah,
            )
        store[mid] = payload
        item["picks"] = list(payload.get("picks") or [])
        item["picks_text"] = str(payload.get("text") or "—")
        item["picks_below_threshold"] = bool(payload.get("below_threshold"))
        item["picks_odds_missing"] = bool(payload.get("odds_missing"))
        out.append(item)
    return out
