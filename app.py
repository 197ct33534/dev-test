"""EPL Value Betting — Streamlit dashboard.

Kết nối Dixon–Coles + LightGBM + Ensemble 1X2 + ValueBetRecommender + CornerModel.
Odds trận sắp tới lấy từ Flashscore (ưu tiên bet365).
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import threading
from typing import Any, Mapping, Sequence

# Windows + asyncio Proactor: remote host đóng socket (Flashscore/Streamlit)
# hay ném ConnectionResetError trong callback — nhiễu, không làm sập app.
if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except AttributeError:
        pass

    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport

        _orig_call_connection_lost = _ProactorBasePipeTransport._call_connection_lost

        def _call_connection_lost(self, exc):  # type: ignore[no-untyped-def]
            try:
                _orig_call_connection_lost(self, exc)
            except (ConnectionResetError, ConnectionAbortedError, OSError):
                pass

        _ProactorBasePipeTransport._call_connection_lost = _call_connection_lost  # type: ignore[method-assign]
    except Exception:  # noqa: BLE001
        pass

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src.corner_model import CornerPredictor
from src.config import (
    MAX_BETS_PER_DAY,
    DEFAULT_KELLY_FRACTION,
    DEFAULT_MIN_EV,
    DEFAULT_W_ML,
    MAX_KELLY_FRACTION,
    MAX_STAKE_PCT,
)
from src.data_loader import (
    get_available_leagues,
    get_league_config,
    get_team_profile_data,
    league_db_path,
    league_label,
    list_teams,
    load_league_data,
    load_upcoming_fixtures,
    normalize_league,
)
from src.dixon_coles import (
    DixonColesModel,
    MIN_TEAM_MATCHES,
    WEAK_TIER_NEW_TEAM_ATTACK,
    WEAK_TIER_NEW_TEAM_DEFENCE,
)
from src.models import (
    ensemble_weight_for_sample,
    maybe_wrap_with_league_weights,
    resolve_fixture_teams,
    teams_needing_data_badge,
)

from src.journal import (
    add_to_journal,
    delete_live_bet,
    journal_bankroll_summary,
    journal_summary_from_frame,
    load_live_bets,
    load_live_bets_all_leagues,
    settle_live_bet,
    sync_closing_odds_from_results,
    update_closing_odds,
)
from src.ml_model import EPLMachineLearningModel
from src.backtester import (
    BacktestConfig,
    format_backtest_report,
    market_breakdown,
    run_ablation,
    run_backtest,
)
from src.notifier import send_telegram_message, send_telegram_value_bets
from src.recommender import (
    ODDS_COLUMN_MAPS,
    ValueBetRecommender,
    format_recommendations,
    implied_probability,
    recommend_fixtures,
    recommend_upcoming,
)
from src.scanner import merge_top_scan_results, scan_top_value_bets
from src.features import attach_rest_days_from_db
from src.global_db import (
    GLOBAL_DB_PATH,
    ensure_global_db_from_legacy,
    global_db_has_matches,
    load_upcoming_from_db,
    read_matches_as_legacy,
    save_upcoming_to_db,
    upcoming_cache_age_minutes,
)
from src.share import (
    build_lite_share_text,
    build_share_text,
    build_share_url,
    find_fixture_for_share,
    match_id_for_fixture,
    parse_match_query,
    pick_top_bet_for_share,
)
from src.pipeline import sync_to_global_db
from src.strategy import (
    calculate_kelly_stake,
    capped_kelly_fraction,
    generate_match_insights,
    select_value_bets,
)
from src.timezone_utils import format_kickoff_vn
from src.japan_schedule import (
    JP_COMP_IDS,
    attach_picks_to_records,
    group_japan_matches,
    japan_match_records,
    load_japan_upcoming,
    load_japan_upcoming_frames,
)

# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="EPL Value Betting",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Responsive CSS (mobile-first) — inject once at startup
# ---------------------------------------------------------------------------

_RESPONSIVE_CSS = """
<style>
/* ----- Base: avoid page-level horizontal overflow ----- */
html, body, [data-testid="stAppViewContainer"] {
  overflow-x: hidden !important;
  max-width: 100vw !important;
}
.block-container {
  max-width: 100% !important;
  padding-top: 1.25rem !important;
  padding-bottom: 2rem !important;
  padding-left: 1.5rem !important;
  padding-right: 1.5rem !important;
}

/* Tabs: allow horizontal swipe/scroll instead of crushing labels */
[data-testid="stTabs"] [role="tablist"] {
  flex-wrap: nowrap !important;
  overflow-x: auto !important;
  overflow-y: hidden !important;
  -webkit-overflow-scrolling: touch;
  gap: 0.15rem;
  scrollbar-width: thin;
}
[data-testid="stTabs"] [role="tab"] {
  white-space: nowrap !important;
  flex-shrink: 0 !important;
}

/* Dataframes / tables: scroll locally, never blow out the page */
[data-testid="stDataFrame"],
[data-testid="stDataFrameResizable"],
[data-testid="stTable"],
div[data-testid="stElementContainer"]:has([data-testid="stDataFrame"]) {
  max-width: 100% !important;
  overflow-x: auto !important;
  -webkit-overflow-scrolling: touch;
}
[data-testid="stDataFrame"] table,
[data-testid="stTable"] table {
  min-width: 28rem;
}

/* Plotly: shrink to parent */
.js-plotly-plot, .plotly, .stPlotlyChart {
  max-width: 100% !important;
  overflow: hidden !important;
}

/* Buttons / inputs fill column width on touch devices */
div.stButton > button,
div.stDownloadButton > button {
  width: 100%;
}

/* Metric cards: prevent label wrap collapse */
[data-testid="stMetric"] {
  min-width: 0 !important;
}
[data-testid="stMetricLabel"] p {
  white-space: normal !important;
  overflow-wrap: anywhere;
  word-break: break-word;
}
[data-testid="stMetricValue"] {
  overflow-wrap: anywhere;
  word-break: break-word;
}

/* Expanders / alert boxes */
[data-testid="stAlert"], .stAlert {
  overflow-wrap: anywhere;
  word-break: break-word;
}

/* ----- Tablet (≤992px): 2-up columns instead of 4–6 squeezed ----- */
@media (max-width: 992px) {
  .block-container {
    padding-left: 1rem !important;
    padding-right: 1rem !important;
  }
  div[data-testid="stHorizontalBlock"] {
    flex-wrap: wrap !important;
    gap: 0.5rem 0.75rem !important;
  }
  div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
    min-width: min(100%, 16rem) !important;
    flex: 1 1 calc(50% - 0.75rem) !important;
  }
}

/* ----- Phone (≤768px): full-width stack ----- */
@media (max-width: 768px) {
  .block-container {
    padding-left: 0.7rem !important;
    padding-right: 0.7rem !important;
    padding-top: 0.75rem !important;
  }
  /* Force every column row to stack vertically */
  div[data-testid="stHorizontalBlock"] {
    flex-direction: column !important;
    flex-wrap: nowrap !important;
    gap: 0.35rem !important;
  }
  div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
    width: 100% !important;
    min-width: 100% !important;
    max-width: 100% !important;
    flex: 1 1 100% !important;
  }
  /* Slightly tighter headings */
  h1 { font-size: 1.55rem !important; }
  h2 { font-size: 1.3rem !important; }
  h3 { font-size: 1.1rem !important; }
  /* Sidebar: denser controls when opened */
  [data-testid="stSidebar"] .block-container {
    padding: 0.75rem 0.6rem !important;
  }
  /* Code blocks / long captions */
  code, pre, [data-testid="stCaption"] {
    overflow-x: auto !important;
    word-break: break-word;
  }
}

/* ----- Very small phones ----- */
@media (max-width: 400px) {
  .block-container {
    padding-left: 0.45rem !important;
    padding-right: 0.45rem !important;
  }
  [data-testid="stMetricValue"] {
    font-size: 1.15rem !important;
  }
}

/* Top-20 value bet cards */
.vb-card-meta {
  font-size: 0.85rem;
  opacity: 0.85;
  margin: 0.15rem 0 0.35rem 0;
}
.vb-card-ev-pos { color: #1b7a3d; font-weight: 600; }
.vb-card-ev-neg { color: #a33b2b; font-weight: 600; }
.vb-thin-badge {
  color: #8a6d1d;
  background: #fff3cd;
  border: 1px solid #e6d28a;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 600;
  padding: 0.05rem 0.35rem;
  margin-left: 0.25rem;
  white-space: nowrap;
}
.vb-ready-badge {
  color: #1b5e20;
  background: #e8f5e9;
  border: 1px solid #a5d6a7;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 600;
  padding: 0.05rem 0.35rem;
  margin-left: 0.25rem;
  white-space: nowrap;
}
.vb-result-w {
  color: #1b5e20;
  background: #e8f5e9;
  border-radius: 3px;
  font-weight: 700;
  padding: 0.05rem 0.35rem;
}
.vb-result-d {
  color: #8a6d1d;
  background: #fff8e1;
  border-radius: 3px;
  font-weight: 700;
  padding: 0.05rem 0.35rem;
}
.vb-result-l {
  color: #b71c1c;
  background: #ffebee;
  border-radius: 3px;
  font-weight: 700;
  padding: 0.05rem 0.35rem;
}
.vb-profile-row {
  font-size: 0.88rem;
  margin: 0.2rem 0 0.35rem 0;
  line-height: 1.35;
}
/* Lite Mode compact cards */
.lite-card-title {
  font-size: 1.05rem;
  font-weight: 700;
  margin: 0 0 0.2rem 0;
  line-height: 1.35;
}
.lite-card-sub {
  font-size: 0.85rem;
  opacity: 0.8;
  margin: 0 0 0.25rem 0;
}
.lite-card-fatigue {
  font-size: 0.8rem;
  opacity: 0.75;
  margin: 0 0 0.45rem 0;
  font-style: italic;
}
.lite-card-pick {
  font-size: 1rem;
  font-weight: 600;
  margin: 0.2rem 0;
}
.lite-card-stars {
  font-size: 1.15rem;
  letter-spacing: 0.02em;
  margin: 0.15rem 0 0.35rem 0;
}
.lite-card-insights {
  font-size: 0.8rem;
  opacity: 0.85;
  margin: 0.15rem 0 0.45rem 0;
  line-height: 1.4;
}
.lite-card-insights-title {
  font-weight: 600;
  margin: 0 0 0.15rem 0;
}
</style>
"""

st.markdown(_RESPONSIVE_CSS, unsafe_allow_html=True)


def _metrics_row(
    items: list[tuple],
    *,
    per_row: int = 2,
) -> None:
    """Render metrics in short rows (default 2) so mobile never squeezes 4–6 cols.

    Each item is ``(label, value)`` or ``(label, value, kwargs_dict)``.
    """
    if not items:
        return
    n = max(1, int(per_row))
    for i in range(0, len(items), n):
        chunk = items[i : i + n]
        cols = st.columns(len(chunk))
        for col, item in zip(cols, chunk):
            label, value = item[0], item[1]
            kwargs = item[2] if len(item) > 2 and isinstance(item[2], dict) else {}
            col.metric(label, value, **kwargs)


def _plotly(fig: go.Figure, *, height: int | None = None) -> None:
    """Plotly chart with mobile-safe margins + responsive config."""
    fig.update_layout(
        autosize=True,
        margin=dict(l=8, r=8, t=48, b=32),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11)),
    )
    if height is not None:
        fig.update_layout(height=int(height))
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "responsive": True,
            "displayModeBar": False,
            "scrollZoom": False,
        },
    )


OU_LINE_OPTIONS = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]
AH_LINE_OPTIONS = [
    -2.0,
    -1.75,
    -1.5,
    -1.25,
    -1.0,
    -0.75,
    -0.5,
    -0.25,
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
]

_MARKET_VI = {
    "1X2": "1X2",
    "OU": "Tài/Xỉu",
    "AH": "Chấp Á",
    "Corners": "Phạt góc",
}
_MARKET_BADGE = {
    "1X2": "[1X2]",
    "OU": "[Tài/Xỉu]",
    "AH": "[Kèo Chấp AH]",
    "Corners": "[Phạt Góc Corners]",
}
_TOP20_DISPLAY_N = 20


def _market_badge(market: str) -> str:
    """Compact market badge shown before pick name on Top-N / Lite cards."""
    code = str(market or "").strip()
    return _MARKET_BADGE.get(code, f"[{_MARKET_VI.get(code, code or '?')}]")
_COL_FIXTURE_VI = {
    "Kickoff": "Giờ đá",
    "HomeTeam": "Chủ nhà",
    "AwayTeam": "Khách",
    "Round": "Vòng",
    "B365H": "Odds chủ",
    "B365D": "Odds hòa",
    "B365A": "Odds khách",
    "OU_Line": "Mốc T/X",
    "OddsOver": "Odds Tài",
    "OddsUnder": "Odds Xỉu",
    "AHh": "Mốc chấp",
    "B365AHH": "Odds chấp chủ",
    "B365AHA": "Odds chấp khách",
    "OddsProvider": "Nguồn odds",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _selection_vi(text: str) -> str:
    if text == "Home":
        return "Chủ nhà"
    if text == "Draw":
        return "Hòa"
    if text == "Away":
        return "Khách"
    if text.startswith("Over "):
        return "Tài " + text[5:]
    if text.startswith("Under "):
        return "Xỉu " + text[6:]
    if text.startswith("AH Home "):
        return "Chấp chủ " + text[8:]
    if text.startswith("AH Away "):
        return "Chấp khách " + text[8:]
    return text


def _ev_to_stars(ev_pct: float) -> str:
    """Map EV% to a Lite Mode star badge.

    Thresholds: ``>20`` → 5🔥, ``10–20`` → 4⚡, ``5–10`` → 3👍;
    below 5% (or missing) → 2⭐.
    """
    try:
        v = float(ev_pct)
    except (TypeError, ValueError):
        return "⭐⭐"
    if pd.isna(v):
        return "⭐⭐"
    if v > 20.0:
        return "🔥🔥🔥🔥🔥"
    if v >= 10.0:
        return "⚡⚡⚡⚡"
    if v >= 5.0:
        return "👍👍👍"
    return "⭐⭐"


_VIEW_LITE = "🟢 Tham Khảo (Lite)"
_VIEW_PRO = "⚙️ Chuyên Gia (Pro)"
_NAV_LITE_TOP = "🔥 Top Kèo Hời Hôm Nay"
_NAV_LITE_SEARCH = "🔍 Tìm Trận"


def _open_pro_match_detail(
    *,
    match_id: str,
    home: str,
    away: str,
    kickoff: Any | None = None,
    league_code: str | None = None,
    current_league: str | None = None,
) -> None:
    """Switch to Pro Mode Chi tiết for a fixture (pending flags + rerun)."""
    if (
        league_code
        and current_league
        and normalize_league(league_code) != normalize_league(current_league)
    ):
        try:
            st.session_state["_pending_sidebar_league"] = str(
                get_league_config(league_code)["label"]
            )
        except ValueError:
            pass
    st.session_state["_pending_view_mode"] = _VIEW_PRO
    st.session_state["selected_match"] = match_id
    st.session_state["selected_home"] = home
    st.session_state["selected_away"] = away
    st.session_state["selected_kickoff"] = kickoff
    st.session_state["_apply_selected_match"] = True
    st.session_state["_goto_detail_tab"] = True
    _set_match_query(str(match_id))
    st.rerun()


_FLASHSCORE_MATCH_URL_EXAMPLE = (
    "https://www.flashscore.com/match/football/"
    "gamba-osaka-zLQAGOBK/tokushima-IcwgwCCt/?mid=zg0G0pvA"
)


def _merge_link_injected_fixture(
    fixtures: pd.DataFrame | None,
    league: str,
) -> pd.DataFrame:
    """Prepend a Flashscore match-link row when it belongs to ``league``."""
    inj = st.session_state.get("_link_injected_fixture")
    inj_lg = st.session_state.get("_link_injected_league")
    if not isinstance(inj, dict):
        return fixtures if fixtures is not None else pd.DataFrame()
    if inj_lg and normalize_league(str(inj_lg)) != normalize_league(league):
        return fixtures if fixtures is not None else pd.DataFrame()

    row = pd.DataFrame([inj])
    if fixtures is None or getattr(fixtures, "empty", True):
        return row.reset_index(drop=True)

    out = pd.concat([row, fixtures], ignore_index=True)
    if "FlashscoreEventId" in out.columns:
        out = out.drop_duplicates(subset=["FlashscoreEventId"], keep="first")
    else:
        out = out.drop_duplicates(subset=["HomeTeam", "AwayTeam"], keep="first")
    return out.reset_index(drop=True)


def _render_flashscore_match_link_analyzer(
    *,
    key_prefix: str,
    current_league: str,
) -> None:
    """Paste a Flashscore match URL → fetch odds → open Pro Chi tiết (all EVs)."""
    from src.fetchers.flashscore_league import (
        fetch_and_persist_match_from_url,
        parse_flashscore_match_url,
    )

    st.subheader("Soi Kèo Theo Match Link Flashscore")
    st.text_input(
        "🔗 Dán Link Trận Đấu Flashscore để phân tích tức thì:",
        placeholder=_FLASHSCORE_MATCH_URL_EXAMPLE,
        key=f"{key_prefix}_match_url",
    )
    if not st.button(
        "🔍 Phân Tích Trận Này",
        key=f"{key_prefix}_analyze_btn",
        use_container_width=True,
        type="primary",
    ):
        return

    raw_url = str(st.session_state.get(f"{key_prefix}_match_url") or "").strip()
    try:
        parse_flashscore_match_url(raw_url)
    except ValueError as exc:
        st.error(str(exc))
        return

    with st.spinner("Đang tải trận + odds từ Flashscore…"):
        try:
            result = fetch_and_persist_match_from_url(
                raw_url,
                default_league=current_league,
                timeout_s=45.0,
            )
        except ValueError as exc:
            st.error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).strip() or type(exc).__name__
            st.error(f"Không phân tích được trận: {msg[:180]}")
            return

    mid = str(result.get("match_id") or "")
    home = str(result.get("home") or "")
    away = str(result.get("away") or "")
    if not mid or not home or not away:
        st.error("Flashscore trả về dữ liệu trận không đủ.")
        return

    # Keep the fetched row for Pro Chi tiết even if DB cache lags a tick.
    row = result.get("row")
    if isinstance(row, dict):
        st.session_state["_link_injected_fixture"] = dict(row)
        st.session_state["_link_injected_league"] = str(result.get("league") or "")

    st.success(
        f"Đã nạp {home} vs {away}"
        + (
            f" · {result.get('tournament_name')}"
            if result.get("tournament_name")
            else ""
        )
        + f" · mid=`{mid}` — mở Pro Chi tiết…"
    )
    _open_pro_match_detail(
        match_id=mid,
        home=home,
        away=away,
        kickoff=result.get("kickoff"),
        league_code=str(result.get("league") or current_league),
        current_league=current_league,
    )


def _parse_row_float(row: Mapping[str, Any] | pd.Series, *keys: str) -> float:
    """First parseable float among ``keys``, else NaN."""
    for key in keys:
        if isinstance(row, pd.Series):
            raw = row.get(key) if key in row.index else None
        else:
            raw = row.get(key)
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return float("nan")


def _format_lite_ai_pick(market: str, selection: str, odds_txt: str) -> str:
    """Human-readable Lite AI pick line (1X2 / AH / OU).

    Examples: ``AH · Chấp chủ -0.5 @ 1.93``, ``Tài 2.5 @ 1.85``,
    ``1X2 · Chủ nhà @ 2.10``.
    """
    mkt = str(market or "").strip().upper()
    sel_raw = str(selection or "").strip()
    sel = _selection_vi(sel_raw) if sel_raw and sel_raw != "—" else "—"
    if mkt == "AH":
        # Drop redundant "AH " prefix from English labels after VI map.
        short = sel
        if short.lower().startswith("chấp "):
            return f"AH · {short} @ {odds_txt}"
        if sel_raw.upper().startswith("AH "):
            short = _selection_vi(sel_raw)
        return f"AH · {short} @ {odds_txt}"
    if mkt == "OU":
        # "Tài 2.5 @ 1.85" — selection already encodes Over/Under.
        return f"{sel} @ {odds_txt}"
    if mkt == "1X2":
        return f"1X2 · {sel} @ {odds_txt}"
    label = _MARKET_VI.get(mkt, mkt or "?")
    return f"{label} · {sel} @ {odds_txt}"


def _render_lite_bet_card(
    row: Mapping[str, Any] | pd.Series,
    *,
    idx: int,
    default_league: str,
    key_prefix: str = "lite",
) -> None:
    """Compact mobile-friendly value-bet card for Lite Mode."""
    home = str(row.get("home") or row.get("home_team") or "")
    away = str(row.get("away") or row.get("away_team") or "")
    ko_vn = str(row.get("kickoff_vn") or format_kickoff_vn(row.get("kickoff")))
    market_code = str(row.get("market", "") or "").strip()
    sel_raw = str(row.get("selection", ""))
    odds = _parse_row_float(row, "odds", "bookmaker_odds")
    p_m = _parse_row_float(row, "p_model")
    ev_pct = _parse_row_float(row, "ev_pct")
    if pd.isna(ev_pct):
        ev_frac = _parse_row_float(row, "ev")
        if not pd.isna(ev_frac):
            ev_pct = float(ev_frac) * 100.0
    mid = str(row.get("match_id") or f"{home}__{away}")
    lg = str(row.get("league") or default_league)
    comp_label = str(
        row.get("competition")
        or (league_label(lg) if lg else league_label(default_league))
    )
    fatigue = str(row.get("fatigue_label") or "").strip()
    stars = _ev_to_stars(ev_pct)
    odds_txt = f"{odds:.2f}" if pd.notna(odds) else "n/a"
    pick_line = _format_lite_ai_pick(market_code, sel_raw, odds_txt)
    insight_lines = generate_match_insights(row)
    share_url = build_share_url(match_id=mid, base_url=_share_base_url())
    share_text = build_lite_share_text(
        home=home,
        away=away,
        league=comp_label,
        kickoff_vn=ko_vn,
        selection=pick_line.replace(f" @ {odds_txt}", "").strip()
        if odds_txt != "n/a"
        else pick_line,
        odds=float(odds) if pd.notna(odds) else None,
        ev=(float(ev_pct) / 100.0) if pd.notna(ev_pct) else None,
        share_url=share_url,
        p_model=float(p_m) if pd.notna(p_m) else None,
        fatigue_label=fatigue or None,
        insight_lines=insight_lines or None,
    )

    with st.container(border=True):
        st.markdown(
            f'<p class="lite-card-title">{home} vs {away}</p>',
            unsafe_allow_html=True,
        )
        _render_team_profile_buttons(
            home,
            away,
            key_prefix=f"{key_prefix}_{idx}_{_sanitize_widget_key(mid)}",
        )
        st.markdown(
            f'<p class="lite-card-sub">{comp_label} · {ko_vn}</p>',
            unsafe_allow_html=True,
        )
        _render_manual_team_import_ui(
            home,
            away,
            key_prefix=f"{key_prefix}_{idx}_{_sanitize_widget_key(mid)}",
            thin_names=None,
            kickoff=row.get("kickoff"),
            comp_id=lg,
        )
        if fatigue:
            st.markdown(
                f'<p class="lite-card-fatigue">{fatigue}</p>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<p class="lite-card-pick">AI pick: <strong>{_market_badge(market_code)} '
            f"{pick_line}</strong></p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p class="lite-card-stars">{stars}'
            + (
                f' · <span class="vb-card-ev-pos">EV {ev_pct:+.1f}%</span>'
                if pd.notna(ev_pct)
                else " · EV n/a"
            )
            + "</p>",
            unsafe_allow_html=True,
        )
        model_line = str(row.get("model_fair_line") or "").strip()
        bookie_line = str(row.get("bookie_market_line") or "").strip()
        line_edge = str(row.get("line_edge") or "").strip()
        if model_line or bookie_line or line_edge:
            bits = []
            if model_line:
                bits.append(f"Model {model_line}")
            if bookie_line:
                bits.append(f"NC {bookie_line}")
            if line_edge:
                bits.append(f"lệch {line_edge}")
            st.markdown(
                f'<p class="lite-card-fatigue">{" · ".join(bits)}</p>',
                unsafe_allow_html=True,
            )
        if insight_lines:
            with st.expander("💡 Tại sao AI chọn cửa này?", expanded=False):
                for line in insight_lines:
                    st.caption(line)
        btn_l, btn_r = st.columns(2)
        with btn_l:
            _clipboard_copy_button(
                share_text,
                label="📋 Copy Nhận Định",
                key=f"{key_prefix}_copy_{idx}_{mid}_{lg}",
            )
        with btn_r:
            if st.button(
                "🔍 Xem Chi Tiết",
                key=f"{key_prefix}_pro_{idx}_{mid}_{lg}",
                use_container_width=True,
            ):
                _open_pro_match_detail(
                    match_id=mid,
                    home=home,
                    away=away,
                    kickoff=row.get("kickoff"),
                    league_code=lg,
                    current_league=default_league,
                )


_THIN_DATA_BADGE_HTML = (
    '<span class="vb-thin-badge" title="Thiếu lịch sử trận đấu trong DB">'
    "[⚠️ Thiếu Data Đội]</span>"
)
_THIN_DATA_BADGE_MD = "[⚠️ Thiếu Data Đội]"
_READY_DATA_BADGE_HTML = (
    '<span class="vb-ready-badge" title="Đã có lịch sử trận đấu trong DB">'
    "[✅ Data Đã Sẵn Sàng]</span>"
)
_READY_DATA_BADGE_MD = "[✅ Data Đã Sẵn Sàng]"
_FLASHSCORE_TEAM_URL_EXAMPLE = (
    "https://www.flashscore.com/team/vissel-kobe/698tGI9q/"
)


def _team_label_with_thin_badge(
    name: str,
    *,
    thin: bool = False,
    ready: bool = False,
    html: bool = False,
) -> str:
    """Append missing-data or ready badge beside a club name."""
    base = str(name or "").strip()
    if not base:
        return base
    if thin:
        return f"{base} {_THIN_DATA_BADGE_HTML if html else _THIN_DATA_BADGE_MD}"
    if ready:
        return f"{base} {_READY_DATA_BADGE_HTML if html else _READY_DATA_BADGE_MD}"
    return base


def _sanitize_widget_key(raw: str) -> str:
    """Stable Streamlit key fragment from a team code / name."""
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", str(raw or "").strip())
    return s.strip("_") or "team"


def _result_badge_html(result: str | None) -> str:
    r = str(result or "").strip().upper()
    if r == "W":
        return '<span class="vb-result-w">W</span>'
    if r == "D":
        return '<span class="vb-result-d">D</span>'
    if r == "L":
        return '<span class="vb-result-l">L</span>'
    return "—"


def _format_profile_kickoff(raw: Any) -> str:
    s = str(raw or "").strip()
    if not s:
        return "—"
    try:
        return format_kickoff_vn(s)
    except Exception:  # noqa: BLE001
        return s[:16]


@st.dialog("📊 Hồ Sơ & Lịch Thi Đấu Đội Bóng")
def show_team_profile(team_id: str | int) -> None:
    """DB-only team dossier: last 5 results + next 5 fixtures."""
    data = get_team_profile_data(team_id, limit=5)
    title = str(data.get("display_name") or data.get("canonical_name") or team_id)
    canon = str(data.get("canonical_name") or "")
    if canon and canon != title:
        st.markdown(f"### {title}")
        st.caption(f"Mã: `{canon}`")
    else:
        st.markdown(f"### {title}")

    past = list(data.get("past_matches") or [])
    upcoming = list(data.get("upcoming_matches") or [])
    col_past, col_up = st.columns(2)

    with col_past:
        st.markdown("#### 🕒 5 trận vừa qua")
        if not past:
            st.info("Chưa có trận đã đấu trong DB.")
        else:
            for m in past:
                date_s = str(m.get("date") or "—")
                comp = str(m.get("competition") or m.get("comp_id") or "—")
                home = str(m.get("home") or "—")
                away = str(m.get("away") or "—")
                score = str(m.get("score") or "—")
                badge = _result_badge_html(m.get("result"))
                st.markdown(
                    f'<p class="vb-profile-row">'
                    f"<strong>{date_s}</strong> · {comp}<br/>"
                    f"{home} vs {away} · <strong>{score}</strong> · {badge}"
                    f"</p>",
                    unsafe_allow_html=True,
                )

    with col_up:
        st.markdown("#### 📅 5 trận sắp tới")
        if not upcoming:
            st.info("Chưa có lịch sắp tới trong DB.")
        else:
            for m in upcoming:
                ko = _format_profile_kickoff(m.get("kickoff"))
                comp = str(m.get("competition") or m.get("comp_id") or "—")
                opponent = str(m.get("opponent") or "—")
                side = " (sân nhà)" if m.get("is_home") else " (sân khách)"
                pair = f"{m.get('home')} vs {m.get('away')}"
                st.markdown(
                    f'<p class="vb-profile-row">'
                    f"<strong>{ko}</strong> · {comp}<br/>"
                    f"vs <strong>{opponent}</strong>{side}"
                    f'<br/><span style="opacity:0.7">{pair}</span>'
                    f"</p>",
                    unsafe_allow_html=True,
                )


def _render_team_profile_buttons(
    home: str,
    away: str,
    *,
    key_prefix: str,
) -> None:
    """📊 buttons beside each side — opens :func:`show_team_profile`."""
    c_h, c_a = st.columns(2)
    home_key = f"{key_prefix}_prof_h_{_sanitize_widget_key(home)}"
    away_key = f"{key_prefix}_prof_a_{_sanitize_widget_key(away)}"
    with c_h:
        label_h = str(home or "Chủ").strip() or "Chủ"
        if st.button(
            f"📊 {label_h}",
            key=home_key,
            use_container_width=True,
            help="Hồ sơ & lịch thi đấu đội chủ",
        ):
            show_team_profile(home)
    with c_a:
        label_a = str(away or "Khách").strip() or "Khách"
        if st.button(
            f"📊 {label_a}",
            key=away_key,
            use_container_width=True,
            help="Hồ sơ & lịch thi đấu đội khách",
        ):
            show_team_profile(away)


def _fixture_team_data_statuses(
    home: str,
    away: str,
    *,
    thin_names: Sequence[str] | None = None,
    comp_id: str | None = None,
    kickoff: Any = None,
) -> dict[str, dict[str, Any]]:
    """Per-side DB history + optional Flashscore hash for import UI.

    Missing-data badge uses :func:`has_team_history` only (live DB, never
    stale scanner ``thin_names`` / ``home_thin``). Manual URL expander still
    opens when history is missing **or** hash is absent.
    """
    from src.data_loader import has_team_history
    from src.fetchers.flashscore_team import team_flashscore_data_status

    _ = thin_names, kickoff  # kept for call-site compat; badge ignores them
    out: dict[str, dict[str, Any]] = {}
    for name in (home, away):
        label = str(name or "").strip()
        if not label:
            continue
        status = dict(team_flashscore_data_status(label, comp_id=comp_id))
        has_hist = has_team_history(label, comp_id=comp_id)
        status["has_sufficient"] = has_hist
        status["has_history"] = bool(status.get("has_history")) or has_hist
        status["missing_badge"] = not has_hist
        # Ready badge = history present (hash optional).
        status["ready"] = has_hist
        # Expander: still need import when history missing or hash missing.
        status["needs_import"] = bool(
            not has_hist or not status.get("has_hash")
        )
        out[label] = status
    return out


def _render_manual_team_import_ui(
    home: str,
    away: str,
    *,
    key_prefix: str,
    thin_names: Sequence[str] | None = None,
    kickoff: Any = None,
    comp_id: str | None = None,
    show_ready_badges: bool = True,
    always_show_badges: bool = False,
) -> list[dict[str, Any]]:
    """Badges + expander to paste Flashscore team URLs for missing sides.

    Warning badge only when DB history is insufficient. Expander also appears
    for hash-less teams (rest refresh) even when history exists.

    Returns the list of sides that still need import (empty when all ready).
    """
    from src.fetchers.flashscore_team import import_team_from_flashscore_url

    statuses = _fixture_team_data_statuses(
        home, away, thin_names=thin_names, comp_id=comp_id, kickoff=kickoff
    )
    missing = [s for s in statuses.values() if s.get("needs_import")]
    session_ready = set(st.session_state.get("_flashscore_ready_teams") or [])
    show_badges = bool(
        any(s.get("missing_badge") for s in statuses.values())
        or always_show_badges
        or any(
            str(s.get("team") or "") in session_ready and s.get("ready")
            for s in statuses.values()
        )
    )

    if show_badges and statuses:
        bits: list[str] = []
        for label, st_info in statuses.items():
            missing_hist = bool(st_info.get("missing_badge"))
            is_ready = bool(st_info.get("ready")) and show_ready_badges
            bits.append(
                _team_label_with_thin_badge(
                    label,
                    thin=missing_hist,
                    ready=is_ready and not missing_hist,
                    html=True,
                )
            )
        if bits:
            st.markdown("<p>" + " · ".join(bits) + "</p>", unsafe_allow_html=True)

    if not missing:
        return []

    with st.expander(
        "🔗 Nạp dữ liệu thủ công cho đội bóng này",
        expanded=any(s.get("missing_badge") for s in missing),
    ):
        st.caption(
            "Dán URL trang đội trên Flashscore "
            f"(ví dụ `{_FLASHSCORE_TEAM_URL_EXAMPLE}`)."
        )
        for st_info in missing:
            team_label = str(st_info.get("team") or "").strip()
            tid_key = _sanitize_widget_key(
                str(st_info.get("db_team_id") or team_label)
            )
            # Streamlit key must be unique across cards; include link_{team_id}.
            widget_key = f"{key_prefix}_link_{tid_key}"
            st.text_input(
                f"URL Flashscore — {team_label}",
                key=widget_key,
                placeholder=_FLASHSCORE_TEAM_URL_EXAMPLE,
                help="Định dạng: https://www.flashscore.com/team/{slug}/{hash}/",
            )
            btn_key = f"{key_prefix}_import_{tid_key}"
            if st.button(
                "📥 Bóc Tách & Nạp Data",
                key=btn_key,
                use_container_width=True,
            ):
                url_val = str(st.session_state.get(widget_key) or "").strip()
                if not url_val:
                    st.error("Hãy dán URL Flashscore team trước khi nạp.")
                else:
                    with st.spinner(f"Đang nạp dữ liệu {team_label}…"):
                        result = import_team_from_flashscore_url(
                            team_label,
                            url_val,
                            upcoming_kickoff=kickoff,
                            upcoming_comp_id=comp_id,
                        )
                    if not result.get("ok"):
                        st.error(str(result.get("error") or "Nạp dữ liệu thất bại."))
                    else:
                        try:
                            from src.global_db import clear_team_history_cache

                            clear_team_history_cache()
                        except Exception:  # noqa: BLE001
                            pass
                        ready_set = st.session_state.setdefault(
                            "_flashscore_ready_teams", []
                        )
                        if not isinstance(ready_set, list):
                            ready_set = list(ready_set)
                            st.session_state["_flashscore_ready_teams"] = ready_set
                        if team_label not in ready_set:
                            ready_set.append(team_label)
                        st.toast("✅ Đã cập nhật thành công dữ liệu đội bóng!")
                        st.rerun()
    return missing


def _snap_line(value: float, options: list[float]) -> float:
    if value in options:
        return float(value)
    return float(min(options, key=lambda x: abs(x - value)))


def _rename_fixture_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df[cols].copy()
    out.columns = [_COL_FIXTURE_VI.get(c, c) for c in cols]
    return out


def _match_id(row: pd.Series | None, home: str, away: str) -> str:
    """Stable widget-key id per fixture (Flashscore mid when available)."""
    return match_id_for_fixture(row, home, away)


def _share_base_url() -> str | None:
    """Best-effort public host for share links (Streamlit ≥1.37 ``st.context``)."""
    try:
        headers = getattr(getattr(st, "context", None), "headers", None)
        if not headers:
            return None
        host = headers.get("Host") or headers.get("host")
        if not host:
            return None
        proto = (
            headers.get("X-Forwarded-Proto")
            or headers.get("x-forwarded-proto")
            or "https"
        )
        return f"{proto}://{host}"
    except Exception:  # noqa: BLE001
        return None


def _clipboard_copy_button(
    text: str,
    *,
    label: str = "📋 Copy Tóm Tắt & Link",
    key: str = "share_copy",
) -> None:
    """One-click clipboard via browser API (HTTPS/localhost); falls back to execCommand."""
    payload = json.dumps(text)
    btn_id = f"vb_copy_{key}".replace("-", "_")
    components.html(
        f"""
        <div style="font-family:sans-serif;">
          <button id="{btn_id}" type="button"
            style="width:100%;padding:0.55rem 0.9rem;border-radius:0.5rem;
                   border:1px solid #2e7d32;background:#1b5e20;color:#fff;
                   font-size:0.95rem;font-weight:600;cursor:pointer;">
            {label}
          </button>
        </div>
        <script>
        (function() {{
          const btn = document.getElementById("{btn_id}");
          if (!btn) return;
          const text = {payload};
          const original = btn.innerText;
          btn.onclick = async function() {{
            let ok = false;
            try {{
              if (navigator.clipboard && navigator.clipboard.writeText) {{
                await navigator.clipboard.writeText(text);
                ok = true;
              }}
            }} catch (e) {{ ok = false; }}
            if (!ok) {{
              try {{
                const ta = document.createElement("textarea");
                ta.value = text;
                ta.style.position = "fixed";
                ta.style.left = "-9999px";
                document.body.appendChild(ta);
                ta.select();
                ok = document.execCommand("copy");
                document.body.removeChild(ta);
              }} catch (e2) {{ ok = false; }}
            }}
            btn.innerText = ok ? "✅ Đã copy!" : "⚠️ Copy thủ công bên dưới";
            setTimeout(function() {{ btn.innerText = original; }}, 2000);
          }};
        }})();
        </script>
        """,
        height=52,
    )


def _set_match_query(match_id: str) -> None:
    """Keep URL ``match_id`` in sync with the open detail fixture."""
    try:
        current = st.query_params.get("match_id")
        if isinstance(current, (list, tuple)):
            current = current[0] if current else None
        if str(current or "") == str(match_id):
            return
        st.query_params["match_id"] = str(match_id)
        if "match" in st.query_params:
            del st.query_params["match"]
    except Exception:  # noqa: BLE001
        pass


def _clear_match_query() -> None:
    """Drop deep-link params so the URL matches Top-20 landing."""
    try:
        for key in ("match_id", "match"):
            if key in st.query_params:
                del st.query_params[key]
    except Exception:  # noqa: BLE001
        pass


def _apply_share_deep_link(
    fx: pd.DataFrame,
    *,
    league: str,
    alt_fixtures: pd.DataFrame | None = None,
    alt_league: str | None = None,
    extra_fixtures: Sequence[tuple[str, pd.DataFrame]] | None = None,
) -> None:
    """Open Chi tiết when ``?match_id=`` / ``?match=`` is present and valid.

    Searches the active league first, then optional ``alt_fixtures`` (other
    league), then ``extra_fixtures`` (e.g. Japan cups). On a cross-league hit,
    switches the sidebar league and reruns so detail loads the correct
    models/fixtures. Never silently drops a bad ``match_id``.
    """
    mid_q, slug_q = parse_match_query(dict(st.query_params))
    if not mid_q and not slug_q:
        st.session_state.pop("_share_deep_link_miss", None)
        st.session_state.pop("_share_deep_link_key", None)
        return

    apply_key = f"{league}|{mid_q or ''}|{slug_q or ''}"
    already = (
        st.session_state.get("_share_deep_link_key") == apply_key
        and st.session_state.get("selected_match")
        and not st.session_state.get("_share_deep_link_miss")
    )
    if already:
        return

    hit = find_fixture_for_share(fx, match_id=mid_q, match_slug=slug_q)
    hit_league = league
    if hit is None and alt_fixtures is not None and alt_league:
        hit = find_fixture_for_share(
            alt_fixtures, match_id=mid_q, match_slug=slug_q
        )
        if hit is not None:
            hit_league = alt_league
    if hit is None and extra_fixtures:
        for lg, efx in extra_fixtures:
            if efx is None or getattr(efx, "empty", True):
                continue
            hit = find_fixture_for_share(
                efx, match_id=mid_q, match_slug=slug_q
            )
            if hit is not None:
                hit_league = str(lg)
                break

    st.session_state["_share_deep_link_key"] = apply_key
    if hit is None:
        st.session_state["_share_deep_link_miss"] = mid_q or slug_q
        return

    # Cross-league deep link: queue league switch then rerun once.
    # Never write ``sidebar_league`` here — the selectbox already owns that key.
    if hit_league != league:
        other_label = str(get_league_config(hit_league)["label"])
        if st.session_state.get("sidebar_league") != other_label:
            st.session_state["_pending_sidebar_league"] = other_label
            st.session_state["selected_match"] = hit["match_id"]
            st.session_state["selected_home"] = hit["home"]
            st.session_state["selected_away"] = hit["away"]
            st.session_state["selected_kickoff"] = hit.get("kickoff")
            st.session_state["_apply_selected_match"] = True
            st.session_state["_goto_detail_tab"] = True
            st.session_state.pop("_share_deep_link_miss", None)
            st.session_state.pop("_share_deep_link_key", None)
            _set_match_query(str(hit["match_id"]))
            st.rerun()

    st.session_state.pop("_share_deep_link_miss", None)
    st.session_state["selected_match"] = hit["match_id"]
    st.session_state["selected_home"] = hit["home"]
    st.session_state["selected_away"] = hit["away"]
    st.session_state["selected_kickoff"] = hit.get("kickoff")
    st.session_state["_apply_selected_match"] = True
    st.session_state["_goto_detail_tab"] = True
    st.session_state["_share_deep_link_ok"] = (
        f"{hit['home']} vs {hit['away']} (`{hit['match_id']}`)"
    )
    _set_match_query(str(hit["match_id"]))


def _fixture_labels(fx: pd.DataFrame) -> list[str]:
    labels: list[str] = []
    for _, r in fx.iterrows():
        ko = pd.Timestamp(r["Kickoff"]).strftime("%Y-%m-%d %H:%M")
        labels.append(f"{ko} | {r['HomeTeam']} vs {r['AwayTeam']}")
    return labels


def _defaults_from_row(row: pd.Series | None) -> dict[str, float]:
    defaults = {
        "odds_h": 2.10,
        "odds_d": 3.40,
        "odds_a": 3.50,
        "ou_line": 2.5,
        "odds_over": 1.90,
        "odds_under": 1.90,
        "ah_line": -0.5,
        "odds_ah_h": 1.95,
        "odds_ah_a": 1.95,
    }
    if row is None:
        return defaults
    for key, col in (("odds_h", "B365H"), ("odds_d", "B365D"), ("odds_a", "B365A")):
        if col in row.index and pd.notna(row[col]):
            defaults[key] = float(row[col])
    if "OU_Line" in row.index and pd.notna(row["OU_Line"]):
        defaults["ou_line"] = float(row["OU_Line"])
    if "OddsOver" in row.index and pd.notna(row["OddsOver"]):
        defaults["odds_over"] = float(row["OddsOver"])
    elif "B365_O25" in row.index and pd.notna(row["B365_O25"]):
        defaults["odds_over"] = float(row["B365_O25"])
    if "OddsUnder" in row.index and pd.notna(row["OddsUnder"]):
        defaults["odds_under"] = float(row["OddsUnder"])
    elif "B365_U25" in row.index and pd.notna(row["B365_U25"]):
        defaults["odds_under"] = float(row["B365_U25"])
    if "AHh" in row.index and pd.notna(row["AHh"]):
        defaults["ah_line"] = float(row["AHh"])
    if "B365AHH" in row.index and pd.notna(row["B365AHH"]):
        defaults["odds_ah_h"] = float(row["B365AHH"])
    if "B365AHA" in row.index and pd.notna(row["B365AHA"]):
        defaults["odds_ah_a"] = float(row["B365AHA"])
    return defaults


def _row_has_api_odds(row: pd.Series | None) -> bool:
    if row is None:
        return False
    return any(c in row.index and pd.notna(row[c]) for c in ("B365H", "B365D", "B365A"))


def _odds_source_label(row: pd.Series | None) -> str:
    if row is None:
        return "manual"
    if "OddsProvider" in row.index and pd.notna(row["OddsProvider"]):
        return str(row["OddsProvider"])
    if _row_has_api_odds(row):
        return "football-data"
    return "manual"


def _resolve_match_odds(row: pd.Series | None) -> tuple[dict[str, float], str]:
    d = _defaults_from_row(row)
    cfg = {
        "odds_home": float(d["odds_h"]),
        "odds_draw": float(d["odds_d"]),
        "odds_away": float(d["odds_a"]),
        "ou_line": float(d["ou_line"]),
        "odds_over": float(d["odds_over"]),
        "odds_under": float(d["odds_under"]),
        "ah_line": float(d["ah_line"]),
        "odds_ah_home": float(d["odds_ah_h"]),
        "odds_ah_away": float(d["odds_ah_a"]),
    }
    return cfg, _odds_source_label(row)


def _comparison_frame(bets) -> pd.DataFrame:
    rows = []
    for b in bets:
        rows.append(
            {
                "Thị trường": _MARKET_VI.get(b.market, b.market),
                "Lựa chọn": _selection_vi(b.selection),
                "Odds nhà cái": b.bookmaker_odds,
                "P mô hình": b.p_model,
                "P nhà cái": implied_probability(b.bookmaker_odds),
                "Odds công bằng": b.fair_odds,
                "EV %": b.ev_pct,
                "Kelly %": b.kelly_pct,
                "Value?": b.recommended,
            }
        )
    return pd.DataFrame(rows)


def _prob_compare_chart(bets) -> go.Figure:
    labels = [_selection_vi(b.selection) for b in bets]
    fig = go.Figure(
        data=[
            go.Bar(
                name="P mô hình",
                x=labels,
                y=[b.p_model for b in bets],
                marker_color="#2ecc71",
            ),
            go.Bar(
                name="P nhà cái (1/Odds)",
                x=labels,
                y=[implied_probability(b.bookmaker_odds) for b in bets],
                marker_color="#3498db",
            ),
        ]
    )
    fig.update_layout(
        barmode="group",
        title="So sánh xác suất: Model vs Nhà cái",
        yaxis_title="Xác suất",
        yaxis_tickformat=".0%",
        height=360,
        margin=dict(l=8, r=8, t=48, b=32),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
        template="plotly_white",
        autosize=True,
    )
    return fig


def _blend_1x2(
    p_dc: dict[str, float],
    p_ml: dict[str, float],
    w_ml: float,
) -> dict[str, float]:
    """Linear Ensemble: P = (1 − w)·P_DC + w·P_ML, renormalised."""
    w = float(max(0.0, min(1.0, w_ml)))
    blended = {
        k: (1.0 - w) * float(p_dc[k]) + w * float(p_ml[k]) for k in ("H", "D", "A")
    }
    total = sum(blended.values())
    if total <= 0:
        return {"H": 1 / 3, "D": 1 / 3, "A": 1 / 3}
    return {k: v / total for k, v in blended.items()}


def _1x2_models_chart(
    p_dc: dict[str, float],
    p_ml: dict[str, float] | None,
    p_ens: dict[str, float],
    odds_h: float,
    odds_d: float,
    odds_a: float,
) -> go.Figure:
    """Grouped bars: Dixon–Coles / LightGBM / Ensemble / Implied 1X2."""
    labels = ["Chủ", "Hòa", "Khách"]
    keys = ("H", "D", "A")
    implied = [
        implied_probability(odds_h),
        implied_probability(odds_d),
        implied_probability(odds_a),
    ]
    traces = [
        go.Bar(
            name="Dixon–Coles",
            x=labels,
            y=[p_dc[k] for k in keys],
            marker_color="#3498db",
        ),
    ]
    if p_ml is not None:
        traces.append(
            go.Bar(
                name="LightGBM",
                x=labels,
                y=[p_ml[k] for k in keys],
                marker_color="#9b59b6",
            )
        )
    traces.extend(
        [
            go.Bar(
                name="Ensemble",
                x=labels,
                y=[p_ens[k] for k in keys],
                marker_color="#2ecc71",
            ),
            go.Bar(
                name="P nhà cái (1/Odds)",
                x=labels,
                y=implied,
                marker_color="#95a5a6",
            ),
        ]
    )
    fig = go.Figure(data=traces)
    fig.update_layout(
        barmode="group",
        title="1X2: Dixon–Coles · LightGBM · Ensemble · Nhà cái",
        yaxis_title="Xác suất",
        yaxis_tickformat=".0%",
        height=360,
        margin=dict(l=8, r=8, t=48, b=32),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
        template="plotly_white",
        autosize=True,
    )
    return fig


def _odds_families_available(df: pd.DataFrame) -> list[str]:
    available = [
        family
        for family, cols in ODDS_COLUMN_MAPS.items()
        if all(c in df.columns for c in cols.values())
    ]
    return available or ["B365"]


# ---------------------------------------------------------------------------
# Cached data / models
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Đang tải kết quả giải đấu…")
def _load_raw_data(
    league: str, n_seasons: int, force_refresh: bool = False, _cache_ver: int = 4
) -> pd.DataFrame:
    # ``db_path=None`` prefers global_matches.db (multi-league), then legacy.
    return load_league_data(
        league=league,
        n_seasons=n_seasons,
        force_refresh=force_refresh,
        db_path=None,
    )


@st.cache_data(show_spinner="Đang cắt mẫu huấn luyện…")
def _train_slice(
    league: str,
    n_seasons: int,
    n_train: int,
    force_refresh: bool = False,
    _cache_ver: int = 3,
) -> pd.DataFrame:
    raw = _load_raw_data(league, n_seasons, force_refresh=force_refresh)
    if raw.empty:
        return raw
    ordered = raw.sort_values("Date")
    out = ordered.tail(int(n_train)).reset_index(drop=True)
    # Preserve source metadata from the full load
    out.attrs.update(dict(raw.attrs))
    out.attrs["n_matches"] = int(len(out))
    out.attrs["n_matches_full"] = int(len(raw))
    return out


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_fixtures_and_odds(
    league: str, n_seasons: int, force_refresh: bool = False, _cache_ver: int = 4
) -> pd.DataFrame:
    """Network fetch of fixtures+odds; persists to ``global_matches.db``.

    Cached 5 minutes so tab switches / widget interactions do not re-hit
    Flashscore. Hashable args only (league key, ints, bool).
    """
    results = _load_raw_data(league, n_seasons, force_refresh=force_refresh)
    fx = load_upcoming_fixtures(
        league=league,
        results=results,
        only_future=True,
        exclude_played=True,
    )
    try:
        save_upcoming_to_db(league, fx)
    except Exception:  # noqa: BLE001
        pass
    return fx


# Stale DB cache → background refresh threshold (minutes).
_FIXTURES_STALE_MINUTES = 15
_FX_BG_LOCK = threading.Lock()
_FX_BG_RUNNING: set[str] = set()


def _bg_refresh_fixtures(league: str, n_seasons: int) -> None:
    """Fetch fixtures+odds off the UI thread; write DB + bump sentinel."""
    code = normalize_league(league)
    try:
        # Avoid Streamlit cache APIs from a worker thread — call loaders directly.
        results = load_league_data(
            league=code,
            n_seasons=int(n_seasons),
            force_refresh=False,
            db_path=None,
        )
        fx = load_upcoming_fixtures(
            league=code,
            results=results,
            only_future=True,
            exclude_played=True,
        )
        save_upcoming_to_db(code, fx)
        try:
            bump = GLOBAL_DB_PATH.parent / f".fx_cache_bump_{code}"
            bump.write_text("1", encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    except Exception as exc:  # noqa: BLE001
        try:
            err_path = GLOBAL_DB_PATH.parent / f".fx_refresh_err_{code}.txt"
            err_path.write_text(str(exc)[:500], encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    finally:
        with _FX_BG_LOCK:
            _FX_BG_RUNNING.discard(code)


def _maybe_spawn_fixture_refresh(league: str, n_seasons: int) -> bool:
    """Start at most one background refresh per league. Returns True if spawned."""
    code = normalize_league(league)
    with _FX_BG_LOCK:
        if code in _FX_BG_RUNNING:
            return False
        _FX_BG_RUNNING.add(code)
    thread = threading.Thread(
        target=_bg_refresh_fixtures,
        args=(code, int(n_seasons)),
        name=f"fx-refresh-{code}",
        daemon=True,
    )
    thread.start()
    return True


def _consume_fixture_cache_bump(league: str) -> None:
    """If a BG refresh finished, clear the 5-min network cache (main thread)."""
    code = normalize_league(league)
    bump = GLOBAL_DB_PATH.parent / f".fx_cache_bump_{code}"
    if not bump.is_file():
        return
    try:
        bump.unlink(missing_ok=True)
        _fetch_fixtures_and_odds.clear()
    except Exception:  # noqa: BLE001
        pass


def _load_fixtures(
    league: str, n_seasons: int, force_refresh: bool = False, _cache_ver: int = 4
) -> pd.DataFrame:
    """DB-first fixtures+odds for the UI (instant), with optional BG refresh.

    1. Prefer ``upcoming_fixtures`` in ``global_matches.db`` (<0.2s).
    2. If cache older than 15 minutes → toast + background network refresh.
    3. Cold start / force_refresh → network via ``_fetch_fixtures_and_odds``.
    """
    code = normalize_league(league)
    _consume_fixture_cache_bump(code)

    if force_refresh:
        return _fetch_fixtures_and_odds(
            code, n_seasons, force_refresh=True, _cache_ver=_cache_ver
        )

    db_df, last_ts = load_upcoming_from_db(code)
    age = upcoming_cache_age_minutes(last_ts)

    if db_df is not None and not db_df.empty:
        if age is None or age > _FIXTURES_STALE_MINUTES:
            toast_key = f"_fx_stale_toast_{code}"
            if not st.session_state.get(toast_key):
                st.session_state[toast_key] = True
                try:
                    st.toast("🔄 Đang cập nhật odds mới ngầm...")
                except Exception:  # noqa: BLE001
                    pass
            _maybe_spawn_fixture_refresh(code, n_seasons)
            # Surface prior BG failure once (non-blocking).
            err_path = GLOBAL_DB_PATH.parent / f".fx_refresh_err_{code}.txt"
            if err_path.is_file():
                try:
                    msg = err_path.read_text(encoding="utf-8").strip()
                    err_path.unlink(missing_ok=True)
                    if msg and not st.session_state.get(f"_fx_err_shown_{code}"):
                        st.session_state[f"_fx_err_shown_{code}"] = True
                        st.warning(
                            f"Cập nhật odds nền thất bại — đang dùng cache DB. ({msg[:120]})"
                        )
                except Exception:  # noqa: BLE001
                    pass
        else:
            # Fresh enough — allow a future stale toast again after next expiry.
            st.session_state.pop(f"_fx_stale_toast_{code}", None)
            st.session_state.pop(f"_fx_err_shown_{code}", None)
        return db_df

    # Cold start: block once, then DB + 5-min cache keep later renders instant.
    with st.spinner("Đang tải lịch + odds…"):
        return _fetch_fixtures_and_odds(
            code, n_seasons, force_refresh=False, _cache_ver=_cache_ver
        )


@st.cache_resource(show_spinner="Đang huấn luyện Dixon–Coles…")
def _fit_dixon_coles(
    league: str,
    n_seasons: int,
    n_train: int,
    xi: float,
    force_refresh: bool = False,
    _cache_ver: int = 4,
) -> DixonColesModel:
    code = normalize_league(league)
    train = _train_slice(league, n_seasons, n_train, force_refresh=force_refresh)
    use_weak = code == "UWCL" or int(len(train)) < 100
    return DixonColesModel(
        xi=xi,
        use_weak_tier_priors=use_weak,
        new_team_attack=WEAK_TIER_NEW_TEAM_ATTACK if use_weak else None,
        new_team_defence=WEAK_TIER_NEW_TEAM_DEFENCE if use_weak else None,
        min_team_matches=MIN_TEAM_MATCHES,
    ).fit(train)


@st.cache_resource(show_spinner="Đang huấn luyện mô hình phạt góc…")
def _fit_corners(
    league: str,
    n_seasons: int,
    n_train: int,
    force_refresh: bool = False,
    _cache_ver: int = 3,
) -> CornerPredictor:
    backend = "poisson" if normalize_league(league) == "UWCL" else "auto"
    return CornerPredictor(backend=backend).fit(
        _train_slice(league, n_seasons, n_train, force_refresh=force_refresh)
    )


@st.cache_resource(show_spinner="Đang huấn luyện LightGBM…")
def _fit_ml_model(
    league: str,
    n_seasons: int,
    n_train: int,
    force_refresh: bool = False,
    _cache_ver: int = 4,
) -> EPLMachineLearningModel:
    return EPLMachineLearningModel().fit(
        _train_slice(league, n_seasons, n_train, force_refresh=force_refresh)
    )


@st.cache_data(ttl=300, show_spinner="Đang quét Top 20 kèo hời (VN)…")
def _cached_top20_scan(
    fixtures: pd.DataFrame,
    league: str,
    n_seasons: int,
    n_train: int,
    xi: float,
    w_ml: float,
    min_ev: float,
    kelly_fraction: float,
    markets: tuple,
    bankroll: float,
    use_ml: bool,
    force_refresh: bool = False,
    _cache_ver: int = 4,
) -> dict:
    """Scan today/tomorrow fixtures; models come from ``@st.cache_resource``."""
    dc = maybe_wrap_with_league_weights(
        _fit_dixon_coles(
            league, n_seasons, n_train, float(xi), force_refresh=force_refresh
        ),
        league,
    )
    ml = None
    if use_ml and float(w_ml) > 0:
        try:
            ml = _fit_ml_model(
                league, n_seasons, n_train, force_refresh=force_refresh
            )
        except Exception:  # noqa: BLE001
            ml = None
    result = scan_top_value_bets(
        fixtures,
        dc,
        ml,
        w_ml=float(w_ml) if ml is not None else 0.0,
        min_ev=float(min_ev),
        kelly_fraction=float(kelly_fraction),
        max_stake_pct=MAX_STAKE_PCT,
        allowed_markets=tuple(markets) or ("1X2",),
        top_n=_TOP20_DISPLAY_N,
        bankroll=float(bankroll),
        league=str(league),
        filter_vn_window=True,
    )
    return result.to_dict()


# Per-league train defaults when the landing scans the *other* competition.
_TOP20_LEAGUE_DEFAULTS: dict[str, dict[str, int]] = {
    "EPL": {"n_seasons": 3, "n_train": 800},
    "UWCL": {"n_seasons": 5, "n_train": 200},
    "LALIGA": {"n_seasons": 3, "n_train": 800},
}


@st.cache_data(ttl=300, show_spinner="Đang quét Top 20 EPL + UWCL (VN)…")
def _cached_top20_scan_multi(
    fixtures_primary: pd.DataFrame,
    fixtures_other: pd.DataFrame,
    league_primary: str,
    league_other: str,
    n_seasons_primary: int,
    n_train_primary: int,
    n_seasons_other: int,
    n_train_other: int,
    xi: float,
    w_ml: float,
    min_ev: float,
    kelly_fraction: float,
    markets: tuple,
    bankroll: float,
    use_ml: bool,
    force_refresh: bool = False,
    _cache_ver: int = 3,
) -> dict:
    """Scan both leagues with league-specific models; merge Top-20 by EV%."""
    results = []
    for lg, fx, n_seas, n_tr in (
        (league_primary, fixtures_primary, n_seasons_primary, n_train_primary),
        (league_other, fixtures_other, n_seasons_other, n_train_other),
    ):
        if fx is None or getattr(fx, "empty", True):
            continue
        try:
            dc = maybe_wrap_with_league_weights(
                _fit_dixon_coles(
                    lg, int(n_seas), int(n_tr), float(xi), force_refresh=force_refresh
                ),
                lg,
            )
        except Exception:  # noqa: BLE001
            continue
        ml = None
        if use_ml and float(w_ml) > 0:
            try:
                ml = _fit_ml_model(
                    lg, int(n_seas), int(n_tr), force_refresh=force_refresh
                )
            except Exception:  # noqa: BLE001
                ml = None
        results.append(
            scan_top_value_bets(
                fx,
                dc,
                ml,
                w_ml=float(w_ml) if ml is not None else 0.0,
                min_ev=float(min_ev),
                kelly_fraction=float(kelly_fraction),
                max_stake_pct=MAX_STAKE_PCT,
                allowed_markets=tuple(markets) or ("1X2",),
                top_n=_TOP20_DISPLAY_N,
                bankroll=float(bankroll),
                league=str(lg),
                filter_vn_window=True,
            )
        )
    if not results:
        return scan_top_value_bets(
            fixtures_primary if fixtures_primary is not None else pd.DataFrame(),
            maybe_wrap_with_league_weights(
                _fit_dixon_coles(
                    league_primary,
                    n_seasons_primary,
                    n_train_primary,
                    float(xi),
                    force_refresh=force_refresh,
                ),
                league_primary,
            ),
            None,
            w_ml=0.0,
            min_ev=float(min_ev),
            top_n=_TOP20_DISPLAY_N,
            bankroll=float(bankroll),
            league=str(league_primary),
        ).to_dict()
    if len(results) == 1:
        return results[0].to_dict()
    return merge_top_scan_results(
        results, top_n=_TOP20_DISPLAY_N, min_ev=float(min_ev)
    ).to_dict()


@st.cache_data(ttl=300, show_spinner=False)
def _cached_global_matches_legacy(_cache_ver: int = 1) -> pd.DataFrame:
    """Load multi-comp history for Lite fatigue labels (empty if unavailable)."""
    try:
        ensure_global_db_from_legacy(GLOBAL_DB_PATH)
        if not global_db_has_matches(GLOBAL_DB_PATH):
            return pd.DataFrame()
        return read_matches_as_legacy(GLOBAL_DB_PATH)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def _enrich_lite_display_bets(display: pd.DataFrame | None) -> pd.DataFrame:
    """Attach fatigue / rest_days from SQLite only (instant; no web I/O).

    Spawns a non-blocking background :func:`sync_upcoming_teams_fast` so the
    next paint picks up fresher ``last_match_date`` values.
    """
    if display is None or getattr(display, "empty", True):
        return display if display is not None else pd.DataFrame()
    enriched = attach_rest_days_from_db(display)
    # Human-readable competition for card headers.
    if "competition" not in enriched.columns:
        enriched = enriched.copy()
        enriched["competition"] = enriched["league"].map(
            lambda c: league_label(str(c)) if pd.notna(c) and str(c) else ""
        )
    _maybe_spawn_team_feed_sync(enriched)
    return enriched


_TEAM_FEED_BG_LOCK = threading.Lock()
_TEAM_FEED_BG_RUNNING = False


def _bg_sync_team_feeds(display_records: list[dict[str, Any]]) -> None:
    global _TEAM_FEED_BG_RUNNING
    try:
        from src.fetchers.flashscore_team import sync_upcoming_teams_fast

        sync_upcoming_teams_fast(display_records, max_teams=10)
    except Exception:  # noqa: BLE001
        pass
    finally:
        with _TEAM_FEED_BG_LOCK:
            _TEAM_FEED_BG_RUNNING = False


def _maybe_spawn_team_feed_sync(display: pd.DataFrame) -> bool:
    """Daemon-thread lazy Team Feed sync for Top-20 sides (at most one at a time)."""
    global _TEAM_FEED_BG_RUNNING
    if display is None or getattr(display, "empty", True):
        return False
    with _TEAM_FEED_BG_LOCK:
        if _TEAM_FEED_BG_RUNNING:
            return False
        _TEAM_FEED_BG_RUNNING = True
    try:
        records = display.head(10).to_dict(orient="records")
    except Exception:  # noqa: BLE001
        with _TEAM_FEED_BG_LOCK:
            _TEAM_FEED_BG_RUNNING = False
        return False
    thread = threading.Thread(
        target=_bg_sync_team_feeds,
        args=(records,),
        name="team-feed-sync-fast",
        daemon=True,
    )
    thread.start()
    return True


def _clear_caches() -> None:
    _load_raw_data.clear()
    _train_slice.clear()
    _fetch_fixtures_and_odds.clear()
    _fit_dixon_coles.clear()
    _fit_corners.clear()
    _fit_ml_model.clear()
    _cached_top20_scan.clear()
    _cached_top20_scan_multi.clear()
    _cached_global_matches_legacy.clear()


def _format_data_source_status(df: pd.DataFrame) -> str:
    source = df.attrs.get("data_source", "unknown")
    n_full = int(df.attrs.get("n_matches_full") or df.attrs.get("n_matches") or len(df))
    if source == "sqlite_local":
        return f"Đang dùng dữ liệu SQLite Local ({n_full} trận)"
    if source == "sqlite_global":
        return f"Đang dùng dữ liệu Global multi-comp ({n_full} trận)"
    if source == "football-data.co.uk":
        return f"Đã cập nhật dữ liệu mới từ football-data.co.uk ({n_full} trận)"
    if source == "fotmob":
        return f"Đã cập nhật dữ liệu mới từ Fotmob/Flashscore ({n_full} trận)"
    if source == "sqlite_fallback":
        return f"Mạng lỗi — khôi phục SQLite Local ({n_full} trận)"
    return f"Nguồn dữ liệu: {source} ({n_full} trận)"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("⚽ Value Betting")

# View mode: apply pending BEFORE the radio widget (same pattern as league).
_pending_vm = st.session_state.pop("_pending_view_mode", None)
if _pending_vm in (_VIEW_LITE, _VIEW_PRO):
    st.session_state["view_mode"] = _pending_vm
if "view_mode" not in st.session_state:
    st.session_state["view_mode"] = _VIEW_LITE

view_mode = st.sidebar.radio(
    "Chế độ hiển thị:",
    [_VIEW_LITE, _VIEW_PRO],
    horizontal=True,
    key="view_mode",
)
is_lite = view_mode == _VIEW_LITE

if is_lite:
    st.sidebar.caption("Chế độ tham khảo — kèo gọn, ít chỉnh kỹ thuật.")
else:
    st.sidebar.caption("Dixon–Coles · LightGBM · Ensemble · Flashscore Odds")

_available = get_available_leagues()
_league_options = [str(row["name"]) for row in _available]
_league_label_to_code = {str(row["name"]): str(row["code"]) for row in _available}
# Apply cross-league switches BEFORE the selectbox widget exists.
# Callers must set ``_pending_sidebar_league`` (never ``sidebar_league``) after
# the widget has been instantiated in the same run.
_pending_lg = st.session_state.pop("_pending_sidebar_league", None)
if _pending_lg is not None and str(_pending_lg) in _league_options:
    st.session_state["sidebar_league"] = str(_pending_lg)

league_ui = st.sidebar.selectbox(
    "Giải đấu",
    options=_league_options,
    index=0,
    help=(
        "Cấu hình trong config/leagues.json — EPL (football-data) · "
        "UWCL (Fotmob) · LaLiga (Flashscore IDs + SP1)."
    ),
    key="sidebar_league",
)
league = _league_label_to_code.get(league_ui, "EPL")
active_db_path = league_db_path(league)

# Bankroll session defaults (UI: Lite sidebar · Pro budget/journal tabs)
if "bankroll" not in st.session_state:
    st.session_state["bankroll"] = 1000.0
if "bankroll_initial" not in st.session_state:
    st.session_state["bankroll_initial"] = float(st.session_state["bankroll"])

if is_lite:
    st.session_state["bankroll"] = float(
        st.sidebar.number_input(
            "Bankroll ($)",
            min_value=10.0,
            max_value=1_000_000.0,
            value=float(st.session_state["bankroll"]),
            step=50.0,
            key="sidebar_bankroll_input",
            help="Vốn dùng để gợi ý mức stake (Kelly).",
        )
    )

    # Fixed defaults — hide technical sliders in Lite Mode.
    xi = 0.0018
    _n_train_default = 200 if league == "UWCL" else 800
    _n_train_max = 500 if league == "UWCL" else 2000
    n_train = min(_n_train_default, _n_train_max)
    use_ensemble = True
    w_ml = float(DEFAULT_W_ML)
    min_ev = float(DEFAULT_MIN_EV)
    kelly_frac = float(DEFAULT_KELLY_FRACTION)
    # Lite: scan all goal markets so Best Value Pick can be AH/OU when 1X2
    # odds are missing (same pool as Pro Mode Chi tiết).
    scan_markets = ["1X2", "OU", "AH"]
    tg_enabled = bool(st.session_state.get("tg_enabled", False))

    st.sidebar.divider()
    if st.sidebar.button(
        "🔄 Làm mới dữ liệu", type="primary", use_container_width=True
    ):
        _clear_caches()
        st.session_state["_force_refresh"] = True
        st.session_state["_retrain_ok"] = True
        st.rerun()
else:
    xi = st.sidebar.slider(
        "ξ (time-decay)",
        min_value=0.0,
        max_value=0.01,
        value=0.0018,
        step=0.0001,
        format="%.4f",
        help="Tăng: Dồn trọng số vào phong độ ngắn hạn (dễ nhiễu).\n"
        "Giảm: Đánh giá dựa trên đẳng cấp dài hạn (chậm thích nghi).",
    )

    _n_train_default = 200 if league == "UWCL" else 800
    _n_train_max = 500 if league == "UWCL" else 2000
    _n_train_min = 50 if league == "UWCL" else 200
    n_train = st.sidebar.slider(
        "Số trận huấn luyện",
        min_value=_n_train_min,
        max_value=_n_train_max,
        value=min(_n_train_default, _n_train_max),
        step=25 if league == "UWCL" else 50,
        help="Tăng: Giúp mô hình có đủ data để hội tụ, tránh EV ảo.\n"
        "Giảm: Chạy nhanh hơn nhưng dễ bị lệch xác suất do thiếu mẫu.",
    )

    st.sidebar.markdown("**Ensemble 1X2**")
    use_ensemble = st.sidebar.toggle(
        "Dùng Ensemble cho Value Bet 1X2",
        value=True,
        help="Bật: P_1X2 = (1−w)·Dixon–Coles + w·LightGBM. Tắt: chỉ Dixon–Coles. "
        "Tài/Xỉu & AH luôn dùng Dixon–Coles.",
    )
    w_ml = (
        st.sidebar.slider(
            "Trọng số LightGBM w (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(DEFAULT_W_ML * 100),
            step=5.0,
            disabled=not use_ensemble,
            help="Tăng: Tin tưởng Machine Learning (cần nhiều data).\n"
            "Giảm: Tin tưởng Poisson Thống kê (An toàn cho giải thiếu data).",
        )
        / 100.0
    )

    min_ev = (
        st.sidebar.slider(
            "Ngưỡng EV tối thiểu (%)",
            min_value=0.0,
            max_value=20.0,
            value=float(DEFAULT_MIN_EV * 100),
            step=1.0,
            help="Tăng: Siết chặt tiêu chuẩn, ít kèo hơn nhưng an toàn (ít Drawdown).\n"
            "Giảm: Nhiều kèo hơn nhưng dễ dính kèo nhiễu.",
        )
        / 100.0
    )

    kelly_frac = (
        st.sidebar.slider(
            "Tỷ lệ Kelly tối đa (%)",
            min_value=5.0,
            max_value=float(MAX_KELLY_FRACTION * 100),
            value=float(DEFAULT_KELLY_FRACTION * 100),
            step=1.0,
            help="Tăng: Cược nhiều tiền hơn (Tăng trưởng nhanh nhưng rủi ro cháy vốn cao).\n"
            "Giảm: Cược bảo thủ, bảo vệ Bankroll an toàn trước chuỗi thua.",
        )
        / 100.0
    )

    st.sidebar.markdown("**Thị trường quét**")
    scan_markets = st.sidebar.multiselect(
        "Markets",
        options=["1X2", "OU", "AH", "Corners"],
        default=["1X2"],
        help="1X2 / Tài–Xỉu bàn thắng / Chấp Á / Phạt góc (CornerPredictor).",
        key="sidebar_scan_markets",
    )
    if not scan_markets:
        scan_markets = ["1X2"]

    st.sidebar.divider()
    if st.sidebar.button(
        "Tải dữ liệu & huấn luyện lại", type="primary", use_container_width=True
    ):
        _clear_caches()
        st.session_state["_force_refresh"] = True
        st.session_state["_retrain_ok"] = True
        st.session_state["_sync_global_after_refresh"] = True
        st.rerun()

    # ----- Telegram alerts (Pro only) -----
    st.sidebar.divider()
    st.sidebar.markdown("**Telegram Alert**")
    tg_enabled = st.sidebar.toggle(
        "Bật thông báo Telegram",
        value=bool(st.session_state.get("tg_enabled", False)),
        key="tg_enabled",
        help="Gửi Value Bet 1X2 (EV ≥ ngưỡng) tới chat Telegram của bạn.",
    )
    if "tg_token" not in st.session_state:
        st.session_state["tg_token"] = ""
    if "tg_chat_id" not in st.session_state:
        st.session_state["tg_chat_id"] = ""

    if tg_enabled:
        st.session_state["tg_token"] = st.sidebar.text_input(
            "Bot token",
            value=st.session_state["tg_token"],
            type="password",
            key="tg_token_input",
            help="Từ @BotFather",
        )
        st.session_state["tg_chat_id"] = st.sidebar.text_input(
            "Chat ID",
            value=st.session_state["tg_chat_id"],
            key="tg_chat_input",
            help="ID nhóm / user nhận tin (có thể lấy từ @userinfobot).",
        )
        tg_c1, tg_c2 = st.sidebar.columns(2)
        with tg_c1:
            if st.button("Gửi thử", use_container_width=True, key="tg_test_btn"):
                try:
                    send_telegram_message(
                        st.session_state["tg_token"],
                        st.session_state["tg_chat_id"],
                        "✅ <b>Value Betting</b>\nKết nối Telegram OK.",
                    )
                    st.sidebar.success("Đã gửi tin thử.")
                except Exception as exc:  # noqa: BLE001
                    st.sidebar.error(f"Lỗi: {exc}")
        with tg_c2:
            if st.button("Quét & gửi VB", use_container_width=True, key="tg_scan_btn"):
                st.session_state["_tg_scan_send"] = True
    else:
        st.sidebar.caption("Bật toggle để cấu hình Bot token / Chat ID.")


force_refresh = bool(st.session_state.pop("_force_refresh", False))
_sync_global = bool(st.session_state.pop("_sync_global_after_refresh", False))

# Data seasons buffer (enough matches for n_train slider)
N_SEASONS_BUFFER = 5 if league == "UWCL" else 3

try:
    data = _train_slice(league, N_SEASONS_BUFFER, n_train, force_refresh=force_refresh)
    fixtures = _load_fixtures(league, N_SEASONS_BUFFER, force_refresh=force_refresh)
    dc_model = _fit_dixon_coles(
        league, N_SEASONS_BUFFER, n_train, float(xi), force_refresh=force_refresh
    )
except Exception as exc:  # noqa: BLE001
    st.error(f"Không tải được dữ liệu / huấn luyện mô hình: {exc}")
    st.stop()

# Match-link paste may inject a single fixture ahead of the DB cache.
fixtures = _merge_link_injected_fixture(fixtures, league)

# After league refresh: sync global_matches.db so Lite / Top-N use fresh history.
if force_refresh or _sync_global:
    try:
        # Prefer full pipeline (all leagues) so UWCL + EPL stay in sync together.
        _gstats = sync_to_global_db(force=True)
        _cached_global_matches_legacy.clear()
        if _gstats.get("ok"):
            st.session_state["_global_sync_msg"] = (
                f"Đã sync global DB ({_gstats.get('matches_upserted', 0)} trận)."
            )
        else:
            st.session_state["_global_sync_msg"] = (
                f"Sync global DB cảnh báo: {_gstats.get('error', 'unknown')}"
            )
    except Exception as _gexc:  # noqa: BLE001
        st.session_state["_global_sync_msg"] = f"Sync global DB lỗi: {_gexc}"

corner_model: CornerPredictor | None = None
corner_error: str | None = None
try:
    corner_model = _fit_corners(
        league, N_SEASONS_BUFFER, n_train, force_refresh=force_refresh
    )
except Exception as exc:  # noqa: BLE001
    corner_error = str(exc)
    if not is_lite:
        if league == "UWCL":
            st.sidebar.info(
                "UWCL Fotmob chưa có HC/AC — tab Phạt góc tạm khóa "
                f"({corner_error[:80]})."
            )
        else:
            st.sidebar.warning(f"Corner model: {corner_error}")

# Thin-sample guard: shrink LightGBM weight when history is short (esp. UWCL)
n_hist_full = int(data.attrs.get("n_matches_full") or len(data))
w_ml_eff, w_ml_warn = ensemble_weight_for_sample(n_hist_full, float(w_ml))
if w_ml_warn:
    if not is_lite:
        st.sidebar.warning(w_ml_warn)
    w_ml = float(w_ml_eff)

ml_model: EPLMachineLearningModel | None = None
ml_error: str | None = None
try:
    ml_model = _fit_ml_model(
        league, N_SEASONS_BUFFER, n_train, force_refresh=force_refresh
    )
except Exception as exc:  # noqa: BLE001
    ml_error = str(exc)

teams = [t for t in list_teams(data) if t in dc_model.teams]
odds_families = _odds_families_available(data)
recommender = ValueBetRecommender(dc_model, min_ev=min_ev, kelly_fraction=kelly_frac)

st.sidebar.caption(f"**{league_label(league)}** · {_format_data_source_status(data)}")
if data.attrs.get("download_warnings"):
    st.sidebar.warning("; ".join(data.attrs["download_warnings"][:2]))

if not is_lite:
    if ml_model is not None:
        ens_note = f"Ensemble w={w_ml:.0%}" if use_ensemble else "Ensemble tắt"
        st.sidebar.caption(
            f"LightGBM sẵn sàng · {ml_model.train_rows_:,} mẫu feature · {ens_note}"
        )
    elif ml_error:
        st.sidebar.warning(f"LightGBM chưa sẵn sàng: {ml_error}")

if st.session_state.pop("_retrain_ok", False):
    src = data.attrs.get("data_source")
    if src == "football-data.co.uk":
        st.sidebar.success("Đã cập nhật dữ liệu mới từ API & huấn luyện lại.")
    elif src == "sqlite_fallback":
        st.sidebar.warning("Không cập nhật được API — đang dùng SQLite Local.")
    else:
        st.sidebar.success("Đã tải dữ liệu & huấn luyện lại thành công.")
_gmsg = st.session_state.pop("_global_sync_msg", None)
if _gmsg:
    st.sidebar.caption(str(_gmsg))

if not is_lite:
    st.sidebar.success(
        f"{len(data):,} trận train · {len(fixtures)} sắp tới · {len(teams)} đội"
    )
else:
    st.sidebar.caption(f"{len(fixtures)} trận sắp tới · bankroll ${st.session_state['bankroll']:,.0f}")

# Deep-link: ?match_id=… or ?match=Home_vs_Away → open Chi tiết
# Prefetch another league's fixtures so cross-league links resolve.
_alt_candidates = [r["code"] for r in _available if r["code"] != league]
_alt_league = "EPL" if "EPL" in _alt_candidates else (_alt_candidates[0] if _alt_candidates else "EPL")
_alt_defaults = _TOP20_LEAGUE_DEFAULTS.get(
    _alt_league, {"n_seasons": 3, "n_train": 800}
)
_alt_fixtures: pd.DataFrame | None = None
try:
    _alt_fixtures = _load_fixtures(
        _alt_league,
        int(_alt_defaults["n_seasons"]),
        force_refresh=False,
    )
except Exception:  # noqa: BLE001
    _alt_fixtures = None

_jp_extra: list[tuple[str, pd.DataFrame]] = []
try:
    _jp_extra = [
        (code, df)
        for code, df in load_japan_upcoming_frames()
        if code not in {league, _alt_league}
    ]
except Exception:  # noqa: BLE001
    _jp_extra = []

_apply_share_deep_link(
    fixtures,
    league=league,
    alt_fixtures=_alt_fixtures,
    alt_league=_alt_league,
    extra_fixtures=_jp_extra or None,
)

# Deep-link / "mở chuyên sâu" while in Lite → switch to Pro so Chi tiết exists.
if is_lite and st.session_state.get("_goto_detail_tab"):
    st.session_state["_pending_view_mode"] = _VIEW_PRO
    st.rerun()

# Sidebar "Quét & gửi VB" → scan fixtures for selected markets and push Telegram
if st.session_state.pop("_tg_scan_send", False):
    if fixtures.empty:
        st.sidebar.warning("Chưa có fixtures để quét.")
    else:
        try:
            mkts = tuple(m for m in scan_markets if m != "Corners") or ("1X2",)
            with st.spinner(f"Quét Value Bet {','.join(scan_markets)} & gửi Telegram…"):
                scan_df = recommend_upcoming(
                    dc_model,
                    fixtures.head(40),
                    odds_family="B365",
                    min_ev=min_ev,
                    kelly_fraction=kelly_frac,
                    only_value=True,
                    include_ou="OU" in mkts,
                    include_ah="AH" in mkts,
                )
                if not scan_df.empty:
                    scan_df = scan_df.loc[scan_df["market"].isin(list(mkts))].copy()
                    scan_df["stake"] = (
                        scan_df["kelly_fraction"] * float(st.session_state["bankroll"])
                    )
                # Optional corner EV alerts (manual default odds if feed lacks them)
                corner_rows: list[dict] = []
                if "Corners" in scan_markets and corner_model is not None:
                    for _, fx in fixtures.head(20).iterrows():
                        h_t, a_t = resolve_fixture_teams(
                            str(fx["HomeTeam"]),
                            str(fx["AwayTeam"]),
                            known_teams=dc_model.teams,
                        )
                        try:
                            legs = corner_model.predict_corner_ev(
                                h_t,
                                a_t,
                                {
                                    "over": float(fx["OddsCornerOver"])
                                    if "OddsCornerOver" in fx.index
                                    and pd.notna(fx.get("OddsCornerOver"))
                                    else 1.90,
                                    "under": float(fx["OddsCornerUnder"])
                                    if "OddsCornerUnder" in fx.index
                                    and pd.notna(fx.get("OddsCornerUnder"))
                                    else 1.90,
                                },
                                line=10.5,
                                min_ev=min_ev,
                            )
                        except Exception:
                            continue
                        for leg in legs:
                            corner_rows.append(
                                {
                                    "home_team": h_t,
                                    "away_team": a_t,
                                    "market": "Corners",
                                    "selection": leg["selection"],
                                    "bookmaker_odds": leg["odds"],
                                    "ev": leg["ev"],
                                    "ev_pct": leg["ev_pct"],
                                    "kelly_fraction": 0.0,
                                    "stake": 0.0,
                                }
                            )
                if corner_rows:
                    cdf = pd.DataFrame(corner_rows)
                    scan_df = (
                        pd.concat([scan_df, cdf], ignore_index=True)
                        if scan_df is not None and not getattr(scan_df, "empty", True)
                        else cdf
                    )
                # De-dupe correlated legs + daily Top-N (same policy as auto_scanner)
                if scan_df is not None and not getattr(scan_df, "empty", True):
                    scan_df = select_value_bets(
                        scan_df,
                        max_per_day=MAX_BETS_PER_DAY,
                        already_today=0,
                        one_per_match=True,
                    )
                result_tg = send_telegram_value_bets(
                    scan_df,
                    st.session_state.get("tg_token", ""),
                    st.session_state.get("tg_chat_id", ""),
                    bankroll=float(st.session_state["bankroll"]),
                    min_ev=min_ev,
                    markets=tuple(scan_markets),
                    league=league,
                )
            if result_tg["sent"]:
                st.sidebar.success(
                    f"Đã gửi {result_tg['sent']} Value Bet ({', '.join(scan_markets)})."
                )
            else:
                st.sidebar.info(
                    f"Không có VB để gửi (skipped={result_tg['skipped']})."
                )
            for err in result_tg.get("errors", [])[:3]:
                st.sidebar.error(err)
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"Telegram scan lỗi: {exc}")

# ---------------------------------------------------------------------------
# Main nav (session-state driven — st.tabs cannot be selected programmatically)
# ---------------------------------------------------------------------------

_NAV_TOP20 = "🔥 Top 20 Kèo Hời (Hôm Nay & Ngày Mai)"
_NAV_JP = "🇯🇵 Lịch Nhật"
_NAV_ABOUT = "Giới thiệu thuật toán"
_NAV_DETAIL = "Chi tiết trận đấu & Mô hình"
_NAV_BUDGET = "Quản lý ngân sách"
_NAV_JOURNAL = "📈 Nhật Ký Cược & Bankroll"
_NAV_HIST = "Dự đoán & Lịch sử"
_NAV_BACKTEST = "Backtest"
_NAV_CORNERS = "Phân tích Phạt Góc"
_NAV_COMPARE = "So sánh đội"
_NAV_OPTIONS = [
    _NAV_TOP20,
    _NAV_JP,
    _NAV_ABOUT,
    _NAV_DETAIL,
    _NAV_BUDGET,
    _NAV_JOURNAL,
    _NAV_HIST,
    _NAV_BACKTEST,
    _NAV_CORNERS,
    _NAV_COMPARE,
]
_NAV_LITE_OPTIONS = [_NAV_LITE_TOP, _NAV_LITE_SEARCH, _NAV_JP]

if "_main_nav" not in st.session_state:
    st.session_state["_main_nav"] = _NAV_LITE_TOP if is_lite else _NAV_TOP20

# Map nav when toggling Lite ↔ Pro so pills never hold an invalid option.
_cur_nav = st.session_state.get("_main_nav")
if is_lite:
    if _cur_nav == _NAV_JP:
        pass  # shared Japan schedule tab
    elif _cur_nav not in _NAV_LITE_OPTIONS:
        if _cur_nav == _NAV_TOP20:
            st.session_state["_main_nav"] = _NAV_LITE_TOP
        elif _cur_nav == _NAV_DETAIL and st.session_state.get("selected_match"):
            # Detail only exists in Pro — bounce to search with selection kept.
            st.session_state["_main_nav"] = _NAV_LITE_SEARCH
        else:
            st.session_state["_main_nav"] = _NAV_LITE_TOP
else:
    if _cur_nav == _NAV_JP:
        pass  # shared Japan schedule tab
    elif _cur_nav in _NAV_LITE_OPTIONS:
        st.session_state["_main_nav"] = (
            _NAV_TOP20 if _cur_nav == _NAV_LITE_TOP else _NAV_DETAIL
        )
    elif _cur_nav not in _NAV_OPTIONS:
        st.session_state["_main_nav"] = _NAV_TOP20

if st.session_state.pop("_goto_detail_tab", False):
    st.session_state["_main_nav"] = _NAV_DETAIL if not is_lite else _NAV_LITE_SEARCH
if st.session_state.pop("_goto_top20_tab", False):
    st.session_state["_main_nav"] = _NAV_LITE_TOP if is_lite else _NAV_TOP20

# Deep-link feedback (always visible, not buried in an unselected tab)
_miss_id = st.session_state.get("_share_deep_link_miss")
if _miss_id:
    st.warning(
        f"Không tìm thấy trận với `match_id={_miss_id}` trong fixtures "
        "(EPL / UWCL / J.League Cup / Emperor's Cup). "
        "Bấm **Tải dữ liệu & huấn luyện lại** ở sidebar hoặc kiểm tra link."
    )
_ok_link = st.session_state.pop("_share_deep_link_ok", None)
if _ok_link:
    st.success(f"Đã mở trận từ link: {_ok_link}")

_nav_options = _NAV_LITE_OPTIONS if is_lite else _NAV_OPTIONS
main_nav = st.pills(
    "Điều hướng",
    options=_nav_options,
    key="_main_nav",
    label_visibility="collapsed",
    required=True,
)

# ===========================================================================
# Lite Mode — Top Kèo Hôm Nay
# ===========================================================================

if is_lite and main_nav == _NAV_LITE_TOP:
    st.header("🔥 Top Kèo Hời Hôm Nay")
    st.caption(
        f"Cửa sổ hôm nay & ngày mai · EV ≥ {min_ev:.0%} · "
        "Sắp xếp theo EV% giảm dần toàn bộ thị trường đã chọn"
    )
    if st.button("🔄 Quét lại", key="lite_top_rescan", use_container_width=True):
        _cached_top20_scan.clear()
        _cached_top20_scan_multi.clear()
        _cached_global_matches_legacy.clear()
        st.rerun()

    if fixtures.empty and (
        _alt_fixtures is None or getattr(_alt_fixtures, "empty", True)
    ):
        st.warning(
            "Không có trận sắp tới. Bấm **Làm mới dữ liệu** ở sidebar."
        )
    else:
        bank = float(st.session_state.get("bankroll", 1000.0))
        other_fx = (
            _alt_fixtures if _alt_fixtures is not None else pd.DataFrame()
        )
        od = _TOP20_LEAGUE_DEFAULTS[_alt_league]
        scan_out = _cached_top20_scan_multi(
            fixtures,
            other_fx,
            league,
            _alt_league,
            N_SEASONS_BUFFER,
            n_train,
            int(od["n_seasons"]),
            int(od["n_train"]),
            float(xi),
            float(w_ml),
            float(min_ev),
            float(kelly_frac),
            tuple(m for m in scan_markets if m != "Corners") or ("1X2", "OU", "AH"),
            bank,
            True,
            force_refresh=False,
        )
        display = _enrich_lite_display_bets(scan_out.get("display_bets"))
        if display is None or getattr(display, "empty", True):
            st.info(
                "ℹ️ Chưa có kèo trong cửa sổ hôm nay → ngày mai (~48 giờ, giờ VN). "
                "Thử **Quét lại** hoặc mở **Tìm Trận**."
            )
        else:
            n_val = int(scan_out.get("n_value") or 0)
            if scan_out.get("odds_missing"):
                st.info("ℹ️ Có trận nhưng chưa đủ odds — vẫn liệt kê bên dưới.")
            elif scan_out.get("below_threshold"):
                st.info(
                    f"ℹ️ Chưa có kèo EV ≥ {min_ev:.0%}. "
                    "Dưới đây là xếp hạng EV cao nhất (có thể dưới ngưỡng)."
                )
            else:
                st.success(
                    f"Tìm thấy **{n_val}** kèo hời (EV ≥ {min_ev:.0%}) · EPL + UWCL."
                )
            for i, (_, row) in enumerate(display.iterrows()):
                _render_lite_bet_card(
                    row,
                    idx=i,
                    default_league=league,
                    key_prefix="lite_top",
                )


# ===========================================================================
# Lite Mode — Tìm Trận
# ===========================================================================

if is_lite and main_nav == _NAV_LITE_SEARCH:
    st.header("🔍 Tìm Trận")
    st.caption("Chọn trận sắp tới để xem thẻ nhận định nhanh hoặc mở phân tích Pro.")

    _render_flashscore_match_link_analyzer(
        key_prefix="lite_search_link",
        current_league=str(league),
    )
    st.divider()

    search_rows: list[dict] = []
    for src_lg, src_fx in ((league, fixtures), (_alt_league, _alt_fixtures)):
        if src_fx is None or getattr(src_fx, "empty", True):
            continue
        for _, fx in src_fx.iterrows():
            h = str(fx.get("HomeTeam", ""))
            a = str(fx.get("AwayTeam", ""))
            mid = _match_id(fx, h, a)
            ko = fx.get("Kickoff")
            search_rows.append(
                {
                    "label": f"{format_kickoff_vn(ko)} · {src_lg} · {h} vs {a}",
                    "home": h,
                    "away": a,
                    "match_id": mid,
                    "kickoff": ko,
                    "league": src_lg,
                    "row": fx,
                }
            )

    if not search_rows:
        st.warning("Không có fixtures để tìm. Làm mới dữ liệu ở sidebar.")
    else:
        labels = [r["label"] for r in search_rows]
        # Prefer deep-link / prior selection when present
        default_idx = 0
        want_mid = st.session_state.get("selected_match")
        if want_mid:
            for i, r in enumerate(search_rows):
                if str(r["match_id"]) == str(want_mid):
                    default_idx = i
                    break
        picked = st.selectbox(
            "Trận đấu",
            options=labels,
            index=min(default_idx, len(labels) - 1),
            key="lite_search_fixture",
        )
        hit = search_rows[labels.index(picked)]

        # Best-effort: attach top scan pick for this match if available
        bank = float(st.session_state.get("bankroll", 1000.0))
        other_fx = (
            _alt_fixtures if _alt_fixtures is not None else pd.DataFrame()
        )
        od = _TOP20_LEAGUE_DEFAULTS[_alt_league]
        scan_out = _cached_top20_scan_multi(
            fixtures,
            other_fx,
            league,
            _alt_league,
            N_SEASONS_BUFFER,
            n_train,
            int(od["n_seasons"]),
            int(od["n_train"]),
            float(xi),
            float(w_ml),
            float(min_ev),
            float(kelly_frac),
            tuple(m for m in scan_markets if m != "Corners") or ("1X2", "OU", "AH"),
            bank,
            True,
            force_refresh=False,
        )
        display = _enrich_lite_display_bets(scan_out.get("display_bets"))
        card_row: dict | pd.Series | None = None
        if display is not None and not getattr(display, "empty", True):
            for _, r in display.iterrows():
                if str(r.get("match_id") or "") == str(hit["match_id"]):
                    card_row = r
                    break
                if (
                    str(r.get("home") or r.get("home_team") or "") == hit["home"]
                    and str(r.get("away") or r.get("away_team") or "") == hit["away"]
                ):
                    card_row = r
                    break

        if card_row is not None:
            _render_lite_bet_card(
                card_row,
                idx=0,
                default_league=str(hit["league"]),
                key_prefix="lite_search",
            )
        else:
            # Minimal card when match is outside the VN window / no EV yet
            fx_row = hit["row"]
            defs, _src = _resolve_match_odds(fx_row)
            mini = {
                "home": hit["home"],
                "away": hit["away"],
                "kickoff": hit["kickoff"],
                "kickoff_vn": format_kickoff_vn(hit["kickoff"]),
                "league": hit["league"],
                "match_id": hit["match_id"],
                "market": "1X2",
                "selection": "—",
                "odds": defs.get("odds_h"),
                "ev_pct": float("nan"),
                "p_model": float("nan"),
            }
            mini_df = _enrich_lite_display_bets(pd.DataFrame([mini]))
            mini_row = mini_df.iloc[0] if not mini_df.empty else mini
            st.info(
                "Chưa có kèo EV trong cửa sổ hôm nay→mai cho trận này — "
                "vẫn có thể mở Pro để phân tích đầy đủ."
            )
            _render_lite_bet_card(
                mini_row,
                idx=0,
                default_league=str(hit["league"]),
                key_prefix="lite_search_mini",
            )


# ===========================================================================
# Tab — Lịch Nhật (J.League Cup + Emperor's Cup)
# ===========================================================================

if main_nav == _NAV_JP:
    st.header("🇯🇵 Lịch Nhật — J.League Cup & Emperor's Cup")
    st.caption(
        "Upcoming từ global_matches.db · Giờ VN · "
        "Kèo gợi ý = mọi market EV≥ngưỡng (1X2/OU/AH) · "
        "Chi tiết / Flashscore giữ nguyên"
    )
    _jp_base = _share_base_url() or "http://localhost:8501"
    _jp_fx = load_japan_upcoming()
    _jp_records = japan_match_records(_jp_fx, base_url=_jp_base)

    _jp_c1, _jp_c2 = st.columns([1, 3])
    with _jp_c1:
        if st.button(
            "🔄 Tính lại kèo",
            key="jp_recalc_picks",
            use_container_width=True,
            help="Xóa cache session và chạy lại Dixon–Coles + EV cho từng trận.",
        ):
            st.session_state["jp_picks_cache"] = {}
            st.rerun()
    with _jp_c2:
        st.caption(
            f"EV ≥ {min_ev:.0%} · markets 1X2 / Tài-Xỉu / Chấp Á · "
            "cache theo match_id trong session"
        )

    if "jp_picks_cache" not in st.session_state:
        st.session_state["jp_picks_cache"] = {}

    if not _jp_records:
        st.warning(
            "Chưa có fixture J.League Cup / Emperor's Cup trong DB. "
            "Chọn giải ở sidebar rồi bấm **Làm mới dữ liệu**."
        )
    else:
        # Fit (cached) Japan cup models — weak priors for thin cup history.
        _jp_models: dict[str, Any] = {}
        _jp_n_seasons = 3
        _jp_n_train = min(int(n_train), 400)
        for _jp_code in JP_COMP_IDS:
            try:
                _jp_models[_jp_code] = maybe_wrap_with_league_weights(
                    _fit_dixon_coles(
                        _jp_code,
                        _jp_n_seasons,
                        _jp_n_train,
                        float(xi),
                        force_refresh=False,
                    ),
                    _jp_code,
                )
            except Exception as _jp_exc:  # noqa: BLE001
                st.caption(f"Model {_jp_code}: {_jp_exc}")

        _jp_need = [
            r["match_id"]
            for r in _jp_records
            if r.get("match_id")
            and str(r["match_id"]) not in st.session_state["jp_picks_cache"]
        ]
        if _jp_need and _jp_models:
            with st.spinner(
                f"Đang tính kèo gợi ý cho {len(_jp_need)} trận…"
            ):
                _jp_records = attach_picks_to_records(
                    _jp_records,
                    _jp_fx,
                    _jp_models,
                    min_ev=float(min_ev),
                    kelly_fraction=float(kelly_frac),
                    cache=st.session_state["jp_picks_cache"],
                    include_ou=True,
                    include_ah=True,
                )
        else:
            _jp_records = attach_picks_to_records(
                _jp_records,
                _jp_fx,
                _jp_models,
                min_ev=float(min_ev),
                kelly_fraction=float(kelly_frac),
                cache=st.session_state["jp_picks_cache"],
                include_ou=True,
                include_ah=True,
            )

        for _g_label, _g_rows in group_japan_matches(_jp_records):
            st.subheader(_g_label)
            _tbl = pd.DataFrame(
                [
                    {
                        "Giải": r["comp_label"],
                        "Trận": r["match"],
                        "Giờ VN": r["kickoff_vn"],
                        "Odds 1X2": r["odds"] or "—",
                        "Kèo gợi ý": r.get("picks_text") or "—",
                        "Chi tiết": r["detail_url"],
                        "Flashscore": r["flashscore_url"],
                    }
                    for r in _g_rows
                ]
            )
            st.dataframe(
                _tbl,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Giải": st.column_config.TextColumn("Giải", width="small"),
                    "Trận": st.column_config.TextColumn("Trận", width="medium"),
                    "Giờ VN": st.column_config.TextColumn("Giờ VN", width="small"),
                    "Odds 1X2": st.column_config.TextColumn(
                        "Odds (H/D/A)", width="small"
                    ),
                    "Kèo gợi ý": st.column_config.TextColumn(
                        "Kèo gợi ý",
                        width="large",
                        help="Mọi market EV≥ngưỡng; nếu không có → best pick + EV.",
                    ),
                    "Chi tiết": st.column_config.LinkColumn(
                        "Chi tiết",
                        display_text="Chi tiết",
                        width="small",
                    ),
                    "Flashscore": st.column_config.LinkColumn(
                        "Flashscore",
                        display_text="Flashscore",
                        width="small",
                    ),
                },
            )
            st.caption(f"{len(_g_rows)} trận")


# ===========================================================================
# Tab — Top 20 Kèo Hời (Hôm Nay & Ngày Mai)  [VN timezone landing]
# ===========================================================================

if (not is_lite) and main_nav == _NAV_TOP20:
    st.header("🔥 Top Kèo Hời (Daily Cap)")
    st.caption(
        f"Cửa sổ hôm nay & ngày mai · EV ≥ {min_ev:.0%} · "
        "Sắp xếp theo EV% giảm dần toàn bộ thị trường đã chọn"
    )

    top_c1, top_c2, top_c3 = st.columns([1, 1, 2])
    with top_c1:
        if st.button("🔄 Quét lại", key="top20_rescan", use_container_width=True):
            _cached_top20_scan.clear()
            _cached_top20_scan_multi.clear()
            st.rerun()
    with top_c2:
        scan_both = st.checkbox(
            "EPL + UWCL",
            value=True,
            key="top20_scan_both",
            help="Quét cả hai giải trong cửa sổ VN, xếp theo EV% "
            f"(tối đa {_TOP20_DISPLAY_N} kèo sau lọc).",
        )
    with top_c3:
        scope_note = "EPL + UWCL" if scan_both else league_label(league)
        st.caption(f"Phạm vi: {scope_note} · markets: {', '.join(scan_markets)} · DB-first · network cache TTL 5 phút")

    if fixtures.empty and not (
        scan_both and _alt_fixtures is not None and not _alt_fixtures.empty
    ):
        st.warning(
            "Không có trận sắp tới. Bấm **Tải dữ liệu & huấn luyện lại** ở sidebar."
        )
    else:
        mkts = tuple(m for m in scan_markets if m != "Corners") or ("1X2",)
        bank = float(st.session_state.get("bankroll", 1000.0))
        # Try LightGBM per league inside the cached scanner when Ensemble is on.
        use_ml_flag = bool(use_ensemble)
        w_for_scan = float(w_ml) if use_ensemble else 0.0

        if scan_both:
            other_fx = (
                _alt_fixtures
                if _alt_fixtures is not None
                else pd.DataFrame()
            )
            od = _TOP20_LEAGUE_DEFAULTS[_alt_league]
            scan_out = _cached_top20_scan_multi(
                fixtures,
                other_fx,
                league,
                _alt_league,
                N_SEASONS_BUFFER,
                n_train,
                int(od["n_seasons"]),
                int(od["n_train"]),
                float(xi),
                w_for_scan,
                float(min_ev),
                float(kelly_frac),
                mkts,
                bank,
                use_ml_flag,
                force_refresh=False,
            )
        else:
            scan_out = _cached_top20_scan(
                fixtures,
                league,
                N_SEASONS_BUFFER,
                n_train,
                float(xi),
                w_for_scan,
                float(min_ev),
                float(kelly_frac),
                mkts,
                bank,
                use_ml_flag,
                force_refresh=False,
            )
        display = scan_out.get("display_bets")
        if display is None or getattr(display, "empty", True):
            n_upcoming = int(len(fixtures)) if fixtures is not None else 0
            if scan_both and _alt_fixtures is not None:
                n_upcoming += int(len(_alt_fixtures))
            if n_upcoming > 0:
                st.info(
                    "ℹ️ Có trận sắp tới nhưng không nằm trong cửa sổ "
                    "hôm nay → ngày mai / ~48 giờ (giờ VN). "
                    "Mở tab Chi tiết hoặc bấm **Quét lại** sau."
                )
            else:
                st.info(
                    "ℹ️ Không tìm thấy trận trong cửa sổ hôm nay → ngày mai "
                    "(~48 giờ, giờ VN). Thử **Quét lại** hoặc đổi giải ở sidebar."
                )
        else:
            below = bool(scan_out.get("below_threshold"))
            odds_missing = bool(scan_out.get("odds_missing"))
            n_val = int(scan_out.get("n_value") or 0)
            if odds_missing:
                st.info(
                    "ℹ️ Có trận trong cửa sổ giờ VN nhưng chưa lấy được odds "
                    "để tính EV. Danh sách bên dưới — mở **Xem Chi Tiết** hoặc "
                    "bấm **Quét lại** / làm mới dữ liệu để tải odds."
                )
            elif below:
                st.info(
                    f"ℹ️ Không tìm thấy kèo hời (EV ≥ {min_ev:.0%}) trong ~48 giờ tới. "
                    "Dưới đây là danh sách các trận sắp diễn ra "
                    "(xếp theo EV cao nhất, có thể dưới ngưỡng)."
                )
            else:
                scope_txt = "EPL + UWCL" if scan_both else league_label(league)
                st.success(
                    f"Tìm thấy **{n_val}** kèo hời (EV ≥ {min_ev:.0%}) "
                    f"trong cửa sổ hôm nay & ngày mai (~48 giờ, giờ VN) · {scope_txt}."
                )

            # Compact header row (desktop); cards stack on mobile via CSS
            for i, row in display.iterrows():
                home = str(row.get("home") or row.get("home_team") or "")
                away = str(row.get("away") or row.get("away_team") or "")
                ko_vn = str(row.get("kickoff_vn") or format_kickoff_vn(row.get("kickoff")))
                mkt_code = str(row.get("market", "") or "").strip()
                badge = _market_badge(mkt_code)
                sel_raw = str(row.get("selection", ""))
                sel = _selection_vi(sel_raw) if sel_raw and sel_raw != "—" else "—"
                odds_raw = row.get("odds") if row.get("odds") is not None else row.get("bookmaker_odds")
                try:
                    odds = float(odds_raw) if odds_raw is not None and pd.notna(odds_raw) else float("nan")
                except (TypeError, ValueError):
                    odds = float("nan")
                p_raw = row.get("p_model")
                try:
                    p_m = float(p_raw) if p_raw is not None and pd.notna(p_raw) else float("nan")
                except (TypeError, ValueError):
                    p_m = float("nan")
                ev_raw = row.get("ev_pct")
                try:
                    ev_pct = float(ev_raw) if ev_raw is not None and pd.notna(ev_raw) else float("nan")
                except (TypeError, ValueError):
                    ev_pct = float("nan")
                mid = str(row.get("match_id") or f"{home}__{away}")
                lg = str(row.get("league") or league)
                row_odds_missing = bool(row.get("odds_missing")) or (
                    odds_missing and (pd.isna(odds) or odds <= 0)
                )
                ev_cls = (
                    "vb-card-ev-neg"
                    if pd.isna(ev_pct)
                    else ("vb-card-ev-pos" if ev_pct >= float(min_ev) * 100 else "vb-card-ev-neg")
                )

                with st.container(border=True):
                    left, right = st.columns([4, 1])
                    with left:
                        from src.data_loader import has_team_history

                        home_thin = not has_team_history(home, comp_id=lg)
                        away_thin = not has_team_history(away, comp_id=lg)
                        home_lbl = _team_label_with_thin_badge(
                            home, thin=home_thin, html=True
                        )
                        away_lbl = _team_label_with_thin_badge(
                            away, thin=away_thin, html=True
                        )
                        st.markdown(
                            f"**{ko_vn}** · `{lg}` · "
                            f"<strong>{home_lbl} vs {away_lbl}</strong>",
                            unsafe_allow_html=True,
                        )
                        _render_team_profile_buttons(
                            home,
                            away,
                            key_prefix=f"top20_{i}_{_sanitize_widget_key(mid)}_{_sanitize_widget_key(lg)}",
                        )
                        if row_odds_missing or pd.isna(ev_pct):
                            st.markdown(
                                '<p class="vb-card-meta">Odds / EV n/a — '
                                "mở chi tiết hoặc làm mới odds</p>",
                                unsafe_allow_html=True,
                            )
                        else:
                            odds_txt = f"{odds:.2f}" if pd.notna(odds) else "n/a"
                            p_txt = f"{p_m:.0%}" if pd.notna(p_m) else "n/a"
                            st.markdown(
                                f'<p class="vb-card-meta">{badge} {sel} @ '
                                f"{odds_txt} · Model {p_txt} · "
                                f'<span class="{ev_cls}">EV {ev_pct:+.1f}%</span></p>',
                                unsafe_allow_html=True,
                            )
                            model_line = str(row.get("model_fair_line") or "").strip()
                            bookie_line = str(row.get("bookie_market_line") or "").strip()
                            line_edge = str(row.get("line_edge") or "").strip()
                            if model_line or bookie_line or line_edge:
                                edge_bits = []
                                if model_line:
                                    edge_bits.append(f"Model {model_line}")
                                if bookie_line:
                                    edge_bits.append(f"NC {bookie_line}")
                                if line_edge:
                                    edge_bits.append(f"lệch {line_edge}")
                                st.markdown(
                                    f'<p class="vb-card-meta">{" · ".join(edge_bits)}</p>',
                                    unsafe_allow_html=True,
                                )
                    with right:
                        if st.button(
                            "🔍 Xem Chi Tiết",
                            key=f"top20_detail_{i}_{mid}_{lg}",
                            use_container_width=True,
                        ):
                            # Cross-league card → queue league switch for next run
                            # (before selectbox); never write ``sidebar_league`` here.
                            lg_code = normalize_league(lg)
                            if lg_code != league:
                                try:
                                    st.session_state["_pending_sidebar_league"] = str(
                                        get_league_config(lg_code)["label"]
                                    )
                                except ValueError:
                                    pass
                            st.session_state["selected_match"] = mid
                            st.session_state["selected_home"] = home
                            st.session_state["selected_away"] = away
                            st.session_state["selected_kickoff"] = row.get("kickoff")
                            st.session_state["_apply_selected_match"] = True
                            st.session_state["_goto_detail_tab"] = True
                            _set_match_query(mid)
                            st.rerun()

# ===========================================================================
# Tab 0 — Giới thiệu thuật toán đề xuất
# ===========================================================================

if (not is_lite) and main_nav == _NAV_ABOUT:
    st.header("Giới thiệu thuật toán đề xuất")
    st.caption("Giải thích ngắn gọn — không cần biết toán vẫn dùng được.")

    # ----- 1. Executive Summary -----
    st.info(
        "**Hệ thống giúp bạn tìm ra các trận đấu mà Nhà cái ra tỷ lệ thưởng cao hơn thực tế "
        "(Kèo Hời), đồng thời tính toán số tiền cược an toàn để bảo vệ vốn.**\n\n"
        f"Chỉ đề xuất khi lợi thế ≥ **{min_ev:.0%}** · mức vốn tối đa theo Kelly "
        f"**{kelly_frac:.0%}** (có thể chỉnh ở thanh bên trái)."
    )

    # ----- 2. Ba trụ cột -----
    st.subheader("Ba trụ cột hoạt động")
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("#### 📊 Đo sức mạnh")
            st.markdown("**Dixon–Coles**")
            st.write(
                "Máy bóc tách chỉ số **tấn công**, **phòng thủ** và "
                "**lợi thế sân nhà** của từng đội, rồi ước lượng xác suất "
                "thắng / hòa / thua thực tế của trận đấu."
            )
    with c2:
        with st.container(border=True):
            st.markdown("#### 💎 Săn kèo hời")
            st.markdown("**Value Bet**")
            st.write(
                "So sánh dự đoán của máy với tỷ lệ thưởng nhà cái. "
                f"Chỉ đề xuất khi phát hiện nhà cái **trả thưởng hớ** "
                f"(lợi thế ≥ **{min_ev:.0%}**)."
            )
    with c3:
        with st.container(border=True):
            st.markdown("#### 🛡️ Bảo vệ vốn")
            st.markdown("**Kelly Criterion**")
            st.write(
                "Khuyên chính xác nên đặt **bao nhiêu % tiền quỹ** cho mỗi kèo, "
                "để hạn chế rủi ro cháy tài khoản khi gặp chuỗi thua."
            )

    # ----- 3. Ví dụ minh họa -----
    st.subheader("Ví dụ minh họa thực tế")
    st.caption("Cùng dự đoán 50% thắng — chỉ khác tỷ lệ nhà cái là quyết định nên / không nên đặt.")

    ex_yes, ex_no = st.columns(2)
    with ex_yes:
        with st.container(border=True):
            st.markdown("##### 🟢 Kèo NÊN ĐẶT")
            st.markdown(
                """
- Máy dự đoán: **50%** thắng (giá công bằng ≈ ăn **2.00**)
- Nhà cái ra: ăn **2.30**
- → Có lợi thế khoảng **+15%**
                """
            )
            st.success("Nhà cái trả thưởng **cao hơn** thực tế → **đề xuất cược**.")

    with ex_no:
        with st.container(border=True):
            st.markdown("##### 🔴 Kèo NÊN BỎ")
            st.markdown(
                """
- Máy dự đoán: **50%** thắng (giá công bằng ≈ ăn **2.00**)
- Nhà cái chỉ cho: ăn **1.80**
- → Bị ép giá khoảng **−10%**
                """
            )
            st.warning("Nhà cái trả thưởng **thấp hơn** thực tế → **khuyên bỏ qua**.")

    # ----- 4. Interactive Sandbox -----
    st.subheader("Bộ công cụ tính thử")
    st.caption(
        "Kéo thanh trượt bên dưới để tự kiểm tra: có phải kèo hời không, "
        "và nên đặt bao nhiêu tiền."
    )

    sb1, sb2, sb3 = st.columns([1.2, 1.2, 1])
    with sb1:
        p_pct = st.slider(
            "Máy dự đoán thắng (%)",
            min_value=1,
            max_value=99,
            value=50,
            step=1,
            key="sandbox_p_pct",
            help="Xác suất máy tin cửa này sẽ thắng.",
        )
    with sb2:
        odds_sb = st.slider(
            "Tỷ lệ thưởng nhà cái (Odds)",
            min_value=1.01,
            max_value=20.0,
            value=2.30,
            step=0.01,
            key="sandbox_odds",
            help="Ví dụ 2.00 = ăn gấp đôi tiền cược nếu thắng.",
        )
    with sb3:
        bank_sb = st.number_input(
            "Số tiền quỹ của bạn",
            min_value=0.0,
            value=float(st.session_state.get("bankroll", 1000.0)),
            step=50.0,
            key="sandbox_bank",
        )

    p_sb = p_pct / 100.0
    fair_sb = 1.0 / p_sb
    ev_sb = p_sb * float(odds_sb) - 1.0
    if float(odds_sb) > 1.0 and 0.0 <= float(p_sb) <= 1.0:
        kelly_sb, stake_sb = calculate_kelly_stake(
            float(p_sb),
            float(odds_sb),
            float(bank_sb),
            kelly_fraction=kelly_frac,
            max_stake_pct=MAX_STAKE_PCT,
        )
    else:
        kelly_sb = 0.0
        stake_sb = 0.0

    _metrics_row(
        [
            ("Giá công bằng nên có", f"{fair_sb:.2f}"),
            ("Lợi thế của bạn", f"{ev_sb * 100:+.1f}%"),
            ("% vốn nên đặt", f"{kelly_sb * 100:.2f}%"),
            ("Tiền cược gợi ý", f"{stake_sb:,.0f}"),
        ],
        per_row=2,
    )

    if ev_sb >= min_ev:
        st.success(
            f"✅ Lợi thế **{ev_sb:+.1%}** ≥ ngưỡng **{min_ev:.0%}** → "
            f"**Nên cân nhắc đặt** khoảng **{stake_sb:,.0f}** "
            f"(trên quỹ {float(bank_sb):,.0f})."
        )
    else:
        st.warning(
            f"⚠️ Lợi thế **{ev_sb:+.1%}** chưa đủ ngưỡng **{min_ev:.0%}** → "
            "**Nên bỏ qua** kèo này để bảo vệ vốn."
        )

    st.caption(
        "Công cụ mang tính hỗ trợ quyết định / nghiên cứu — không đảm bảo thắng cược."
    )

# ===========================================================================
# Tab 1 — Chi tiết kèo trận sắp diễn ra
# ===========================================================================

if (not is_lite) and main_nav == _NAV_DETAIL:
    st.header("Chi tiết trận đấu & Mô hình")
    st.caption(
        "Odds từ tab Odds Flashscore (ưu tiên bet365) · "
        + (
            f"[{league_label(league)}]("
            f"{get_league_config(league).get('flashscore_fixtures_url') or '#'})"
            if get_league_config(league).get("flashscore_fixtures_url")
            else league_label(league)
        )
        + " · so sánh Dixon–Coles / LightGBM / Ensemble realtime."
    )

    _render_flashscore_match_link_analyzer(
        key_prefix="pro_detail_link",
        current_league=str(league),
    )
    st.divider()

    odds_src = fixtures.attrs.get("odds_sources") if not fixtures.empty else None
    if odds_src:
        st.caption(f"Nguồn odds đã tải: `{', '.join(odds_src)}`")

    if fixtures.empty:
        st.warning("Không có trận sắp tới. Bấm **Tải dữ liệu & huấn luyện lại** ở sidebar.")
    else:
        # ----- 1) Chọn trận -----
        st.subheader("1. Chọn trận đấu")
        n_show = min(40, len(fixtures))
        fx_view = fixtures.head(n_show).copy()
        # Ensure Top-20 selection is visible even if outside the first 40
        sel_home = st.session_state.get("selected_home")
        sel_away = st.session_state.get("selected_away")
        if sel_home and sel_away:
            hit = fixtures[
                (fixtures["HomeTeam"] == sel_home) & (fixtures["AwayTeam"] == sel_away)
            ]
            if not hit.empty:
                fx_view = (
                    pd.concat([hit, fx_view], ignore_index=True)
                    .drop_duplicates(subset=["HomeTeam", "AwayTeam"], keep="first")
                    .reset_index(drop=True)
                )
        labels = _fixture_labels(fx_view)
        if st.session_state.pop("_apply_selected_match", False) and labels:
            target = None
            for lab in labels:
                if sel_home and sel_away and f"{sel_home} vs {sel_away}" in lab:
                    target = lab
                    break
            if target is not None:
                st.session_state["detail_select"] = target
                st.info(f"Đã mở trận từ link / Top 20: **{sel_home} vs {sel_away}**")
        clr_c1, clr_c2 = st.columns([3, 1])
        with clr_c1:
            picked = st.selectbox("Trận sắp diễn ra", labels, key="detail_select")
        with clr_c2:
            st.write("")  # vertical align with selectbox
            if st.button(
                "✕ Bỏ chọn",
                key="detail_clear_selection",
                use_container_width=True,
                help="Xóa match_id trên URL và về tab Top 20",
            ):
                for k in (
                    "selected_match",
                    "selected_home",
                    "selected_away",
                    "selected_kickoff",
                    "_detail_synced_mid",
                    "_apply_selected_match",
                    "_share_deep_link_miss",
                    "_share_deep_link_key",
                    "_share_deep_link_ok",
                ):
                    st.session_state.pop(k, None)
                _clear_match_query()
                st.session_state["_goto_top20_tab"] = True
                st.rerun()

        _, match_part = picked.split(" | ", 1)
        home, away = match_part.split(" vs ", 1)
        row_hit = fx_view[
            (fx_view["HomeTeam"] == home) & (fx_view["AwayTeam"] == away)
        ]
        row = row_hit.iloc[0] if not row_hit.empty else None
        odds_cfg, src_note = _resolve_match_odds(row)
        mid = _match_id(row, home, away)
        kickoff = picked.split(" | ", 1)[0]
        # Keep URL consistent while a deep-link / Top-20 drill-down is active.
        prev_synced = st.session_state.get("_detail_synced_mid")
        share_active = bool(
            st.session_state.get("selected_match")
            or "match_id" in st.query_params
            or "match" in st.query_params
        )
        if share_active and prev_synced != mid:
            st.session_state["_detail_synced_mid"] = mid
            st.session_state["selected_match"] = mid
            st.session_state["selected_home"] = home
            st.session_state["selected_away"] = away
            _set_match_query(mid)

        if src_note == "manual" or not _row_has_api_odds(row):
            st.warning(
                f"**Đang phân tích:** {home} vs {away} · {kickoff} — "
                "chưa có odds API, hãy nhập tay ở phần 2."
            )
        else:
            st.info(
                f"**Đang phân tích:** {home} vs {away} · {kickoff} · "
                f"nguồn odds: `{src_note}` · match_id=`{mid}`"
            )

        # Early thin/missing-data badges + manual Flashscore URL import.
        _hdr_home, _hdr_away = resolve_fixture_teams(
            home, away, known_teams=dc_model.teams
        )
        _detail_kick = None
        if row is not None:
            _detail_kick = row.get("Kickoff") if "Kickoff" in getattr(row, "index", []) else row.get("Date")
        if _detail_kick is None:
            try:
                _detail_kick = pd.Timestamp(kickoff)
            except (TypeError, ValueError):
                _detail_kick = kickoff
        _thin_hdr = teams_needing_data_badge(
            dc_model,
            _hdr_home,
            _hdr_away,
            known_teams=dc_model.teams,
            upcoming_date=_detail_kick,
            min_matches=1,
            comp_id=str(league) if league else None,
        )
        _render_manual_team_import_ui(
            _hdr_home,
            _hdr_away,
            key_prefix=f"detail_{_sanitize_widget_key(mid)}",
            thin_names=_thin_hdr,
            kickoff=_detail_kick,
            comp_id=str(league) if league else None,
            always_show_badges=True,
        )
        _render_team_profile_buttons(
            _hdr_home,
            _hdr_away,
            key_prefix=f"detail_prof_{_sanitize_widget_key(mid)}",
        )

        # Dynamic keys theo match_id — reset khi đổi trận
        if st.session_state.get("_detail_last_pick") != picked:
            st.session_state["_detail_last_pick"] = picked
            st.session_state[f"o_h_{mid}"] = float(odds_cfg["odds_home"])
            st.session_state[f"o_d_{mid}"] = float(odds_cfg["odds_draw"])
            st.session_state[f"o_a_{mid}"] = float(odds_cfg["odds_away"])
            st.session_state[f"o_ov_{mid}"] = float(odds_cfg["odds_over"])
            st.session_state[f"o_un_{mid}"] = float(odds_cfg["odds_under"])
            st.session_state[f"o_ahh_{mid}"] = float(odds_cfg["odds_ah_home"])
            st.session_state[f"o_aha_{mid}"] = float(odds_cfg["odds_ah_away"])
            st.session_state[f"ou_line_{mid}"] = _snap_line(
                float(odds_cfg["ou_line"]), OU_LINE_OPTIONS
            )
            st.session_state[f"ah_line_{mid}"] = _snap_line(
                float(odds_cfg["ah_line"]), AH_LINE_OPTIONS
            )

        for key, val in {
            f"o_h_{mid}": float(odds_cfg["odds_home"]),
            f"o_d_{mid}": float(odds_cfg["odds_draw"]),
            f"o_a_{mid}": float(odds_cfg["odds_away"]),
            f"o_ov_{mid}": float(odds_cfg["odds_over"]),
            f"o_un_{mid}": float(odds_cfg["odds_under"]),
            f"o_ahh_{mid}": float(odds_cfg["odds_ah_home"]),
            f"o_aha_{mid}": float(odds_cfg["odds_ah_away"]),
            f"ou_line_{mid}": _snap_line(float(odds_cfg["ou_line"]), OU_LINE_OPTIONS),
            f"ah_line_{mid}": _snap_line(float(odds_cfg["ah_line"]), AH_LINE_OPTIONS),
        }.items():
            if key not in st.session_state:
                st.session_state[key] = val

        # ----- 2) Odds nhà cái -----
        st.subheader("2. Tỷ lệ kèo nhà cái")
        st.caption("Đổi trận = nạp odds API mới · chỉnh tay = cập nhật realtime (key theo match_id).")
        c_1x2, c_ou, c_ah = st.columns(3)

        with c_1x2:
            st.markdown("**1X2 (Châu Âu)**")
            odds_h = st.number_input("Odds Chủ", min_value=1.01, step=0.05, key=f"o_h_{mid}")
            odds_d = st.number_input("Odds Hòa", min_value=1.01, step=0.05, key=f"o_d_{mid}")
            odds_a = st.number_input("Odds Khách", min_value=1.01, step=0.05, key=f"o_a_{mid}")

        with c_ou:
            st.markdown("**Tài / Xỉu**")
            ou_line = st.selectbox("Mốc", OU_LINE_OPTIONS, key=f"ou_line_{mid}")
            odds_over = st.number_input("Odds Tài", min_value=1.01, step=0.05, key=f"o_ov_{mid}")
            odds_under = st.number_input("Odds Xỉu", min_value=1.01, step=0.05, key=f"o_un_{mid}")

        with c_ah:
            st.markdown("**Asian Handicap**")
            ah_line = st.selectbox(
                "Mốc chấp (nhà)",
                AH_LINE_OPTIONS,
                format_func=lambda x: f"{x:+g}",
                key=f"ah_line_{mid}",
            )
            odds_ah_h = st.number_input(
                "Odds chấp chủ", min_value=1.01, step=0.05, key=f"o_ahh_{mid}"
            )
            odds_ah_a = st.number_input(
                "Odds chấp khách", min_value=1.01, step=0.05, key=f"o_aha_{mid}"
            )

        # ----- 3) So sánh Model vs Nhà cái -----
        st.subheader("3. So sánh Model vs Nhà cái")

        # Align Flashscore names (Leuven) with fitted spelling (Oud-Heverlee Leuven).
        home, away = resolve_fixture_teams(home, away, known_teams=dc_model.teams)

        thin_for_badge = teams_needing_data_badge(
            dc_model,
            home,
            away,
            known_teams=dc_model.teams,
            upcoming_date=_detail_kick,
            min_matches=1,
            comp_id=str(league) if league else None,
        )
        if thin_for_badge:
            st.warning(
                f"⚠️ Đội thiếu lịch sử trong DB: {', '.join(thin_for_badge)}. "
                "Nạp Team Feed / dán URL Flashscore để có rest_days và hồ sơ đội."
            )
        elif home not in dc_model.teams or away not in dc_model.teams:
            missing = [
                t for t in (home, away) if t not in dc_model.teams
            ]
            st.warning(
                f"⚠️ Đội mới chưa có trong lịch sử huấn luyện: {', '.join(missing)}. "
                f"Dùng prior yếu (α={WEAK_TIER_NEW_TEAM_ATTACK}, "
                f"δ={WEAK_TIER_NEW_TEAM_DEFENCE}); Kelly tối đa 1% bankroll."
            )
        try:
            lam, mu = dc_model.expected_goals(home, away)
            probs_dc = dc_model.predict_match_probs(home, away)
            ou_probs = dc_model.predict_over_under(
                home, away, line=float(ou_line)
            )

            kick_ts = None
            if row is not None and "Kickoff" in row.index and pd.notna(row["Kickoff"]):
                kick_ts = pd.Timestamp(row["Kickoff"])

            probs_ml: dict[str, float] | None = None
            if ml_model is not None:
                try:
                    probs_ml = ml_model.predict_proba(
                        home, away, match_date=kick_ts
                    )
                except Exception:  # noqa: BLE001
                    probs_ml = None

            probs_ens = (
                _blend_1x2(probs_dc, probs_ml, w_ml)
                if probs_ml is not None
                else None
            )
            if use_ensemble and probs_ens is not None:
                probs_1x2 = probs_ens
                p_source = f"Ensemble (w_ML={w_ml:.0%})"
            else:
                probs_1x2 = probs_dc
                p_source = "Dixon–Coles"

            all_bets = recommender.evaluate_match(
                home,
                away,
                odds_1x2={"H": odds_h, "D": odds_d, "A": odds_a},
                over_under={
                    "line": float(ou_line),
                    "over": odds_over,
                    "under": odds_under,
                },
                asian_handicap={
                    "handicap": float(ah_line),
                    "home": odds_ah_h,
                    "away": odds_ah_a,
                },
                only_value=False,
                probs_1x2=probs_1x2,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Lỗi tính toán: {exc}")
        else:
                _metrics_row(
                    [
                        ("xG Chủ (DC)", f"{lam:.2f}"),
                        ("xG Khách (DC)", f"{mu:.2f}"),
                        ("P Hòa (dùng để EV)", f"{probs_1x2['D']:.1%}"),
                        (
                            f"Tài / Xỉu {float(ou_line):g}",
                            f"{ou_probs['over']:.0%} / {ou_probs['under']:.0%}",
                        ),
                    ],
                    per_row=2,
                )
                st.caption(
                    f"1X2 Value Bet dùng **{p_source}** · "
                    "Tài/Xỉu & Asian Handicap luôn từ Dixon–Coles (ma trận tỷ số)."
                )

                # Bảng xác suất 1X2 theo từng mô hình
                st.markdown("##### Xác suất 1X2 theo mô hình")
                prob_rows = [
                    {
                        "Mô hình": "Dixon–Coles",
                        "P(Chủ)": probs_dc["H"],
                        "P(Hòa)": probs_dc["D"],
                        "P(Khách)": probs_dc["A"],
                    }
                ]
                if probs_ml is not None:
                    prob_rows.append(
                        {
                            "Mô hình": "LightGBM",
                            "P(Chủ)": probs_ml["H"],
                            "P(Hòa)": probs_ml["D"],
                            "P(Khách)": probs_ml["A"],
                        }
                    )
                    assert probs_ens is not None
                    prob_rows.append(
                        {
                            "Mô hình": f"Ensemble (w={w_ml:.0%})",
                            "P(Chủ)": probs_ens["H"],
                            "P(Hòa)": probs_ens["D"],
                            "P(Khách)": probs_ens["A"],
                        }
                    )
                elif ml_error:
                    st.info(f"LightGBM chưa chạy được: {ml_error}")
                # else: ML probs already included (incl. new-team mean impute)

                show_p = pd.DataFrame(prob_rows)
                for col in ("P(Chủ)", "P(Hòa)", "P(Khách)"):
                    show_p[col] = show_p[col].map(lambda x: f"{x:.1%}")
                st.dataframe(show_p, use_container_width=True, hide_index=True)

                _plotly(
                    _1x2_models_chart(
                        probs_dc,
                        probs_ml,
                        probs_ens if probs_ens is not None else probs_dc,
                        float(odds_h),
                        float(odds_d),
                        float(odds_a),
                    ),
                    height=360,
                )
                _plotly(_prob_compare_chart(all_bets), height=360)

                cmp_df = _comparison_frame(all_bets)
                show = cmp_df.copy()
                show["P mô hình"] = show["P mô hình"].map(lambda x: f"{x:.1%}")
                show["P nhà cái"] = show["P nhà cái"].map(lambda x: f"{x:.1%}")
                show["Odds nhà cái"] = show["Odds nhà cái"].map(lambda x: f"{x:.2f}")
                show["Odds công bằng"] = show["Odds công bằng"].map(lambda x: f"{x:.2f}")
                show["EV %"] = show["EV %"].map(lambda x: f"{x:+.1f}%")
                show["Kelly %"] = show["Kelly %"].map(lambda x: f"{x:.2f}%")
                show["Value?"] = show["Value?"].map(lambda x: "✅" if x else "—")
                st.dataframe(show, use_container_width=True, hide_index=True)

                # ----- 4) Value Bet -----
                st.subheader("4. Đề xuất Value Bet")
                bankroll = float(st.session_state["bankroll"])
                recommended = [b for b in all_bets if b.recommended]
                selected: list[tuple[Any, float, float]] = []
                if recommended:
                    vb_rows = []
                    for b in recommended:
                        k_pct, _ = calculate_kelly_stake(
                            float(b.p_model),
                            float(b.bookmaker_odds),
                            bankroll=1.0,
                            kelly_fraction=kelly_frac,
                            max_stake_pct=MAX_STAKE_PCT,
                        )
                        vb_rows.append(
                            {
                                "home_team": home,
                                "away_team": away,
                                "match_id": mid,
                                "kickoff": kickoff,
                                "market": b.market,
                                "selection": b.selection,
                                "p_model": float(b.p_model),
                                "bookmaker_odds": float(b.bookmaker_odds),
                                "ev": float(b.ev),
                                "ev_pct": float(b.ev_pct),
                                "kelly_fraction": float(k_pct),
                                "kelly_pct": float(k_pct) * 100.0,
                                "_bet": b,
                            }
                        )
                    vb_df = select_value_bets(
                        pd.DataFrame(vb_rows),
                        max_per_day=max(MAX_BETS_PER_DAY, len(vb_rows)),
                        already_today=0,
                        allow_multi_picks_per_match=True,
                    )
                    for _, r in vb_df.iterrows():
                        bet = r["_bet"]
                        k = float(r["kelly_fraction"])
                        selected.append((bet, k, k * bankroll))

                if selected:
                    st.success(format_recommendations([b for b, _, _ in selected]))
                    tg_rows: list[dict] = []
                    for i, (bet, bet_kelly, stake) in enumerate(selected, start=1):
                        with st.container(border=True):
                            st.markdown(
                                f"**#{i} · {_market_badge(bet.market)} "
                                f"{_selection_vi(bet.selection)}**"
                            )
                            _metrics_row(
                                [
                                    ("Odds nhà cái", f"{bet.bookmaker_odds:.2f}"),
                                    ("EV %", f"{bet.ev_pct:+.1f}%"),
                                    ("Kelly %", f"{bet_kelly * 100:.2f}%"),
                                    ("Tiền cược gợi ý", f"{stake:,.2f}"),
                                ],
                                per_row=2,
                            )
                            src_cap = (
                                p_source if bet.market == "1X2" else "Dixon–Coles"
                            )
                            st.caption(
                                f"P_model={bet.p_model:.1%} ({src_cap}) · "
                                f"Fair={bet.fair_odds:.2f} · "
                                f"P_implied={implied_probability(bet.bookmaker_odds):.1%}"
                            )
                            jkey = (
                                f"journal_{mid}_{bet.market}_{bet.selection}_{i}_"
                                f"{bet.bookmaker_odds:.2f}"
                            )
                            if st.button(
                                "➕ Đặt cược vào Journal",
                                key=jkey,
                                use_container_width=True,
                            ):
                                try:
                                    jr = add_to_journal(
                                        home_team=home,
                                        away_team=away,
                                        selection=_selection_vi(bet.selection),
                                        odds=float(bet.bookmaker_odds),
                                        stake_amount=float(stake),
                                        match_date=kickoff,
                                        market=str(bet.market),
                                        stake_pct=float(bet_kelly),
                                        ev=float(bet.ev),
                                        p_model=float(bet.p_model),
                                        db_path=active_db_path,
                                    )
                                    if jr["created"]:
                                        st.success(
                                            f"Đã lưu Journal #{jr['id']} · "
                                            f"{_selection_vi(bet.selection)} @ "
                                            f"{bet.bookmaker_odds:.2f} · stake {stake:,.2f}"
                                        )
                                    else:
                                        st.info(
                                            f"Đã có PENDING #{jr['id']} (dedup) — "
                                            "không ghi trùng."
                                        )
                                except Exception as exc:  # noqa: BLE001
                                    st.error(f"Không lưu được Journal: {exc}")

                            if bet.market == "1X2":
                                tg_rows.append(
                                    {
                                        "home_team": home,
                                        "away_team": away,
                                        "market": bet.market,
                                        "selection": _selection_vi(bet.selection),
                                        "bookmaker_odds": bet.bookmaker_odds,
                                        "ev": bet.ev,
                                        "kelly_fraction": bet_kelly,
                                        "stake": stake,
                                        "match_date": kickoff,
                                    }
                                )

                    if (
                        tg_enabled
                        and tg_rows
                        and st.button(
                            "📨 Gửi các VB 1X2 trận này lên Telegram",
                            key=f"tg_send_match_{mid}",
                        )
                    ):
                        try:
                            out = send_telegram_value_bets(
                                tg_rows,
                                st.session_state.get("tg_token", ""),
                                st.session_state.get("tg_chat_id", ""),
                                bankroll=bankroll,
                                min_ev=min_ev,
                                markets=("1X2",),
                            )
                            st.success(f"Telegram: đã gửi {out['sent']} tin.")
                            for err in out.get("errors", [])[:3]:
                                st.error(err)
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"Telegram lỗi: {exc}")
                else:
                    st.warning(
                        f"Không có Value Bet — mọi cửa có EV < {min_ev:.0%} "
                        "(ngưỡng sidebar)."
                    )

                # ----- 4b) Chia sẻ Phân tích AI -----
                st.subheader("📤 Chia sẻ Phân tích AI")
                ko_vn = format_kickoff_vn(
                    row["Kickoff"]
                    if row is not None and "Kickoff" in row.index and pd.notna(row["Kickoff"])
                    else kickoff
                )
                top_bet = pick_top_bet_for_share(all_bets)
                share_url = build_share_url(
                    match_id=mid, base_url=_share_base_url()
                )
                top_sel = (
                    _selection_vi(str(top_bet.selection)) if top_bet is not None else None
                )
                share_text = build_share_text(
                    home=home,
                    away=away,
                    league=league_label(league),
                    kickoff_vn=ko_vn,
                    model_source=p_source,
                    p_home=float(probs_1x2["H"]),
                    p_draw=float(probs_1x2["D"]),
                    p_away=float(probs_1x2["A"]),
                    share_url=share_url,
                    top_selection=top_sel,
                    top_odds=(
                        float(top_bet.bookmaker_odds) if top_bet is not None else None
                    ),
                    top_ev=float(top_bet.ev) if top_bet is not None else None,
                    top_kelly_pct=(
                        float(top_bet.kelly_pct) if top_bet is not None else None
                    ),
                )
                st.caption(
                    "Bấm nút để copy tóm tắt + deep-link (Telegram / Zalo). "
                    f"Link: `{share_url}`"
                )
                _clipboard_copy_button(
                    share_text, key=f"share_copy_{mid}"
                )
                st.code(share_text, language=None)

        # ----- 5) Lịch thi đấu -----
        st.subheader("5. Lịch thi đấu sắp tới")
        preview = fx_view.copy()
        preview["Kickoff"] = pd.to_datetime(preview["Kickoff"]).dt.strftime(
            "%Y-%m-%d %H:%M"
        )
        cols = [
            c
            for c in (
                "Kickoff",
                "HomeTeam",
                "AwayTeam",
                "Round",
                "B365H",
                "B365D",
                "B365A",
                "OU_Line",
                "OddsOver",
                "AHh",
                "OddsProvider",
            )
            if c in preview.columns
        ]
        st.dataframe(
            _rename_fixture_cols(preview, cols),
            use_container_width=True,
            hide_index=True,
        )

# ===========================================================================
# Tab 2 — Quản lý ngân sách
# ===========================================================================

if (not is_lite) and main_nav == _NAV_BUDGET:
    st.header("Quản lý ngân sách")
    st.caption(
        f"Kelly {DEFAULT_KELLY_FRACTION:.0%} × full · trần stake {MAX_STAKE_PCT:.1%} "
        "bankroll → số tiền cược gợi ý trên thẻ Value Bet."
    )

    bankroll = st.number_input(
        "Bankroll (vốn)",
        min_value=0.0,
        value=float(st.session_state["bankroll"]),
        step=50.0,
        key="budget_bankroll_input",
    )
    st.session_state["bankroll"] = float(bankroll)

    _metrics_row(
        [
            ("Vốn hiện tại", f"{bankroll:,.2f}"),
            ("Kelly fraction", f"{kelly_frac:.0%}"),
            ("Ngưỡng EV", f"{min_ev:.0%}"),
        ],
        per_row=2,
    )

    st.info(
        "Stake gợi ý = bankroll × kelly_fraction × (EV / (Odds − 1)), "
        "đã nhân hệ số Fractional Kelly ở sidebar."
    )

    st.subheader("Quét value bet hàng loạt (có odds)")
    if fixtures.empty:
        st.warning("Chưa có fixtures.")
    else:
        family = st.selectbox(
            "Nguồn cột odds để quét",
            ["B365", "Avg", "Max"],
            index=0,
            key="scan_family",
        )
        if st.button("Quét Value Bet", type="primary", key="scan_btn"):
            with st.spinner("Đang quét…"):
                recs = recommend_upcoming(
                    dc_model,
                    fixtures.head(40),
                    odds_family=family,
                    min_ev=min_ev,
                    kelly_fraction=kelly_frac,
                    only_value=True,
                )
            if recs.empty:
                st.info("Không có value bet trên các trận có cột odds.")
            else:
                recs = recs.copy()
                kelly_vals = [
                    capped_kelly_fraction(
                        float(r["p_model"]),
                        float(r["bookmaker_odds"]),
                        kelly_fraction=kelly_frac,
                        max_stake_pct=MAX_STAKE_PCT,
                    )
                    for _, r in recs.iterrows()
                ]
                recs["kelly_fraction"] = kelly_vals
                recs["kelly_pct"] = recs["kelly_fraction"] * 100.0
                recs = select_value_bets(
                    recs,
                    max_per_day=MAX_BETS_PER_DAY,
                    already_today=0,
                    one_per_match=True,
                )
                if recs.empty:
                    st.info(
                        "Không còn kèo sau bộ lọc rủi ro "
                        f"(1/trận · Top {MAX_BETS_PER_DAY}/ngày · sanity)."
                    )
                else:
                    recs["stake"] = (recs["kelly_fraction"] * bankroll).round(2)
                    st.success(
                        f"Tìm thấy {len(recs)} lựa chọn value "
                        f"(≤ {MAX_BETS_PER_DAY}/ngày, cap stake {MAX_STAKE_PCT:.1%})"
                    )
                    display = recs[
                        [
                            "kickoff",
                            "home_team",
                            "away_team",
                            "market",
                            "selection",
                            "p_model",
                            "bookmaker_odds",
                            "ev_pct",
                            "kelly_pct",
                            "stake",
                        ]
                    ].copy()
                    display["kickoff"] = pd.to_datetime(display["kickoff"]).dt.strftime(
                        "%Y-%m-%d %H:%M"
                    )
                    display["market"] = display["market"].map(
                        lambda m: _MARKET_VI.get(m, m)
                    )
                    display["selection"] = display["selection"].map(_selection_vi)
                    display["p_model"] = display["p_model"].map(lambda x: f"{x:.1%}")
                    display["ev_pct"] = display["ev_pct"].map(lambda x: f"{x:+.1f}%")
                    display["kelly_pct"] = display["kelly_pct"].map(lambda x: f"{x:.2f}%")
                    display.columns = [
                        "Giờ đá",
                        "Chủ",
                        "Khách",
                        "Thị trường",
                        "Lựa chọn",
                        "P model",
                        "Odds",
                        "EV %",
                        "Kelly %",
                        "Tiền gợi ý",
                    ]
                    st.dataframe(display, use_container_width=True, hide_index=True)

# ===========================================================================
# Tab 2b — 📈 Nhật Ký Cược & Bankroll (Paper Trading)
# ===========================================================================

if (not is_lite) and main_nav == _NAV_JOURNAL:
    st.header("📈 Nhật Ký Cược & Bankroll")
    st.caption(
        "Paper trading: lệnh từ tab Chi tiết kèo · settle Thắng/Thua/Hòa · "
        "bankroll, ROI, hit rate & CLV trên EPL + UWCL."
    )

    init_br = float(
        st.session_state.get("bankroll_initial", st.session_state.get("bankroll", 1000.0))
    )
    new_init = st.number_input(
        "Bankroll gốc (paper)",
        min_value=0.0,
        value=float(init_br) if init_br > 0 else 1000.0,
        step=50.0,
        key="journal_init_bank",
        help="Mốc vốn ban đầu (mặc định $1000). Bankroll hiện tại = gốc + Σ PnL đã settle.",
    )
    st.session_state["bankroll_initial"] = float(new_init)

    # All-league journal (per-DB live_bets tagged with league)
    live_all = load_live_bets_all_leagues(("EPL", "UWCL"))
    if live_all.empty:
        live_all = pd.DataFrame(
            columns=[
                "id",
                "match_date",
                "home_team",
                "away_team",
                "market",
                "selection",
                "odds",
                "stake_amount",
                "status",
                "pnl",
                "closing_odds",
                "clv_pct",
                "settled_at",
                "league",
            ]
        )

    filt_c1, filt_c2, filt_c3 = st.columns([1, 1, 1])
    with filt_c1:
        league_filter = st.selectbox(
            "Giải",
            options=["Tất cả", "EPL", "UWCL"],
            index=0,
            key="journal_league_filter",
        )
    with filt_c2:
        result_filter = st.selectbox(
            "Kết quả",
            options=[
                "Tất cả",
                "Win / Thắng",
                "Loss / Thua",
                "Push / Hòa",
                "Pending / Đang chờ",
            ],
            index=0,
            key="journal_result_filter",
        )
    with filt_c3:
        if st.button("↻ Đồng bộ CLV (giải đang chọn)", key="journal_sync_clv"):
            try:
                sync_leagues = (
                    ("EPL", "UWCL")
                    if league_filter == "Tất cả"
                    else (league_filter,)
                )
                updated = skipped = 0
                for lg in sync_leagues:
                    hist = _load_raw_data(lg, N_SEASONS_BUFFER)
                    sync = sync_closing_odds_from_results(hist, league_db_path(lg))
                    updated += int(sync["updated"])
                    skipped += int(sync["skipped"])
                st.success(f"CLV sync: cập nhật {updated} · bỏ qua {skipped}")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"CLV sync lỗi: {exc}")

    filtered = live_all.copy()
    if league_filter != "Tất cả" and "league" in filtered.columns:
        filtered = filtered.loc[filtered["league"] == league_filter]
    _RESULT_STATUS = {
        "Win / Thắng": "WIN",
        "Loss / Thua": "LOSS",
        "Push / Hòa": "PUSH",
        "Pending / Đang chờ": "PENDING",
    }
    if result_filter in _RESULT_STATUS and not filtered.empty:
        filtered = filtered.loc[filtered["status"] == _RESULT_STATUS[result_filter]]

    # KPIs from league scope (ignore result filter so cards stay overall for that league)
    kpi_base = live_all
    if league_filter != "Tất cả" and not live_all.empty:
        kpi_base = live_all.loc[live_all["league"] == league_filter]
    summary_j = journal_summary_from_frame(kpi_base, float(new_init))
    # Sync session bankroll from active sidebar league (Kelly sizing elsewhere)
    st.session_state["bankroll"] = float(
        journal_bankroll_summary(float(new_init), active_db_path)["current_bankroll"]
    )

    avg_clv_pct = summary_j.get("avg_clv_pct", float("nan"))
    n_clv = int(summary_j.get("n_clv") or 0)
    clv_display = (
        f"{avg_clv_pct:+.2f}%"
        if avg_clv_pct == avg_clv_pct and n_clv > 0
        else "—"
    )
    clv_help = (
        f"CLV = (odds vào / odds đóng) − 1 · {n_clv} lệnh có closing"
        if n_clv > 0
        else "Chưa có closing_odds trên journal."
    )

    _metrics_row(
        [
            (
                "Current Bankroll ($)",
                f"${summary_j['current_bankroll']:,.2f}",
                {
                    "delta": f"PnL {summary_j['realised_pnl']:+,.2f}",
                    "delta_color": (
                        "normal" if summary_j["realised_pnl"] >= 0 else "inverse"
                    ),
                },
            ),
            (
                "Cumulative ROI (%)",
                f"{summary_j['roi_pct']:+.2f}%",
                {
                    "help": "ROI = Σ PnL / Σ stake đã settle.",
                },
            ),
            (
                "Hit Rate (%)",
                f"{summary_j['hit_rate'] * 100:.1f}%",
                {
                    "delta": (
                        f"{summary_j['wins']}W / {summary_j['losses']}L"
                        f" / {summary_j['pushes']}P"
                    ),
                    "delta_color": "off",
                },
            ),
            (
                "Avg CLV (%)",
                clv_display,
                {"help": clv_help},
            ),
        ],
        per_row=2,
    )

    # --- Charts: bankroll growth + daily PnL ---
    settled_for_charts = (
        kpi_base.loc[kpi_base["status"].isin(["WIN", "LOSS", "PUSH"])].copy()
        if not kpi_base.empty
        else kpi_base
    )
    chart_c1, chart_c2 = st.columns(2)
    with chart_c1:
        st.subheader("Bankroll growth")
        if settled_for_charts.empty:
            st.caption("Chưa có lệnh đã settle để vẽ equity.")
        else:
            chron = settled_for_charts.copy()
            if "settled_at" in chron.columns:
                chron["_ts"] = pd.to_datetime(chron["settled_at"], errors="coerce", utc=True)
            else:
                chron["_ts"] = pd.NaT
            chron = chron.sort_values(
                ["_ts", "id"], ascending=True, na_position="last"
            ).reset_index(drop=True)
            chron["session"] = chron.index + 1
            chron["cum_pnl"] = pd.to_numeric(chron["pnl"], errors="coerce").fillna(0).cumsum()
            chron["bankroll"] = float(new_init) + chron["cum_pnl"]
            # Seed start point at $initial
            seed = pd.DataFrame(
                {"session": [0], "bankroll": [float(new_init)], "cum_pnl": [0.0]}
            )
            plot_df = pd.concat(
                [seed, chron[["session", "bankroll", "cum_pnl"]]],
                ignore_index=True,
            )
            fig_eq = px.line(
                plot_df,
                x="session",
                y="bankroll",
                title=f"Bankroll theo phiên settle (start ${new_init:,.0f})",
                labels={"session": "Settlement session", "bankroll": "Bankroll ($)"},
            )
            fig_eq.update_layout(template="plotly_white")
            fig_eq.add_hline(
                y=float(new_init),
                line_dash="dot",
                line_color="#888",
                annotation_text=f"Start ${new_init:,.0f}",
            )
            _plotly(fig_eq, height=320)

    with chart_c2:
        st.subheader("Daily PnL")
        if settled_for_charts.empty:
            st.caption("Chưa có PnL theo ngày.")
        else:
            daily = settled_for_charts.copy()
            ts_src = (
                daily["settled_at"]
                if "settled_at" in daily.columns
                else daily.get("match_date")
            )
            ts = pd.to_datetime(ts_src, errors="coerce", utc=True)
            # Drop tz for calendar-day grouping (UTC day is fine for paper journal).
            daily["_day"] = ts.dt.strftime("%Y-%m-%d")
            daily["pnl"] = pd.to_numeric(daily["pnl"], errors="coerce").fillna(0.0)
            by_day = (
                daily.dropna(subset=["_day"])
                .groupby("_day", as_index=False)["pnl"]
                .sum()
                .rename(columns={"_day": "day"})
                .sort_values("day")
            )
            if by_day.empty:
                st.caption("Thiếu settled_at — chưa gom được Daily PnL.")
            else:
                fig_pnl = px.bar(
                    by_day,
                    x="day",
                    y="pnl",
                    title="PnL theo ngày settle",
                    labels={"day": "Ngày", "pnl": "PnL ($)"},
                    color="pnl",
                    color_continuous_scale=["#c0392b", "#ecf0f1", "#27ae60"],
                    color_continuous_midpoint=0,
                )
                fig_pnl.update_layout(template="plotly_white", coloraxis_showscale=False)
                _plotly(fig_pnl, height=320)

    # --- Filterable history + CSV ---
    st.subheader("Lịch sử paper bets")
    if filtered.empty:
        st.info("Không có lệnh khớp bộ lọc.")
    else:
        hist_cols = [
            c
            for c in (
                "id",
                "league",
                "match_date",
                "home_team",
                "away_team",
                "market",
                "selection",
                "odds",
                "closing_odds",
                "clv_pct",
                "stake_amount",
                "status",
                "pnl",
                "settled_at",
            )
            if c in filtered.columns
        ]
        show_h = filtered[hist_cols].copy()
        # Display-friendly copy
        display_h = show_h.copy()
        _status_vn = {
            "WIN": "Win / Thắng",
            "LOSS": "Loss / Thua",
            "PUSH": "Push / Hòa",
            "PENDING": "Pending / Đang chờ",
        }
        if "status" in display_h.columns:
            display_h["status"] = display_h["status"].map(
                lambda s: _status_vn.get(str(s), str(s))
            )
        if "odds" in display_h.columns:
            display_h["odds"] = display_h["odds"].map(
                lambda x: f"{float(x):.2f}" if pd.notna(x) else "—"
            )
        if "closing_odds" in display_h.columns:
            display_h["closing_odds"] = display_h["closing_odds"].map(
                lambda x: f"{float(x):.2f}" if pd.notna(x) else "—"
            )
        if "clv_pct" in display_h.columns:
            display_h["clv_pct"] = display_h["clv_pct"].map(
                lambda x: f"{float(x) * 100:+.2f}%" if pd.notna(x) else "—"
            )
        if "stake_amount" in display_h.columns:
            display_h["stake_amount"] = display_h["stake_amount"].map(
                lambda x: f"{float(x):,.2f}" if pd.notna(x) else "—"
            )
        if "pnl" in display_h.columns:
            display_h["pnl"] = display_h["pnl"].map(
                lambda x: f"{float(x):+,.2f}" if pd.notna(x) else "—"
            )
        st.dataframe(display_h, use_container_width=True, hide_index=True)

        export_df = show_h.copy()
        if "clv_pct" in export_df.columns:
            export_df["clv_pct"] = pd.to_numeric(export_df["clv_pct"], errors="coerce")
        csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 Export CSV",
            data=csv_bytes,
            file_name="paper_bets_journal.csv",
            mime="text/csv",
            key="journal_export_csv",
            use_container_width=True,
        )

    # --- Pending settle (active sidebar league DB) ---
    st.subheader("⏳ Đang chờ (PENDING) — giải sidebar")
    pending_df = load_live_bets(active_db_path, status="PENDING")
    if pending_df.empty:
        st.info(
            "Chưa có lệnh treo trên giải đang chọn. Vào tab **Chi tiết kèo** → "
            "**➕ Đặt cược vào Journal**."
        )
    else:
        for _, prow in pending_df.iterrows():
            bid = int(prow["id"])
            with st.container(border=True):
                st.markdown(
                    f"**#{bid}** · {league} · {prow.get('match_date') or '—'} · "
                    f"**{prow['home_team']}** vs **{prow['away_team']}** · "
                    f"{prow['selection']} @ **{float(prow['odds']):.2f}** · "
                    f"stake **{float(prow['stake_amount']):,.2f}**"
                )
                settle_cols = st.columns(2)
                with settle_cols[0]:
                    if st.button(
                        "✅ Thắng", key=f"settle_win_{bid}", use_container_width=True
                    ):
                        pnl = settle_live_bet(bid, "WIN", active_db_path)
                        st.session_state["bankroll"] = float(
                            journal_bankroll_summary(
                                float(st.session_state["bankroll_initial"]),
                                active_db_path,
                            )["current_bankroll"]
                        )
                        st.success(f"#{bid} WIN · PnL {pnl:+,.2f}")
                        st.rerun()
                    if st.button(
                        "↩ Hòa (Push)", key=f"settle_push_{bid}", use_container_width=True
                    ):
                        settle_live_bet(bid, "PUSH", active_db_path)
                        st.info(f"#{bid} PUSH · hoàn stake")
                        st.rerun()
                with settle_cols[1]:
                    if st.button(
                        "❌ Thua", key=f"settle_loss_{bid}", use_container_width=True
                    ):
                        pnl = settle_live_bet(bid, "LOSS", active_db_path)
                        st.session_state["bankroll"] = float(
                            journal_bankroll_summary(
                                float(st.session_state["bankroll_initial"]),
                                active_db_path,
                            )["current_bankroll"]
                        )
                        st.warning(f"#{bid} LOSS · PnL {pnl:+,.2f}")
                        st.rerun()
                    if st.button(
                        "🗑 Xóa", key=f"del_bet_{bid}", use_container_width=True
                    ):
                        delete_live_bet(bid, active_db_path)
                        st.rerun()
                with st.expander(f"Nhập closing odds #{bid}"):
                    c_odds = st.number_input(
                        "Closing odds",
                        min_value=1.01,
                        value=float(prow["odds"]),
                        step=0.01,
                        key=f"clv_input_{bid}",
                    )
                    if st.button("Lưu CLV", key=f"clv_save_{bid}"):
                        try:
                            clv = update_closing_odds(
                                bid, float(c_odds), active_db_path
                            )
                            st.success(f"CLV = {clv * 100:+.2f}%")
                            st.rerun()
                        except Exception as exc:  # noqa: BLE001
                            st.error(str(exc))

# ===========================================================================
# Tab 3 — Dự đoán & Lịch sử
# ===========================================================================

if (not is_lite) and main_nav == _NAV_HIST:
    st.header("Dự đoán & Lịch sử")

    st.subheader("Dữ liệu huấn luyện")
    st.caption(
        f"{len(data)} trận · {data['Date'].min().date()} → {data['Date'].max().date()}"
    )
    by_season = (
        data.groupby("Season")
        .size()
        .rename("Số trận")
        .reset_index()
        .rename(columns={"Season": "Mùa giải"})
    )
    st.dataframe(by_season, use_container_width=True, hide_index=True)

    n_hist = st.slider("Hiện N trận gần nhất", 10, 100, 30, 5, key="hist_n")
    hist = data.sort_values("Date", ascending=False).head(n_hist).copy()
    hist["Date"] = hist["Date"].dt.date
    cols = ["Date", "Season", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    if "B365H" in hist.columns:
        cols += ["B365H", "B365D", "B365A"]
    st.dataframe(
        hist[cols].rename(
            columns={
                "Date": "Ngày",
                "Season": "Mùa",
                "HomeTeam": "Chủ",
                "AwayTeam": "Khách",
                "FTHG": "Bàn chủ",
                "FTAG": "Bàn khách",
                "FTR": "KQ",
                "B365H": "Odds chủ",
                "B365D": "Odds hòa",
                "B365A": "Odds khách",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Walk-forward Backtest đầy đủ nằm ở tab **Backtest**.")

# ===========================================================================
# Tab 3b — Walk-forward Backtest
# ===========================================================================

if (not is_lite) and main_nav == _NAV_BACKTEST:
    st.header("Walk-forward Backtest")
    st.caption(
        "Fit mô hình chỉ trên trận **trước** ngày t · calibrate xác suất · "
        "lọc thị trường / odds / stake · đo ROI / drawdown / Brier."
    )

    _BT_MARKET_LABELS = {
        "1X2": "1X2",
        "Over/Under": "OU",
        "Asian Handicap": "AH",
    }
    bt_markets_ui = st.multiselect(
        "Chọn thị trường đặt cược",
        options=list(_BT_MARKET_LABELS.keys()),
        default=["1X2"],
        key="bt_markets_ui",
        help="Equity curve & bankroll chỉ tính trên các thị trường được chọn.",
    )
    bt_allowed = tuple(_BT_MARKET_LABELS[m] for m in bt_markets_ui if m in _BT_MARKET_LABELS)
    if not bt_allowed:
        st.warning("Hãy chọn ít nhất một thị trường.")
        bt_allowed = ("1X2",)

    ev_cols = st.columns(3)
    with ev_cols[0]:
        bt_ev_1x2 = (
            st.slider(
                "EV tối thiểu 1X2 (%)",
                0.0,
                20.0,
                float(min_ev * 100),
                1.0,
                key="bt_ev_1x2",
                disabled="1X2" not in bt_allowed,
            )
            / 100.0
        )
    with ev_cols[1]:
        bt_ev_ou = (
            st.slider(
                "EV tối thiểu Over/Under (%)",
                0.0,
                20.0,
                10.0,
                1.0,
                key="bt_ev_ou",
                disabled="OU" not in bt_allowed,
                help="OU thường nhiễu hơn — ngưỡng cao hơn mặc định 10%.",
            )
            / 100.0
        )
    with ev_cols[2]:
        bt_ev_ah = (
            st.slider(
                "EV tối thiểu Asian Handicap (%)",
                0.0,
                20.0,
                float(min_ev * 100),
                1.0,
                key="bt_ev_ah",
                disabled="AH" not in bt_allowed,
            )
            / 100.0
        )

    bt1, bt2, bt3 = st.columns(3)
    with bt1:
        bt_use_ensemble = st.checkbox(
            "Dùng Ensemble (Dixon–Coles + LightGBM)",
            value=False,
            key="bt_use_ensemble",
            help="Tắt = chỉ Dixon–Coles. Bật = blend 1X2 với LightGBM.",
        )
        bt_w_ml = (
            st.slider(
                "Trọng số LightGBM w (%)",
                0.0,
                100.0,
                float(DEFAULT_W_ML * 100),
                5.0,
                key="bt_w_ml",
                disabled=not bt_use_ensemble,
            )
            / 100.0
        )
    with bt2:
        bt_calib = st.selectbox(
            "Calibration",
            options=["isotonic", "platt", "none"],
            index=0,
            key="bt_calib",
            help="Isotonic / Platt hạ xác suất ảo ở cửa odds cao (longshot).",
        )
        bt_odds_lo, bt_odds_hi = st.slider(
            "Khoảng Odds cho phép",
            min_value=1.01,
            max_value=10.0,
            value=(1.40, 3.50),
            step=0.05,
            key="bt_odds_band",
        )
    with bt3:
        bt_max_stake = (
            st.slider(
                "Trần stake / trận (% vốn)",
                0.5,
                10.0,
                float(MAX_STAKE_PCT * 100),
                0.5,
                key="bt_max_stake",
                help="Kelly bị cắt trần — mặc định 1% bankroll (giảm drawdown).",
            )
            / 100.0
        )
        bt_refit = st.selectbox(
            "Refit mỗi N ngày đấu",
            options=[1, 3, 7, 14],
            index=2,
            key="bt_refit",
            help="1 = walk-forward chặt (chậm). 7 = nhanh, vẫn causal.",
        )

    bt_a, bt_b = st.columns(2)
    with bt_a:
        bt_min_train = st.number_input(
            "Số trận train tối thiểu",
            min_value=50,
            max_value=500,
            value=120,
            step=10,
            key="bt_min_train",
        )
    with bt_b:
        bt_bank = st.number_input(
            "Bankroll giả lập",
            min_value=100.0,
            value=float(st.session_state.get("bankroll", 1000.0)),
            step=100.0,
            key="bt_bank",
        )

    bt_c1, bt_c2, bt_c3 = st.columns(3)
    with bt_c1:
        bt_commission = (
            st.slider(
                "Commission trên lãi (%)",
                0.0,
                5.0,
                0.0,
                0.5,
                key="bt_commission",
                help="Trừ % trên PnL thắng (kiểu sàn exchange). 0 = không trừ.",
            )
            / 100.0
        )
    with bt_c2:
        bt_run_ablation = st.checkbox(
            "Ablation DC vs Ensemble",
            value=False,
            key="bt_run_ablation",
            help="Chạy lưới w_ML = 0 / 20 / 40 / 60% (chậm hơn).",
        )
    with bt_c3:
        st.caption("Odds vào = B365 closing · CLV so Avg closing.")

    st.caption(
        f"Đang chạy trên: **{', '.join(bt_allowed)}** · "
        f"EV 1X2≥{bt_ev_1x2:.0%} · OU≥{bt_ev_ou:.0%} · AH≥{bt_ev_ah:.0%} · "
        f"commission={bt_commission:.1%}"
    )

    run_bt = st.button("Chạy Walk-forward Backtest", type="primary", key="bt_run_full")

    if run_bt:
        full = _load_raw_data(league, N_SEASONS_BUFFER).sort_values("Date")
        cfg = BacktestConfig(
            min_ev=float(min_ev),
            kelly_fraction=float(kelly_frac),
            initial_bankroll=float(bt_bank),
            min_train_matches=int(bt_min_train),
            xi=float(xi),
            odds_family="B365",
            use_ml=bool(bt_use_ensemble),
            ensemble_w_ml=float(bt_w_ml),
            allowed_markets=bt_allowed,
            min_ev_1x2=float(bt_ev_1x2),
            min_ev_ou=float(bt_ev_ou),
            min_ev_ah=float(bt_ev_ah),
            persist=True,
            replace_history=True,
            refit_every=int(bt_refit),
            calibration=str(bt_calib),  # type: ignore[arg-type]
            min_cal_samples=80,
            min_odds=float(bt_odds_lo),
            max_odds=float(bt_odds_hi),
            max_stake_pct=float(bt_max_stake),
            closing_ref_family="Avg",
            commission_pct=float(bt_commission),
            adapt_w_ml=True,
        )
        progress = st.progress(0.0, text="Đang chạy backtest…")
        status = st.empty()

        def _ui_progress(step: int, total: int, msg: str) -> None:
            progress.progress(min(1.0, step / max(total, 1)), text=msg)
            status.caption(msg)

        try:
            with st.spinner("Walk-forward đang chạy (có thể mất vài phút)…"):
                result = run_backtest(full, config=cfg, progress=_ui_progress)
                if bt_run_ablation:
                    status.caption("Đang chạy ablation DC vs Ensemble…")
                    abl = run_ablation(
                        full,
                        base_config=cfg,
                        w_ml_grid=(0.0, 0.2, 0.4, 0.6),
                        progress=_ui_progress,
                    )
                    st.session_state["bt_last_ablation"] = abl
            progress.progress(1.0, text="Hoàn tất")
            st.session_state["bt_last_result"] = result
            clv_pct = result.summary.get("avg_clv_pct")
            clv_txt = (
                f" · CLV={clv_pct:+.2f}%"
                if clv_pct is not None and clv_pct == clv_pct
                else ""
            )
            st.success(
                f"Xong · {result.summary['total_bets']} cược · "
                f"markets={list(bt_allowed)} · "
                f"model={'Ensemble' if bt_use_ensemble else 'Dixon–Coles'} · "
                f"calib={bt_calib}{clv_txt}"
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Backtest lỗi: {exc}")

    result = st.session_state.get("bt_last_result")
    if result is not None:
        summary = result.summary
        brier = summary.get("brier_score")
        _metrics_row(
            [
                ("Số cược", f"{summary['total_bets']}"),
                ("Hit rate", f"{summary['hit_rate_pct']:.1f}%"),
                ("ROI", f"{summary['roi_pct']:+.1f}%"),
                ("Max DD", f"{summary['max_drawdown_pct']:.1f}%"),
                (
                    "Brier",
                    f"{brier:.3f}" if brier == brier else "n/a",
                ),
                (
                    "CLV TB",
                    (
                        f"{summary['avg_clv_pct']:+.2f}%"
                        if summary.get("avg_clv_pct") == summary.get("avg_clv_pct")
                        else "n/a"
                    ),
                ),
            ],
            per_row=2,
        )

        st.markdown("##### Tóm tắt")
        st.code(format_backtest_report(summary), language="text")

        # Calibration report + reliability diagram
        st.subheader("Calibration (độ tin cậy xác suất)")
        cal = getattr(result, "calibration", None) or summary.get("calibration") or {}
        overall = cal.get("overall") or {}
        if overall:
            _metrics_row(
                [
                    ("N preds", f"{overall.get('n', 0)}"),
                    (
                        "Brier preds",
                        f"{overall['brier']:.4f}"
                        if overall.get("brier") == overall.get("brier")
                        else "n/a",
                    ),
                    (
                        "Log-loss",
                        f"{overall['log_loss']:.4f}"
                        if overall.get("log_loss") == overall.get("log_loss")
                        else "n/a",
                    ),
                    (
                        "ECE",
                        f"{overall['ece']:.4f}"
                        if overall.get("ece") == overall.get("ece")
                        else "n/a",
                    ),
                ],
                per_row=2,
            )
        by_m_cal = cal.get("by_market")
        if isinstance(by_m_cal, pd.DataFrame) and not by_m_cal.empty:
            st.markdown("**Theo thị trường (1X2 / OU / AH)**")
            st.dataframe(by_m_cal, use_container_width=True, hide_index=True)
        by_s_cal = cal.get("by_season")
        if isinstance(by_s_cal, pd.DataFrame) and not by_s_cal.empty:
            st.markdown("**Theo mùa**")
            st.dataframe(by_s_cal, use_container_width=True, hide_index=True)
        rel = cal.get("reliability")
        if isinstance(rel, pd.DataFrame) and not rel.empty:
            rel_plot = rel.dropna(subset=["p_mean", "y_freq"])
            if not rel_plot.empty:
                fig_rel = go.Figure()
                fig_rel.add_trace(
                    go.Scatter(
                        x=rel_plot["p_mean"],
                        y=rel_plot["y_freq"],
                        mode="markers+lines",
                        name="Model",
                        marker=dict(size=10),
                    )
                )
                fig_rel.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        name="Perfect",
                        line=dict(dash="dash", color="#888"),
                    )
                )
                fig_rel.update_layout(
                    title="Reliability diagram",
                    xaxis_title="P dự đoán (bin mean)",
                    yaxis_title="Tần suất thực",
                    height=360,
                    template="plotly_white",
                    margin=dict(l=8, r=8, t=48, b=32),
                )
                _plotly(fig_rel, height=360)

        abl = st.session_state.get("bt_last_ablation")
        if isinstance(abl, pd.DataFrame) and not abl.empty:
            st.subheader("Ablation: Dixon–Coles vs Ensemble")
            st.dataframe(abl, use_container_width=True, hide_index=True)
            if "roi_pct" in abl.columns and "w_ml" in abl.columns:
                fig_ab = px.line(
                    abl,
                    x="w_ml",
                    y="roi_pct",
                    markers=True,
                    title="ROI theo w_ML (0 = DC thuần)",
                    labels={"w_ml": "w_ML", "roi_pct": "ROI %"},
                )
                fig_ab.update_layout(
                    height=320,
                    template="plotly_white",
                    margin=dict(l=8, r=8, t=48, b=32),
                )
                _plotly(fig_ab, height=320)

        # Market breakdown
        st.subheader("Phân tích theo thị trường")
        mb = summary.get("market_breakdown")
        if mb is None or (isinstance(mb, pd.DataFrame) and mb.empty):
            mb = market_breakdown(result.bets)
        show_mb = mb.copy()
        show_mb["hit_rate_pct"] = show_mb["hit_rate_pct"].map(lambda x: f"{x:.1f}%")
        show_mb["roi_pct"] = show_mb["roi_pct"].map(lambda x: f"{x:+.1f}%")
        show_mb["total_pnl"] = show_mb["total_pnl"].map(lambda x: f"{x:+.2f}")
        show_mb["avg_odds"] = show_mb["avg_odds"].map(
            lambda x: f"{x:.2f}" if x == x else "—"
        )
        show_mb["brier_score"] = show_mb["brier_score"].map(
            lambda x: f"{x:.3f}" if x == x else "—"
        )
        show_mb = show_mb.rename(
            columns={
                "market": "Thị trường",
                "n_bets": "Số cược",
                "wins": "Thắng",
                "losses": "Thua",
                "hit_rate_pct": "Hit %",
                "total_pnl": "PnL",
                "total_staked": "Tổng stake",
                "roi_pct": "ROI %",
                "avg_odds": "Odds TB",
                "brier_score": "Brier",
            }
        )
        st.dataframe(show_mb, use_container_width=True, hide_index=True)

        fig_roi = px.bar(
            mb,
            x="market",
            y="roi_pct",
            color="market",
            text=mb["roi_pct"].map(lambda x: f"{x:+.1f}%"),
            title="ROI (%) theo thị trường",
            labels={"market": "Thị trường", "roi_pct": "ROI %"},
        )
        fig_roi.update_traces(textposition="outside")
        fig_roi.update_layout(
            showlegend=False,
            height=340,
            margin=dict(l=8, r=8, t=48, b=32),
            template="plotly_white",
            yaxis_title="ROI %",
            autosize=True,
        )
        _plotly(fig_roi, height=340)

        # Equity curve
        st.subheader("Đường cong ngân sách (Equity)")
        eq = result.equity_curve
        if not eq.empty:
            fig_eq = px.line(
                eq,
                x="date",
                y="bankroll",
                title="Bankroll theo thời gian",
                labels={"date": "Ngày", "bankroll": "Bankroll"},
            )
            fig_eq.update_layout(
                height=340,
                template="plotly_white",
                margin=dict(l=8, r=8, t=48, b=32),
                autosize=True,
            )
            _plotly(fig_eq, height=340)

        with st.expander("Chi tiết các lệnh cược"):
            if result.bets.empty:
                st.info("Không có lệnh nào vượt bộ lọc an toàn.")
            else:
                detail = result.bets.copy()
                detail["match_date"] = pd.to_datetime(detail["match_date"]).dt.date
                detail["p_model"] = detail["p_model"].map(lambda x: f"{x:.1%}")
                detail["ev"] = detail["ev"].map(lambda x: f"{x:+.1%}")
                detail["stake_pct"] = detail["stake_pct"].map(lambda x: f"{x:.2%}")
                detail["pnl"] = detail["pnl"].map(lambda x: f"{x:+.2f}")
                st.dataframe(detail, use_container_width=True, hide_index=True)
    else:
        st.info("Chọn tham số rồi bấm **Chạy Walk-forward Backtest**.")

# ===========================================================================
# Tab 4 — Phân tích Phạt Góc (CornerPredictor)
# ===========================================================================

if (not is_lite) and main_nav == _NAV_CORNERS:
    st.header("Phân tích Phạt Góc")
    if corner_model is None:
        st.warning(
            "Chưa fit được CornerPredictor"
            + (f": {corner_error}" if corner_error else "")
            + ". UWCL từ Fotmob hiện chỉ có tỷ số — chưa có HC/AC."
        )
        st.stop()
    summary_c = corner_model.summary()
    st.caption(
        f"CornerPredictor · backend=`{summary_c.get('backend')}` · "
        f"league avg HC/AC = {summary_c.get('league_avg_hc', 0):.1f} / "
        f"{summary_c.get('league_avg_ac', 0):.1f}"
    )

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        c_home = st.selectbox("Chủ nhà", teams, key="corner_home")
    with cc2:
        c_away_opts = [t for t in teams if t != c_home] or teams
        c_away = st.selectbox("Khách", c_away_opts, key="corner_away")
    with cc3:
        corner_line = st.selectbox(
            "Mốc tổng góc",
            [9.5, 10.5, 11.5, 8.5, 12.5],
            index=1,
            key="corner_line_sel",
        )

    try:
        exp = corner_model.expected_corners(c_home, c_away)
        ou = corner_model.predict_over_under(c_home, c_away, line=float(corner_line))
        pmf = corner_model.predict_total_pmf(c_home, c_away)
        lines_df = corner_model.predict_lines(c_home, c_away)
        ah = corner_model.predict_handicap(c_home, c_away, handicap=-0.5)
    except Exception as exc:  # noqa: BLE001
        st.warning(str(exc))
    else:
        _metrics_row(
            [
                ("E[Góc chủ] C_home", f"{exp['hc']:.1f}"),
                ("E[Góc khách] C_away", f"{exp['ac']:.1f}"),
                ("E[Tổng góc]", f"{exp['total']:.1f}"),
                (
                    f"Tài/Xỉu {float(corner_line):g}",
                    f"{ou['over']:.0%} / {ou['under']:.0%}",
                ),
            ],
            per_row=2,
        )

        st.subheader("Xác suất Over/Under theo mốc")
        show_lines = lines_df.copy()
        show_lines["over"] = show_lines["over"].map(lambda x: f"{x:.1%}")
        show_lines["under"] = show_lines["under"].map(lambda x: f"{x:.1%}")
        show_lines["expected_total"] = show_lines["expected_total"].map(
            lambda x: f"{x:.2f}"
        )
        st.dataframe(
            show_lines[["line", "over", "under", "expected_total"]].rename(
                columns={
                    "line": "Mốc",
                    "over": "P(Tài)",
                    "under": "P(Xỉu)",
                    "expected_total": "E[Total]",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            f"Corner Handicap −0.5 · P(Home)={ah['home']:.1%} · P(Away)={ah['away']:.1%} "
            f"(E[diff]={ah['expected_diff']:+.2f})"
        )

        fig_c = go.Figure(
            data=[go.Bar(x=list(range(len(pmf))), y=pmf, marker_color="#3498db")]
        )
        fig_c.add_vline(x=float(corner_line), line_dash="dash", line_color="#e67e22")
        fig_c.update_layout(
            title=f"{c_home} vs {c_away} — phân phối tổng phạt góc",
            xaxis_title="Tổng số phạt góc",
            yaxis_title="Xác suất",
            height=340,
            margin=dict(l=8, r=8, t=40, b=32),
            template="plotly_white",
            autosize=True,
        )
        _plotly(fig_c, height=340)

        # ----- Trend charts (last 10) -----
        st.subheader("Xu hướng phạt góc (10 trận gần nhất)")
        t1, t2 = st.columns(2)
        for col, team_name, color in (
            (t1, c_home, "#2ecc71"),
            (t2, c_away, "#e74c3c"),
        ):
            with col:
                trend = corner_model.trend(team_name, last_n=10)
                if trend.empty:
                    st.info(f"Chưa đủ dữ liệu góc cho {team_name}.")
                    continue
                fig_tr = go.Figure()
                fig_tr.add_trace(
                    go.Scatter(
                        x=trend["Date"],
                        y=trend["corners_for"],
                        mode="lines+markers",
                        name="Góc tạo",
                        line=dict(color=color),
                    )
                )
                fig_tr.add_trace(
                    go.Scatter(
                        x=trend["Date"],
                        y=trend["corners_against"],
                        mode="lines+markers",
                        name="Góc thủng",
                        line=dict(color="#95a5a6", dash="dot"),
                    )
                )
                fig_tr.update_layout(
                    title=team_name,
                    height=280,
                    margin=dict(l=8, r=8, t=40, b=24),
                    legend=dict(orientation="h", y=1.12),
                    template="plotly_white",
                    autosize=True,
                )
                _plotly(fig_tr, height=280)

        # ----- EV sandbox -----
        st.subheader("Tính EV kèo góc (nhập odds nhà cái)")
        e1, e2, e3 = st.columns(3)
        with e1:
            odds_cov = st.number_input("Odds Tài", min_value=1.01, value=1.90, step=0.05, key="c_od_over")
        with e2:
            odds_cun = st.number_input("Odds Xỉu", min_value=1.01, value=1.90, step=0.05, key="c_od_under")
        with e3:
            ev_line = st.number_input("Mốc EV", min_value=0.5, max_value=20.5, value=float(corner_line), step=0.5, key="c_ev_line")
        legs = corner_model.predict_corner_ev(
            c_home,
            c_away,
            {"over": float(odds_cov), "under": float(odds_cun)},
            line=float(ev_line),
            min_ev=-1.0,
        )
        if legs:
            show_ev = pd.DataFrame(legs)[
                ["selection", "p_model", "odds", "ev_pct"]
            ].copy()
            show_ev["p_model"] = show_ev["p_model"].map(lambda x: f"{x:.1%}")
            show_ev["odds"] = show_ev["odds"].map(lambda x: f"{x:.2f}")
            show_ev["ev_pct"] = show_ev["ev_pct"].map(lambda x: f"{x:+.1f}%")
            st.dataframe(
                show_ev.rename(
                    columns={
                        "selection": "Cửa",
                        "p_model": "P mô hình",
                        "odds": "Odds",
                        "ev_pct": "EV %",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
            best = legs[0]
            if best["ev"] >= min_ev:
                st.success(
                    f"Value: **{best['selection']}** @ {best['odds']:.2f} · "
                    f"EV {best['ev_pct']:+.1f}% (≥ ngưỡng {min_ev:.0%})"
                )
            else:
                st.info("Chưa có cửa góc đạt ngưỡng EV sidebar.")

# ===========================================================================
# Tab 5 — So sánh đội
# ===========================================================================

if (not is_lite) and main_nav == _NAV_COMPARE:
    st.header("So sánh đội")
    st.caption("Attack (α) cao = tấn công mạnh · Defence (δ) cao = phòng ngự tốt.")

    strengths = dc_model.team_strengths().rename(
        columns={
            "Team": "Đội",
            "Attack": "Tấn công (α)",
            "Defence": "Phòng ngự (δ)",
            "Net": "Điểm ròng",
        }
    )
    st.dataframe(strengths.round(3), use_container_width=True, hide_index=True)

    fig_s = px.scatter(
        strengths,
        x="Phòng ngự (δ)",
        y="Tấn công (α)",
        text="Đội",
        color="Điểm ròng",
        color_continuous_scale="RdYlGn",
        title="Tấn công vs Phòng ngự",
    )
    fig_s.update_traces(textposition="top center", marker=dict(size=12))
    fig_s.update_layout(
        height=480,
        margin=dict(l=8, r=8, t=40, b=32),
        template="plotly_white",
        autosize=True,
    )
    _plotly(fig_s, height=480)

    st.subheader("So sánh cặp đấu nhanh")
    s1, s2 = st.columns(2)
    with s1:
        h = st.selectbox("Đội nhà", teams, key="cmp_home")
    with s2:
        a_opts = [t for t in teams if t != h] or teams
        a = st.selectbox("Đội khách", a_opts, key="cmp_away")
    try:
        lam, mu = dc_model.expected_goals(h, a)
        p_dc = dc_model.predict_match_probs(h, a)
        p_ml = None
        if ml_model is not None:
            try:
                p_ml = ml_model.predict_proba(h, a)
            except Exception:  # noqa: BLE001
                p_ml = None
        p_ens = _blend_1x2(p_dc, p_ml, w_ml) if p_ml is not None else p_dc
    except Exception as exc:  # noqa: BLE001
        st.warning(str(exc))
    else:
        _metrics_row(
            [
                ("xG chủ (DC)", f"{lam:.2f}"),
                ("xG khách (DC)", f"{mu:.2f}"),
                (
                    "P(Chủ) Ensemble" if p_ml is not None else "P(Chủ) DC",
                    f"{p_ens['H']:.1%}",
                ),
                (
                    "P(Hòa) / P(Khách)",
                    f"{p_ens['D']:.1%} / {p_ens['A']:.1%}",
                ),
            ],
            per_row=2,
        )
        if p_ml is not None:
            cmp_models = pd.DataFrame(
                [
                    {
                        "Mô hình": "Dixon–Coles",
                        "P(Chủ)": f"{p_dc['H']:.1%}",
                        "P(Hòa)": f"{p_dc['D']:.1%}",
                        "P(Khách)": f"{p_dc['A']:.1%}",
                    },
                    {
                        "Mô hình": "LightGBM",
                        "P(Chủ)": f"{p_ml['H']:.1%}",
                        "P(Hòa)": f"{p_ml['D']:.1%}",
                        "P(Khách)": f"{p_ml['A']:.1%}",
                    },
                    {
                        "Mô hình": f"Ensemble w={w_ml:.0%}",
                        "P(Chủ)": f"{p_ens['H']:.1%}",
                        "P(Hòa)": f"{p_ens['D']:.1%}",
                        "P(Khách)": f"{p_ens['A']:.1%}",
                    },
                ]
            )
            st.dataframe(cmp_models, use_container_width=True, hide_index=True)
            with st.expander("Feature importance LightGBM (top 10)"):
                st.dataframe(
                    ml_model.feature_importance().head(10),
                    use_container_width=True,
                    hide_index=True,
                )
