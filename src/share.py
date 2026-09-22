"""Share URL / deep-link helpers for Streamlit match detail.

Pure functions for ``match_id`` encode/decode, query-param parsing, and
Telegram/Zalo paste text. Kept free of Streamlit imports so unit tests stay light.
"""

from __future__ import annotations

import re
from typing import Any, Mapping
from urllib.parse import quote

import pandas as pd

_VS_SPLIT = re.compile(r"_vs_", re.IGNORECASE)


def normalize_team_token(name: str) -> str:
    """Normalize a team name for stable ids / slugs (spaces → ``_``)."""
    return str(name).strip().replace(" ", "_").replace("'", "")


def fallback_match_id(home: str, away: str) -> str:
    """Fallback ``match_id`` when Flashscore event id is absent: ``Home__Away``."""
    return f"{normalize_team_token(home)}__{normalize_team_token(away)}"


def match_slug_from_teams(home: str, away: str) -> str:
    """Human-readable fallback query value: ``Home_vs_Away``."""
    return f"{normalize_team_token(home)}_vs_{normalize_team_token(away)}"


def match_id_for_fixture(
    row: pd.Series | Mapping[str, Any] | None,
    home: str,
    away: str,
) -> str:
    """Stable id: FlashscoreEventId when present, else ``Home__Away``."""
    if row is None:
        return fallback_match_id(home, away)
    if isinstance(row, pd.Series):
        if "FlashscoreEventId" in row.index and pd.notna(row.get("FlashscoreEventId")):
            return str(row["FlashscoreEventId"])
    elif isinstance(row, Mapping):
        mid = row.get("FlashscoreEventId")
        if mid is not None and not (isinstance(mid, float) and pd.isna(mid)):
            return str(mid)
    return fallback_match_id(home, away)


def _first_param(value: Any) -> str | None:
    """Normalize Streamlit query-param value (str or list) to a single string."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    text = str(value).strip()
    return text or None


def parse_match_query(params: Mapping[str, Any] | None) -> tuple[str | None, str | None]:
    """Extract ``(match_id, match_slug)`` from URL query params.

    Supports ``match_id`` and optional fallback ``match=Home_vs_Away``.
    """
    if not params:
        return None, None
    mid = _first_param(params.get("match_id"))
    slug = _first_param(params.get("match"))
    return mid, slug


def parse_home_vs_away(slug: str) -> tuple[str, str] | None:
    """Parse ``Home_vs_Away`` into ``(home, away)`` with underscores → spaces."""
    text = str(slug).strip()
    if not text:
        return None
    parts = _VS_SPLIT.split(text, maxsplit=1)
    if len(parts) != 2:
        return None
    home = parts[0].replace("_", " ").strip()
    away = parts[1].replace("_", " ").strip()
    if not home or not away:
        return None
    return home, away


def _norm_name(name: str) -> str:
    return " ".join(str(name).replace("_", " ").split()).casefold()


def find_fixture_for_share(
    fixtures: pd.DataFrame | None,
    *,
    match_id: str | None = None,
    match_slug: str | None = None,
) -> dict[str, Any] | None:
    """Locate a fixture row matching ``match_id`` or ``match`` slug.

    Returns ``{home, away, match_id, kickoff, row}`` or ``None`` if invalid/missing.
    """
    if fixtures is None or getattr(fixtures, "empty", True):
        return None
    if not match_id and not match_slug:
        return None

    slug_pair = parse_home_vs_away(match_slug) if match_slug else None
    want_id = str(match_id).strip() if match_id else None

    has_fs_col = "FlashscoreEventId" in getattr(fixtures, "columns", [])

    for _, row in fixtures.iterrows():
        home = str(row.get("HomeTeam", ""))
        away = str(row.get("AwayTeam", ""))
        mid = match_id_for_fixture(row, home, away)
        raw_fs = row.get("FlashscoreEventId") if has_fs_col else None
        fs_ok = raw_fs is not None and pd.notna(raw_fs) and str(raw_fs).strip() != ""
        if want_id and (
            mid == want_id
            or (fs_ok and str(raw_fs).strip() == want_id)
            or fallback_match_id(home, away) == want_id
            or match_slug_from_teams(home, away) == want_id
        ):
            return {
                "home": home,
                "away": away,
                "match_id": mid,
                "kickoff": row.get("Kickoff"),
                "row": row,
            }
        if slug_pair is not None:
            sh, sa = slug_pair
            if _norm_name(home) == _norm_name(sh) and _norm_name(away) == _norm_name(sa):
                return {
                    "home": home,
                    "away": away,
                    "match_id": mid,
                    "kickoff": row.get("Kickoff"),
                    "row": row,
                }
    return None


def build_share_url(*, match_id: str, base_url: str | None = None) -> str:
    """Build a deep-link URL (absolute if ``base_url`` given, else ``?match_id=…``)."""
    q = f"match_id={quote(str(match_id), safe='')}"
    base = (base_url or "").strip().rstrip("/")
    if not base:
        return f"?{q}"
    # Drop existing match_id / match query fragments for a clean link
    if "?" in base:
        path, _, query = base.partition("?")
        keep: list[str] = []
        for part in query.split("&"):
            key = part.split("=", 1)[0].lower()
            if key in ("match_id", "match", ""):
                continue
            keep.append(part)
        base = path if not keep else f"{path}?{'&'.join(keep)}"
    sep = "&" if "?" in base else "?"
    return f"{base}{sep}{q}"


def build_share_text(
    *,
    home: str,
    away: str,
    league: str,
    kickoff_vn: str,
    model_source: str,
    p_home: float,
    p_draw: float,
    p_away: float,
    share_url: str,
    top_selection: str | None = None,
    top_odds: float | None = None,
    top_ev: float | None = None,
    top_kelly_pct: float | None = None,
) -> str:
    """Ready-to-paste Telegram/Zalo analysis blurb.

    ``top_ev`` is a fraction (e.g. ``0.08`` → ``+8.0%``). ``top_kelly_pct`` is
    already in percent points (e.g. ``1.25`` → ``1.25%``).
    """
    lines = [
        f"⚽ {home} vs {away}",
        f"🏆 {league} · 🕒 {kickoff_vn}",
        f"📊 Model: {model_source}",
        f"1X2: H {p_home:.0%} · D {p_draw:.0%} · A {p_away:.0%}",
    ]
    if (
        top_selection
        and top_odds is not None
        and top_ev is not None
        and top_kelly_pct is not None
    ):
        lines.append(
            f"💎 Top pick: {top_selection} @ {float(top_odds):.2f} · "
            f"EV {float(top_ev):+.1%} · Kelly {float(top_kelly_pct):.2f}%"
        )
    else:
        lines.append("💎 Top pick: (chưa có value bet đạt ngưỡng)")
    lines.append(f"🔗 {share_url}")
    lines.append("⚠️ Nghiên cứu / không đảm bảo thắng cược")
    return "\n".join(lines)


def pick_top_bet_for_share(
    bets: list[Any] | None,
    *,
    prefer_recommended: bool = True,
) -> Any | None:
    """Choose the highest-EV bet (recommended first) for the share card."""
    if not bets:
        return None
    pool = list(bets)
    if prefer_recommended:
        rec = [b for b in pool if getattr(b, "recommended", False)]
        if rec:
            pool = rec
    return max(pool, key=lambda b: float(getattr(b, "ev", float("-inf"))))
