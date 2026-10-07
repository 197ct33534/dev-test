"""Configuration-driven league registry (``config/leagues.json``).

JSON is the preferred source of truth for multi-league support. Hardcoded
``LEAGUE_CONFIG`` in :mod:`src.data_loader` remains a fallback for EPL/UWCL
when the JSON file is missing or incomplete.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEAGUES_JSON_PATH = PROJECT_ROOT / "config" / "leagues.json"

# Keys that are documentation / non-league entries in leagues.json.
_META_KEYS = frozenset({"_documentation", "_comment", "documentation"})


def _as_path(raw: str | Path | None) -> Path | None:
    if raw is None or raw == "":
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


@lru_cache(maxsize=1)
def load_leagues_json(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Load ``config/leagues.json`` → ``{LEAGUE_KEY: config}`` (no meta keys)."""
    json_path = Path(path) if path is not None else LEAGUES_JSON_PATH
    if not json_path.is_file():
        return {}
    with json_path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"leagues.json must be an object, got {type(raw).__name__}")
    out: dict[str, dict[str, Any]] = {}
    for key, value in raw.items():
        if key in _META_KEYS or str(key).startswith("_"):
            continue
        if not isinstance(value, dict):
            continue
        out[str(key).strip().upper()] = dict(value)
    return out


def clear_leagues_cache() -> None:
    """Drop cached JSON (tests / hot-reload)."""
    load_leagues_json.cache_clear()


def get_league_entry(league_key: str) -> dict[str, Any] | None:
    """Return one league block from JSON, or ``None`` if absent."""
    key = str(league_key or "").strip().upper()
    return load_leagues_json().get(key)


def league_config_from_json(league_key: str) -> dict[str, object] | None:
    """Map a JSON league block → ``LEAGUE_CONFIG``-compatible dict.

    Returns ``None`` when the key is not in ``leagues.json``.
    """
    entry = get_league_entry(league_key)
    if entry is None:
        return None

    code = str(league_key).strip().upper()
    name = str(entry.get("name") or code)
    path = entry.get("flashscore_path") or ""
    fixtures_url = ""
    if path:
        fixtures_url = f"https://www.flashscore.com{path.rstrip('/')}/fixtures/"

    history = entry.get("history_source")
    if not history:
        history = "football-data" if entry.get("fd_div") else "flashscore"

    db_raw = entry.get("db_path")
    db_path = _as_path(db_raw) if db_raw else (PROJECT_ROOT / "data" / f"{code.lower()}_matches.db")

    cfg: dict[str, object] = {
        "label": name,
        "short": code,
        "db_path": db_path,
        "fotmob_id": entry.get("fotmob_id"),
        "flashscore_id": entry.get("flashscore_id"),
        "flashscore_path": path or None,
        "flashscore_fixtures_url": fixtures_url or None,
        "fd_div": entry.get("fd_div"),
        "history_source": str(history),
        "telegram_tag": str(entry.get("telegram_tag") or code),
        "time_zone": entry.get("time_zone"),
        "league_weight": float(entry.get("league_weight") or 1.0),
        "is_cup": bool(entry.get("is_cup") or False),
        "country": entry.get("country"),
        "seasons": dict(entry.get("seasons") or {}),
        "team_aliases": dict(entry.get("team_aliases") or {}),
    }
    return cfg


# Built-in cup codes when JSON omits ``is_cup`` (team-feed / global_db comps).
_DEFAULT_CUP_COMPS: frozenset[str] = frozenset(
    {
        "EMPERORS_CUP",
        "J_LEAGUE_CUP",
        "FA_CUP",
        "EFL_CUP",
        "COPA_DEL_REY",
    }
)


def is_cup_competition(comp_id: str | None) -> bool:
    """True when ``comp_id`` is a national / knockout cup.

    Prefers ``is_cup`` from ``config/leagues.json``; falls back to a small
    built-in set (``EMPERORS_CUP``, ``J_LEAGUE_CUP``, …).
    """
    code = str(comp_id or "").strip().upper()
    if not code:
        return False
    entry = get_league_entry(code)
    if entry is not None and "is_cup" in entry:
        return bool(entry.get("is_cup"))
    # Merged config may carry is_cup from JSON overlay.
    try:
        _c, cfg = resolve_league_config(code)
        if "is_cup" in cfg:
            return bool(cfg.get("is_cup"))
    except ValueError:
        pass
    return code in _DEFAULT_CUP_COMPS


def merged_league_config() -> dict[str, dict[str, object]]:
    """JSON leagues overlay hardcoded ``LEAGUE_CONFIG`` (JSON wins on conflict)."""
    from src.data_loader import LEAGUE_CONFIG

    out: dict[str, dict[str, object]] = {
        str(k).upper(): dict(v) for k, v in LEAGUE_CONFIG.items()
    }
    for key, _entry in load_leagues_json().items():
        mapped = league_config_from_json(key)
        if mapped is None:
            continue
        if key in out:
            # Preserve hardcoded URLs / paths when JSON omits them.
            base = dict(out[key])
            for mk, mv in mapped.items():
                if mv is None or mv == "" or mv == {}:
                    continue
                base[mk] = mv
            out[key] = base
        else:
            out[key] = mapped
    return out


def resolve_league_config(league: str) -> tuple[str, dict[str, object]]:
    """Return ``(canonical_code, config)`` from JSON ∪ ``LEAGUE_CONFIG``.

    Raises
    ------
    ValueError
        Unknown league key.
    """
    # Lazy import aliases from data_loader after JSON merge awareness.
    from src.data_loader import LEAGUE_CONFIG

    raw = str(league or "EPL").strip().upper()
    aliases = {
        "EPL": "EPL",
        "PL": "EPL",
        "PREMIER": "EPL",
        "PREMIER LEAGUE": "EPL",
        "E0": "EPL",
        "UWCL": "UWCL",
        "WCL": "UWCL",
        "LALIGA": "LALIGA",
        "LA LIGA": "LALIGA",
        "LA_LIGA": "LALIGA",
        "SPAIN": "LALIGA",
        "SP1": "LALIGA",
        "EMPERORS_CUP": "EMPERORS_CUP",
        "EMPEROR_CUP": "EMPERORS_CUP",
        "EMPERORS CUP": "EMPERORS_CUP",
        "EMPEROR'S CUP": "EMPERORS_CUP",
        "JAPAN CUP": "EMPERORS_CUP",
        "CUP HOANG DE": "EMPERORS_CUP",
        "J_LEAGUE_CUP": "J_LEAGUE_CUP",
        "JLEAGUE_CUP": "J_LEAGUE_CUP",
        "J.LEAGUE CUP": "J_LEAGUE_CUP",
        "J LEAGUE CUP": "J_LEAGUE_CUP",
        "LEVAIN": "J_LEAGUE_CUP",
        "LEVAIN CUP": "J_LEAGUE_CUP",
        "YBC LEVAIN CUP": "J_LEAGUE_CUP",
    }
    # Also map label strings from JSON.
    for code, entry in load_leagues_json().items():
        label = str(entry.get("name") or "").strip().upper()
        if label:
            aliases[label] = code
        aliases[code] = code

    for code, cfg in LEAGUE_CONFIG.items():
        label = str(cfg.get("label") or "").strip().upper()
        if label:
            aliases[label] = str(code).upper()

    key = aliases.get(raw, raw)
    merged = merged_league_config()
    if key not in merged:
        raise ValueError(
            f"Unsupported league={league!r}. Choose one of {sorted(merged)}"
        )
    return key, merged[key]


def get_available_leagues() -> list[dict[str, Any]]:
    """Sidebar-friendly list of leagues from JSON (+ hardcoded fallback).

    Each item: ``{code, name, league_weight, flashscore_id, ...}``.
    """
    merged = merged_league_config()
    rows: list[dict[str, Any]] = []
    # Prefer JSON key order, then any hardcoded-only leftovers.
    ordered_keys = list(load_leagues_json().keys())
    for code in merged:
        if code not in ordered_keys:
            ordered_keys.append(code)

    for code in ordered_keys:
        cfg = merged[code]
        rows.append(
            {
                "code": code,
                "name": str(cfg.get("label") or code),
                "league_weight": float(cfg.get("league_weight") or 1.0),
                "flashscore_id": cfg.get("flashscore_id"),
                "fotmob_id": cfg.get("fotmob_id"),
                "fd_div": cfg.get("fd_div"),
                "time_zone": cfg.get("time_zone"),
                "country": cfg.get("country"),
            }
        )
    return rows


def season_starts_from_labels(
    seasons: dict[str, str] | None,
    *,
    n_seasons: int | None = None,
) -> list[tuple[str, str, int]]:
    """Parse JSON ``seasons`` map → ``[(label, season_id, start_year), ...]`` desc."""
    items: list[tuple[str, str, int]] = []
    for label, sid in (seasons or {}).items():
        label_s = str(label).strip()
        start = int(label_s.split("-")[0])
        items.append((label_s, str(sid), start))
    items.sort(key=lambda x: x[2], reverse=True)
    if n_seasons is not None:
        items = items[: max(0, int(n_seasons))]
    return items
