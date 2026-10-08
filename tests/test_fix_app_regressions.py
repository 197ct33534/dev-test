"""Regression checks for app.py fixes (no Streamlit runtime needed).

Pure helpers are extracted from ``app.py`` via AST and executed in a small
namespace, so the heavy Streamlit script never runs at import time.
"""

from __future__ import annotations

import ast
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
APP_SRC = APP_PATH.read_text(encoding="utf-8")
APP_TREE = ast.parse(APP_SRC)


def _func_defs() -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in APP_TREE.body
        if isinstance(node, ast.FunctionDef)
    }


def _load_helpers(names: list[str], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    defs = _func_defs()
    module = ast.Module(
        body=[
            ast.FunctionDef(**{**defs[n].__dict__, "decorator_list": []})
            for n in names
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    ns: dict[str, Any] = {
        "Any": Any,
        "pd": pd,
        "threading": threading,
        "time": time,
        "annotations": None,
    }
    ns.update(extra or {})
    exec(compile(module, str(APP_PATH), "exec"), ns)  # noqa: S102
    return ns


def _decorator_names(fn: ast.FunctionDef) -> list[str]:
    out = []
    for dec in fn.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        out.append(ast.unparse(target))
    return out


def _decorator_kwargs(fn: ast.FunctionDef) -> dict[str, str]:
    for dec in fn.decorator_list:
        if isinstance(dec, ast.Call):
            return {kw.arg: ast.unparse(kw.value) for kw in dec.keywords if kw.arg}
    return {}


# ---------------------------------------------------------------------------
# Bug 1 — corner alerts never priced on assumed odds
# ---------------------------------------------------------------------------


def test_no_assumed_corner_odds_in_app() -> None:
    assert "else 1.90" not in APP_SRC
    assert "line=10.5" not in APP_SRC


@pytest.mark.parametrize(
    "row, expected",
    [
        ({"HomeTeam": "A", "AwayTeam": "B"}, None),
        ({"OddsCornerOver": 1.9, "OddsCornerUnder": float("nan"), "OddsCornerLine": 10.5}, None),
        ({"OddsCornerOver": 1.9, "OddsCornerUnder": 1.95}, None),  # no line
        ({"OddsCornerOver": 1.0, "OddsCornerUnder": 1.95, "OddsCornerLine": 9.5}, None),
        ({"OddsCornerOver": 1.85, "OddsCornerUnder": 1.95, "OddsCornerLine": 9.5}, (1.85, 1.95, 9.5)),
        ({"OddsCornerOver": 2.0, "OddsCornerUnder": 1.8, "CornerLine": 11.5}, (2.0, 1.8, 11.5)),
    ],
)
def test_real_corner_odds_requires_feed_values(row: dict, expected: Any) -> None:
    ns = _load_helpers(
        ["_real_corner_odds"],
        {"_CORNER_LINE_COLS": ("OddsCornerLine", "CornerLine")},
    )
    assert ns["_real_corner_odds"](pd.Series(row)) == expected


# ---------------------------------------------------------------------------
# Bug 2 — process-wide background guards with failure backoff
# ---------------------------------------------------------------------------


def test_bg_guard_is_cache_resource_not_module_global() -> None:
    defs = _func_defs()
    assert "st.cache_resource" in _decorator_names(defs["_bg_guard_state"])
    for stale in ("_FX_BG_RUNNING", "_FX_BG_LOCK", "_TEAM_FEED_BG_RUNNING"):
        assert stale not in APP_SRC


def test_bg_guard_dedupes_and_backs_off() -> None:
    ns = _load_helpers(
        ["_bg_try_acquire", "_bg_release"], {"_BG_FAILURE_BACKOFF_S": 300.0}
    )
    acquire, release = ns["_bg_try_acquire"], ns["_bg_release"]
    guard = {"lock": threading.Lock(), "running": set(), "last_failure": {}}

    assert acquire(guard, "fx:EPL") is True
    assert acquire(guard, "fx:EPL") is False  # already running
    assert acquire(guard, "fx:UWCL") is True  # other league independent

    release(guard, "fx:EPL", failed=True)
    assert acquire(guard, "fx:EPL") is False  # inside backoff window

    guard["last_failure"]["fx:EPL"] = time.monotonic() - 301.0
    assert acquire(guard, "fx:EPL") is True  # backoff elapsed
    release(guard, "fx:EPL", failed=False)
    assert "fx:EPL" not in guard["last_failure"]
    assert acquire(guard, "fx:EPL") is True


# ---------------------------------------------------------------------------
# Bug 4 — cache keys / TTL / bounds
# ---------------------------------------------------------------------------


def _cached_functions() -> dict[str, ast.FunctionDef]:
    return {
        name: fn
        for name, fn in _func_defs().items()
        if any(d in ("st.cache_data", "st.cache_resource") for d in _decorator_names(fn))
    }


def test_cached_functions_do_not_key_on_force_refresh_or_hidden_version() -> None:
    cached = _cached_functions()
    assert {"_load_raw_data", "_fit_dixon_coles", "_fit_ml_model"} <= set(cached)
    for name, fn in cached.items():
        params = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
        assert "force_refresh" not in params, name
        assert "_cache_ver" not in params, name


def test_raw_data_has_ttl_and_models_are_bounded() -> None:
    cached = _cached_functions()
    assert "ttl" in _decorator_kwargs(cached["_load_raw_data"])
    for name in ("_fit_dixon_coles", "_fit_corners", "_fit_ml_model"):
        assert "max_entries" in _decorator_kwargs(cached[name]), name


def test_refresh_clears_targeted_entries_not_everything() -> None:
    assert "def _clear_caches" not in APP_SRC
    assert "st.cache_data.clear()" not in APP_SRC
    assert "st.cache_resource.clear()" not in APP_SRC
    assert "_load_raw_data.clear(code, n_seasons)" in APP_SRC


# ---------------------------------------------------------------------------
# Bug 5 / 7 / 8 — misc
# ---------------------------------------------------------------------------


def test_japan_picks_cache_scoped_to_context() -> None:
    assert "jp_picks_cache_ctx" in APP_SRC
    assert "http://localhost:8501" not in APP_SRC
    assert "PUBLIC_BASE_URL" in APP_SRC


def test_top20_defaults_fallback_and_scope_label() -> None:
    ns = _load_helpers(
        ["_top20_defaults", "_scope_label"],
        {
            "_TOP20_LEAGUE_DEFAULTS": {"EPL": {"n_seasons": 3, "n_train": 800}},
            "_TOP20_FALLBACK_DEFAULTS": {"n_seasons": 3, "n_train": 800},
            "league_label": lambda c: {"EPL": "Premier League"}.get(c, c),
        },
    )
    assert ns["_top20_defaults"]("EMPERORS_CUP") == {"n_seasons": 3, "n_train": 800}
    assert ns["_top20_defaults"]("epl")["n_train"] == 800
    assert ns["_scope_label"]("EPL", "UWCL") == "Premier League + UWCL"
    assert ns["_scope_label"]("EPL", "EPL") == "Premier League"
    assert "EPL + UWCL\"" not in APP_SRC
    assert "_TOP20_LEAGUE_DEFAULTS[_alt_league]" not in APP_SRC


def test_no_deprecated_use_container_width() -> None:
    assert "use_container_width" not in APP_SRC


def test_telegram_toggle_persisted_outside_widget_key() -> None:
    assert 'key="tg_enabled"' not in APP_SRC
    assert "tg_enabled_saved" in APP_SRC


def test_telegram_scan_uses_top20_model_path() -> None:
    """Telegram scan shares _scan_league_models + scan_top_value_bets."""
    start = APP_SRC.index('if st.session_state.pop("_tg_scan_send", False):')
    block = APP_SRC[start : start + 4000]
    assert "_scan_league_models(" in block
    assert "scan_top_value_bets(" in block
    assert 'odds_family="B365"' not in block
