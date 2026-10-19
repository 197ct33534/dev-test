"""Japanese cups price clubs from a pooled J1/J2/J3 + cups Dixon–Coles fit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.api.services import value_bets as vb_svc


def _synthetic(comp: str, teams: list[str], n_rounds: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    day = pd.Timestamp("2026-05-01")
    for r in range(n_rounds):
        order = rng.permutation(teams)
        for i in range(0, len(order) - 1, 2):
            rows.append(
                {
                    "Date": day + pd.Timedelta(days=7 * r),
                    "HomeTeam": order[i],
                    "AwayTeam": order[i + 1],
                    "FTHG": int(rng.poisson(1.4)),
                    "FTAG": int(rng.poisson(1.1)),
                    "comp_id": comp,
                }
            )
    df = pd.DataFrame(rows)
    df["FTR"] = np.where(df.FTHG > df.FTAG, "H", np.where(df.FTHG < df.FTAG, "A", "D"))
    return df


J1 = [f"J1_{i}" for i in range(10)]
J2 = [f"J2_{i}" for i in range(10)]


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    vb_svc.clear_model_cache()
    yield
    vb_svc.clear_model_cache()


def _patch_db(monkeypatch: pytest.MonkeyPatch, frames: dict[str, pd.DataFrame]) -> list[str]:
    seen: list[str] = []

    def _read(*, comp_id: str | None = None, **kw) -> pd.DataFrame:
        seen.append(str(comp_id))
        return frames.get(str(comp_id), pd.DataFrame()).copy()

    monkeypatch.setattr(vb_svc, "_read_global_matches", _read)
    # Keep the test hermetic: no global-DB team weights.
    monkeypatch.setattr(vb_svc, "maybe_wrap_with_league_weights", lambda m, code: m)
    return seen


def test_cup_fit_pools_domestic_leagues(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = {
        "J1": _synthetic("J1", J1, 20, 1),
        "J2": _synthetic("J2", J2, 20, 2),
        # Thin cup history: each club only appears once or twice.
        "J_LEAGUE_CUP": _synthetic("J_LEAGUE_CUP", J1[:4] + J2[:4], 1, 3),
    }
    seen = _patch_db(monkeypatch, frames)
    dc, ml, w_ml, note = vb_svc._resolve_models_uncached("J_LEAGUE_CUP")
    assert "pooled" in note
    assert set(vb_svc.POOLED_FIT_COMPETITIONS["J_LEAGUE_CUP"]) <= set(seen)
    # Clubs thin in the cup are well supported by their league matches.
    assert dc.unknown_teams("J1_0", "J2_0") == []
    assert ml is None and w_ml == 0.0


def test_cup_fit_falls_back_when_pool_is_small(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_db(monkeypatch, {"J1": _synthetic("J1", J1, 2, 1)})
    called: list[str] = []

    def _fake_fit(code, n_seasons, n_train, xi=0.0):
        called.append(code)
        return vb_svc._fit_dc_frame(_synthetic("J1", J1, 14, 4))

    monkeypatch.setattr(vb_svc, "_fit_dc", _fake_fit)
    _dc, _ml, _w, note = vb_svc._resolve_models_uncached("J_LEAGUE_CUP")
    assert called == ["J_LEAGUE_CUP"]
    assert note == "dc_fitted_cold"


def test_pooled_note_is_cold_fit() -> None:
    assert vb_svc._model_flags("dc_pooled_fitted_cold") == {
        "ml_enabled": False,
        "model_source": "cold_fit",
    }
