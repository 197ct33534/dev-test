"""Top-N scan: one pick per match (correlated legs) + new-team prior gate."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from src import scanner as scanner_mod
from src.scanner import TopScanResult, merge_top_scan_results, scan_top_value_bets
from src.timezone_utils import VN_TZ

NOW_VN = datetime(2026, 9, 22, 10, 0, tzinfo=VN_TZ)


class _ToyDC:
    """Minimal DC stand-in: ``thin`` clubs are priced with the new-team prior."""

    def __init__(self, thin: set[str] | None = None) -> None:
        self.thin = set(thin or ())

    def unknown_teams(self, *teams: str) -> list[str]:
        return [t for t in teams if t in self.thin]

    def has_new_team(self, home: str, away: str) -> bool:
        return bool(self.unknown_teams(home, away))

    def predict_match_probs(self, home: str, away: str) -> dict[str, float]:
        return {"H": 0.55, "D": 0.25, "A": 0.20}

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        return 1.6, 1.1

    def predict_score_matrix(self, home: str, away: str, max_goals: int = 10):
        import numpy as np
        from scipy.stats import poisson

        xs = np.arange(0, max_goals + 1)
        return np.outer(poisson.pmf(xs, 1.6), poisson.pmf(xs, 1.1))


# (market, selection, p, odds) legs per fixture — EV = p·odds − 1.
LEGS = {
    ("Alpha", "Beta"): [
        ("1X2", "Home", 0.55, 2.20),  # EV 0.21
        ("AH", "AH Home -0.5", 0.55, 2.10),  # EV 0.155 — same payoff as 1X2 Home
        ("OU", "Over 2.5", 0.60, 1.90),  # EV 0.14
    ],
    ("Gamma", "Delta"): [
        ("1X2", "Away", 0.40, 3.00),  # EV 0.20
    ],
}


@pytest.fixture()
def fixtures(monkeypatch: pytest.MonkeyPatch) -> pd.DataFrame:
    def _fake_recommend(model, fx_df, **kwargs):
        rows = []
        for _, fx in fx_df.iterrows():
            home, away = str(fx["HomeTeam"]), str(fx["AwayTeam"])
            for market, sel, p, odds in LEGS[(home, away)]:
                ev = p * odds - 1.0
                rows.append(
                    {
                        "home_team": home,
                        "away_team": away,
                        "market": market,
                        "selection": sel,
                        "p_model": p,
                        "fair_odds": 1.0 / p,
                        "bookmaker_odds": odds,
                        "ev": ev,
                        "ev_pct": ev * 100.0,
                        "kelly_fraction": 0.005,
                        "kelly_pct": 0.5,
                        "recommended": ev >= 0.05,
                        "kickoff": fx["Kickoff"],
                        "match_date": fx["Kickoff"],
                    }
                )
        return pd.DataFrame(rows)

    monkeypatch.setattr(scanner_mod, "recommend_upcoming", _fake_recommend)
    return pd.DataFrame(
        [
            {
                "Kickoff": pd.Timestamp("2026-09-22 12:00:00"),
                "HomeTeam": "Alpha",
                "AwayTeam": "Beta",
                "B365H": 2.20,
                "B365D": 3.40,
                "B365A": 3.50,
            },
            {
                "Kickoff": pd.Timestamp("2026-09-22 15:00:00"),
                "HomeTeam": "Gamma",
                "AwayTeam": "Delta",
                "B365H": 1.50,
                "B365D": 3.80,
                "B365A": 3.00,
            },
        ]
    )


def _scan(fixtures: pd.DataFrame, dc: _ToyDC, **kw) -> TopScanResult:
    params = dict(
        ml_model=None,
        w_ml=0.0,
        min_ev=0.05,
        allowed_markets=("1X2", "AH", "OU"),
        top_n=20,
        league="EPL",
        filter_vn_window=True,
        now=NOW_VN,
    )
    params.update(kw)
    return scan_top_value_bets(fixtures, dc, **params)  # type: ignore[arg-type]


def _pairs(df: pd.DataFrame) -> list[tuple[str, str]]:
    return list(zip(df["home"], df["market"]))


# ---------------------------------------------------------------------------
# Issue 1 — correlated legs on one match
# ---------------------------------------------------------------------------


def test_default_keeps_every_leg(fixtures: pd.DataFrame) -> None:
    """Research views (Streamlit) still see all legs unless they opt in."""
    res = _scan(fixtures, _ToyDC())
    assert len(res.value_bets) == 4


def test_one_per_match_keeps_best_leg(fixtures: pd.DataFrame) -> None:
    res = _scan(fixtures, _ToyDC(), one_per_match=True)
    assert sorted(_pairs(res.value_bets)) == [("Alpha", "1X2"), ("Gamma", "1X2")]
    # Highest-EV leg survives for Alpha (1X2 Home, EV 0.21).
    alpha = res.value_bets.loc[res.value_bets["home"] == "Alpha"].iloc[0]
    assert alpha["ev"] == pytest.approx(0.21)
    assert res.fallback_bets["match_id"].is_unique


def test_merge_one_per_match(fixtures: pd.DataFrame) -> None:
    r1 = _scan(fixtures, _ToyDC())
    merged = merge_top_scan_results([r1], top_n=20, min_ev=0.05, one_per_match=True)
    assert merged.display_bets["match_id"].is_unique
    assert len(merged.display_bets) == 2


# ---------------------------------------------------------------------------
# Issue 2 — EV built on new-team priors
# ---------------------------------------------------------------------------


def test_league_one_thin_team_kept(fixtures: pd.DataFrame) -> None:
    """Domestic league: promoted club prior is the documented policy (capped stake)."""
    res = _scan(fixtures, _ToyDC(thin={"Beta"}), league="EPL")
    assert "Alpha" in set(res.value_bets["home"])
    assert res.excluded_prior == 0


def test_both_teams_thin_excluded_everywhere(fixtures: pd.DataFrame) -> None:
    res = _scan(fixtures, _ToyDC(thin={"Alpha", "Beta"}), league="EPL")
    assert "Alpha" not in set(res.value_bets["home"])
    assert "Alpha" not in set(res.fallback_bets["home"])
    assert res.excluded_prior == 3  # every Alpha–Beta leg
    assert set(res.value_bets["home"]) == {"Gamma"}


def test_cup_any_thin_team_excluded(fixtures: pd.DataFrame) -> None:
    """Cups: thin cup history says nothing about strength (Urawa ≠ weak)."""
    res = _scan(fixtures, _ToyDC(thin={"Beta"}), league="J_LEAGUE_CUP")
    assert "Alpha" not in set(res.display_bets["home"])
    assert res.excluded_prior == 3
    assert set(res.value_bets["home"]) == {"Gamma"}


def test_all_excluded_yields_empty_not_fallback(fixtures: pd.DataFrame) -> None:
    res = _scan(
        fixtures, _ToyDC(thin={"Alpha", "Beta", "Gamma", "Delta"}), league="J_LEAGUE_CUP"
    )
    assert res.display_bets.empty
    assert res.below_threshold
    assert res.excluded_prior == 4


def test_merge_sums_excluded(fixtures: pd.DataFrame) -> None:
    r1 = _scan(fixtures, _ToyDC(thin={"Beta"}), league="J_LEAGUE_CUP")
    r2 = _scan(fixtures, _ToyDC(), league="EPL")
    merged = merge_top_scan_results([r1, r2], top_n=20, min_ev=0.05)
    assert merged.excluded_prior == 3


def test_models_without_team_api_are_not_gated(fixtures: pd.DataFrame) -> None:
    class _Bare(_ToyDC):
        unknown_teams = None  # type: ignore[assignment]

    res = _scan(fixtures, _Bare(), league="J_LEAGUE_CUP")
    assert res.excluded_prior == 0
    assert len(res.value_bets) == 4


# ---------------------------------------------------------------------------
# Other scan paths share the gate
# ---------------------------------------------------------------------------


def test_drop_prior_gated_legs_on_recommender_rows() -> None:
    """auto_scanner feeds raw recommender rows (home_team / away_team)."""
    from src.scanner import drop_prior_gated_legs

    recs = pd.DataFrame(
        [
            {"home_team": "Alpha", "away_team": "Beta", "ev": 0.3},
            {"home_team": "Gamma", "away_team": "Delta", "ev": 0.1},
        ]
    )
    kept, n = drop_prior_gated_legs(recs, _ToyDC(thin={"Beta"}), "EMPERORS_CUP")
    assert n == 1
    assert list(kept["home_team"]) == ["Gamma"]
    kept, n = drop_prior_gated_legs(recs, _ToyDC(thin={"Beta"}), "EPL")
    assert n == 0


def test_japan_fixture_picks_gated() -> None:
    from src.japan_schedule import evaluate_japan_fixture_picks

    row = {
        "HomeTeam": "Alpha",
        "AwayTeam": "Beta",
        "comp_id": "J_LEAGUE_CUP",
        "B365H": 2.2,
        "B365D": 3.4,
        "B365A": 3.5,
    }
    out = evaluate_japan_fixture_picks(_ToyDC(thin={"Beta"}), row)
    assert out["picks"] == []
    assert out["prior_gated"] == "cup_new_team_prior"


def test_api_scan_defaults_to_one_per_match(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.api.services import value_bets as vb_svc

    seen: dict[str, object] = {}

    def _fake_scan(fx, dc, ml, **kw):
        seen.update(kw)
        return TopScanResult(
            value_bets=pd.DataFrame(),
            fallback_bets=pd.DataFrame(),
            display_bets=pd.DataFrame(),
            below_threshold=True,
            n_value=0,
            excluded_prior=2,
        )

    monkeypatch.setattr(
        vb_svc,
        "load_fixtures_with_meta",
        lambda lg: {"df": pd.DataFrame([{"x": 1}]), "source": "db_cache", "stale": False, "age_minutes": 1.0},
    )
    monkeypatch.setattr(vb_svc, "_resolve_models", lambda lg: (object(), None, 0.0, "dc_pickle"))
    monkeypatch.setattr(vb_svc, "scan_top_value_bets", _fake_scan)
    out = vb_svc.scan_value_bets_api(leagues="EPL")
    assert seen["one_per_match"] is True
    assert out["excluded_prior"] == 2
    assert out["league_meta"][0]["excluded_prior"] == 2
    assert "new-team priors" in (out["notes"] or "")
