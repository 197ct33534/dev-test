"""Regression tests: relative new-team priors, league-weight proxy, ensemble, Kelly cap."""

from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd
import pytest

from src.config import MAX_STAKE_PCT
from src.dixon_coles import DixonColesModel, score_matrix, score_probability
from src.models import (
    NEW_TEAM_KELLY_CAP,
    LeagueWeightDixonColesProxy,
    apply_new_team_kelly_cap,
)
from src.models.ensemble_v2 import EnsembleEngineV2


def _simulated_league(seed: int = 7, n_rounds: int = 4) -> pd.DataFrame:
    """Double round-robin x n_rounds for 8 clubs with a spread of strengths."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(8)]
    attack = np.linspace(0.35, -0.35, len(teams))
    defence = np.linspace(0.25, -0.45, len(teams))  # higher = better
    rows = []
    day = pd.Timestamp("2023-08-01")
    for _ in range(n_rounds):
        for i, h in enumerate(teams):
            for j, a in enumerate(teams):
                if i == j:
                    continue
                lam = np.exp(attack[i] - defence[j] + 0.25)
                mu = np.exp(attack[j] - defence[i])
                rows.append(
                    {
                        "Date": day,
                        "HomeTeam": h,
                        "AwayTeam": a,
                        "FTHG": int(rng.poisson(lam)),
                        "FTAG": int(rng.poisson(mu)),
                    }
                )
                day += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def fitted() -> DixonColesModel:
    return DixonColesModel(xi=0.0).fit(_simulated_league())


def test_newcomer_strength_below_league_mean(fitted: DixonColesModel) -> None:
    mean_a, mean_d = fitted.league_mean_strengths()
    atk, deff = fitted.new_team_priors()
    assert atk < mean_a - 0.1
    assert deff < mean_d - 0.1  # higher δ = better defence → newcomer leakier
    fitted.expected_goals("NewFC", "T3")
    assert fitted.attack["NewFC"] == pytest.approx(atk)
    assert fitted.defence["NewFC"] == pytest.approx(deff)
    assert fitted.has_new_team("NewFC", "T3")


def test_newcomer_home_win_prob_not_extreme(fitted: DixonColesModel) -> None:
    ts = fitted.team_strengths()
    mid = str(ts.iloc[len(ts) // 2]["Team"])
    p_new = fitted.predict_match_probs("NewFC", mid)
    p_mid_home = fitted.predict_match_probs(mid, "NewFC")
    # Newcomer is an underdog at home vs a mid-table club, but not absurdly so.
    assert 0.08 < p_new["H"] < 0.45
    assert p_new["H"] < p_new["A"]
    assert p_mid_home["H"] > p_new["H"]
    assert abs(sum(p_new.values()) - 1.0) < 1e-9


def test_stale_absolute_prior_is_refreshed(fitted: DixonColesModel) -> None:
    """Old pickles cached absolute 0.4/0.4 priors for unseen clubs."""
    fitted.attack["Legacy FC"] = 0.4
    fitted.defence["Legacy FC"] = 0.4
    atk, deff = fitted._strengths_for("Legacy FC")
    assert (atk, deff) == pytest.approx(fitted.new_team_priors())


def test_explicit_prior_override_still_honoured() -> None:
    model = DixonColesModel(xi=0.0, new_team_attack=-1.0, new_team_defence=-1.0)
    model.fit(_simulated_league(n_rounds=1))
    assert model.new_team_priors() == (-1.0, -1.0)


def test_new_team_kelly_cap_binds() -> None:
    assert NEW_TEAM_KELLY_CAP < MAX_STAKE_PCT
    assert apply_new_team_kelly_cap(MAX_STAKE_PCT, has_new_team=True) == NEW_TEAM_KELLY_CAP
    assert apply_new_team_kelly_cap(MAX_STAKE_PCT, has_new_team=False) == MAX_STAKE_PCT


def test_score_matrix_matches_scalar_reference() -> None:
    lam, mu, rho, g = 1.7, 0.9, -0.08, 10
    ref = np.array(
        [[score_probability(x, y, lam, mu, rho) for y in range(g + 1)] for x in range(g + 1)]
    )
    ref /= ref.sum()
    np.testing.assert_allclose(score_matrix(lam, mu, rho, g), ref, atol=1e-15)
    # Cached result is not mutated through the returned copy.
    m = score_matrix(lam, mu, rho, g)
    m[0, 0] = 99.0
    assert score_matrix(lam, mu, rho, g)[0, 0] == pytest.approx(ref[0, 0])


# ---------------------------------------------------------------------------
# League-weight proxy
# ---------------------------------------------------------------------------


class _Inner:
    fitted_ = True
    max_goals = 10
    rho = -0.05

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        return 1.6, 1.2

    def predict_over_under(self, *a, **k):  # pragma: no cover - must not be used
        raise AssertionError("proxy fell through to unweighted inner O/U")

    def predict_asian_handicap(self, *a, **k):  # pragma: no cover
        raise AssertionError("proxy fell through to unweighted inner AH")


def test_proxy_all_markets_use_weighted_matrix() -> None:
    proxy = LeagueWeightDixonColesProxy(_Inner(), lambda t: 0.5 if t == "A" else 1.0)  # type: ignore[arg-type]
    lam, mu = proxy.expected_goals("A", "B")
    assert (lam, mu) == pytest.approx((0.8, 1.2))
    mat = score_matrix(lam, mu, -0.05, 10)

    ou = proxy.predict_over_under("A", "B", line=2.5)
    over_ref = float(sum(mat[x, y] for x in range(11) for y in range(11) if x + y > 2.5))
    assert ou["over"] == pytest.approx(over_ref)

    ah = proxy.predict_asian_handicap("A", "B", handicap=-0.5)
    assert ah["home"] == pytest.approx(float(np.tril(mat, k=-1).sum()))
    assert proxy.predict_match_probs("A", "B")["H"] == pytest.approx(ah["home"])


def test_domestic_fixture_not_scaled_by_own_league_weight(monkeypatch) -> None:
    import src.global_db as gdb
    from src.models.league_fit import maybe_wrap_with_league_weights

    weights = {"LALIGA": 0.9, "CHAMP": 0.6}
    team_league = {"Home FC": "LALIGA", "Away FC": "LALIGA", "Promoted": "CHAMP"}
    monkeypatch.setattr(gdb, "global_db_has_matches", lambda path: True)
    monkeypatch.setattr(
        gdb, "connect_global_db", lambda *a, **k: contextlib.nullcontext(object())
    )
    monkeypatch.setattr(
        gdb, "get_competition_weight", lambda conn, code, default=1.0: weights.get(code, default)
    )
    monkeypatch.setattr(
        gdb, "resolve_team_id", lambda conn, name, create=False, comp_id=None: name
    )
    monkeypatch.setattr(
        gdb,
        "team_weight_for_match",
        lambda conn, tid, code, default=1.0: weights[team_league[tid]],
    )

    inner = _Inner()
    proxy = maybe_wrap_with_league_weights(inner, "LALIGA")  # type: ignore[arg-type]
    # Both clubs play in LALIGA → relative weight 1.0 → untouched λ/μ.
    assert proxy.expected_goals("Home FC", "Away FC") == pytest.approx((1.6, 1.2))
    # A club from a weaker domestic league is scaled relative to LALIGA.
    lam, mu = proxy.expected_goals("Promoted", "Home FC")
    assert lam == pytest.approx(1.6 * 0.6 / 0.9)
    assert mu == pytest.approx(1.2)
    # Explicit w_ref keeps the legacy absolute behaviour.
    legacy = maybe_wrap_with_league_weights(inner, "LALIGA", w_ref=1.0)  # type: ignore[arg-type]
    assert legacy.expected_goals("Home FC", "Away FC") == pytest.approx((1.6 * 0.9, 1.2 * 0.9))


# ---------------------------------------------------------------------------
# Ensemble v2
# ---------------------------------------------------------------------------

DC = {"H": 0.50, "D": 0.27, "A": 0.23}
MKT = {"H": 0.45, "D": 0.28, "A": 0.27}


def test_ensemble_missing_lgbm_drops_and_renormalizes() -> None:
    eng = EnsembleEngineV2()
    res = eng.predict(dixon_coles=DC, lightgbm=None, market=MKT, aggregate_data_score=95)
    assert res.mix_weights["lgbm"] == 0.0
    assert res.mix_weights["dc"] == pytest.approx(0.40 / 0.65)
    assert res.mix_weights["market"] == pytest.approx(0.25 / 0.65)
    expected_raw = {
        k: (0.40 * DC[k] + 0.25 * MKT[k]) / 0.65 for k in ("H", "D", "A")
    }
    for k in ("H", "D", "A"):
        assert res.raw_probabilities[k] == pytest.approx(expected_raw[k])
    assert res.meta["missing_components"] == ["lgbm"]
    # No uniform 1/3 leaking in: home stays well above 1/3 blend dilution.
    assert res.raw_probabilities["H"] > 0.47


def test_ensemble_missing_market_skips_market_pull() -> None:
    eng = EnsembleEngineV2()
    res = eng.predict(dixon_coles=DC, lightgbm=None, market=None, aggregate_data_score=70)
    assert res.mix_weights == {"dc": pytest.approx(1.0), "lgbm": 0.0, "market": 0.0}
    for k in ("H", "D", "A"):
        assert res.fair_probabilities[k] == pytest.approx(DC[k])


def test_ensemble_no_components_is_no_bet() -> None:
    eng = EnsembleEngineV2()
    res = eng.predict(dixon_coles=None, lightgbm=None, market=None, aggregate_data_score=95)
    assert res.no_bet is True
    assert res.expected_value_percent == 0.0
