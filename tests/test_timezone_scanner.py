"""Vietnam timezone window + Top-20 scanner unit tests."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from src.scanner import TopScanResult, _blend_1x2, scan_top_value_bets
from src.timezone_utils import (
    VN_TZ,
    filter_matches_today_tomorrow,
    format_kickoff_vn,
    get_vn_today_tomorrow_window,
    kickoff_to_vn,
)


def test_vn_window_boundaries_around_midnight() -> None:
    """VN midnight maps correctly; end is max(tomorrow, now+48h)."""
    # 2026-03-15 00:30 VN = 2026-03-14 17:30 UTC
    now_vn = datetime(2026, 3, 15, 0, 30, tzinfo=VN_TZ)
    start, end = get_vn_today_tomorrow_window(now=now_vn, as_naive_utc=True)

    # today 00:00 VN → 2026-03-14 17:00 UTC
    assert start == pd.Timestamp("2026-03-14 17:00:00")
    # tomorrow 23:59:59 VN = 2026-03-16 16:59:59 UTC
    # now+48h = 2026-03-17 00:30 VN = 2026-03-16 17:30 UTC → wins
    assert end == pd.Timestamp("2026-03-16 17:30:00")


def test_vn_window_includes_day2_early_kickoff() -> None:
    """European Wed night (~02:00 VN Thu) stays inside ~48h from Tue evening."""
    now_vn = datetime(2026, 9, 22, 20, 0, tzinfo=VN_TZ)
    start, end = get_vn_today_tomorrow_window(now=now_vn, as_naive_utc=True)
    # Sep 22 00:00 VN → Sep 21 17:00 UTC
    assert start == pd.Timestamp("2026-09-21 17:00:00")
    # now+48h = Sep 24 20:00 VN = Sep 24 13:00 UTC (later than tomorrow end)
    assert end == pd.Timestamp("2026-09-24 13:00:00")
    # Barcelona-style kickoff: Sep 23 19:00 UTC = Sep 24 02:00 VN
    assert pd.Timestamp("2026-09-23 19:00:00") <= end


def test_filter_matches_today_tomorrow_midnight_edges() -> None:
    """Include kickoffs on both sides of VN midnight; exclude outside window."""
    now_vn = datetime(2026, 9, 22, 12, 0, tzinfo=VN_TZ)
    # Calendar tomorrow end: Sep 23 16:59:59 UTC
    # now+48h: Sep 24 12:00 VN = Sep 24 05:00 UTC → window end

    rows = [
        # Just before today VN start → exclude
        {"Kickoff": "2026-09-21 16:59:00", "HomeTeam": "A", "AwayTeam": "B"},
        # Exactly today start (00:00 VN) → include
        {"Kickoff": "2026-09-21 17:00:00", "HomeTeam": "C", "AwayTeam": "D"},
        # Mid-window → include
        {"Kickoff": "2026-09-22 12:00:00", "HomeTeam": "E", "AwayTeam": "F"},
        # Tomorrow evening VN (23:00 VN = 16:00 UTC Sep 23) → include
        {"Kickoff": "2026-09-23 16:00:00", "HomeTeam": "G", "AwayTeam": "H"},
        # Day+2 early VN (02:00 VN Sep 24 = 19:00 UTC Sep 23) → include via 48h
        {"Kickoff": "2026-09-23 19:00:00", "HomeTeam": "Barca", "AwayTeam": "PFC"},
        # Just after now+48h end (Sep 24 05:00 UTC) → exclude
        {"Kickoff": "2026-09-24 06:00:00", "HomeTeam": "I", "AwayTeam": "J"},
    ]
    df = pd.DataFrame(rows)
    df["Kickoff"] = pd.to_datetime(df["Kickoff"])

    out = filter_matches_today_tomorrow(df, now=now_vn)
    pairs = set(zip(out["HomeTeam"], out["AwayTeam"]))
    assert ("C", "D") in pairs
    assert ("E", "F") in pairs
    assert ("G", "H") in pairs
    assert ("Barca", "PFC") in pairs
    assert ("A", "B") not in pairs
    assert ("I", "J") not in pairs


def test_kickoff_to_vn_and_format() -> None:
    # Naive UTC noon → 19:00 VN (UTC+7, no DST)
    ts = pd.Timestamp("2026-06-01 12:00:00")
    vn = kickoff_to_vn(ts)
    assert vn.tzinfo is not None
    assert vn.hour == 19
    assert "19:00" in format_kickoff_vn(ts)


def test_blend_1x2_renormalises() -> None:
    p_dc = {"H": 0.5, "D": 0.3, "A": 0.2}
    p_ml = {"H": 0.4, "D": 0.3, "A": 0.3}
    out = _blend_1x2(p_dc, p_ml, 0.2)
    assert abs(sum(out.values()) - 1.0) < 1e-9
    # w=0.2 → H = 0.8*0.5 + 0.2*0.4 = 0.48
    assert out["H"] == pytest.approx(0.48)


class _ToyDC:
    """Minimal Dixon–Coles stand-in for scanner ranking tests."""

    teams = ("Alpha", "Beta", "Gamma", "Delta")

    def predict_match_probs(self, home: str, away: str) -> dict[str, float]:
        # Strong home favourite for Alpha, weaker otherwise
        if home == "Alpha":
            return {"H": 0.55, "D": 0.25, "A": 0.20}
        return {"H": 0.35, "D": 0.30, "A": 0.35}

    def predict_score_matrix(self, home: str, away: str, max_goals: int = 10):
        import numpy as np
        from scipy.stats import poisson

        lam_h, lam_a = 1.6, 1.1
        xs = np.arange(0, max_goals + 1)
        ph = poisson.pmf(xs, lam_h)
        pa = poisson.pmf(xs, lam_a)
        return np.outer(ph, pa)


def test_scan_top_value_bets_ranks_and_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Toy fixtures: high-EV home wins top; below-threshold fills fallback."""
    # Bypass real Dixon–Coles score matrix path by stubbing recommend_upcoming
    from src import scanner as scanner_mod

    def _fake_recommend(model, fixtures, **kwargs):
        rows = []
        for _, fx in fixtures.iterrows():
            home, away = str(fx["HomeTeam"]), str(fx["AwayTeam"])
            # Alpha home @ 2.20 with p=0.55 → EV = 0.55*2.20-1 = 0.21
            # Beta home @ 1.50 with p=0.35 → EV = 0.35*1.50-1 = -0.475
            if home == "Alpha":
                p, odds = 0.55, 2.20
            else:
                p, odds = 0.35, 1.50
            ev = p * odds - 1.0
            rows.append(
                {
                    "home_team": home,
                    "away_team": away,
                    "market": "1X2",
                    "selection": "Home",
                    "p_model": p,
                    "fair_odds": 1.0 / p,
                    "bookmaker_odds": odds,
                    "ev": ev,
                    "ev_pct": ev * 100.0,
                    "kelly_fraction": 0.01,
                    "kelly_pct": 1.0,
                    "recommended": ev >= 0.05,
                    "kickoff": fx["Kickoff"],
                    "match_date": fx["Kickoff"],
                }
            )
        return pd.DataFrame(rows)

    monkeypatch.setattr(scanner_mod, "recommend_upcoming", _fake_recommend)

    now_vn = datetime(2026, 9, 22, 10, 0, tzinfo=VN_TZ)
    # Both kickoffs inside VN today/tomorrow window (naive UTC)
    fixtures = pd.DataFrame(
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
                "B365A": 6.00,
            },
        ]
    )

    result = scan_top_value_bets(
        fixtures,
        _ToyDC(),  # type: ignore[arg-type]
        ml_model=None,
        w_ml=0.0,
        min_ev=0.05,
        top_n=20,
        bankroll=1000.0,
        league="EPL",
        filter_vn_window=True,
        now=now_vn,
    )
    assert isinstance(result, TopScanResult)
    assert result.n_value == 1
    assert not result.below_threshold
    assert result.value_bets.iloc[0]["home"] == "Alpha"
    assert result.value_bets.iloc[0]["ev"] == pytest.approx(0.21)
    # Fallback still ranks Alpha first
    assert result.fallback_bets.iloc[0]["home"] == "Alpha"
    assert "kickoff_vn" in result.display_bets.columns
    assert "match_id" in result.display_bets.columns
    assert result.odds_missing is False


def test_scan_below_threshold_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from src import scanner as scanner_mod

    def _fake_recommend(model, fixtures, **kwargs):
        rows = []
        for _, fx in fixtures.iterrows():
            p, odds = 0.40, 1.80  # EV = -0.28
            ev = p * odds - 1.0
            rows.append(
                {
                    "home_team": fx["HomeTeam"],
                    "away_team": fx["AwayTeam"],
                    "market": "1X2",
                    "selection": "Home",
                    "p_model": p,
                    "fair_odds": 1.0 / p,
                    "bookmaker_odds": odds,
                    "ev": ev,
                    "ev_pct": ev * 100.0,
                    "kelly_fraction": 0.0,
                    "kelly_pct": 0.0,
                    "recommended": False,
                    "kickoff": fx["Kickoff"],
                    "match_date": fx["Kickoff"],
                }
            )
        return pd.DataFrame(rows)

    monkeypatch.setattr(scanner_mod, "recommend_upcoming", _fake_recommend)

    now_vn = datetime(2026, 9, 22, 10, 0, tzinfo=VN_TZ)
    fixtures = pd.DataFrame(
        [
            {
                "Kickoff": pd.Timestamp("2026-09-22 12:00:00"),
                "HomeTeam": "Alpha",
                "AwayTeam": "Beta",
                "B365H": 1.80,
                "B365D": 3.40,
                "B365A": 4.50,
            }
        ]
    )
    result = scan_top_value_bets(
        fixtures,
        _ToyDC(),  # type: ignore[arg-type]
        min_ev=0.05,
        now=now_vn,
        filter_vn_window=True,
    )
    assert result.below_threshold is True
    assert result.value_bets.empty
    assert not result.display_bets.empty
    assert result.display_bets is result.fallback_bets or len(result.display_bets) == len(
        result.fallback_bets
    )


def test_scan_no_odds_still_lists_fixtures(monkeypatch: pytest.MonkeyPatch) -> None:
    """When recommend_upcoming returns empty, still show window fixtures (EV n/a)."""
    from src import scanner as scanner_mod

    monkeypatch.setattr(
        scanner_mod, "recommend_upcoming", lambda *a, **k: pd.DataFrame()
    )

    now_vn = datetime(2026, 9, 22, 10, 0, tzinfo=VN_TZ)
    fixtures = pd.DataFrame(
        [
            {
                "Kickoff": pd.Timestamp("2026-09-22 19:00:00"),
                "HomeTeam": "Juventus",
                "AwayTeam": "Benfica",
                "FlashscoreEventId": "abc123",
            },
            {
                "Kickoff": pd.Timestamp("2026-09-23 19:00:00"),
                "HomeTeam": "Barcelona",
                "AwayTeam": "Paris FC",
            },
        ]
    )
    result = scan_top_value_bets(
        fixtures,
        _ToyDC(),  # type: ignore[arg-type]
        league="UWCL",
        min_ev=0.05,
        now=now_vn,
        filter_vn_window=True,
    )
    assert result.odds_missing is True
    assert result.below_threshold is True
    assert result.n_value == 0
    assert len(result.display_bets) == 2
    assert result.display_bets.iloc[0]["home"] == "Juventus"
    assert bool(result.display_bets.iloc[0]["odds_missing"]) is True
    assert pd.isna(result.display_bets.iloc[0]["ev"])
    assert "02:00" in str(result.display_bets.iloc[0]["kickoff_vn"])


def test_scan_multi_market_picks_per_match(monkeypatch: pytest.MonkeyPatch) -> None:
    """Multiple value legs on one fixture → all retained, sorted by EV desc."""
    from src import scanner as scanner_mod

    def _fake_recommend(model, fixtures, **kwargs):
        rows = []
        for _, fx in fixtures.iterrows():
            home, away = str(fx["HomeTeam"]), str(fx["AwayTeam"])
            # Two correlated 1X2 legs; Away EV higher
            for sel, p, odds in (
                ("Home", 0.40, 2.50),  # EV = 0.00
                ("Away", 0.30, 4.00),  # EV = 0.20
                ("Draw", 0.25, 4.50),  # EV = 0.125
            ):
                ev = p * odds - 1.0
                rows.append(
                    {
                        "home_team": home,
                        "away_team": away,
                        "market": "1X2",
                        "selection": sel,
                        "p_model": p,
                        "fair_odds": 1.0 / p,
                        "bookmaker_odds": odds,
                        "ev": ev,
                        "ev_pct": ev * 100.0,
                        "kelly_fraction": 0.01,
                        "kelly_pct": 1.0,
                        "recommended": ev >= 0.05,
                        "kickoff": fx["Kickoff"],
                        "match_date": fx["Kickoff"],
                    }
                )
        return pd.DataFrame(rows)

    monkeypatch.setattr(scanner_mod, "recommend_upcoming", _fake_recommend)

    now_vn = datetime(2026, 9, 22, 10, 0, tzinfo=VN_TZ)
    fixtures = pd.DataFrame(
        [
            {
                "Kickoff": pd.Timestamp("2026-09-22 12:00:00"),
                "HomeTeam": "Alpha",
                "AwayTeam": "Beta",
                "B365H": 2.50,
                "B365D": 4.50,
                "B365A": 4.00,
                "FlashscoreEventId": "fx-1",
            }
        ]
    )
    result = scan_top_value_bets(
        fixtures,
        _ToyDC(),  # type: ignore[arg-type]
        min_ev=0.05,
        now=now_vn,
        filter_vn_window=True,
        top_n=20,
    )
    assert result.n_value == 2
    assert list(result.value_bets["selection"]) == ["Away", "Draw"]
    assert result.value_bets.iloc[0]["ev"] == pytest.approx(0.20)
    assert result.value_bets.iloc[1]["ev"] == pytest.approx(0.125)


def test_merge_top_scan_results_ranks_across_leagues() -> None:
    from src.scanner import TopScanResult, merge_top_scan_results

    epl = pd.DataFrame(
        [
            {
                "home": "Arsenal",
                "away": "Chelsea",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "match_id": "epl-1",
                "league": "EPL",
                "market": "1X2",
                "selection": "Home",
                "ev": 0.10,
                "ev_pct": 10.0,
                "kickoff": "2026-09-22 15:00",
            }
        ]
    )
    uwcl = pd.DataFrame(
        [
            {
                "home": "Barcelona",
                "away": "Lyon",
                "home_team": "Barcelona",
                "away_team": "Lyon",
                "match_id": "uwcl-1",
                "league": "UWCL",
                "market": "1X2",
                "selection": "Home",
                "ev": 0.18,
                "ev_pct": 18.0,
                "kickoff": "2026-09-22 19:00",
            },
            {
                "home": "Barcelona",
                "away": "Lyon",
                "home_team": "Barcelona",
                "away_team": "Lyon",
                "match_id": "uwcl-1",
                "league": "UWCL",
                "market": "1X2",
                "selection": "Draw",
                "ev": 0.12,
                "ev_pct": 12.0,
                "kickoff": "2026-09-22 19:00",
            },
        ]
    )
    r1 = TopScanResult(
        value_bets=epl,
        fallback_bets=epl,
        display_bets=epl,
        below_threshold=False,
        n_value=1,
    )
    r2 = TopScanResult(
        value_bets=uwcl,
        fallback_bets=uwcl,
        display_bets=uwcl,
        below_threshold=False,
        n_value=2,
    )
    merged = merge_top_scan_results([r1, r2], top_n=20, min_ev=0.05)
    assert merged.n_value == 3
    assert merged.display_bets.iloc[0]["league"] == "UWCL"
    assert merged.display_bets.iloc[0]["selection"] == "Home"
    assert merged.display_bets.iloc[0]["ev"] == pytest.approx(0.18)
    # Multi-market: both UWCL legs kept, ranked by EV before EPL
    assert list(merged.display_bets["selection"]) == ["Home", "Draw", "Home"]
    assert list(merged.display_bets["match_id"]) == ["uwcl-1", "uwcl-1", "epl-1"]
    assert list(merged.display_bets["ev"]) == sorted(
        merged.display_bets["ev"], reverse=True
    )
