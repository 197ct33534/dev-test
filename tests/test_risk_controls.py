"""Risk controls: one-bet-per-match, daily Top-5, CLV closing-odds fallback."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.config import MAX_BETS_PER_DAY, MAX_STAKE_PCT, DEFAULT_KELLY_FRACTION
from src.journal import (
    add_live_bet,
    closing_odds_from_row,
    compute_clv_pct,
    evaluate_pending_against_results,
    load_live_bets,
    snapshot_closing_odds_near_kickoff,
    sync_closing_odds_from_results,
)
from src.strategy import match_group_key, select_value_bets


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "risk_journal.db"


def _candidate_bets() -> pd.DataFrame:
    """Arsenal vs Koge with Draw+Away (the double-bet bug) plus extra matches."""
    day = "2026-09-22"
    rows = [
        {
            "match_id": "ars_koge",
            "home_team": "Arsenal",
            "away_team": "Koge",
            "kickoff": f"{day} 17:00",
            "market": "1X2",
            "selection": "Draw",
            "ev": 0.12,
            "ev_pct": 12.0,
            "bookmaker_odds": 4.0,
        },
        {
            "match_id": "ars_koge",
            "home_team": "Arsenal",
            "away_team": "Koge",
            "kickoff": f"{day} 17:00",
            "market": "1X2",
            "selection": "Away",
            "ev": 0.18,
            "ev_pct": 18.0,
            "bookmaker_odds": 5.5,
        },
        {
            "match_id": "liv_eve",
            "home_team": "Liverpool",
            "away_team": "Everton",
            "kickoff": f"{day} 19:00",
            "market": "1X2",
            "selection": "Home",
            "ev": 0.08,
            "ev_pct": 8.0,
            "bookmaker_odds": 1.70,
        },
        {
            "home_team": "Chelsea",
            "away_team": "Spurs",
            "kickoff": f"{day} 15:00",
            "market": "OU",
            "selection": "Over 2.5",
            "ev": 0.15,
            "ev_pct": 15.0,
            "bookmaker_odds": 1.95,
        },
        {
            "home_team": "Chelsea",
            "away_team": "Spurs",
            "kickoff": f"{day} 15:00",
            "market": "1X2",
            "selection": "Home",
            "ev": 0.06,
            "ev_pct": 6.0,
            "bookmaker_odds": 2.10,
        },
        {
            "match_id": "mci_mun",
            "home_team": "Man City",
            "away_team": "Man United",
            "kickoff": f"{day} 20:00",
            "market": "1X2",
            "selection": "Home",
            "ev": 0.09,
            "ev_pct": 9.0,
            "bookmaker_odds": 1.55,
        },
        {
            "match_id": "new_whu",
            "home_team": "Newcastle",
            "away_team": "West Ham",
            "kickoff": f"{day} 14:00",
            "market": "AH",
            "selection": "AH Home -0.5",
            "ev": 0.07,
            "ev_pct": 7.0,
            "bookmaker_odds": 1.90,
        },
        {
            "match_id": "bri_ful",
            "home_team": "Brighton",
            "away_team": "Fulham",
            "kickoff": f"{day} 16:00",
            "market": "Corners",
            "selection": "Over 10.5",
            "ev": 0.11,
            "ev_pct": 11.0,
            "bookmaker_odds": 1.90,
        },
        {
            "match_id": "wol_bou",
            "home_team": "Wolves",
            "away_team": "Bournemouth",
            "kickoff": f"{day} 18:00",
            "market": "1X2",
            "selection": "Away",
            "ev": 0.05,
            "ev_pct": 5.0,
            "bookmaker_odds": 3.20,
        },
    ]
    return pd.DataFrame(rows)


def test_one_bet_per_match_keeps_highest_ev() -> None:
    bets = _candidate_bets()
    filtered = select_value_bets(
        bets,
        max_per_day=20,
        already_today=0,
        allow_multi_picks_per_match=False,
    )
    # Arsenal/Koge: Away (18%) beats Draw (12%)
    ars = filtered.loc[filtered["home_team"] == "Arsenal"]
    assert len(ars) == 1
    assert ars.iloc[0]["selection"] == "Away"
    assert ars.iloc[0]["ev"] == pytest.approx(0.18)
    # Chelsea/Spurs: OU Over (15%) beats Home (6%) via home|away|date key
    che = filtered.loc[filtered["home_team"] == "Chelsea"]
    assert len(che) == 1
    assert che.iloc[0]["market"] == "OU"
    # No duplicate match keys
    keys = [match_group_key(r) for _, r in filtered.iterrows()]
    assert len(keys) == len(set(keys))


def test_multi_picks_per_match_retains_markets_sorted_by_ev() -> None:
    """Same match_id, two markets both EV≥threshold → both kept when multi=True."""
    day = "2026-09-22"
    bets = pd.DataFrame(
        [
            {
                "match_id": "ars_koge",
                "home_team": "Arsenal",
                "away_team": "Koge",
                "kickoff": f"{day} 17:00",
                "market": "1X2",
                "selection": "Away",
                "ev": 0.12,
                "ev_pct": 12.0,
                "bookmaker_odds": 5.5,
            },
            {
                "match_id": "ars_koge",
                "home_team": "Arsenal",
                "away_team": "Koge",
                "kickoff": f"{day} 17:00",
                "market": "OU",
                "selection": "Over 2.5",
                "ev": 0.18,
                "ev_pct": 18.0,
                "bookmaker_odds": 1.95,
            },
            {
                "match_id": "liv_eve",
                "home_team": "Liverpool",
                "away_team": "Everton",
                "kickoff": f"{day} 19:00",
                "market": "AH",
                "selection": "AH Home -0.5",
                "ev": 0.08,
                "ev_pct": 8.0,
                "bookmaker_odds": 1.90,
            },
        ]
    )
    multi = select_value_bets(
        bets, max_per_day=20, already_today=0, allow_multi_picks_per_match=True
    )
    assert len(multi) == 3
    assert list(multi["ev"]) == sorted(multi["ev"], reverse=True)
    assert multi.iloc[0]["market"] == "OU"
    assert multi.iloc[0]["ev"] == pytest.approx(0.18)
    ars = multi.loc[multi["match_id"] == "ars_koge"]
    assert len(ars) == 2
    assert set(ars["market"]) == {"1X2", "OU"}

    single = select_value_bets(
        bets, max_per_day=20, already_today=0, allow_multi_picks_per_match=False
    )
    assert len(single) == 2
    ars_one = single.loc[single["match_id"] == "ars_koge"]
    assert len(ars_one) == 1
    assert ars_one.iloc[0]["market"] == "OU"


def test_one_per_match_kwarg_overrides_allow_multi() -> None:
    bets = _candidate_bets()
    filtered = select_value_bets(
        bets,
        max_per_day=20,
        already_today=0,
        allow_multi_picks_per_match=True,
        one_per_match=True,
    )
    ars = filtered.loc[filtered["home_team"] == "Arsenal"]
    assert len(ars) == 1
    assert ars.iloc[0]["selection"] == "Away"


def test_daily_top5_after_dedupe() -> None:
    bets = _candidate_bets()
    filtered = select_value_bets(
        bets,
        max_per_day=MAX_BETS_PER_DAY,
        already_today=0,
        allow_multi_picks_per_match=False,
    )
    assert len(filtered) == MAX_BETS_PER_DAY
    # Sorted by EV desc: Away Arsenal 18, OU Chelsea 15, Brighton 11, City 9, Liv 8
    assert list(filtered["ev"]) == sorted(filtered["ev"], reverse=True)
    assert filtered.iloc[0]["selection"] == "Away"
    assert filtered.iloc[0]["home_team"] == "Arsenal"
    # Wolves 5% and Newcastle 7% drop out of Top-5 after dedupe (6 unique matches
    # with EV≥7% wait — unique after dedupe: Ars18, Che15, Bri11, City9, Liv8, New7, Wol5
    # Top5 excludes New + Wol
    assert "Newcastle" not in set(filtered["home_team"])
    assert "Wolves" not in set(filtered["home_team"])


def test_daily_cap_respects_already_today() -> None:
    bets = _candidate_bets()
    filtered = select_value_bets(
        bets,
        max_per_day=MAX_BETS_PER_DAY,
        already_today=3,
        allow_multi_picks_per_match=False,
    )
    assert len(filtered) == 2
    empty = select_value_bets(
        bets,
        max_per_day=MAX_BETS_PER_DAY,
        already_today=5,
        allow_multi_picks_per_match=False,
    )
    assert empty.empty


def test_config_kelly_and_stake_caps() -> None:
    assert DEFAULT_KELLY_FRACTION == pytest.approx(0.10)
    assert MAX_STAKE_PCT == pytest.approx(0.01)
    assert MAX_BETS_PER_DAY == 5


def test_clv_fallback_from_results_at_settle(tmp_db: Path) -> None:
    """Settle must set CLV from Avg/B365 even when closing_odds was never snapshotted."""
    past = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d %H:%M")
    add_live_bet(
        home_team="Arsenal",
        away_team="Chelsea",
        selection="Home",
        odds=2.00,
        stake_amount=10.0,
        match_date=past,
        market="1X2",
        p_model=0.55,
        db_path=tmp_db,
    )
    day = past[:10]
    # Only AvgH present (no prior closing_odds on journal) — fallback path.
    results = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp(f"{day} 15:00"),
                "HomeTeam": "Arsenal",
                "AwayTeam": "Chelsea",
                "FTHG": 2,
                "FTAG": 1,
                "FTR": "H",
                "AvgH": 1.85,
                "AvgD": 3.50,
                "AvgA": 4.00,
            }
        ]
    )
    session = evaluate_pending_against_results(results, tmp_db, sync_clv=False)
    assert session["settled"] == 1
    bet = load_live_bets(tmp_db).iloc[0]
    assert bet["closing_odds"] == pytest.approx(1.85)
    assert bet["clv_pct"] == pytest.approx(compute_clv_pct(2.00, 1.85))


def test_closing_odds_from_row_prefers_b365_then_avg() -> None:
    row = pd.Series({"B365H": 1.80, "AvgH": 1.90, "PSH": 1.85})
    assert closing_odds_from_row("Home", row) == pytest.approx(1.80)
    row2 = pd.Series({"AvgH": 1.92})
    assert closing_odds_from_row("Home", row2) == pytest.approx(1.92)
    assert closing_odds_from_row("Home", pd.Series({"HC": 5})) is None


def test_snapshot_closing_odds_near_kickoff(tmp_db: Path) -> None:
    now = datetime(2026, 9, 22, 16, 40, tzinfo=timezone.utc)
    # Kickoff in 20 minutes → inside 15–30 window
    ko = (now + timedelta(minutes=20)).strftime("%Y-%m-%d %H:%M")
    add_live_bet(
        home_team="Arsenal",
        away_team="Koge",
        selection="Away",
        odds=5.50,
        stake_amount=10.0,
        match_date=ko,
        market="1X2",
        db_path=tmp_db,
    )
    fixtures = pd.DataFrame(
        [
            {
                "HomeTeam": "Arsenal",
                "AwayTeam": "Koge",
                "Kickoff": pd.Timestamp(ko),
                "B365A": 5.20,
                "B365H": 1.40,
                "B365D": 4.50,
            }
        ]
    )
    snap = snapshot_closing_odds_near_kickoff(fixtures, tmp_db, now=now)
    assert snap["updated"] == 1
    bet = load_live_bets(tmp_db).iloc[0]
    assert bet["closing_odds"] == pytest.approx(5.20)
    assert bet["clv_pct"] == pytest.approx(compute_clv_pct(5.50, 5.20))


def test_sync_closing_odds_alias_odds_close(tmp_db: Path) -> None:
    """sync_closing_odds_from_results still fills closing_odds (canonical)."""
    past = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d %H:%M")
    add_live_bet(
        home_team="Liverpool",
        away_team="Everton",
        selection="Draw",
        odds=4.20,
        stake_amount=5.0,
        match_date=past,
        db_path=tmp_db,
    )
    day = past[:10]
    results = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp(f"{day} 17:30"),
                "HomeTeam": "Liverpool",
                "AwayTeam": "Everton",
                "FTHG": 1,
                "FTAG": 1,
                "FTR": "D",
                "B365D": 3.90,
            }
        ]
    )
    out = sync_closing_odds_from_results(results, tmp_db)
    assert out["updated"] == 1
    bet = load_live_bets(tmp_db).iloc[0]
    assert bet["closing_odds"] == pytest.approx(3.90)
