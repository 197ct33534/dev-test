"""Unit tests for post-match settlement + performance analytics."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.journal import (
    add_live_bet,
    ensure_live_bets_table,
    load_live_bets,
    settle_live_bet,
)
from src.services.settlement_service import (
    compute_performance_analytics,
    payout_from_outcome,
    resolve_bet_settlement,
    settle_ah_detailed,
    settle_completed_matches,
    settle_ou_detailed,
    settle_pending_against_frame,
)


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    path = tmp_path / "journal.db"
    ensure_live_bets_table(path)
    return path


# ---------------------------------------------------------------------------
# Pure settlement math
# ---------------------------------------------------------------------------


def test_payout_half_win_half_loss_void() -> None:
    stake, odds = 100.0, 1.90
    pnl, pay = payout_from_outcome("HALF_WIN", odds, stake)
    assert pnl == pytest.approx(45.0)  # 0.5 * 100 * 0.9
    assert pay == pytest.approx(145.0)  # 0.5*190 + 0.5*100

    pnl, pay = payout_from_outcome("HALF_LOSS", odds, stake)
    assert pnl == pytest.approx(-50.0)
    assert pay == pytest.approx(50.0)

    pnl, pay = payout_from_outcome("VOID", odds, stake)
    assert pnl == pytest.approx(0.0)
    assert pay == pytest.approx(100.0)

    pnl, pay = payout_from_outcome("PUSH", odds, stake)
    assert pnl == pytest.approx(0.0)
    assert pay == pytest.approx(100.0)


def test_1x2_settlement() -> None:
    st, pnl, pay = resolve_bet_settlement(
        market="1X2",
        selection="Home",
        odds=2.0,
        stake=10.0,
        fthg=2,
        ftag=1,
        ftr="H",
    )
    assert st == "WIN"
    assert pnl == pytest.approx(10.0)
    assert pay == pytest.approx(20.0)

    st, pnl, _ = resolve_bet_settlement(
        market="1X2",
        selection="Draw",
        odds=3.2,
        stake=10.0,
        fthg=2,
        ftag=1,
        ftr="H",
    )
    assert st == "LOSS"
    assert pnl == pytest.approx(-10.0)


def test_ou_quarter_half_win() -> None:
    # Over 2.25 with exactly 3 goals → win both halves? 3 > 2.0 and 3 > 2.5 → WIN
    # Over 2.25 with exactly 2 goals → push on 2.0, lose on 2.5 → HALF_LOSS
    st, pnl, pay = settle_ou_detailed("Over", 1, 1, 2.25, 1.90, 100.0)
    assert st == "HALF_LOSS"
    assert pnl == pytest.approx(-50.0)
    assert pay == pytest.approx(50.0)

    # Under 2.25 with 2 goals → push 2.0 + win 2.5 → HALF_WIN
    st, pnl, pay = settle_ou_detailed("Under", 1, 1, 2.25, 1.90, 100.0)
    assert st == "HALF_WIN"
    assert pnl == pytest.approx(45.0)


def test_ou_push_void_line() -> None:
    # Over 2.5 with exactly 3 → win; with 2 → lose; with 2.5 whole line push
    st, pnl, pay = settle_ou_detailed("Over", 1, 1, 2.0, 1.95, 50.0)
    assert st == "PUSH"
    assert pnl == pytest.approx(0.0)
    assert pay == pytest.approx(50.0)


def test_ah_half_win_half_loss_void() -> None:
    # Home -0.25, score 1-1 → margin = 0 + (-0.0/-0.5) split:
    # -0.0 → push, -0.5 → lose → HALF_LOSS
    st, pnl, pay = settle_ah_detailed("AH Home", 1, 1, -0.25, 1.95, 100.0)
    assert st == "HALF_LOSS"
    assert pnl == pytest.approx(-50.0)

    # Home -0.25, score 1-0 → margin +1-0.0 / +1-0.5 → both win
    st, pnl, pay = settle_ah_detailed("AH Home", 1, 0, -0.25, 1.95, 100.0)
    assert st == "WIN"
    assert pnl == pytest.approx(95.0)

    # Home 0.0, score 1-1 → push / VOID stake return
    st, pnl, pay = settle_ah_detailed("AH Home", 1, 1, 0.0, 1.90, 80.0)
    assert st == "PUSH"
    assert pnl == pytest.approx(0.0)
    assert pay == pytest.approx(80.0)

    # Home -0.25, score 2-1 → win on both halves of -0/-0.5
    # Away +0.25 on 1-1: mirror of home -0.25 on 1-1 → HALF_WIN
    st, pnl, pay = settle_ah_detailed("AH Away", 1, 1, -0.25, 1.90, 100.0)
    # Wait: settle_ah_detailed with Away mirrors home settlement on same handicap.
    # Home -0.25 @ 1-1 = HALF_LOSS → Away mirror = HALF_WIN
    assert st == "HALF_WIN"
    assert pnl == pytest.approx(45.0)


def test_settle_live_bet_writes_payout(tmp_db: Path) -> None:
    bet_id = add_live_bet(
        home_team="A",
        away_team="B",
        selection="Home",
        odds=2.0,
        stake_amount=20.0,
        match_date="2026-01-01 15:00",
        db_path=tmp_db,
    )
    settle_live_bet(bet_id, "HALF_WIN", tmp_db, pnl=9.0, payout=29.0)
    row = load_live_bets(tmp_db).iloc[0]
    assert row["status"] == "HALF_WIN"
    assert float(row["pnl"]) == pytest.approx(9.0)
    assert float(row["payout"]) == pytest.approx(29.0)


def test_settle_pending_against_frame(tmp_db: Path) -> None:
    past = (datetime.now(timezone.utc) - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M")
    add_live_bet(
        home_team="Arsenal",
        away_team="Chelsea",
        selection="Over 2.25",
        odds=1.90,
        stake_amount=100.0,
        match_date=past,
        market="OU",
        p_model=0.55,
        ev=0.08,
        db_path=tmp_db,
    )
    day = past[:10]
    results = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp(f"{day} 15:00"),
                "HomeTeam": "Arsenal",
                "AwayTeam": "Chelsea",
                "FTHG": 1,
                "FTAG": 1,
                "FTR": "D",
            }
        ]
    )
    session = settle_pending_against_frame(results, tmp_db)
    assert session["settled"] == 1
    assert session["half_losses"] == 1
    bet = load_live_bets(tmp_db).iloc[0]
    assert bet["status"] == "HALF_LOSS"
    assert float(bet["pnl"]) == pytest.approx(-50.0)
    assert float(bet["payout"]) == pytest.approx(50.0)


def test_settle_completed_matches_uses_grace(tmp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Kickoff only 30 min ago → should NOT settle with 120 min grace.
    recent = (datetime.now(timezone.utc) - timedelta(minutes=30)).strftime(
        "%Y-%m-%d %H:%M"
    )
    add_live_bet(
        home_team="Arsenal",
        away_team="Chelsea",
        selection="Home",
        odds=2.0,
        stake_amount=10.0,
        match_date=recent,
        db_path=tmp_db,
    )
    day = recent[:10]
    fake = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp(f"{day} 12:00"),
                "HomeTeam": "Arsenal",
                "AwayTeam": "Chelsea",
                "FTHG": 2,
                "FTAG": 0,
                "FTR": "H",
            }
        ]
    )
    monkeypatch.setattr(
        "src.services.settlement_service._load_ft_results",
        lambda *a, **k: fake,
    )
    out = settle_completed_matches(db_path=tmp_db, grace_minutes=120)
    assert out["settled"] == 0
    assert load_live_bets(tmp_db).iloc[0]["status"] == "PENDING"


def test_performance_aggregation() -> None:
    now = datetime.now(timezone.utc)
    df = pd.DataFrame(
        [
            {
                "status": "WIN",
                "pnl": 10.0,
                "stake_amount": 10.0,
                "ev": 0.10,
                "p_model": 0.60,
                "created_at": now.isoformat(),
                "league": "EPL",
            },
            {
                "status": "LOSS",
                "pnl": -10.0,
                "stake_amount": 10.0,
                "ev": 0.08,
                "p_model": 0.40,
                "created_at": now.isoformat(),
                "league": "EPL",
            },
            {
                "status": "HALF_WIN",
                "pnl": 4.5,
                "stake_amount": 10.0,
                "ev": 0.05,
                "p_model": 0.52,
                "created_at": now.isoformat(),
                "league": "EPL",
            },
            {
                "status": "PENDING",
                "pnl": 0.0,
                "stake_amount": 10.0,
                "ev": 0.12,
                "p_model": 0.55,
                "created_at": now.isoformat(),
                "league": "EPL",
            },
        ]
    )
    perf = compute_performance_analytics(df, league="EPL")
    assert perf["total_bets_placed"] == 4
    assert perf["total_bets_settled"] == 3
    # WIN + HALF_WIN = 2 decisive wins out of 3 (WIN/LOSS/HALF_WIN)
    assert perf["win_rate_percent"] == pytest.approx(100.0 * 2 / 3, rel=1e-3)
    assert perf["net_pnl"] == pytest.approx(4.5)
    assert perf["realized_roi_percent"] == pytest.approx(100.0 * 4.5 / 30.0, rel=1e-3)
    assert perf["ev_vs_realized_gap"] is not None
    assert perf["brier_score"] is not None


def test_performance_brier_insufficient() -> None:
    df = pd.DataFrame(
        [
            {
                "status": "PUSH",
                "pnl": 0.0,
                "stake_amount": 10.0,
                "ev": 0.05,
                "p_model": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        ]
    )
    perf = compute_performance_analytics(df)
    assert perf["brier_score"] is None
    assert perf["brier_note"]
