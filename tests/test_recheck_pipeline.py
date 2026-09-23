"""Unit tests for post-match settle / CLV / Brier helpers (no network)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.journal import (
    add_live_bet,
    compute_clv_pct,
    compute_settled_brier,
    evaluate_pending_against_results,
    find_match_result,
    journal_bankroll_summary,
    kickoff_has_passed,
    load_live_bets,
    parse_selection_line,
    resolve_bet_outcome,
    summarise_evaluation_session,
)
from src.notifier import format_recheck_summary


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test_journal.db"


def _results_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2026-09-20 15:00"),
                "HomeTeam": "Arsenal",
                "AwayTeam": "Chelsea",
                "FTHG": 2,
                "FTAG": 1,
                "FTR": "H",
                "HC": 7,
                "AC": 4,
                "B365H": 1.80,
                "B365D": 3.60,
                "B365A": 4.20,
            },
            {
                "Date": pd.Timestamp("2026-09-20 17:30"),
                "HomeTeam": "Liverpool",
                "AwayTeam": "Everton",
                "FTHG": 1,
                "FTAG": 1,
                "FTR": "D",
                "HC": 5,
                "AC": 5,
                "B365H": 1.55,
                "B365D": 4.00,
                "B365A": 6.00,
            },
        ]
    )


def test_parse_selection_line() -> None:
    assert parse_selection_line("Over 2.5") == ("Over", 2.5)
    assert parse_selection_line("Under 10.5") == ("Under", 10.5)
    assert parse_selection_line("AH Home -0.25") == ("AH Home", -0.25)
    assert parse_selection_line("AH Away +0.25") == ("AH Away", 0.25)
    assert parse_selection_line("Home") == ("Home", None)


def test_compute_clv_pct() -> None:
    assert compute_clv_pct(2.10, 2.00) == pytest.approx(0.05)
    with pytest.raises(ValueError):
        compute_clv_pct(1.0, 2.0)


def test_kickoff_has_passed_with_grace() -> None:
    now = datetime(2026, 9, 20, 18, 0, tzinfo=timezone.utc)
    assert kickoff_has_passed(
        "2026-09-20 15:00", now=now, grace=timedelta(hours=2)
    )
    assert not kickoff_has_passed(
        "2026-09-20 17:00", now=now, grace=timedelta(hours=2)
    )
    # Date-only: treated as 15:00 UTC + grace
    assert kickoff_has_passed("2026-09-19", now=now, grace=timedelta(hours=2))
    # Same calendar day, before 15:00+grace → not yet settleable
    early = datetime(2026, 9, 20, 16, 0, tzinfo=timezone.utc)
    assert not kickoff_has_passed("2026-09-20", now=early, grace=timedelta(hours=2))
    assert kickoff_has_passed("2026-09-20", now=now, grace=timedelta(hours=2))


def test_resolve_1x2_ou_ah() -> None:
    status, pnl = resolve_bet_outcome(
        market="1X2",
        selection="Home",
        odds=2.0,
        stake=10.0,
        fthg=2,
        ftag=1,
        ftr="H",
    )
    assert status == "WIN"
    assert pnl == pytest.approx(10.0)

    status, pnl = resolve_bet_outcome(
        market="OU",
        selection="Over 2.5",
        odds=1.90,
        stake=10.0,
        fthg=2,
        ftag=1,
        ftr="H",
    )
    assert status == "WIN"
    assert pnl == pytest.approx(9.0)

    status, pnl = resolve_bet_outcome(
        market="OU",
        selection="Under 2.5",
        odds=1.90,
        stake=10.0,
        fthg=2,
        ftag=1,
        ftr="H",
    )
    assert status == "LOSS"
    assert pnl == pytest.approx(-10.0)

    # Home -0.5 covers 2-1
    status, pnl = resolve_bet_outcome(
        market="AH",
        selection="AH Home -0.5",
        odds=1.95,
        stake=10.0,
        fthg=2,
        ftag=1,
        ftr="H",
    )
    assert status == "WIN"

    # Away +0.5 when home won 2-1 → away loses
    status, pnl = resolve_bet_outcome(
        market="AH",
        selection="AH Away +0.5",
        odds=1.95,
        stake=10.0,
        fthg=2,
        ftag=1,
        ftr="H",
    )
    assert status == "LOSS"


def test_resolve_corners_skip_without_hc() -> None:
    with pytest.raises(ValueError, match="HC/AC"):
        resolve_bet_outcome(
            market="Corners",
            selection="Over 10.5",
            odds=1.90,
            stake=5.0,
            fthg=1,
            ftag=0,
            ftr="H",
            hc=None,
            ac=None,
        )

    status, pnl = resolve_bet_outcome(
        market="Corners",
        selection="Over 10.5",
        odds=1.90,
        stake=5.0,
        fthg=1,
        ftag=0,
        ftr="H",
        hc=7,
        ac=5,
    )
    assert status == "WIN"
    assert pnl == pytest.approx(4.5)


def test_find_match_result() -> None:
    res = _results_frame()
    row = find_match_result(
        res, home_team="Arsenal", away_team="Chelsea", match_date="2026-09-20"
    )
    assert row is not None
    assert int(row["FTHG"]) == 2
    assert find_match_result(
        res, home_team="Nobody", away_team="Else", match_date="2026-09-20"
    ) is None


def test_evaluate_pending_settles_and_brier(tmp_db: Path) -> None:
    past = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d %H:%M")
    add_live_bet(
        home_team="Arsenal",
        away_team="Chelsea",
        selection="Home",
        odds=2.0,
        stake_amount=10.0,
        match_date=past,
        market="1X2",
        p_model=0.55,
        db_path=tmp_db,
    )
    add_live_bet(
        home_team="Liverpool",
        away_team="Everton",
        selection="Home",
        odds=1.70,
        stake_amount=10.0,
        match_date=past,
        market="1X2",
        p_model=0.60,
        db_path=tmp_db,
    )
    # Future kickoff — must stay PENDING
    future = (datetime.now(timezone.utc) + timedelta(days=3)).strftime("%Y-%m-%d %H:%M")
    add_live_bet(
        home_team="Spurs",
        away_team="Wolves",
        selection="Away",
        odds=3.0,
        stake_amount=5.0,
        match_date=future,
        market="1X2",
        p_model=0.40,
        db_path=tmp_db,
    )

    res = _results_frame()
    # Align result dates to the past kickoff day used above
    day = past[:10]
    res = res.copy()
    res.loc[0, "Date"] = pd.Timestamp(f"{day} 15:00")
    res.loc[1, "Date"] = pd.Timestamp(f"{day} 17:30")

    session = evaluate_pending_against_results(res, tmp_db, sync_clv=True)
    assert session["settled"] == 2
    assert session["wins"] == 1  # Arsenal Home
    assert session["losses"] == 1  # Liverpool Home on draw
    assert session["pnl"] == pytest.approx(10.0 - 10.0)

    bets = load_live_bets(tmp_db)
    pending = bets.loc[bets["status"] == "PENDING"]
    assert len(pending) == 1
    assert pending.iloc[0]["home_team"] == "Spurs"

    # CLV backfilled for Arsenal Home via B365H
    arsenal = bets.loc[
        (bets["home_team"] == "Arsenal") & (bets["status"] == "WIN")
    ].iloc[0]
    assert arsenal["closing_odds"] == pytest.approx(1.80)
    assert arsenal["clv_pct"] == pytest.approx(compute_clv_pct(2.0, 1.80))

    brier = compute_settled_brier(db_path=tmp_db)
    # WIN p=0.55 → (0.55-1)^2; LOSS p=0.60 → (0.60-0)^2
    assert brier == pytest.approx(((0.55 - 1) ** 2 + (0.60 - 0) ** 2) / 2)

    summary = summarise_evaluation_session(
        session, db_path=tmp_db, initial_bankroll=1000.0
    )
    assert summary["journal"]["n_settled"] == 2
    assert summary["journal"]["realised_pnl"] == pytest.approx(0.0)
    assert summary["brier"] == pytest.approx(brier)

    bank = journal_bankroll_summary(1000.0, tmp_db)
    assert bank["n_pending"] == 1
    assert "roi" in bank
    assert "pushes" in bank


def test_evaluate_idempotent(tmp_db: Path) -> None:
    past = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d %H:%M")
    add_live_bet(
        home_team="Arsenal",
        away_team="Chelsea",
        selection="Home",
        odds=2.0,
        stake_amount=10.0,
        match_date=past,
        db_path=tmp_db,
    )
    res = _results_frame()
    day = past[:10]
    res = res.copy()
    res.loc[0, "Date"] = pd.Timestamp(f"{day} 15:00")

    first = evaluate_pending_against_results(res, tmp_db, sync_clv=False)
    second = evaluate_pending_against_results(res, tmp_db, sync_clv=False)
    assert first["settled"] == 1
    assert second["settled"] == 0


def test_format_recheck_summary() -> None:
    text = format_recheck_summary(
        league="EPL",
        settled=3,
        wins=2,
        losses=1,
        pushes=0,
        session_pnl=12.5,
        realised_pnl=12.5,
        roi=0.042,
        avg_clv=0.018,
        brier=0.214,
        retrain_ok=True,
        retrain_detail="DC + LGBM(calibrated)",
    )
    assert "[EPL Recheck]" in text
    assert "2W/1L/0P" in text
    assert "Retrain" in text
    assert "OK" in text
