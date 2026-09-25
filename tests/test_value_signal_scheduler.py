"""Unit tests for value-signal filter, dedupe store, and Telegram HTML format."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.api.services.scheduler import (
    filter_hot_signals,
    is_hot_signal,
    kickoff_within_hours,
    market_line_delta,
)
from src.bot.notifier import format_hot_value_signal, webapp_inline_keyboard
from src.bot.signal_store import (
    already_notified,
    list_active_chat_ids,
    make_signal_key,
    mark_notified,
    upsert_subscriber,
)


def _ko_in(hours: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()


def test_is_hot_signal_ev_threshold() -> None:
    assert is_hot_signal({"ev_pct": 8.0, "market": "1X2"}, min_ev_pct=8.0)
    assert is_hot_signal({"ev": 0.09, "market": "1X2"}, min_ev_pct=8.0)
    assert not is_hot_signal({"ev_pct": 7.9, "market": "1X2"}, min_ev_pct=8.0)


def test_is_hot_signal_line_delta() -> None:
    bet = {
        "ev_pct": 2.0,
        "market": "OU",
        "ou_line_delta": 0.5,
        "ah_line_delta": 0.0,
    }
    assert is_hot_signal(bet, min_ev_pct=8.0, min_line_delta=0.5)
    bet_low = {**bet, "ou_line_delta": 0.4}
    assert not is_hot_signal(bet_low, min_ev_pct=8.0, min_line_delta=0.5)


def test_market_line_delta_prefers_market() -> None:
    bet = {"market": "AH", "ou_line_delta": 1.0, "ah_line_delta": -0.25}
    assert market_line_delta(bet) == pytest.approx(0.25)
    bet_ou = {"market": "OU", "ou_line_delta": -0.75, "ah_line_delta": 2.0}
    assert market_line_delta(bet_ou) == pytest.approx(0.75)


def test_kickoff_within_hours() -> None:
    now = datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc)
    inside = {
        "kickoff": (now + timedelta(hours=12)).isoformat(),
    }
    outside = {
        "kickoff": (now + timedelta(hours=30)).isoformat(),
    }
    past = {
        "kickoff": (now - timedelta(hours=1)).isoformat(),
    }
    assert kickoff_within_hours(inside, hours=24, now=now)
    assert not kickoff_within_hours(outside, hours=24, now=now)
    assert not kickoff_within_hours(past, hours=24, now=now)


def test_is_hot_signal_data_score_gate() -> None:
    hot = {"ev_pct": 10.0, "market": "1X2", "aggregate_data_score": 80.0}
    cold = {"ev_pct": 10.0, "market": "1X2", "aggregate_data_score": 60.0}
    legacy = {"ev_pct": 10.0, "market": "1X2"}  # missing score → still allowed
    assert is_hot_signal(hot, min_ev_pct=5.0, min_data_score=75.0)
    assert not is_hot_signal(cold, min_ev_pct=5.0, min_data_score=75.0)
    assert is_hot_signal(legacy, min_ev_pct=5.0, min_data_score=75.0)
    assert is_hot_signal(cold, min_ev_pct=5.0, min_data_score=0.0)


def test_filter_hot_signals_combines_rules() -> None:
    now = datetime(2026, 9, 23, 10, 0, tzinfo=timezone.utc)
    bets = [
        {
            "match_id": "a",
            "market": "1X2",
            "selection": "H",
            "ev_pct": 10.0,
            "kickoff": (now + timedelta(hours=5)).isoformat(),
        },
        {
            "match_id": "b",
            "market": "OU",
            "selection": "Over",
            "ev_pct": 1.0,
            "ou_line_delta": 0.6,
            "kickoff": (now + timedelta(hours=5)).isoformat(),
        },
        {
            "match_id": "c",
            "market": "1X2",
            "selection": "A",
            "ev_pct": 3.0,
            "kickoff": (now + timedelta(hours=5)).isoformat(),
        },
        {
            "match_id": "d",
            "market": "1X2",
            "selection": "D",
            "ev_pct": 12.0,
            "kickoff": (now + timedelta(hours=48)).isoformat(),
        },
    ]
    hot = filter_hot_signals(bets, min_ev_pct=8.0, min_line_delta=0.5, now=now)
    ids = {b["match_id"] for b in hot}
    assert ids == {"a", "b"}


def test_signal_store_dedupe_and_subscribers(tmp_path: Path) -> None:
    db = tmp_path / "notified_signals.db"
    upsert_subscriber(111, username="alice", db_path=db)
    upsert_subscriber(222, first_name="Bob", db_path=db)
    assert list_active_chat_ids(db) == ["111", "222"]

    key = make_signal_key(match_id="m1", market="OU", selection="Over 2.5")
    assert mark_notified(key, match_id="m1", market="OU", selection="Over 2.5", db_path=db)
    assert already_notified(key, db_path=db)
    assert not mark_notified(key, db_path=db)


def test_format_hot_value_signal_html_escapes() -> None:
    bet = {
        "home": "A <B>",
        "away": "C & D",
        "competition": "EPL",
        "kickoff_vn": "23/09 19:00",
        "market": "OU",
        "selection": "Over 2.5",
        "odds": 1.95,
        "ev_pct": 9.5,
        "model_fair_line": "Tài Xỉu 2.75",
        "bookie_market_line": "Tài Xỉu 2.5",
        "ai_reasons": ["Form tốt", "Line lệch <0.5>"],
    }
    text = format_hot_value_signal(bet)
    assert "🚨" in text and "TÍN HIỆU VALUE BET NÓNG" in text
    assert "A &lt;B&gt;" in text
    assert "C &amp; D" in text
    assert "EV = +9.5%" in text
    assert "Kèo Đề Xuất: Tài Xỉu 2.75 vs Nhà cái: Tài Xỉu 2.5" in text
    assert "Form tốt" in text
    assert "&lt;0.5&gt;" in text
    assert "AI Pick: OU - Over 2.5 @ 1.95" in text


def test_webapp_inline_keyboard() -> None:
    kb = webapp_inline_keyboard("https://example.com/webapp/")
    btn = kb["inline_keyboard"][0][0]
    assert btn["text"] == "🔥 Mở WebApp Soi Kèo"
    assert btn["web_app"]["url"] == "https://example.com/webapp/"
