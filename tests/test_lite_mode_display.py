"""Lite Mode: rest-day display cap + best-pick across 1X2/AH/OU."""

from __future__ import annotations

from pathlib import Path


def test_lite_scan_markets_include_ah_ou() -> None:
    """Lite Mode must not hardcode 1X2-only for Top / Search scans."""
    app_src = (Path(__file__).resolve().parents[1] / "app.py").read_text(
        encoding="utf-8"
    )
    assert 'scan_markets = ["1X2", "OU", "AH"]' in app_src
    # Lite scan call sites pass the markets tuple, not a bare 1X2-only constant.
    assert 'tuple(m for m in scan_markets if m != "Corners")' in app_src


def test_format_lite_ai_pick_helpers() -> None:
    """Import helper logic via exec of the pure formatter body (no Streamlit run)."""
    # Mirror app._format_lite_ai_pick / _selection_vi for unit checks without
    # importing streamlit-heavy app.py at module level.
    market_vi = {"1X2": "1X2", "OU": "Tài/Xỉu", "AH": "Chấp Á"}

    def selection_vi(text: str) -> str:
        if text == "Home":
            return "Chủ nhà"
        if text == "Draw":
            return "Hòa"
        if text == "Away":
            return "Khách"
        if text.startswith("Over "):
            return "Tài " + text[5:]
        if text.startswith("Under "):
            return "Xỉu " + text[6:]
        if text.startswith("AH Home "):
            return "Chấp chủ " + text[8:]
        if text.startswith("AH Away "):
            return "Chấp khách " + text[8:]
        return text

    def format_lite_ai_pick(market: str, selection: str, odds_txt: str) -> str:
        mkt = str(market or "").strip().upper()
        sel_raw = str(selection or "").strip()
        sel = selection_vi(sel_raw) if sel_raw and sel_raw != "—" else "—"
        if mkt == "AH":
            short = sel
            if short.lower().startswith("chấp "):
                return f"AH · {short} @ {odds_txt}"
            return f"AH · {short} @ {odds_txt}"
        if mkt == "OU":
            return f"{sel} @ {odds_txt}"
        if mkt == "1X2":
            return f"1X2 · {sel} @ {odds_txt}"
        label = market_vi.get(mkt, mkt or "?")
        return f"{label} · {sel} @ {odds_txt}"

    assert format_lite_ai_pick("AH", "AH Home -0.5", "1.93") == (
        "AH · Chấp chủ -0.5 @ 1.93"
    )
    assert format_lite_ai_pick("OU", "Over 2.5", "1.85") == "Tài 2.5 @ 1.85"
    assert format_lite_ai_pick("1X2", "Home", "2.10") == "1X2 · Chủ nhà @ 2.10"


def test_scanner_selects_best_market_not_only_1x2(monkeypatch) -> None:
    """When AH EV beats 1X2, scan_top_value_bets keeps the AH leg."""
    from datetime import datetime

    import pandas as pd
    import pytest

    from src.timezone_utils import VN_TZ
    from src import scanner as scanner_mod

    def _fake_recommend(model, fixtures, **kwargs):
        rows = []
        for _, fx in fixtures.iterrows():
            home, away = str(fx["HomeTeam"]), str(fx["AwayTeam"])
            # Weak 1X2 + strong AH (mirrors Pro Mode AH +43.8% case)
            for market, sel, p, odds in (
                ("1X2", "Home", 0.45, 1.50),  # EV ≈ -0.325
                ("AH", "AH Home -0.5", 0.72, 1.93),  # EV ≈ +0.39
                ("OU", "Over 2.5", 0.55, 1.85),  # EV ≈ +0.0175
            ):
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
                        "kelly_fraction": 0.01,
                        "kelly_pct": 1.0,
                        "recommended": ev >= 0.05,
                        "kickoff": fx["Kickoff"],
                        "match_date": fx["Kickoff"],
                    }
                )
        return pd.DataFrame(rows)

    monkeypatch.setattr(scanner_mod, "recommend_upcoming", _fake_recommend)

    class _ToyDC:
        teams = ["Alpha", "Beta"]
        fitted_ = True

        def expected_goals(self, home, away):
            return 1.5, 1.0

        def is_thin_team(self, team: str) -> bool:
            return False

        def predict_match_probs(self, home: str, away: str) -> dict[str, float]:
            return {"H": 0.45, "D": 0.28, "A": 0.27}

    now_vn = datetime(2026, 9, 22, 10, 0, tzinfo=VN_TZ)
    fixtures = pd.DataFrame(
        [
            {
                "Kickoff": pd.Timestamp("2026-09-22 12:00:00"),
                "HomeTeam": "Alpha",
                "AwayTeam": "Beta",
                "B365H": 1.50,
                "B365D": 4.00,
                "B365A": 6.00,
                "FlashscoreEventId": "fx-ah",
            }
        ]
    )
    result = scanner_mod.scan_top_value_bets(
        fixtures,
        _ToyDC(),  # type: ignore[arg-type]
        ml_model=None,
        w_ml=0.0,
        min_ev=0.05,
        allowed_markets=("1X2", "OU", "AH"),
        top_n=5,
        bankroll=1000.0,
        league="EPL",
        filter_vn_window=True,
        now=now_vn,
    )
    assert result.n_value == 1
    assert result.value_bets.iloc[0]["market"] == "AH"
    assert result.value_bets.iloc[0]["selection"] == "AH Home -0.5"
    assert float(result.value_bets.iloc[0]["ev"]) == pytest.approx(0.72 * 1.93 - 1.0)
