"""Unit tests for Lite Mode AI reason generator (match insights)."""

from __future__ import annotations

import pandas as pd

from src.strategy import generate_match_insights


def test_generate_match_insights_full_features() -> None:
    data = {
        "home": "Arsenal",
        "away": "Chelsea",
        "home_rest_days": 5,
        "away_rest_days": 2,
        "home_matches_last_14d": 1,
        "away_matches_last_14d": 3,
        "home_rolling_xg_5": 1.85,
        "away_rolling_xg_5": 1.20,
        "home_is_rotation_risk": 0,
        "away_is_rotation_risk": 1,
        "p_model": 0.55,
        "odds": 2.05,
        "ev": 0.1275,
    }
    lines = generate_match_insights(data)
    assert isinstance(lines, list)
    assert 1 <= len(lines) <= 3
    assert any(line.startswith("Thể lực:") for line in lines)
    assert any("xG5" in line for line in lines)
    assert any(line.startswith("EV:") for line in lines)
    fitness = next(line for line in lines if line.startswith("Thể lực:"))
    assert "Arsenal" in fitness and "Chelsea" in fitness
    assert "tươi hơn" in fitness
    assert "xoay tua" in fitness
    ev_line = next(line for line in lines if line.startswith("EV:"))
    assert "55%" in ev_line
    assert "2.05" in ev_line


def test_generate_match_insights_missing_features() -> None:
    # Only EV inputs — fitness/form skipped, still no crash.
    lines = generate_match_insights(
        {"home": "A", "away": "B", "p_model": 0.48, "odds": 2.20, "ev_pct": 5.6}
    )
    assert len(lines) == 1
    assert lines[0].startswith("EV:")
    assert "48%" in lines[0]

    assert generate_match_insights({}) == []
    assert generate_match_insights(None) == []
    # Sparse Series with NaNs must not raise.
    sparse = pd.Series(
        {
            "home": "Liverpool",
            "away": "Everton",
            "home_rest_days": float("nan"),
            "away_rolling_xg_5": float("nan"),
            "p_model": float("nan"),
        }
    )
    assert generate_match_insights(sparse) == []


def test_generate_match_insights_form_goals_fallback() -> None:
    lines = generate_match_insights(
        {
            "home": "Tottenham",
            "away": "Fulham",
            "home_rolling_gf_5": 2.1,
            "away_rolling_gf_5": 1.0,
        }
    )
    assert len(lines) == 1
    assert "Phong độ" in lines[0] or "bàn" in lines[0]
    assert "Tottenham" in lines[0]
