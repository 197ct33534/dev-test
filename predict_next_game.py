r"""Dự đoán trận EPL sắp đá với odds nhà cái thực tế → Match Card.

Chạy:
    python predict_next_game.py

Hoặc import:
    from src.recommender import predict_upcoming_match, format_match_card
"""

from __future__ import annotations

import sys

from src.recommender import format_match_card, predict_upcoming_match


# ---------------------------------------------------------------------------
# Mẫu odds giả định: Arsenal (nhà) vs Chelsea (khách)
# Thay số này bằng odds thực tế từ nhà cái khi dùng hàng ngày.
# ---------------------------------------------------------------------------

SAMPLE_HOME = "Arsenal"
SAMPLE_AWAY = "Chelsea"

SAMPLE_BOOKMAKER_ODDS: dict[str, float] = {
    # 1X2
    "odds_home": 1.85,
    "odds_draw": 3.60,
    "odds_away": 4.20,
    # Tài / Xỉu bàn thắng
    "line": 2.5,
    "odds_over": 1.90,
    "odds_under": 1.95,
    # Asian Handicap (mốc đội nhà)
    "handicap": -0.75,
    "odds_home_handicap": 1.95,
    "odds_away_handicap": 1.90,
}


def main() -> None:
    # Windows console thường dùng cp1252 — bật UTF-8 để in tiếng Việt / emoji.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    print("Đang tải 3 mùa EPL + fit Dixon–Coles… (có thể mất ~10–20s)")
    prediction = predict_upcoming_match(
        SAMPLE_HOME,
        SAMPLE_AWAY,
        SAMPLE_BOOKMAKER_ODDS,
        n_seasons=3,
        xi=0.0018,
        min_ev=0.05,
        kelly_fraction=0.10,
    )
    print(format_match_card(prediction))


if __name__ == "__main__":
    main()
