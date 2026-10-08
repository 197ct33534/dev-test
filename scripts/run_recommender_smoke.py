"""Integration smoke run: data → Dixon–Coles fit → ValueBetRecommender (EPL).

Not a pytest test (formerly ``test_recommender.py`` at repo root).
Usage: ``python scripts/run_recommender_smoke.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_epl_data  # noqa: E402
from src.dixon_coles import DixonColesModel  # noqa: E402
from src.recommender import ValueBetRecommender, format_recommendations  # noqa: E402


def main() -> None:
    print("=== 1) Load EPL data (2 seasons) ===")
    data = load_epl_data(n_seasons=2)
    print(f"Loaded {len(data)} matches, {data['HomeTeam'].nunique()} home teams")

    print("\n=== 2) Fit Dixon-Coles ===")
    model = DixonColesModel(xi=0.0018).fit(data)
    print(model.summary())

    home, away = "Arsenal", "Chelsea"
    if home not in model.teams or away not in model.teams:
        # Fall back to first two known teams if names differ in the CSV.
        home, away = model.teams[0], model.teams[1]
        print(f"(fallback fixture) {home} vs {away}")

    lam, mu = model.expected_goals(home, away)
    print(f"\n=== 3) {home} vs {away} ===")
    print(f"lambda={lam:.2f}, mu={mu:.2f}")
    print("1X2:", model.predict_match_probs(home, away))
    print("O/U 2.5:", {k: model.predict_over_under(home, away, 2.5)[k] for k in ("over", "under")})
    print("O/U 2.25:", {k: model.predict_over_under(home, away, 2.25)[k] for k in ("over", "under")})
    print(
        "AH -0.25:",
        {k: model.predict_asian_handicap(home, away, -0.25)[k] for k in ("home", "away")},
    )
    print(
        "AH -0.75:",
        {k: model.predict_asian_handicap(home, away, -0.75)[k] for k in ("home", "away")},
    )

    print("\n=== 4) ValueBetRecommender (hypothetical book odds) ===")
    # Slightly soft book prices so the demo surfaces at least some EV >= 5% edges.
    recommender = ValueBetRecommender(model, min_ev=0.05, kelly_fraction=0.25)
    value_bets = recommender.evaluate_match(
        home,
        away,
        odds_1x2={"H": 2.20, "D": 3.60, "A": 3.40},
        over_under=[
            {"line": 2.5, "over": 2.05, "under": 1.85},
            {"line": 2.25, "over": 1.95, "under": 1.95},
            {"line": 3.5, "over": 2.40, "under": 1.60},
        ],
        asian_handicap=[
            {"handicap": -0.25, "home": 2.05, "away": 1.85},
            {"handicap": -0.5, "home": 2.15, "away": 1.75},
            {"handicap": -0.75, "home": 2.30, "away": 1.65},
        ],
        only_value=True,
    )

    print(f"Recommendations (EV >= 5%): {len(value_bets)}")
    print(format_recommendations(value_bets))

    if value_bets:
        df = recommender.evaluate_match_dataframe(
            home,
            away,
            odds_1x2={"H": 2.20, "D": 3.60, "A": 3.40},
            over_under={"line": 2.5, "over": 2.05, "under": 1.85},
            asian_handicap={"handicap": -0.25, "home": 2.05, "away": 1.85},
            only_value=False,
        )
        cols = [
            "market",
            "selection",
            "p_model",
            "fair_odds",
            "bookmaker_odds",
            "ev_pct",
            "kelly_pct",
            "recommended",
        ]
        print("\nFull evaluation (sample markets):")
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
