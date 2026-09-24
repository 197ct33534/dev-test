# EPL Value Betting & Corner Recommender

Predict Premier League 1X2 / totals / Asian Handicap with a Dixon–Coles model,
compare to bookmaker odds, and surface value bets (EV ≥ 5%) with Quarter-Kelly stakes.

## Setup

```bash
cd d:\nghia\score
python -m pip install -r requirements.txt
```

## Run the app

```bash
python -m streamlit run app.py
```

Browser opens at `http://localhost:8501`.

## Practical workflow

1. **Sidebar** — seasons, ξ, min EV %, Kelly fraction, bankroll.
2. **Upcoming** — auto-load EPL fixtures (Flashscore calendar via Fotmob) + book
   odds from the [Flashscore Odds tab](https://www.flashscore.com/football/england/premier-league/fixtures/)
   (prefer bet365), with ESPN / [`fixtures.csv`](https://www.football-data.co.uk/fixtures.csv)
   filling gaps.
3. **Evaluate match** — pick a fixture (odds auto-filled from API) or enter manually.
4. **Results history** — collected finished matches used to fit the model.
5. Optional: Backtest scan, Corners, Team strengths.

### Data sources

| Feed | URL | Content |
|------|-----|---------|
| Results | `mmz4281/{season}/E0.csv` → cache `data/epl_matches.db` | Finished EPL matches + closing odds (SQLite offline after first fetch) |
| Upcoming schedule | [Flashscore PL fixtures](https://www.flashscore.com/football/england/premier-league/fixtures/) (via Fotmob API) | Remaining season fixtures |
| Live odds (primary) | Flashscore Odds GraphQL (`oce`) | bet365 / WH / … 1X2 · O/U · AH (same as Match → Odds) |
| Odds fallback | ESPN scoreboard + `fixtures.csv` | DraftKings / Bet365 when Flashscore gap |

## CLI smoke test

```bash
python test_recommender.py
```

## Project layout

| Path | Role |
|------|------|
| `app.py` | Streamlit UI |
| `src/data_loader.py` | Download & clean EPL CSVs |
| `src/dixon_coles.py` | Dixon–Coles fit + 1X2 / O/U / AH probs |
| `src/recommender.py` | EV, Fair Odds, Quarter Kelly, `ValueBetRecommender` |
| `src/corner_model.py` | Poisson corner GLM |
| `test_recommender.py` | End-to-end integration check |
| `DEPLOY.md` | Docker / VPS 24/7 deploy (nginx + web + bot) |

## Rules (from `.cursorrules`)

- Fair Odds = `1 / P_model`
- EV = `(P_model × Odds) - 1`
- Recommend only when `EV ≥ 0.05` (5%)
