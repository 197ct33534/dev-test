# Football Value Betting — Multi-League Quant Stack

Estimate match probabilities (Dixon–Coles + LightGBM ensemble), compare them
with bookmaker odds and surface **value bets** (EV ≥ 5%) with a conservative
fractional-Kelly stake. Covers 1X2, goal totals (O/U incl. quarter lines),
Asian Handicap and corners (CornerPredictor).

> Full project report (Vietnamese): [`BAO_CAO_DU_AN.md`](BAO_CAO_DU_AN.md) ·
> portfolio write-up: [`PORTFOLIO.md`](PORTFOLIO.md) · deploy: [`DEPLOY.md`](DEPLOY.md)

## Leagues

Configured in [`config/leagues.json`](config/leagues.json) — adding a league is a
config change (Flashscore id, seasons, optional football-data / Fotmob ids):

| Key | Competition | History source |
|-----|-------------|----------------|
| `EPL` | Premier League | football-data.co.uk (E0) |
| `LALIGA` | LaLiga | football-data.co.uk (SP1) + Flashscore |
| `UWCL` | UEFA Women's Champions League | Fotmob |
| `J_LEAGUE_CUP` | J.League Cup | Flashscore |
| `EMPERORS_CUP` | Emperor's Cup (Japan) | Flashscore |

All results / upcoming fixtures / odds snapshots are consolidated in
`data/global_matches.db` (multi-competition SQLite; legacy per-league DBs such
as `data/epl_matches.db` are still read as a fallback). Upcoming odds come from
Flashscore (bet365 preferred).

## Stake policy

Values live in [`src/config.py`](src/config.py) — UI, API, bot and backtests read them from there:

| Rule | Value |
|------|-------|
| Fair odds | `1 / P_model` |
| EV | `P_model × Odds − 1` (AH / O/U use exact settlement EV incl. push / half) |
| Recommend only when | `EV ≥ 0.05` (`DEFAULT_MIN_EV`), sanity cap `EV ≤ 0.50` |
| Kelly | fractional Kelly **0.10** (`DEFAULT_KELLY_FRACTION`) |
| Max stake | **1% of bankroll** per bet (`MAX_STAKE_PCT`), 0.5% for thin-history teams |
| Daily cap | 5 bets per VN calendar day, one per match (`MAX_BETS_PER_DAY`) |
| Ensemble | `P = (1 − w)·P_DC + w·P_ML`, `w = 0.20` (`DEFAULT_W_ML`) |

## Setup

```bash
python -m pip install -r requirements.txt
cp .env.example .env   # Telegram token, WEBAPP_URL, … (optional for the UI)
```

## Run

| Component | Command | Notes |
|-----------|---------|-------|
| Streamlit UI | `python -m streamlit run app.py` | Lite / Pro modes, Top-20 value bets (VN today+tomorrow), Japan schedule, journal, backtest. Set `PUBLIC_BASE_URL` for absolute share links. |
| FastAPI | `python run_api.py` | `/health`, `/docs`, `/api/v1/*`, `/api/v2/*` (Telegram WebApp `initData` auth; `TELEGRAM_AUTH_DISABLED=true` for local dev), WebApp at `/webapp/` |
| Telegram bot | `python -m src.bot.telegram_bot` | Polling + WebApp button + value alerts (`ENABLE_VALUE_SCHEDULER`) |
| Workers | `python -m src.workers.fixtures_worker` · `python -m src.workers.settlement_worker` · `python -m src.workers.data_monitor` | Fixtures/odds refresh, settlement + CLV, data-quality monitor |
| CLI scans | `python scripts/auto_scanner.py`, `python scripts/fetch_league.py`, `python scripts/backfill_leagues.py` | Scheduled scan / league import / backfill |
| Dashboard | `cd dashboard && npm install && npm run dev` | React + Vite prototype — **mock data only (WIP)**, not wired to the API yet |

Docker / VPS (nginx + web_api + workers + telegram_bot + certbot): see
[`DEPLOY.md`](DEPLOY.md) and `docker-compose.yml`.

## Tests

```bash
python -m pytest            # pytest.ini → testpaths = tests
cd dashboard && npx tsc -b --noEmit
```

Manual smoke runs (network / real data, not part of pytest):

```bash
python scripts/run_recommender_smoke.py
python scripts/run_backtest_smoke.py          # bets go to a temp DB
python scripts/run_backtest_smoke.py --db X   # REPLACES bet_history in X
python predict_next_game.py                   # legacy EPL-only demo
```

## Project layout

| Path | Role |
|------|------|
| `app.py` | Streamlit UI |
| `run_api.py`, `src/api/` | FastAPI app (v1 + v2 routes, auth, analytics) |
| `src/bot/`, `src/notifier.py` | Telegram bot, alert formatting / sending |
| `src/workers/` | Long-running fixtures / settlement / monitoring workers |
| `src/data_loader.py`, `src/fetchers/`, `src/pipeline/` | Download, clean and sync data into `global_matches.db` |
| `src/global_db.py`, `src/league_registry.py` | Multi-league DB + league registry (`config/leagues.json`) |
| `src/dixon_coles.py`, `src/models/` | Dixon–Coles fit (time decay ξ, weak-tier priors, league weights) |
| `src/ml_model.py`, `src/features.py`, `src/calibration.py` | LightGBM 1X2 model, point-in-time features, calibration |
| `src/recommender.py`, `src/scanner.py`, `src/strategy.py` | EV / fair odds / Kelly, Top-N scan, risk filters |
| `src/corner_model.py` | Corner predictor (Poisson / GBM) |
| `src/backtester.py`, `src/backtest/`, `src/validation/` | Walk-forward backtests |
| `src/execution/`, `src/risk/`, `src/services/` | Paper trading, Kelly engine / exposure, settlement |
| `src/japan_schedule.py` | J.League Cup / Emperor's Cup schedule + picks |
| `dashboard/` | React dashboard prototype (mock data) |
| `tests/` | pytest suite |

## Disclaimer

Decision-support / research tool. No model guarantees profit; bet responsibly.
