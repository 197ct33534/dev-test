# Deploy Quant Engine (Docker / VPS) — Dockerization V2

24/7 stack: **nginx** → **web_api** (FastAPI, 2 uvicorn workers) + dedicated
workers + **telegram_bot** (polling + value alerts).

## Architecture notes

| Service | Role |
|---------|------|
| `web_api` | `uvicorn --workers 2 --proxy-headers` — API + WebApp static. Schedulers **forced off**. Healthcheck on `/health` |
| `fixtures_worker` | Every 20 min: refresh upcoming fixtures + odds (`FIXTURES_REFRESH_LEAGUES`) so prices are never older than `FIXTURES_MAX_AGE_MINUTES` |
| `data_monitor_worker` | Every 15 min: unmapped teams, freshness, PIT integrity |
| `settlement_worker` | Every 30 min: journal PnL + `SettlementEngineV2` (Pinnacle CLV) |
| `telegram_bot` | Telegram polling / WebApp + value alerts (`ENABLE_VALUE_SCHEDULER`) |
| `nginx` | Reverse proxy `DOMAIN` → `web_api:8000`; Certbot ACME on `:80` |

With multiple uvicorn workers, never enable `ENABLE_*_SCHEDULER` on `web_api`
(jobs would fire twice). Settlement and data monitoring live in workers;
value alerts live on `telegram_bot`.

## VPS quickstart

```bash
# 1. Clone / pull
git clone <your-repo> /opt/score && cd /opt/score

# 2. Env
cp .env.example .env
# Edit: DOMAIN, TELEGRAM_BOT_TOKEN, WEBAPP_URL=https://YOUR_DOMAIN/webapp/
#       ADMIN_TELEGRAM_IDS=<your numeric Telegram user id>
#       DASHBOARD_API_TOKENS=<random >=16 chars> (React dashboard outside Telegram)
# Keep APP_ENV=prod and TELEGRAM_AUTH_DISABLED=false (the API refuses to
# start with auth disabled in prod).
# Value alerts (telegram_bot): SIGNAL_MIN_EV=5, SIGNAL_MIN_DATA_SCORE=75
# Leave ENABLE_*_SCHEDULER unset on web — compose forces them false there.

# 3. Persist DBs + models
mkdir -p data models nginx/certbot/www nginx/certbot/conf

# 4. Deploy
chmod +x deploy.sh
./deploy.sh
# or: docker compose build && docker compose up -d
```

Open: `http://YOUR_DOMAIN/health`, `http://YOUR_DOMAIN/webapp/` and the React
dashboard at `http://YOUR_DOMAIN/dashboard/` (built in the Dockerfile's Node stage;
asks for a `DASHBOARD_API_TOKENS` token on first visit outside Telegram).
`/docs` is hidden in prod (set `ENABLE_DOCS=true` to expose it).

### Security model

* All `/api/v1/*` and `/api/v2/*` routes require Telegram Mini App `initData`
  (HMAC + mandatory `auth_date`, max age `TELEGRAM_AUTH_MAX_AGE`) or a read-only
  dashboard token (`Authorization: Bearer …`, listed in `DASHBOARD_API_TOKENS`).
  Dashboard tokens carry no Telegram user id, so they can never call write endpoints.
* Write endpoints (`POST /api/v2/bets/place`, `POST /api/v1/teams/import-url`)
  additionally require the Telegram user id to be in `ADMIN_TELEGRAM_IDS`.
  `bets/place` defaults to `persist=false`; persisted bets must use an
  `as_of_time` within `PLACE_AS_OF_MAX_SKEW_SECONDS` (300 s) of server time.
* nginx rate-limits `/api/` (10 req/s per IP, burst 20) and sends
  `nosniff` / `Referrer-Policy` / CSP `frame-ancestors` (Telegram web may frame
  the WebApp, so no `X-Frame-Options`).

BotFather → set Web App URL to `https://YOUR_DOMAIN/webapp/` (HTTPS required in production).

## SSL (Certbot companion)

HTTP proxy works immediately. For Telegram production HTTPS:

```bash
# Issue cert (nginx must already serve /.well-known/acme-challenge/)
docker compose --profile certs run --rm certbot certonly \
  --webroot -w /var/www/certbot \
  -d YOUR_DOMAIN \
  --email you@example.com --agree-tos --no-eff-email

# Enable HTTPS server block
cp nginx/templates/ssl.conf.template.example nginx/templates/ssl.conf.template
# Redirect HTTP → HTTPS (ACME challenge stays on HTTP)
sed -i 's/^NGINX_FORCE_HTTPS=.*/NGINX_FORCE_HTTPS=1/' .env
docker compose up -d nginx
```

Renew (cron / systemd timer):

```bash
docker compose --profile certs run --rm certbot renew
docker compose exec nginx nginx -s reload
```

## Local without Docker

```bash
# Auth off for browser/Swagger smoke tests — local only; APP_ENV=dev is
# required (with APP_ENV=prod or a public DOMAIN the API refuses to start).
APP_ENV=dev TELEGRAM_AUTH_DISABLED=true python run_api.py

# Workers (optional local smoke — same entrypoints as Compose)
python -m src.workers.data_monitor
python -m src.workers.settlement_worker
python -m src.workers.fixtures_worker --once   # refresh odds cache once

# Bot + value alerts (enable scheduler only in this process)
ENABLE_VALUE_SCHEDULER=true ENABLE_SETTLE_SCHEDULER=false \
  SIGNAL_MIN_EV=5 SIGNAL_MIN_DATA_SCORE=75 \
  python -m src.bot.telegram_bot
```

`run_api.py` is unchanged — Docker is an alternate entrypoint. Do **not** enable
settle/value schedulers on `run_api.py` if you also run the worker / bot processes.

### Smoke check (Docker)

```bash
docker compose up -d
docker compose ps
# Expect: web_api (healthy), data_monitor_worker, fixtures_worker,
#         settlement_worker, telegram_bot, nginx
curl -sS http://127.0.0.1/health
docker compose logs -f --tail=50 fixtures_worker data_monitor_worker settlement_worker
```

## Volumes

| Host | Container | Contents |
|------|-----------|----------|
| `./data` | `/app/data` | `global_matches.db`, quant_engine_v2, journal / notified DBs |
| `./models` | `/app/models` | `*.pkl` model pickles |
| `.env` | env_file | secrets (not baked into image) |

## Useful commands

```bash
docker compose logs -f web_api telegram_bot fixtures_worker data_monitor_worker settlement_worker
docker compose ps
docker compose restart web_api
./deploy.sh          # git pull + build + up -d
```
