# Deploy Quant Engine (Docker / VPS) — Dockerization V2

24/7 stack: **nginx** → **web_api** (FastAPI, 2 uvicorn workers) + dedicated
workers + **telegram_bot** (polling + value alerts).

## Architecture notes

| Service | Role |
|---------|------|
| `web_api` | `uvicorn --workers 2` — API + WebApp static. Schedulers **forced off** |
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
#       TELEGRAM_AUTH_DISABLED=false
# Value alerts (telegram_bot): SIGNAL_MIN_EV=5, SIGNAL_MIN_DATA_SCORE=75
# Leave ENABLE_*_SCHEDULER unset on web — compose forces them false there.

# 3. Persist DBs + models
mkdir -p data models nginx/certbot/www nginx/certbot/conf

# 4. Deploy
chmod +x deploy.sh
./deploy.sh
# or: docker compose build && docker compose up -d
```

Open: `http://YOUR_DOMAIN/health` and `http://YOUR_DOMAIN/webapp/`.

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
docker compose up -d nginx
```

Optional: edit `nginx/templates/default.conf.template` so non-ACME HTTP returns
`301 https://$host$request_uri`.

Renew (cron / systemd timer):

```bash
docker compose --profile certs run --rm certbot renew
docker compose exec nginx nginx -s reload
```

## Local without Docker

```bash
# Auth off for browser/Swagger smoke tests (unchanged)
TELEGRAM_AUTH_DISABLED=true python run_api.py

# Workers (optional local smoke — same entrypoints as Compose)
python -m src.workers.data_monitor
python -m src.workers.settlement_worker

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
# Expect: web_api, data_monitor_worker, settlement_worker, telegram_bot, nginx
curl -sS http://127.0.0.1/health
docker compose logs -f --tail=50 data_monitor_worker settlement_worker
```

## Volumes

| Host | Container | Contents |
|------|-----------|----------|
| `./data` | `/app/data` | `global_matches.db`, quant_engine_v2, journal / notified DBs |
| `./models` | `/app/models` | `*.pkl` model pickles |
| `.env` | env_file | secrets (not baked into image) |

## Useful commands

```bash
docker compose logs -f web_api telegram_bot data_monitor_worker settlement_worker
docker compose ps
docker compose restart web_api
./deploy.sh          # git pull + build + up -d
```
