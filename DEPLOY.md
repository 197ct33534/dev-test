# Deploy Quant Engine (Docker / VPS)

24/7 stack: **nginx** → **web** (FastAPI + APScheduler) + **bot** (Telegram polling).

## Architecture notes

| Process | Role |
|---------|------|
| `web` | `uvicorn` + value/settle schedulers in FastAPI lifespan |
| `bot` | Telegram polling / WebApp button only — schedulers **forced off** |
| `nginx` | Reverse proxy `DOMAIN` → `web:8000`; Certbot ACME on `:80` |

Do not enable `ENABLE_*_SCHEDULER` on both services — jobs would fire twice.

## VPS quickstart

```bash
# 1. Clone / pull
git clone <your-repo> /opt/score && cd /opt/score

# 2. Env
cp .env.example .env
# Edit: DOMAIN, TELEGRAM_BOT_TOKEN, WEBAPP_URL=https://YOUR_DOMAIN/webapp/
#       ENABLE_VALUE_SCHEDULER=true, ENABLE_SETTLE_SCHEDULER=true
#       TELEGRAM_AUTH_DISABLED=false

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
# Auth off for browser/Swagger smoke tests
TELEGRAM_AUTH_DISABLED=true python run_api.py
python -m src.bot.telegram_bot
```

`run_api.py` is unchanged — Docker is an alternate entrypoint.

## Volumes

| Host | Container | Contents |
|------|-----------|----------|
| `./data` | `/app/data` | `global_matches.db`, journal / notified DBs |
| `./models` | `/app/models` | `*.pkl` model pickles |
| `.env` | env_file | secrets (not baked into image) |

## Useful commands

```bash
docker compose logs -f web bot
docker compose ps
docker compose restart web
./deploy.sh          # git pull + build + up -d
```
