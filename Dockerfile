# Quant Engine — slim runtime for FastAPI + workers + Telegram bot

# --- Stage 1: React dashboard (served by FastAPI at /dashboard) -------------
FROM node:20-alpine AS dashboard
WORKDIR /build
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY dashboard/ ./
RUN npm run build

# --- Stage 2: Python runtime ------------------------------------------------
FROM python:3.11-slim

# LightGBM needs OpenMP; keep apt footprint minimal
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home --uid 1000 --shell /bin/bash appuser \
    && mkdir -p /app/data /app/models \
    && chown -R appuser:appuser /app

COPY --chown=appuser:appuser . .
COPY --from=dashboard --chown=appuser:appuser /build/dist /app/dashboard/dist

USER appuser

EXPOSE 8000

# Compose overrides CMD for web_api / workers / telegram_bot
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=*"]
