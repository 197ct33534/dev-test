# Quant Engine — slim runtime image for FastAPI + Telegram bot
FROM python:3.11-slim

# LightGBM needs OpenMP; keep apt footprint minimal
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home --uid 1000 --shell /bin/bash appuser \
    && mkdir -p /app/data /app/models \
    && chown -R appuser:appuser /app

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

# Compose overrides CMD for web vs bot
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
