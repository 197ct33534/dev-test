#!/usr/bin/env bash
# Pull latest code and recreate the Quant Engine stack.
set -euo pipefail

cd "$(dirname "$0")"

echo "==> git pull"
git pull

echo "==> docker compose build"
docker compose build

echo "==> docker compose up -d"
docker compose up -d

echo "==> done"
docker compose ps
