#!/usr/bin/env bash
# Bring up the whole read-only stack (Phase 3).
#
# Three processes: the Python risk engine, the Node backend that enforces the
# §6 degradation contract, and the Next.js frontend. The backend is the only
# thing that talks to the engine, and the browser never talks to it at all.
#
#   ./scripts/run-stack.sh            # synthetic market data
#   ./scripts/run-stack.sh --live     # live Hyperliquid data (needs API access)
set -euo pipefail
cd "$(dirname "$0")/.."

MODE_FLAG="--fixture"
[[ "${1:-}" == "--live" ]] && MODE_FLAG=""

cleanup() { kill 0 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "risk engine  -> http://127.0.0.1:8787"
python3 -m risk_engine.service $MODE_FLAG --port 8787 &

echo "backend      -> http://127.0.0.1:8080"
( cd services/backend && npm run --silent build && PORT=8080 RISK_SERVICE_URL=http://127.0.0.1:8787 node dist/main.js ) &

echo "frontend     -> http://127.0.0.1:3000"
( cd apps/web && BACKEND_URL=http://127.0.0.1:8080 npm run --silent dev ) &

wait
