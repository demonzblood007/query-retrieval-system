#!/bin/bash
set -euo pipefail

# start qdrant in background if requested
if [ "${START_LOCAL_QDRANT:-false}" = "true" ]; then
  echo "Starting local Qdrant..."
  qdrant --config-path /qdrant/config/production.yaml &
  QDRANT_PID=$!
  # point app to local qdrant if not set
  export QDRANT_HOST=${QDRANT_HOST:-http://localhost:6333}
fi

echo "Starting API server..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8080}"