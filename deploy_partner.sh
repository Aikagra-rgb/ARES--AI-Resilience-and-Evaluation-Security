#!/usr/bin/env bash
# ==============================================================================
# ARES Partner Sidecar Deployment Script
# Generates a hardened per-client environment bundle and launches the local ARES
# gateway sidecar pointing to the client's isolated models and canary token.
#
# Usage:
#   ./deploy_partner.sh --org=org_acme_fintech --api-key=YOUR_KEY [--app=AcmeFintechRAG] [--compose-file=docker-compose.yml]
#   (For local full-stack development only, pass --compose-file=docker-compose.full.yml)
# ==============================================================================

set -euo pipefail

ORG_ID=""
APP_NAME=""
API_KEY="${ARES_API_KEY:-}"
GATEWAY_PORT="${ARES_SIDECAR_PORT:-8080}"
COMPOSE_FILE="docker-compose.yml"

for arg in "$@"; do
  case "$arg" in
    --org=*) ORG_ID="${arg#*=}" ;;
    --app=*) APP_NAME="${arg#*=}" ;;
    --api-key=*) API_KEY="${arg#*=}" ;;
    --port=*) GATEWAY_PORT="${arg#*=}" ;;
    --compose-file=*) COMPOSE_FILE="${arg#*=}" ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$ORG_ID" ]]; then
  echo "Error: --org=<org_id> is required (e.g. --org=org_acme_fintech)" >&2
  exit 1
fi

if [[ -z "$API_KEY" ]]; then
  echo "Error: --api-key=<key> or ARES_API_KEY environment variable is required" >&2
  exit 1
fi

APP_NAME="${APP_NAME:-$ORG_ID}"
ENV_FILE=".env.partner.${ORG_ID}"
MODEL_DIR="backend/models/${ORG_ID}"

echo "[ARES] Preparing partner deployment for org='${ORG_ID}' (app='${APP_NAME}')..."

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "[ARES] Warning: ${MODEL_DIR} not found yet. Run:"
  echo "  PYTHONPATH=backend python -m ares.training.client_model_trainer --org=${ORG_ID}"
fi

cat > "$ENV_FILE" <<EOF
ARES_ORG_ID=${ORG_ID}
ARES_APP_NAME=${APP_NAME}
ARES_API_KEY=${API_KEY}
ARES_SIDECAR_PORT=${GATEWAY_PORT}
ARES_GATEWAY_CANARY_TOKEN=CANARY_${ORG_ID^^}_01
ARES_PER_CLIENT_MODEL_DIR=/app/${MODEL_DIR}
EOF

echo "[ARES] Wrote partner sidecar environment to ${ENV_FILE}"
if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "[ARES] Warning: compose file '${COMPOSE_FILE}' was not found in $(pwd)." >&2
  echo "[ARES] Pass --compose-file=<path> pointing to a valid Docker Compose manifest before starting containers." >&2
else
  echo "[ARES] Launch gateway sidecar with:"
  echo "  docker compose --env-file ${ENV_FILE} -f ${COMPOSE_FILE} up -d"
fi
