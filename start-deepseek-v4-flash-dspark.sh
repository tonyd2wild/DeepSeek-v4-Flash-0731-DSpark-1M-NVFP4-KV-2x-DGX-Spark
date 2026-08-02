#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.dspark}"
COMPOSE_FILE="${COMPOSE_FILE:-$SCRIPT_DIR/docker-compose.dspark.yml}"
API_URL="${API_URL:-http://127.0.0.1:8888/v1/models}"
CHAT_URL="${CHAT_URL:-http://127.0.0.1:8888/v1/chat/completions}"
WAIT_ATTEMPTS="${WAIT_ATTEMPTS:-100}"
WAIT_SECONDS="${WAIT_SECONDS:-15}"

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing $ENV_FILE. Copy .env.dspark.example to .env.dspark and edit node-specific values." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

: "${WORKER_HOST:?WORKER_HOST must be set in $ENV_FILE}"
: "${MASTER_ADDR:?MASTER_ADDR must be set in $ENV_FILE}"
: "${NCCL_IB_HCA:?NCCL_IB_HCA must be set in $ENV_FILE}"
: "${NCCL_SOCKET_IFNAME:?NCCL_SOCKET_IFNAME must be set in $ENV_FILE}"
: "${VLLM_HOST_IP:?VLLM_HOST_IP must be set to the head node fabric IP in $ENV_FILE}"
: "${WORKER_VLLM_HOST_IP:?WORKER_VLLM_HOST_IP must be set to the worker node fabric IP in $ENV_FILE}"

if [ "${SKIP_CONFIG_VALIDATION:-0}" != "1" ]; then
  "$SCRIPT_DIR/validate-dspark-config.sh"
fi

cd "$SCRIPT_DIR"

# DSpark source patches ship inside the runtime image (recipe/overlay/), not
# as runtime bind-mounts. Launching an image that predates the current overlay
# would run stale proposer code, so verify and rebuild first.
# Skip with SKIP_OVERLAY_CHECK=1 (e.g. offline, or a deliberate old image).
if [ "${SKIP_OVERLAY_CHECK:-0}" != "1" ]; then
  IMG="${DSPARK_VLLM_IMAGE:-vllm-dspark-runtime:dspark-nvfp4-stage-c}"
  OVERLAY_DOCKERFILE="$SCRIPT_DIR/recipe/Dockerfile.dspark-runtime-overlay"
  STALE=0
  if ! docker image inspect "$IMG" >/dev/null 2>&1; then
    echo "Runtime image $IMG not present; building it."
    STALE=1
  else
    IMAGE_SUMS="$(docker run --rm --entrypoint bash "$IMG" -c \
      "sha256sum $(awk '$1 == "COPY" { printf "%s ", $3 }' "$OVERLAY_DOCKERFILE")" 2>/dev/null || true)"
    while read -r src dst; do
      want="$(sha256sum "$SCRIPT_DIR/recipe/overlay/$src" | awk '{ print $1 }')"
      have="$(printf '%s\n' "$IMAGE_SUMS" | awk -v f="$dst" '$2 == f { print $1 }')"
      if [ "$want" != "$have" ]; then
        echo "Runtime image $IMG is stale against recipe/overlay/$src; rebuilding."
        STALE=1
        break
      fi
    done < <(awk '$1 == "COPY" { print $2, $3 }' "$OVERLAY_DOCKERFILE")
  fi
  if [ "$STALE" = "1" ]; then
    "$SCRIPT_DIR/build-dspark-vllm-runtime.sh"
  fi
fi

WORKER_DIR="${WORKER_SCRIPT_DIR:-${WORKER_DIR:-$SCRIPT_DIR}}"
WORKER_HF_CACHE="${WORKER_HF_CACHE:-${HF_CACHE:-}}"
REMOTE_WORKER_DIR="$(printf '%q' "$WORKER_DIR")"
REMOTE_COMPOSE="cd $REMOTE_WORKER_DIR && env -u MASTER_ADDR -u MASTER_PORT -u NODE_RANK -u HEADLESS COMPOSE_DISABLE_ENV_FILE=1"

echo "Syncing DSpark deployment files to ${WORKER_HOST}:${WORKER_DIR}"
ssh "$WORKER_HOST" "mkdir -p $REMOTE_WORKER_DIR"
scp "$COMPOSE_FILE" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/docker-compose.dspark.yml"
scp "$ENV_FILE" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/.env.dspark"

echo "Starting DSpark worker on ${WORKER_HOST}..."
ssh "$WORKER_HOST" "$REMOTE_COMPOSE NODE_RANK=1 HEADLESS=1 HF_CACHE='$WORKER_HF_CACHE' VLLM_HOST_IP='$WORKER_VLLM_HOST_IP' docker compose --env-file .env.dspark -f docker-compose.dspark.yml up -d"

echo "Starting DSpark head..."
COMPOSE_DISABLE_ENV_FILE=1 NODE_RANK=0 HEADLESS= docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d

echo "Both ranks have been started. Run this launcher only on the head node."
echo "A cold start can spend several minutes loading weights and compiling/capturing kernels."
echo "Rank 0 may pause at 'world_size=2 ... distributed_init_method=...' while rank 1 joins; this is expected."
echo "For recent logs from both nodes, run in another head-node terminal:"
echo "  $SCRIPT_DIR/logs-deepseek-v4-flash-dspark.sh"
echo "The readiness wait below allows up to $((WAIT_ATTEMPTS * WAIT_SECONDS)) seconds."

echo "Waiting for DSpark vLLM API..."
for attempt in $(seq 1 "$WAIT_ATTEMPTS"); do
  if curl -fsS --max-time 5 "$API_URL" >/dev/null; then
    echo "DeepSeek V4 Flash DSpark is running: $API_URL"
    COMPOSE_DISABLE_ENV_FILE=1 docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" ps
    ssh "$WORKER_HOST" "$REMOTE_COMPOSE docker compose --env-file .env.dspark -f docker-compose.dspark.yml ps"
    echo "Running minimal OpenAI-compatible chat request..."
    curl -fsS --max-time 60 "$CHAT_URL" \
      -H "Content-Type: application/json" \
      -d '{"model":"'"${SERVED_MODEL_NAME:-deepseek-v4-flash-dspark}"'","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":8,"temperature":0.0}' >/dev/null
    echo "Minimal chat request succeeded."
    exit 0
  fi
  if ((attempt % 4 == 0)); then
    echo "Still waiting for the API ($((attempt * WAIT_SECONDS)) seconds elapsed)..."
  fi
  sleep "$WAIT_SECONDS"
done

echo "Timed out waiting for DSpark API. Recent head logs:" >&2
COMPOSE_DISABLE_ENV_FILE=1 docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" logs --tail=120 vllm-dspark >&2 || true
echo "Recent worker logs:" >&2
ssh "$WORKER_HOST" "$REMOTE_COMPOSE docker compose --env-file .env.dspark -f docker-compose.dspark.yml logs --tail=120 vllm-dspark" >&2 || true
exit 1
