#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.dspark}"
COMPOSE_FILE="${COMPOSE_FILE:-$SCRIPT_DIR/docker-compose.dspark.yml}"

fail() {
  echo "Invalid DSpark config: $*" >&2
  exit 1
}

reject_placeholder() {
  local name="$1"
  local value="$2"
  case "$value" in
    *-roce-ip | worker-host-or-roce-ip | rocepXsYfZ | enpXsYfZnpN)
      fail "$name still has the example placeholder '$value'"
      ;;
  esac
}

require_ipv4() {
  local name="$1"
  local value="$2"
  local octet
  local -a octets
  [[ "$value" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] ||
    fail "$name must be an IPv4 fabric address, got '$value'"
  IFS=. read -r -a octets <<<"$value"
  for octet in "${octets[@]}"; do
    ((10#$octet <= 255)) || fail "$name has an invalid IPv4 octet in '$value'"
  done
}

require_uint() {
  local name="$1"
  local value="$2"
  [[ "$value" =~ ^[0-9]+$ ]] ||
    fail "$name must be a decimal integer, got '$value'"
}

require_absolute_path() {
  local name="$1"
  local value="$2"
  [[ "$value" = /* ]] ||
    fail "$name must be an absolute path, got '$value'"
}

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing $ENV_FILE. Copy .env.dspark.example to .env.dspark and edit it." >&2
  exit 1
fi

if [ ! -f "$COMPOSE_FILE" ]; then
  echo "Missing $COMPOSE_FILE." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

: "${WORKER_HOST:?WORKER_HOST must be set in $ENV_FILE}"
: "${MASTER_ADDR:?MASTER_ADDR must be set in $ENV_FILE}"
: "${MASTER_PORT:?MASTER_PORT must be set in $ENV_FILE}"
: "${NCCL_IB_HCA:?NCCL_IB_HCA must be set in $ENV_FILE}"
: "${NCCL_SOCKET_IFNAME:?NCCL_SOCKET_IFNAME must be set in $ENV_FILE}"
: "${NCCL_IB_GID_INDEX:?NCCL_IB_GID_INDEX must be set in $ENV_FILE}"
: "${HF_CACHE:?HF_CACHE must be set in $ENV_FILE}"
: "${VLLM_HOST_IP:?VLLM_HOST_IP must be set in $ENV_FILE}"
: "${WORKER_VLLM_HOST_IP:?WORKER_VLLM_HOST_IP must be set in $ENV_FILE}"
: "${DSPARK_VLLM_IMAGE:?DSPARK_VLLM_IMAGE must be set in $ENV_FILE}"

reject_placeholder WORKER_HOST "$WORKER_HOST"
reject_placeholder MASTER_ADDR "$MASTER_ADDR"
reject_placeholder NCCL_IB_HCA "$NCCL_IB_HCA"
reject_placeholder NCCL_SOCKET_IFNAME "$NCCL_SOCKET_IFNAME"
reject_placeholder VLLM_HOST_IP "$VLLM_HOST_IP"
reject_placeholder WORKER_VLLM_HOST_IP "$WORKER_VLLM_HOST_IP"
require_ipv4 MASTER_ADDR "$MASTER_ADDR"
require_ipv4 VLLM_HOST_IP "$VLLM_HOST_IP"
require_ipv4 WORKER_VLLM_HOST_IP "$WORKER_VLLM_HOST_IP"
require_uint MASTER_PORT "$MASTER_PORT"
require_uint NCCL_IB_GID_INDEX "$NCCL_IB_GID_INDEX"
require_absolute_path HF_CACHE "$HF_CACHE"
if [ -n "${WORKER_HF_CACHE:-}" ]; then
  require_absolute_path WORKER_HF_CACHE "$WORKER_HF_CACHE"
fi
if [ "$VLLM_HOST_IP" = "$WORKER_VLLM_HOST_IP" ]; then
  fail "VLLM_HOST_IP and WORKER_VLLM_HOST_IP must identify different nodes"
fi
if ((MASTER_PORT < 1 || MASTER_PORT > 65535)); then
  fail "MASTER_PORT must be between 1 and 65535, got '$MASTER_PORT'"
fi

echo "DSpark config:"
echo "  worker: ${WORKER_HOST}"
echo "  master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  image: ${DSPARK_VLLM_IMAGE}"
echo "  model: ${DSPARK_MODEL:-deepseek-ai/DeepSeek-V4-Flash-DSpark}"
echo "  served model: ${SERVED_MODEL_NAME:-deepseek-v4-flash-dspark}"
echo "  max model len: ${MAX_MODEL_LEN:-1048576}"
echo "  max num seqs: ${MAX_NUM_SEQS:-12}"
echo "  max batched tokens: ${MAX_NUM_BATCHED_TOKENS:-8192}"
echo "  gpu memory utilization: ${GPU_MEMORY_UTILIZATION:-0.85}"
echo "  spec tokens (MTP_NUM_TOKENS): ${MTP_NUM_TOKENS:-5}"
echo "  sampling override: none (no --override-generation-config; --generation-config vllm only)"
echo "  WO projection: ${VLLM_USE_B12X_WO_PROJECTION:-1}"
echo "  host bind: ${VLLM_HOST:-127.0.0.1}"
echo
echo "Rendered vLLM command:"
env -u MASTER_PORT -u NODE_RANK -u HEADLESS -u WORKER_HOST -u MASTER_ADDR \
  COMPOSE_DISABLE_ENV_FILE=1 \
  docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" config \
  | grep -E -- '--max-model-len|--max-num-seqs|--max-num-batched-tokens|--max-cudagraph-capture-size|--gpu-memory-utilization|--master-port|--kv-cache-dtype|--speculative-config|--async-scheduling|--enable-chunked-prefill|--generation-config|image:|VLLM_USE_B12X_WO_PROJECTION|VLLM_USE_FLASHINFER_SAMPLER|MTP_NUM_TOKENS'
