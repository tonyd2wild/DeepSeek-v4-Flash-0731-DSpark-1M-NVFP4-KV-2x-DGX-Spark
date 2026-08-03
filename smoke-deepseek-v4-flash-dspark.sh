#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.dspark}"
CONCURRENCY="${CONCURRENCY:-6}"
MAX_TOKENS="${MAX_TOKENS:-32}"

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

# Derived after the source so VLLM_PORT reaches it; an explicit override still wins.
CHAT_URL="${CHAT_URL:-http://127.0.0.1:${VLLM_PORT:-8888}/v1/chat/completions}"

MODEL="${SERVED_MODEL_NAME:-deepseek-v4-flash-dspark}"
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

echo "Running ${CONCURRENCY}-way smoke test against ${CHAT_URL}"

for i in $(seq 1 "$CONCURRENCY"); do
  (
    curl -fsS --max-time 180 "$CHAT_URL" \
      -H "Content-Type: application/json" \
      -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"Reply with OK and the number '"$i"'."}],"max_tokens":'"$MAX_TOKENS"',"temperature":0.0}' \
      >"$tmpdir/$i.json"
  ) &
done

fail=0
for job in $(jobs -p); do
  if ! wait "$job"; then
    fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  echo "Smoke test failed. Responses are in $tmpdir until this script exits." >&2
  exit 1
fi

for i in $(seq 1 "$CONCURRENCY"); do
  if ! grep -q '"choices"' "$tmpdir/$i.json"; then
    echo "Smoke response $i did not contain choices." >&2
    cat "$tmpdir/$i.json" >&2
    exit 1
  fi
done

echo "Smoke test passed: ${CONCURRENCY}/${CONCURRENCY} requests succeeded."
