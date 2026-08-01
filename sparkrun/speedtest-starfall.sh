#!/usr/bin/env bash
# Warm-up + reference speed test for the DeepSeek V4 Flash DSpark recipes.
#
# Sends 3 x ~1500-token warm-up generations (the engine runs ~30% slow after
# boot or ~30 min idle), then the "Protocol Starfall" prompt from the original
# demo video, measured the right way (stream:false, usage.completion_tokens —
# counting SSE chunks under spec decode measures steps/s, not tok/s).
#
# Usage:
#   ./speedtest-starfall.sh [base_url] [model]
#   ./speedtest-starfall.sh                                     # localhost:8888, 0731
#   ./speedtest-starfall.sh http://10.0.0.25:8888               # remote head
#   ./speedtest-starfall.sh http://10.0.0.25:8888 deepseek-v4-flash-dspark   # preview recipe
set -eu

BASE_URL="${1:-http://127.0.0.1:8888}"
MODEL="${2:-deepseek-v4-flash-0731}"

python3 - "$BASE_URL" "$MODEL" <<'PY'
import json, sys, time, urllib.request

base_url, model = sys.argv[1].rstrip("/"), sys.argv[2]
url = base_url + "/v1/chat/completions"

def ask(prompt, max_tokens, temperature):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    req = urllib.request.Request(url, json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    body = json.load(urllib.request.urlopen(req, timeout=600))
    dt = time.monotonic() - t0
    return body["usage"]["completion_tokens"], dt

warm = ("Generate 60 SQL INSERT statements for table users(id, email, created_at). "
        "Use the exact form: INSERT INTO users (id, email, created_at) VALUES "
        "(N, 'user_N@example.com', '2026-01-01'); - ids 1 to 60, one per line. SQL only.")
print("[speedtest] warming up the engine (3 x ~1500-token generations)...")
for i in range(3):
    tok, dt = ask(warm, 1500, 0.0)
    print("[speedtest] warmup %d/3: %d tokens in %.1fs (%.1f tok/s)" % (i + 1, tok, dt, tok / dt))

starfall = ("Build a complete single-file HTML5 canvas game called Protocol Starfall: "
            "a top-down space shooter with waves of enemies, score, health, and a "
            "game-over screen. Return only the full HTML.")
tok, dt = ask(starfall, 4096, 0.2)
print("[speedtest] Protocol Starfall: %d tokens in %.1fs = %.1f tok/s "
      "(warm, single-stream, includes prefill)" % (tok, dt, tok / dt))
PY
