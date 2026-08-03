#!/usr/bin/env python3
"""Small OpenAI-compatible stability/concurrency check for agent endpoints."""

import concurrent.futures
import json
import os
import statistics
import sys
import time
import urllib.request


BASE_URL = os.environ.get("DSPARK_BASE_URL", "http://127.0.0.1:8888/v1")
MODEL = os.environ.get("DSPARK_MODEL", "deepseek-v4-flash-dspark")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "256"))
CONCURRENCY_LIST = [
    int(x) for x in os.environ.get("CONCURRENCY", "1,2,4,6").split(",") if x.strip()
]


def make_prompt(i: int) -> str:
    filler = " ".join(f"token{i}_{j}" for j in range(420))
    return (
        "Write a concise implementation note in plain English. "
        "Stay in English. Do not repeat characters. Do not output XML. "
        "Keep writing useful detail until the answer is complete.\n\n"
        f"Context salt {i}: {filler}"
    )


def is_empty(text: str) -> bool:
    """Soft-empty completion: tokens were generated but nothing is visible.

    This is a real, silent failure mode. With thinking enabled the model opens a
    <think> block; if max_tokens runs out before </think>, the reasoning parser
    emits NEITHER content NOR reasoning_content and the caller gets "" while the
    whole budget was spent. Measured here at temperature 0.5, thinking on:
    max_tokens 256 -> 83% empty, 512 -> 50%, 768 -> 17%, 1024 -> 0/18.

    It must be checked separately from looks_bad(): empty text contains no CJK,
    no repeats and no leaked markup, so it passes every other check trivially and
    the gate reports success on a server that answers nothing.
    """
    return not text.strip()


def looks_bad(text: str) -> bool:
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    repeated = any(ch * 18 in text for ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    # <think> only counts as leakage when the reasoning parser should have
    # stripped it; a correctly parsed thinking response never carries it.
    leaked = any(marker in text.lower() for marker in ("<available_skills", "<tool", "</tool", "<think>"))
    return cjk > 0 or repeated or leaked


def request_one(i: int) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": make_prompt(i)}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
    }
    req = urllib.request.Request(
        BASE_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=420) as r:
        data = json.load(r)
    dt = time.perf_counter() - t0
    usage = data.get("usage") or {}
    content = data["choices"][0]["message"].get("content") or ""
    completion = usage.get("completion_tokens") or 0
    return {
        "id": i,
        "seconds": round(dt, 3),
        "completion_tokens": completion,
        "tok_s": round(completion / dt, 2) if dt else 0,
        "finish_reason": data["choices"][0].get("finish_reason"),
        "bad_output": looks_bad(content),
        "empty_output": is_empty(content),
        "reasoning_chars": len(data["choices"][0]["message"].get("reasoning_content") or ""),
        "sample": content[:200],
    }


def run(concurrency: int) -> dict:
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        rows = list(ex.map(request_one, range(concurrency)))
    wall = time.perf_counter() - start
    total = sum(r["completion_tokens"] for r in rows)
    return {
        "concurrency": concurrency,
        "success": f"{sum(not (r['bad_output'] or r['empty_output']) for r in rows)}/{len(rows)}",
        "max_tokens": MAX_TOKENS,
        "wall_seconds": round(wall, 3),
        "completion_tokens": total,
        "aggregate_tok_s": round(total / wall, 2) if wall else 0,
        "per_request_tok_s_mean": round(statistics.mean(r["tok_s"] for r in rows), 2),
        "bad_outputs": sum(1 for r in rows if r["bad_output"]),
        "empty_outputs": sum(1 for r in rows if r["empty_output"]),
        "rows": rows,
    }


def main() -> int:
    failed = False
    for concurrency in CONCURRENCY_LIST:
        result = run(concurrency)
        print(json.dumps(result, indent=2))
        if result["empty_outputs"]:
            print(
                f"FAIL: {result['empty_outputs']}/{len(result['rows'])} responses at "
                f"concurrency {result['concurrency']} had empty visible content while "
                f"consuming their token budget. With thinking enabled this means "
                f"MAX_TOKENS={MAX_TOKENS} is too small to reach </think>; raise it "
                f"(>=1024 measured clean) or disable thinking.",
                file=sys.stderr,
            )
        failed = failed or result["bad_outputs"] > 0 or result["empty_outputs"] > 0
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

