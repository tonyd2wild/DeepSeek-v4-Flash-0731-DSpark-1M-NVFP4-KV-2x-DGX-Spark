# DeepSeek V4 Flash (DSpark) on 2x DGX Spark — 1M context, NVFP4 KV, ~50-60 tok/s

> Self-contained two-node DGX Spark recipe for serving `DeepSeek-V4-Flash-DSpark`
> with vLLM TP=2, DSpark speculative decoding, and an experimental `nvfp4_ds_mla`
> KV cache — 1M-token calibrated context (pushed to 1.5M), clean under agent
> concurrency.

## TL;DR

- **What you get:** a validated Stage C NVFP4 runtime for `fraserprice/DeepSeek-V4-Flash-DSpark`,
  serving over 2x DGX Spark at TP=2 with DSpark speculative decoding and the
  `nvfp4_ds_mla` KV-cache path. Captures the validated Stage C NVFP4 runtime,
  the 2026-06-30 agent-stability refresh, and the 2026-07-02 Keys C12 checkpoint.
- **Key numbers (C12 1.5M NVFP4 checkpoint):** `max_model_len=1500000`,
  `max_num_seqs=12`, `kv_cache_dtype=nvfp4_ds_mla`, reported KV pool
  `3,225,280 tokens`, reported max concurrency for 1.5M requests `2.15x`.
  Single-stream decode stayed above `50 tok/s`; 2/4/6/12 concurrent code-gate
  prompts completed cleanly with no Chinese drift or repeated junk. The DSpark
  in-server concurrency patch validated at `max_model_len=200000`,
  `max_num_seqs=16`, static C16 `315.1 tok/s` aggregate and staggered C16
  `205.0 tok/s` aggregate.
- **Who it is for:** agent/supervisor fleets on a two-node Blackwell-class/DGX
  Spark setup that want long context plus clean concurrent output.
- **Already saw agent garble, loops, Chinese drift, or prompt/tool XML leaking
  into replies?** Start with [`AGENT_GARBLE_FIX.md`](AGENT_GARBLE_FIX.md). The fix
  path keeps the C12 NVFP4 profile; it does not switch production to fp8 or a
  smaller fallback model. See [Troubleshooting](#troubleshooting).

> **Context length — 1M is the ceiling, not 1.5M.** The model's `config.json` uses
> YaRN with `original_max_position_embeddings=65536 × factor 16 = 1,048,576` and
> `max_position_embeddings=1048576`, so **1M (1048576) is the true, calibrated ceiling.**
> Earlier profiles here ran `1500000` by setting `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`, which
> only forces vLLM to *accept* a longer ceiling — it boots and the throughput benchmarks
> below are valid, but any request growing past 1M extrapolates past what YaRN was tuned
> for, so coherent output past 1M is not guaranteed. **Recommended default: `MAX_MODEL_LEN=1048576`.**
> The 1.5M figures retained below are kept as "how far it was pushed" records, not a quality claim past 1M.

## Hardware

- **2x DGX Spark**, one GPU per node, TP=2 (Blackwell-class GB10).
- **Fabric:** RoCE/InfiniBand NCCL. The verified deployment used a MikroTik
  CRS812 switched fabric on `192.168.192.0/24` (head `192.168.192.3`, worker
  `192.168.192.4`, `MASTER_PORT=25440`).
- **Container:** `network_mode: host`, `ipc: host`, `shm_size: 64gb`, `gpus: all`,
  `-v /dev/infiniband:/dev/infiniband`, memlock unlimited.
- Requires matching images on both nodes, correct NCCL/RoCE settings, and a
  two-node Blackwell-class/DGX Spark setup. Full node/network table is in
  [`DEFAULT-CONFIG.md`](DEFAULT-CONFIG.md).

## Quick start

Run from the head node.

```bash
cp .env.dspark.example .env.dspark
```

Edit these values for your cluster:

- `WORKER_HOST`
- `WORKER_SCRIPT_DIR` if the worker checkout/deployment path differs from the head
- `MASTER_ADDR`
- `NCCL_IB_HCA`
- `NCCL_SOCKET_IFNAME`
- `NCCL_IB_GID_INDEX`
- `HF_CACHE`
- `WORKER_HF_CACHE` if the worker cache path differs from the head
- `VLLM_HOST_IP` and `WORKER_VLLM_HOST_IP` for each node's fabric IP

Then build, prepare the cache, and start (worker-first):

```bash
./build-dspark-vllm-runtime.sh      # builds the base overlay + Stage C NVFP4 image
./prepare-dspark-model-cache.sh     # downloads/verifies the model cache
./start-deepseek-v4-flash-dspark.sh # worker-first launch and smoke test
```

The API serves at:

```text
http://HEAD_NODE_IP:8888/v1
```

For head-node-only tests, set `VLLM_HOST=127.0.0.1`. For Hermes/OpenClaw or
another machine to use the endpoint, keep `VLLM_HOST=0.0.0.0` and control access
at the network/firewall layer. The API binds to `127.0.0.1` by default; exposing
it is a deliberate security choice.

## Setup (detailed)

### Agent-serving defaults

Keep these `.env.dspark` values unless you are deliberately experimenting:

- `VLLM_HOST=0.0.0.0` if Hermes/OpenClaw or another machine must reach the API
- `MAX_MODEL_LEN=1048576` — **1M, the model's true YaRN ceiling** (see the note above)
- `MAX_NUM_SEQS=12`
- `GPU_MEMORY_UTILIZATION=0.85`
- `MTP_NUM_TOKENS=3` (with `draft_sample_method=probabilistic`; see the [garble fix](#garble-fix-2026-07-03))
- `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`
- `VLLM_USE_B12X_WO_PROJECTION=1`
- `VLLM_USE_FLASHINFER_SAMPLER=1`
- `VLLM_DSPARK_REPLICATE_MARKOV_W1=1`
- `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`
- no `--override-generation-config` (removed in the 2026-07-03 garble fix)

### Weights

- HF model: [`fraserprice/DeepSeek-V4-Flash-DSpark`](https://huggingface.co/fraserprice/DeepSeek-V4-Flash-DSpark)
  (in the HF cache as `/cache/huggingface/fraserprice/DeepSeek-V4-Flash-DSpark`).
- The compose/env default `DSPARK_MODEL=deepseek-ai/DeepSeek-V4-Flash-DSpark`.
- `./prepare-dspark-model-cache.sh` downloads the snapshot into `HF_CACHE`,
  verifies every safetensor shard is present, and mirrors the download to the
  worker node.

### Image / build

- `./build-dspark-vllm-runtime.sh` builds the base DSpark overlay
  (`vllm-dspark-runtime:mia-raf-pr1`), then NVFP4 stages A → B → C, producing the
  canonical image `vllm-dspark-runtime:dspark-nvfp4-stage-c`. It builds on the
  head and, by default, rsyncs and rebuilds on `WORKER_HOST`.
- A fresh Stage C build already contains Patch 3 (commit `e83606a`), so no
  bind-mount is needed. As deployed on the pre-Patch-3 `probe-c-p2b` image,
  Patch 3 was injected via bind-mount (see [`DEFAULT-CONFIG.md`](DEFAULT-CONFIG.md)).

> **vLLM version note (concurrency fix).** The concurrency-crash fix is a source
> patch to `dspark_proposer.py`, bind-mounted via compose. The committed
> `recipe/vllm/v1/spec_decode/dspark_proposer.py` is **version-matched to this repo's
> default image** (`vllm-dspark-runtime:dspark-nvfp4-stage-c`). If you run a **different
> vLLM build**, its `propose()` signature may differ and mounting the committed file
> will crash with `propose() got an unexpected keyword argument ...`. In that case,
> patch your own image's proposer instead:
> ```bash
> docker create --name t <your-image>
> docker cp t:/opt/env/lib/python3.12/site-packages/vllm/v1/spec_decode/dspark_proposer.py ./myproposer.py
> docker rm t
> python3 scripts/apply-nonuniform-guard.py ./myproposer.py   # version-independent
> # then bind-mount ./myproposer.py at that same container path
> ```

### Launch

```bash
./start-deepseek-v4-flash-dspark.sh
```

Worker-first startup avoids a race during multi-node `mp` initialization. The
launcher runs `--generation-config vllm` with **no** `--override-generation-config`
and `--default-chat-template-kwargs '{"thinking":false}'`. Exact flags and env
are in [Configuration](#configuration); `docker-compose.dspark.yml` is the source
of truth.

Other lifecycle scripts: `stop-deepseek-v4-flash-dspark.sh`,
`status-deepseek-v4-flash-dspark.sh`, `logs-deepseek-v4-flash-dspark.sh`,
`smoke-deepseek-v4-flash-dspark.sh`, `validate-dspark-config.sh`,
`update-and-restart.sh`.

### Verify

After launch:

```bash
curl -fsS http://127.0.0.1:8888/v1/models
```

Confirm the returned model entry reports (C12 1.5M checkpoint):

```json
"max_model_len": 1500000
```

Then check logs:

```bash
docker compose --env-file .env.dspark -f docker-compose.dspark.yml logs vllm-dspark \
  | grep -E "GPU KV cache size|Maximum concurrency"
```

Expected C12 checkpoint values are around:

```text
GPU KV cache size: 3.2M tokens
Maximum concurrency for 1,500,000 tokens per request: 2.1x
```

Before pointing an agent harness at the endpoint, run the direct sanity bench:

```bash
DSPARK_BASE_URL=http://HEAD_NODE_IP:8888/v1 \
CONCURRENCY=1,2,4,6 \
python3 scripts/agent_sanity_bench.py
```

Every row should report `bad_outputs: 0`. If this direct test is clean but an
agent still garbles, investigate the agent session, fallback list, or harness
prompt replay before blaming the DSpark weights.

Capture runtime evidence before and after any fix:

```bash
scripts/capture_runtime.sh runtime-before-change
scripts/capture_runtime.sh runtime-after-change
```

## Benchmarks

### 2026-07-02 Keys C12 1.5M NVFP4 checkpoint

The current high-concurrency lane keeps Tony's known-good Stage C NVFP4 image and
applies Keys' C12 serving profile.

Runtime:

- endpoint tested: `http://100.90.25.78:8888/v1`
- served model: `deepseek-v4-flash-dspark`
- image: `vllm-dspark-runtime:dspark-nvfp4-stage-c`
- model path: `/cache/huggingface/fraserprice/DeepSeek-V4-Flash-DSpark`
- `kv_cache_dtype=nvfp4_ds_mla`
- `max_model_len=1500000`
- `max_num_seqs=12`
- `max_num_batched_tokens=8192`
- `gpu_memory_utilization=0.85`
- `MTP_NUM_TOKENS=5`
- `VLLM_USE_B12X_WO_PROJECTION=1`
- `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`
- `thinking=false`
- `--generation-config vllm`
- `--override-generation-config '{"temperature":0.0,"top_p":1.0,"top_k":40,"repetition_penalty":1.05}'`

Boot evidence:

```text
GPU KV cache size: 3,225,280 tokens
Maximum concurrency for 1,500,000 tokens per request: 2.15x
Application startup complete.
```

Code-gate validation:

| concurrency | success | server generation tok/s | acceptance | bad outputs |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1/1 | 52.79 | 0.585 | 0 |
| 2 | 2/2 | 79.76 | 0.600 | 0 |
| 4 | 4/4 | 134.70 | 0.602 | 0 |
| 6 | 6/6 | 127.78 | 0.615 | 0 |
| 12 | 12/12 | 230.10 | 0.602 | 0 |

Full note: [`benchmarks/20260702-keys-c12-1p5m-nvfp4-checkpoint.md`](benchmarks/20260702-keys-c12-1p5m-nvfp4-checkpoint.md).

Do not enable `VLLM_USE_B12X_FP8_GEMM=1` on this Stage C image. That flag hit a
DeepGEMM layout assertion during DSpark drafter warmup in testing.

### 2026-06-30 clean agent-serving checkpoint

The prior conservative clean endpoint was reproduced on Asusi/Spark4 before
sending the model back through Hermes/OpenClaw-style harnesses.

Runtime:

- endpoint tested: `http://100.90.25.78:8888/v1`
- served model: `deepseek-v4-flash-dspark`
- image used on that lane: `vllm-dspark-runtime:mia-raf-pr1-nvfp4-keys-c`
- model path: `/cache/huggingface/fraserprice/DeepSeek-V4-Flash-DSpark`
- `kv_cache_dtype=nvfp4_ds_mla`
- `max_model_len=1048576`
- `max_num_seqs=6`
- `max_num_batched_tokens=8192`
- `gpu_memory_utilization=0.80`
- `MTP_NUM_TOKENS=5`
- `thinking=false`
- `--generation-config vllm`
- `--override-generation-config '{"temperature":0.0,"top_p":1.0}'`
- explicit per-node `VLLM_HOST_IP` values

Boot evidence:

```text
GPU KV cache size: 1,990,142 tokens
Maximum concurrency for 1,048,576 tokens per request: 1.90x
Application startup complete.
```

Direct validation:

- `/v1/models` reported `"max_model_len": 1048576`
- deterministic sanity prompt returned `NVFP4 DSPARK OK`
- five longer English prompts completed with no CJK drift and no repeated junk
- code-gate server decode mean: `54.22 tok/s`
- 2/4/6 concurrent direct prompts all succeeded cleanly

Concurrency:

| concurrency | success | aggregate tok/s | stability |
| ---: | ---: | ---: | --- |
| 2 | 2/2 | 60.95 | no CJK/repeat junk |
| 4 | 4/4 | 83.21 | no CJK/repeat junk |
| 6 | 6/6 | 104.11 | no CJK/repeat junk |

Full note: [`benchmarks/20260630-asusi-spark4-nvfp4-1m-agent-stability.md`](benchmarks/20260630-asusi-spark4-nvfp4-1m-agent-stability.md).

### 1M NVFP4 profile (single stream)

Validated on 2x DGX Spark, one GPU per node, TP=2, single stream.

| Case | server tok/s | TTFC | acceptance | accepted/draft |
| --- | ---: | ---: | ---: | ---: |
| p256/g64 | 54.46 | 0.506s | 0.667 | 3.33 |
| p256/g256 | 65.38 | 0.324s | 0.718 | 3.59 |
| p512/g64 | 56.26 | 2.738s | 0.625 | 3.13 |
| p512/g256 | 54.41 | 0.422s | 0.550 | 2.75 |
| p512/g256 warmup1 | 56.73 | 0.417s | 0.585 | 2.92 |

Boot logs reported:

```text
GPU KV cache size: 2,044,166 tokens
Maximum concurrency for 1,048,576 tokens per request: 1.95x
```

The API reported:

```json
{"max_model_len":1048576}
```

Full note: [`benchmarks/20260629-dspark-nvfp4-1m-context-checkpoint.md`](benchmarks/20260629-dspark-nvfp4-1m-context-checkpoint.md).

### DSpark concurrency profile (200K / 16)

Validated on the same 2x DGX Spark TP=2 deployment using Keys' DSpark concurrency
patch, `kv_cache_dtype=nvfp4_ds_mla`, `max_model_len=200000`, `max_num_seqs=16`,
`MTP_NUM_TOKENS=5`, and `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`.

Patch source:

- [drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash](https://github.com/drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash)
- Tested patch commit: `7e4d94bbcec95223550517c0fa9244e59f9f6483`

The live fix documented here keeps `kv_cache_dtype=nvfp4_ds_mla` and refreshes the
repo's already-vendored Keys overlay with the path-adjusted Patch 2b update from
that commit. In Patch 2b, ragged `query_start_loc` detection no longer depends on
`num_rejected_tokens_gpu`. Treat the service as validated only after the built-in
OpenAI-compatible chat smoke request plus agent-client validation pass on the live
service.

Static simultaneous batch, one TP=2 replica:

| concurrency | best aggregate tok/s | per-stream tok/s | acceptance |
| ---: | ---: | ---: | ---: |
| 1 | 57.6 | 57.6 | 0.635 |
| 4 | 140.8 | 35.2 | 0.619 |
| 8 | 252.6 | 31.6 | 0.635 |
| 16 | 315.1 | 19.7 | 0.609 |

Staggered independent arrivals, one TP=2 replica:

| concurrency | success | aggregate tok/s | acceptance |
| ---: | ---: | ---: | ---: |
| 4 | 4/4 | 109.2 | 0.544 |
| 8 | 8/8 | 147.3 | 0.534 |
| 16 | 16/16 | 205.0 | 0.567 |

Correctness sanity check: deterministic victim output remained byte-identical
under churn. A medium-churn condense test measured `0.529` acceptance and
`99.7 tok/s` across the churn window.

Full note: [`benchmarks/20260629-dspark-keys-concurrency-checkpoint.md`](benchmarks/20260629-dspark-keys-concurrency-checkpoint.md).

### 2026-06-29 full-1M concurrency microbench (1M / 6)

The 200K/16 profile above maximizes raw concurrency. For agent fleets that want
the **full 1M context ceiling AND concurrency**, run `max_model_len=1048576` with
`max_num_seqs=6`. Every request can still grow to 1M while up to 6 sessions run at
once, because the shared KV pool — not a per-slot reservation — is the real limit
(see [How the KV cache works](#how-the-kv-cache-works-why-1m--concurrency-is-safe)).

Validated on the 2026-06-29 code-completion microbench deployment (NVFP4,
`max_model_len=1048576`, `max_num_seqs=6`,
`VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`, `VLLM_USE_B12X_WO_PROJECTION=1`):

- Boot: `GPU KV cache size: 1,901,239 tokens`, `Maximum concurrency for 1,048,576 tokens per request: 1.81x`
- 6 concurrent requests: **6/6 success**, **~182 tok/s aggregate** (~30 tok/s per stream), no OOM / no preemption failures
- Single-stream decode on this same profile: ~67 tok/s (code)

This is the right shape when most sessions sit far below 1M (typical agent turns)
but you still want the 1M ceiling available. The 2026-06-30 agent-stability
checkpoint above is the safer number to cite for Hermes/OpenClaw harness
validation.

> Higher concurrency is not free: under sustained pressure you can see added
> scheduler churn, prefill contention, and KV fragmentation. 1M/6 is validated
> for normal-length agent traffic; for guaranteed deep-context work under load,
> 1M/2 is conservative and 500K/4 is a balanced middle.

### Honest per-category benchmark (verified live 2026-07-04)

Temp 0, authoritative `completion_tokens`/wall, 5 varied prompts, TP=2 on
Asusi (rank0/head) + Spark4 (rank1/worker), served `:8888`, clean output. Full
config in [`DEFAULT-CONFIG.md`](DEFAULT-CONFIG.md).

| category | tok/s |
| --- | ---: |
| Math | 60.1 |
| JSON | 54.0 |
| Code | 53.8 |
| Communication | 42.0 |
| Narrative | 33.7 |
| **mixed avg** | **48.7** |

Structured/agentic (JSON/code/math — the supervisor workload) = 54–60 tok/s.
Deterministic best-case ~75 (not representative). DSpark draft acceptance ~92% on
structured, ~40% on creative. Decode speed is workload-dependent because DSpark
spec-decode acceptance varies with the text.

### Historical 60 tok/s DSpark baseline (diagnostic)

The older ~60 tok/s number was reproduced, but it is a separate diagnostic
profile, not this repo's default 1M NVFP4 deployment:

- image rebuilt from `rafaelcaricio/vllm#1` commit `3519c3b88`
- `max_model_len=262144`
- `max_num_seqs=1`
- `kv_cache_dtype=fp8`
- `MTP_NUM_TOKENS=5`
- `thinking=false`
- `temperature=0.0`, `top_p=1.0`
- measured `63.97 tok/s` on the `code_completion` gate with `67.9%` DSpark acceptance

Use this to diagnose image/runtime drift. Do not confuse it with the production
1M NVFP4 path. Full note:
[`benchmarks/20260630-dspark-pr-head-262k-fp8-speed-baseline.md`](benchmarks/20260630-dspark-pr-head-262k-fp8-speed-baseline.md).

### Benchmark notes

- The old speed checkpoint is single stream, not aggregate throughput.
- The high-concurrency benchmark is aggregate throughput and was validated at
  `max_model_len=200000`, not full 1M context.
- Full context and high concurrency compete for the same KV pool. The C12 1.5M
  profile is intended for normal agent traffic where most sessions sit far below
  the 1.5M ceiling; it is not twelve simultaneous full-1.5M requests.
- 1M was validated as booted/advertised `max_model_len` with KV headroom and
  short-prompt speed probes. This repo does not claim a full 1M-token retrieval or
  correctness benchmark.
- The measured probes were p256/p512 with g64/g256. Rebenchmark if you change
  sampling, batching, context length, WO projection, compressed MLA, or the
  confidence scheduler.
- The throughput tables above were captured on the pre-fix greedy-draft MTP5
  configuration.

## Configuration

### The knobs

Three independent knobs, often confused, plus the spec-decode depth:

| knob | what it is | this build |
| --- | --- | --- |
| **KV cache pool** | total shared KV memory in tokens, sized from `gpu_memory_utilization` after weights load | ~1.9–2.04M tokens (NVFP4) |
| `max_model_len` | per-request **ceiling** — how long any one request may grow | 1,048,576 (1M) |
| `max_num_seqs` | **concurrency cap** — max active sequences the scheduler runs at once | 6 |

### 1M Keys-concurrency profile

Core vLLM flags:

- `--tensor-parallel-size 2`
- `--distributed-executor-backend mp`
- `--nnodes 2`
- `--kv-cache-dtype nvfp4_ds_mla`
- `--block-size 256`
- `--max-model-len 1048576`
- `--max-num-seqs 6`
- `--max-num-batched-tokens 8192`
- `--max-cudagraph-capture-size 6` (must equal `--max-num-seqs`)
- `--gpu-memory-utilization 0.80`
- `--enable-prefix-caching`
- `--async-scheduling`
- `--enable-chunked-prefill`
- `--generation-config vllm` (no `--override-generation-config`)
- `--speculative-config '{"method":"dspark","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}'`
  - **This 1M / `max-num-seqs 6` profile MUST use `num_speculative_tokens: 3`.** Spec-decode requires the
    cudagraph capture sizes to be a multiple of `num_speculative_tokens + 1`. At `max-num-seqs 6` vLLM's
    ladder is `[1,2,4]`: spec 3 → multiple of **4** (the `4` satisfies it ✓); **spec 5 → multiple of 6,
    which `[1,2,4]` cannot satisfy → `No valid cudagraph sizes` and engine init FAILS.** (Verified 2026-07-04.)
  - **`num_speculative_tokens: 5` is faster but requires `max-num-seqs` to be a multiple of 6** (e.g. 12) so
    the ladder yields a valid captured size. That's the lower-context / higher-concurrency lane — see
    [`DEFAULT-CONFIG.md`](DEFAULT-CONFIG.md) (350K ctx, seqs 12, spec 5), benchmarked **~49 tok/s mixed /
    54–60 structured / ~75 best-case**; spec 3 ≈ 40 avg. At full 1M the seqs-12 KV won't fit on 2 Sparks,
    so **1M pairs with seqs 6 + spec 3**; the seqs-12 + spec-5 lane pairs with a shorter context.
  - Keep `draft_sample_method:probabilistic` — it beats greedy for DSpark's calibrated draft heads.

Key runtime env:

- `VLLM_USE_FLASHINFER_SAMPLER=1`
- `VLLM_USE_B12X_MOE=1`
- `VLLM_USE_B12X_WO_PROJECTION=1`
- `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`
- `VLLM_DSPARK_CONFIDENCE_SCHEDULER=off`
- `VLLM_DSPARK_LOCAL_ARGMAX=1`
- `VLLM_DSPARK_REPLICATE_MARKOV_W1=1`
- `VLLM_DSPARK_FUSED_MARKOV_ARGMAX=0`
- `VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT=0`
- `VLLM_DSV4_B12X_COMPRESSED_MLA=0`
- `VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE=0`
- `B12X_W4A16_TC_DECODE=0`

> **`VLLM_USE_B12X_MOE=1` is essential — this one env var is the entire speed
> difference.** With `=1` the boot log reads `Using 'B12X' Mxfp4 MoE backend` and
> decode runs at the full ~50-60+ tok/s. Setting it to `0` silently falls back to
> the `DEEPGEMM_MXFP4` MoE path and tanks decode to ~29 tok/s. Keep it `1`.

### 200K concurrency profile

For DSpark concurrency, use the included overlay files with Keys' concurrency
patch and set:

- `MAX_MODEL_LEN=200000`
- `MAX_NUM_SEQS=16`
- `VLLM_USE_B12X_WO_PROJECTION=1`
- `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`

Run the static and staggered checks:

```bash
python3 benchmarks/keys-concurrency/bench_concurrent.py http://127.0.0.1:8888 1,4,8,16
python3 benchmarks/keys-concurrency/staggered_bench.py http://127.0.0.1:8888 16 0.4
python3 benchmarks/keys-concurrency/correctness_test.py http://127.0.0.1:8888
```

### 1M single-stream legacy profile

For conservative single-stream testing, set `MAX_NUM_SEQS=1` and
`VLLM_USE_B12X_WO_PROJECTION=0`. The default `MTP_NUM_TOKENS=3` with
`draft_sample_method=probabilistic` (2026-07-03 garble fix) applies here too;
older runs used greedy-draft MTP5, which upstream Mia and Keys had validated but
which caused the cold-start concurrent garble in agent serving.

### Tuning to combine context + concurrency

- To combine DSpark concurrency with longer context, pick a lower context target
  first, then raise concurrency slowly while watching boot logs, KV allocation,
  acceptance, and request errors.
- The current validated agent-serving profile is `MAX_MODEL_LEN=1500000`,
  `MAX_NUM_SEQS=12`, `GPU_MEMORY_UTILIZATION=0.85`, `MTP_NUM_TOKENS=3` with
  `draft_sample_method=probabilistic`, `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`,
  `VLLM_USE_FLASHINFER_SAMPLER=1`, `VLLM_USE_B12X_WO_PROJECTION=1`, no
  `--override-generation-config` (2026-07-03 garble fix), and
  `VLLM_DSV4_B12X_COMPRESSED_MLA=0`.
- The next max-sequence ladder to try is approximately 1.25M, 1.5M, then 1.75M,
  with the same boot/log/speed gates. Raw KV math alone is not enough because
  DeepSeek V4 sparse MLA also allocates max-length-dependent workspaces.

### How the KV cache works (why 1M + concurrency is safe)

The pool is **shared and allocated on demand**: PagedAttention hands KV blocks to
each request as it generates tokens and frees them when it finishes.
`max_model_len` and `max_num_seqs` are **ceilings, not reservations** — vLLM does
NOT pre-allocate `max_num_seqs × max_model_len` of KV. So the real constraint is:

```
sum(live tokens across all active requests) <= KV pool (~1.9M)
```

Worked examples at 1M ceiling / 6 slots:

```
6 requests x  50k tokens =  300k   fits easily
6 requests x 200k tokens =  1.2M   fits
6 requests x 317k tokens =  1.9M   ~at the limit
2 requests x 1M   tokens =  2.0M   ~at the limit  (this is the "1.81x" boot number)
6 requests x 1M   tokens =  6.0M   impossible — excess requests queue/preempt
```

The boot log's `Maximum concurrency for 1,048,576 tokens per request: 1.81x` only
means ~1.8 *simultaneous full-1M* requests fit. But agent turns are almost never
near 1M, so 6 normal-length sessions share the pool comfortably while the 1M
ceiling stays available for the rare long one. That is exactly why
`1M + max_num_seqs=6` is safe: you are not reserving 6×1M, you are sharing one
~1.9M pool across short requests under a high ceiling.

## Troubleshooting

### Garble fix (2026-07-03)

**Symptom.** On a cold server, the *first* prompt of a brand-new session — fired
under concurrency (several fresh sessions hitting the endpoint at once) — dumps
tool-call fragments / skill XML / parse-broken junk, then the same session
recovers and answers cleanly on later turns. Long or heavy sessions could also
drift into BOS or multilingual salad.

**Root cause.** This is a DSpark speculative-decoding **cold-start draft/target
mismatch**, not a sampling problem. Our spec-config used a **greedy draft**
(`draft_sample_method` was unset, so it defaulted to greedy). At a cold, concurrent
first batch the greedy draft distribution diverges from the target model, and the
accepted tokens come out as corrupted tool-call fragments that reach the tool
parser as garbage. It was compounded by `num_speculative_tokens=5` (a larger
mismatch window) and by not pinning `--max-cudagraph-capture-size` (the concurrent
first batch hit an uncaptured CUDA graph). Separately, the old
`--override-generation-config` carried `repetition_penalty=1.05`, which is a
documented DSpark spec-decode **crash risk** (illegal memory access), not a fix.

**Fix.** Five changes. They keep `--kv-cache-dtype nvfp4_ds_mla`, the 1.5M context,
`max_num_seqs`, TP, and the RoCE/NCCL/fabric config untouched:

1. **Spec-config → `{"method":"dspark","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}`.**
   This is the key change: a probabilistic draft matches the target distribution,
   and 3 tokens shrinks the cold-start mismatch window (was greedy draft with 5).
2. **`--max-cudagraph-capture-size` set equal to `--max-num-seqs`** so the first
   concurrent batch hits a captured graph instead of wedging on capture.
3. **`--async-scheduling` and `--enable-chunked-prefill`.**
4. **Env `VLLM_USE_FLASHINFER_SAMPLER=1` and `VLLM_DSPARK_REPLICATE_MARKOV_W1=1`.**
5. **Removed `--override-generation-config`** (kills the `repetition_penalty`
   crash risk). `--generation-config vllm` is kept; no sampling override is added,
   and explicit client request parameters still win.

Diagnosed by comparing this launch against Aiden's clean `production-3.2` DSpark
recipe (which stayed clean under the same oh-my-pi + Hermes concurrency that
garbled ours), and **verified live under concurrency on 2026-07-03** across two
independent TP=2 instances: four concurrent fresh-session first-prompts per
instance, all clean — zero tool-call dumps, zero salad.

**The garble fix costs ~zero speed.** Head-to-head on identical hardware, nvfp4
4-bit KV, 300K context, code workload: the original greedy-draft config measured
`61.8 tok/s` and the fixed probabilistic-draft config measured `61.3 tok/s` —
statistically identical, and only the fixed one is garble-free under concurrency.
Decode speed is workload-dependent because DSpark spec-decode acceptance varies
with the text: code ~61 tok/s, technical ~40 tok/s, essay ~31 tok/s. The "50-60"
figure is the code/agent number, not a flat rate.

**Image-compat notes for the `dspark-nvfp4-stage-c`-class image:**

- Do **not** pass Aiden's `--attention-backend FLASHINFER_MLA_SPARSE_DSV4`; that
  backend name does not exist on this image (`ValueError: unknown backend`). Leave
  the attention backend on AUTO — it selects the DeepSeek-V4 sparse MLA path that
  works with `nvfp4_ds_mla`.
- Do **not** set `VLLM_USE_V2_MODEL_RUNNER=1`; it is incompatible with DSpark
  speculative decoding (hard reject at startup). Keep it unset (`0`).

### Gibberish, loops, Chinese drift, or prompt/XML leakage

If the model boots and basic prompts like `hi` work, but real agent traffic
randomly turns into repeated characters, Chinese drift, leaked tool/schema XML, or
Telegram-visible junk, do not assume the weights are bad. On this deployment there
are three checks to make before blaming the weights:

1. **Runtime concurrency safety:** make sure the Keys Patch 2b logic is present in
   `recipe/overlay/vllm/v1/spec_decode/dspark_proposer.py`. The important behavior
   is that ragged `query_start_loc` handling does not depend on
   `num_rejected_tokens_gpu`, and the no-rejection path creates a zero rejected
   token tensor instead of falling through to unsafe request reshaping. Without
   this, concurrent DSpark requests can mix context.
2. **Runtime image provenance:** make sure the image really contains the current
   DSpark overlay. A reused local tag named `vllm-dspark-runtime:clean` caused
   misleading failures even though a nearby PR-head image worked. Rebuild from the
   intended overlay commit when in doubt.
3. **Decode safety (updated 2026-07-03):** do **not** apply a server-side
   `repetition_penalty` on the DSpark speculative-decode path — it is a documented
   spec-decode crash risk (illegal memory access), and it is not what fixes the
   garble. The actual cold-start garble fix is the DSpark spec-config change
   (`draft_sample_method=probabilistic`, `num_speculative_tokens=3`) plus
   `--max-cudagraph-capture-size == --max-num-seqs`; see
   [Garble fix (2026-07-03)](#garble-fix-2026-07-03). The compose launcher runs
   with `--generation-config vllm`, **no** `--override-generation-config`, and
   `--default-chat-template-kwargs '{"thinking":false}'` so default requests do not
   inherit unstable model-card sampling. Explicit client request parameters still
   win. For exact deterministic curl checks, send `temperature: 0` in the request
   body.

Also clear agent fallback lists during validation. A model that looks fixed in
direct vLLM tests can still appear poisoned if the orchestration layer silently
falls back, reboots a session, or replays a stale prompt/tool transcript into the
visible message stream. Keep OpenClaw/Hermes changes separate from model runtime
validation unless you are deliberately testing that harness.

Validation gates to run after a live fix:

```text
direct vLLM prompts: clean
direct concurrent vLLM prompts: clean
agent harness prompts: clean, DeepSeek, no fallback
MTP5 accepted-token positions 0..4 active
```

This keeps NVFP4 KV and MTP5. Do not switch to fp8 or drop to a smaller fallback
model just to hide the symptom unless you intentionally accept the context and
quality tradeoff.

### Still getting gibberish after the fix? → go fp8

If gibberish / multilingual (Chinese / BOS) salad persists **after** applying the
sampling and spec-decode fixes, the most reliable fix is to switch the KV cache
from 4-bit `nvfp4_ds_mla` to `fp8` and run Aiden Le's `production-3.2` image — see
the FP8 build:
[**DeepSeek-v4-Flash-Official-vLLM-DSpark-NVFP4-2x-DGX-Spark**](https://github.com/tonyd2wild/DeepSeek-v4-Flash-Official-vLLM-DSpark-NVFP4-2x-DGX-Spark).
The 4-bit KV can collapse into salad under long, heavy agentic context; fp8 KV
stays clean (Aiden serves 500K context on fp8). Keep this NVFP4 repo's path when
you need the 1.5M-token context; use the fp8 build when clean output under
concurrency matters more than max context.

### Important caveat — Stage C padded NVFP4

This is the **Stage C padded NVFP4** path. It keeps DeepSeek V4's known-good
584-byte sparse-MLA cache envelope while routing the runtime through
`nvfp4_ds_mla`.

It is **not** the unresolved true-layout 416-byte NVFP4 kernel fix. The
true-layout experiments were useful for diagnosis but failed past roughly 411 real
prompt tokens, so they are intentionally not presented here as the reproducible
recipe.

## Repository layout

| path | purpose |
| --- | --- |
| `recipe/overlay/` | base DSpark vLLM overlay files |
| `recipe/Dockerfile.dspark-runtime-overlay` | builds the base DSpark runtime overlay |
| `recipe/nvfp4/Dockerfile.stage-a` | adds `nvfp4_ds_mla` dtype plumbing |
| `recipe/nvfp4/Dockerfile.stage-b` | enables DeepSeek V4 `nvfp4_ds_mla` probe path |
| `recipe/nvfp4/Dockerfile.stage-c` | switches DeepSeek V4 NVFP4 to the validated 584-byte padded envelope |
| `docker-compose.dspark.yml` | two-node vLLM/DSpark service |
| `.env.dspark.example` | sanitized cluster configuration template |
| `build-dspark-vllm-runtime.sh` | builds the Stage C image locally and on the worker |
| `prepare-dspark-model-cache.sh` | downloads/verifies the model cache |
| `start-deepseek-v4-flash-dspark.sh` | worker-first launch and smoke test; honors worker path/cache/IP overrides |
| `stop-deepseek-v4-flash-dspark.sh` | stops head and worker services |
| `status-deepseek-v4-flash-dspark.sh` | shows head/worker container state |
| `logs-deepseek-v4-flash-dspark.sh` | tails head/worker DSpark logs |
| `smoke-deepseek-v4-flash-dspark.sh` | direct concurrent OpenAI-compatible smoke test |
| `validate-dspark-config.sh` | renders and checks the local DSpark compose/env config |
| `patches/keys-concurrency.patch` | full path-adjusted Keys concurrency patch reference |
| `docs/PATCHES.md` | plain-English Patch 1 / Patch 2 / Patch 2b concurrency explanation |
| `UPSTREAM_V024_STATUS.md` | current vLLM v0.24.0 vs DSpark PR #46995 upgrade notes |
| `AGENT_GARBLE_FIX.md` | update path for older deployments that saw agent garble/drift/loops |
| `scripts/agent_sanity_bench.py` | direct OpenAI-compatible 1/2/4/6 concurrency and garble check |
| `scripts/capture_runtime.sh` | captures head/worker Docker inspect, ps, and log tails before/after changes |
| `benchmarks/keys-concurrency/` | benchmark scripts from Keys' patch repo |
| `benchmarks/` | measured 1M and concurrency checkpoint evidence |

## Credits & links

See [`CREDITS.md`](CREDITS.md) for the full attribution and license notes.

This recipe stands on prior public work:

- **Keys / drowzeys' DSpark in-server concurrency patch:**
  [drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash](https://github.com/drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash).
  This patch fixes the request-stable DSpark main-KV slot mapping and the ragged
  `query_start_loc` path needed for real independent-arrival continuous batching.
  The concurrency results in this repo depend directly on that work.
- **drowzeys ("Keys")** — origin of wiring the `nvfp4_ds_mla` KV-cache dtype into
  a DGX Spark launch recipe
  ([Keys---Full-GLM-5.2-Quantrio…](https://github.com/drowzeys/Keys---Full-GLM-5.2-Quantrio-INT4-INT8-mixed-8bit-Attention-on-4-x-DGX-Spark-GB10-Cluster)).
  This build's 1M NVFP4 KV path descends from that `nvfp4_ds_mla` work.
- **Rafael Caricio's DSpark vLLM integration:**
  [rafaelcaricio/vllm#1](https://github.com/rafaelcaricio/vllm/pull/1) and the
  DSpark deployment/runbook PR
  [rafaelcaricio/spark_vllm_docker#1](https://github.com/rafaelcaricio/spark_vllm_docker/pull/1)
- **Fraser Price's DeepSeek V4 Flash DSpark model/runtime work:**
  [fraserprice/DeepSeek-V4-Flash-DSpark](https://huggingface.co/fraserprice/DeepSeek-V4-Flash-DSpark)
  and [fraserprice/dspark-vllm](https://github.com/fraserprice/dspark-vllm)
- **MiaAI-Lab's two-node DGX Spark packaging and worker-first launch runbook:**
  [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
- **Aiden Le** — clean `production-3.2` fp8 DSpark recipe used to diagnose the
  garble and as the fp8 fallback build.
- **Roady001** and **Fable** — DSpark cold-start garble root-cause fix (Patch 3);
  see [`CREDITS.md`](CREDITS.md) and [`docs/PATCHES.md`](docs/PATCHES.md).
- **Wpnx330** — CUDA-graph capture-size fix
  ([PR #5](https://github.com/tonyd2wild/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark/pull/5)).
- **0rand** — parameterized the API port (`VLLM_PORT`, PR #1) and made the early,
  correct MTP=3 call.
- **paulbrav** — long-context engine-death crash report + fix (PR #4) and
  slot-corruption instrumentation.
- **DaveCharland** — reported/characterized the episodic soft-failure (#6).
- Upstream **vLLM**, **FlashInfer**, **NVIDIA** Blackwell/CUDA/NCCL tooling, and
  **DeepSeek V4 Flash**.
- **DeepSeek-AI's DeepSpec** work as the public DSpark/speculative decoding
  foundation.

Our contribution here is the 1M NVFP4-KV checkpoint recipe, the Stage A/B/C
runtime patches, sanitized two-node launch config, applying and validating Keys'
concurrency patch on the NVFP4 profile, and measured benchmark artifacts from the
validated runs.

### License

Repo scripts and docs are published under this repo's [`LICENSE`](LICENSE) (MIT).
The vLLM overlay/runtime files and `patches/keys-concurrency.patch` are
vLLM/DSpark-derived and retain their Apache-2.0 lineage and SPDX headers where
present. Base images, FlashInfer/TileLang/Triton/CUDA/NCCL, and model weights are
separate upstream artifacts with their own licenses and usage terms.
