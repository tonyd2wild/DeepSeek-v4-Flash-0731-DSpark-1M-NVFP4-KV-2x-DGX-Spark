# sparkrun recipe — DeepSeek V4 Flash 0731, 1M NVFP4 KV, 2x DGX Spark

One-command, copy-paste reproducible deployment of this repo's setup via
[sparkrun](https://github.com/spark-arena/sparkrun). No image build: the
recipe pulls the public base image (digest-pinned, so it can never silently
change) and rebuilds the `dspark-nvfp4-stage-c` runtime inside every
container at start — overlay, stage A/B/C patches, Patch 3 and Patch 4
included, pinned to commit `d728fae` of this repo. It fail-fast-verifies
Patch 3 and Patch 4 actually landed before serving, so you can't
accidentally run the half-speed stock loader.

Everything here was deployed and measured on a real 2x DGX Spark pair on
2026-08-01. For tuning, patch history and troubleshooting, the
[main README](../README.md) is the reference — this file is just how to run
it.

## Quick start

Once, on your head node (~200 GB free disk per node):

```bash
uvx sparkrun setup     # wizard: cluster, SSH mesh, ConnectX-7 fabric detection
```

Then:

```bash
sparkrun run ./deepseek-v4-flash-0731-dspark-nvfp4-1m-vllm.yaml
```

sparkrun syncs the image (~13 GB) and model (167 GB, first time only) to
both nodes, injects per-host NCCL/RoCE settings, launches the worker
headless and the head with the API on port 8888. First-ever boot is ~9 min,
warm boots ~5 min. `Ctrl+C` detaches; `sparkrun logs` / `status` / `stop`
manage it afterwards.

Verify the boot:

```bash
curl -fsS http://<head-ip>:8888/v1/models   # expect "max_model_len": 1048576
sparkrun logs deepseek-v4-flash-0731-dspark-nvfp4-1m-vllm | grep -E "B12X|KV cache size"
# want:  Using 'B12X' Mxfp4 MoE backend     (missing = half-speed fallback)
#        GPU KV cache size: ~1.5-1.7M tokens (varies per boot, that's normal)
```

## Measured (2x DGX Spark, TP=2, warm, 2026-08-01)

| workload | tok/s |
|---|---:|
| Protocol Starfall prompt (the video demo; video showed 50.9 live) | **71.7** |
| bulk SQL INSERTs (~100% draft acceptance) | **82–83** |
| hard code corpus (checker.ts), single-stream | 46.3 |
| hard code corpus, aggregate at c6 | 74.8 |
| hard code corpus, single-stream at 32K depth | 45.1 |

Decode is acceptance-bound and content-driven — the spread above is one
server on one config, exactly as the main README describes.

## Speed test

```bash
./speedtest-starfall.sh http://<head-ip>:8888
```

Warms the engine (first requests after boot or ~30 min idle run ~30% slow),
then measures the Starfall prompt the right way: `stream: false` +
`usage.completion_tokens`. Don't count SSE chunks — under spec decode vLLM
emits one chunk per decode *step*, so stream-delta counting reports steps/s
and under-reads by the acceptance length.

## Benchmark

The recipe ships a coding-flavored `benchmark:` block (llama-benchy's
default corpus is a novel, which under-reports this model for code work):

```bash
sparkrun benchmark ./deepseek-v4-flash-0731-dspark-nvfp4-1m-vllm.yaml --skip-run
```

runs depth 0/32K x concurrency 1/2/6 (~11 min) against a running server.
Corpus A is TypeScript compiler internals (conservative); switch to
boilerplate-heavy HTML for the upper bound with
`-b book_url=https://raw.githubusercontent.com/whatwg/html/88ae68cb961651f0f92c5d2046049f53ecdfc6cf/source`.
Two caveats: warm up first, and ignore the depth-32K cells at concurrency >1
unless you raise `-b tg=1024` — at the default `tg=128` those cells measure
the chunked-prefill storm, not decode.
