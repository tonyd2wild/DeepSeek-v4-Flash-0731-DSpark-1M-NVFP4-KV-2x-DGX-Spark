# CURRENT — what this repo actually runs (2026-09-02)

One launcher per topology, at the top level, in [`launchers/`](launchers/). Everything else on
this repo is reference, history, or a second documented recipe. If a doc and a launcher disagree,
**the launcher wins**.

---

## DeepSeek-V4-Flash-Vision-Exp, TP2 (two Sparks)

**Launcher:** [`launchers/ds4-vision-tp2.sh <0|1>`](launchers/ds4-vision-tp2.sh)

| rank | node | fabric IP | role |
|---|---|---|---|
| 0 | **Asusi** | `192.168.192.3` | head — serves `:8888` |
| 1 | **Bluey** | `192.168.192.1` | worker (`--headless`) |

**Launch worker first**, then the head:

```bash
./launchers/ds4-vision-tp2.sh 1     # bluey  (holds the weights, NFS-exports them)
./launchers/ds4-vision-tp2.sh 0     # asusi  (head, serves :8888)
```

**Recipe:**

- **Served model id:** `deepseek-v4-flash-dspark` (not `...-vision-exp` — the vision build is a
  drop-in on the same endpoint, so agents wired to the text id do not get rewired).
- **DSpark speculative decoding:** `k=5`, `draft_sample_method=probabilistic`.
- **Patch 3 + Patch 4** bind-mounted from `/var/tmp` **on both nodes**. Patch 4 fails *silently*
  when it is missing — the draft's always-on shared expert loads uninitialised and decode runs at
  roughly half speed with perfect-looking output. **Run
  [`scripts/check-patch4.sh <head-container> <worker-container>`](scripts/check-patch4.sh) before
  trusting any number.**
- **KV cache dtype:** `nvfp4_ds_mla`, `--block-size 256`.
- **Context:** `1048576` (1M). *Changed in this commit:* the launcher previously passed
  `1500000`; 1M is the model's calibrated YaRN ceiling and is now the standard on both topologies.
- **`--gpu-memory-utilization 0.85`**, `--max-num-seqs 12`, `--max-cudagraph-capture-size 12`,
  `--max-num-batched-tokens 8192`.
- **Image:** `ghcr.io/tonyd2wild/vllm-dspark-runtime:v2026.09.02` — the same digest as the local
  tag `mia-raf-pr1-nvfp4-probe-c-keys-concurrency-p2b` that the launcher names.
- **Weights:** on Bluey at `/var/tmp/models/DeepSeek-V4-Flash-Vision-Exp`, exported over NFS
  (Asusi mounts it at `/mnt/bluey-models`).
- **Uncensored variant:** set `MODEL_DIR=keys-DeepSeekV4Flash-Vision-EXP-ablit`. Drop-in — same
  shards, same format, same tokenizer. Both launchers read `MODEL_DIR` (default
  `DeepSeek-V4-Flash-Vision-Exp`); the uncensored build is the same launcher with that one variable set.

**Expected (TP2):**

- Single-stream **real-prompt decode ≈ 53 tok/s**.
- The counting prompt reaches a higher number; that is the **draft-acceptance ceiling**, not
  throughput. See "How we quote numbers".
- **KV pool ≈ 2.79M tokens.**

---

## DeepSeek-V4-Flash-Vision-Exp, TP4 (all four Sparks)

**Launcher:** [`launchers/ds4-vision-tp4.sh <0|1|2|3>`](launchers/ds4-vision-tp4.sh)

| rank | node | fabric IP | role |
|---|---|---|---|
| 0 | **Asusi** | `192.168.192.3` | head — serves `:8888` |
| 1 | **Bluey** | `192.168.192.1` | worker (local weights) |
| 2 | **Reddie** | `192.168.192.2` | worker (NFS) |
| 3 | **Spark4** | `192.168.192.4` | worker (NFS) |

**Launch order: 3, 2, 1, then 0.**

```bash
./launchers/ds4-vision-tp4.sh 3     # spark4
./launchers/ds4-vision-tp4.sh 2     # reddie
./launchers/ds4-vision-tp4.sh 1     # bluey
./launchers/ds4-vision-tp4.sh 0     # asusi (head, serves :8888)
```

**Preflight, on all four nodes:**

- Patch 3 (`patch3-scheduler.py`), Patch 4 (`spec-dspark.py`) **and** the four vision port files
  (`ds4v_model.py`, `ds4v_vision.py`, `ds4v_mm.py`, `ds4v_registry.py`) staged at `/var/tmp`.
  The launcher checks all six and exits if any is missing.
- Workers mount Bluey's weights export at `/mnt/bluey-models`.
- **Drop page cache on all four nodes before launch.**

**Recipe:** same image, same `k=5` probabilistic DSpark, same `nvfp4_ds_mla` KV, same
`--gpu-memory-utilization 0.85`, `--max-model-len 1048576`. Differences from TP2:
`--tensor-parallel-size 4`, `--nnodes 4`, **`--max-num-seqs 64`**, and
**`--max-cudagraph-capture-size 64`** (the workspace copy of this launcher still said 12/12; the
validated run used 64/64). CUDA graphs are on: `--enforce-eager` is not passed,
`--max-cudagraph-capture-size 64` is, and the head log shows `Graph capturing finished in 10 secs,
took 0.68 GiB`; the mode is vLLM's default for this image, not pinned by the launcher.

**Expected (measured 2026-09-02, TP4):**

| | |
|---|---|
| KV cache pool | **8,328,795 tokens** |
| Time to healthy | **~7 min** |
| Real-prompt decode, single stream | **prose 42 tok/s · code 98 tok/s** |
| Mixed, 16 streams | **124 tok/s aggregate** |
| Counting ceiling (labeled draft-acceptance only) | **95 tok/s at C1 · 1,073 tok/s at C48** |
| Cold prefill | **~4.6K tok/s**, flat from 14K to 182K tokens |

---

## How we quote numbers

- **Decode is quoted from real prompts** — prose, code, and the like. Those are the numbers to
  compare against anything else.
- **The counting prompt is a labeled draft-acceptance ceiling only.** "Count to 300" is nearly
  perfectly predictable, so the DSpark drafter accepts almost every token and the tok/s figure
  measures how fast speculation can run, not how fast the model serves work. Never quote a
  counting number as throughput.
- **Prefill is cold only.** Prefix caching is on, so a warm prefill number measures the cache.

---

## Do not use

**Do not use the following as a recipe.** These are dated status notes and superseded snapshots.
They will move under `archive/` in the next cleanup commit; **nothing is being deleted**, and
nothing a PR or issue links to changes path — `vision-exp/ds4-vision-tp2.sh` stays as a symlink to
`launchers/ds4-vision-tp2.sh` for exactly that reason.

- Dated status / point-in-time notes: `SPEED-UPDATE-2026-07-16.md`, `SPEED-UPDATE-2026-07-29.md`,
  `RUNTIME-BAKEOFF-2026-07-29.md`, `KAI-DS4-UPDATE-NOTE.md`, `OFFICIAL_MAIN_PORT_PLAN.md`,
  `UPSTREAM_V024_STATUS.md`, `AGENT_GARBLE_FIX.md`, `verified-deployed-2026-07-04/`.
- Any `MTP_NUM_TOKENS=3` / `k=3` guidance anywhere in this repo. That A/B was measured without the
  Patch 4 mount, which silently collapses draft acceptance. It is retracted (issue #48). **k=5.**
- Any `MAX_MODEL_LEN=1500000` guidance, including the KV/context figures still carried in
  `vision-exp/README.md`'s "Measured" table. Standard is **1,048,576**.

**Two things stay put and are NOT archive candidates:**

1. **The text-lane 0731 compose recipe** — `DEFAULT-CONFIG.md`, `docker-compose.dspark.yml`,
   `.env.dspark.example`, and the root `*-deepseek-v4-flash-dspark.sh` / `validate-dspark-config.sh`
   scripts. That is a **second documented recipe** for the text checkpoint, still supported, and it
   stays exactly where it is unless the archive commit says otherwise. It is not the vision recipe:
   if you are serving the vision build, use the launchers above.
2. **`sparkrun/`** — the self-contained sparkrun recipes for both checkpoints, and **`parity/`**.

The self-contained `sparkrun/` recipe serves under the id `deepseek-v4-flash-vision-exp`; the launchers here serve `deepseek-v4-flash-dspark`. Clients pointed at :8888 use the launcher's id.

<!-- launcher hashes, maintained by tools/check-current.sh --write -->
sha256 310e74b5a459f3cab876cd4d5edcbe67416ed74e408dfd965fd592a0bd0b1409  launchers/ds4-vision-tp2.sh
sha256 3ace8ad8172c5a9c146e7dacc385b16d1e60049f65b7697527ec14eeead35ea4  launchers/ds4-vision-tp4.sh
