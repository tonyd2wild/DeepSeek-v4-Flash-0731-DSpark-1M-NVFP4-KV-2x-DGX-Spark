# DeepSeek-V4-Flash-Vision-Exp (DSpark) on DGX Spark — TP2 (2 nodes) or TP4 (4 nodes), 1M context, NVFP4 KV

> **2026-09-10: DeepSeek-V4.1-Flash is out.** The four-Spark vLLM TP4 recipe for it (Engram-on-disk patch, sm121 kernel build, launchers, measured numbers) lives in its own repo: **[tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark)**. This repo stays the DeepSeek-V4-Flash-Vision-Exp (DS4) recipe.

**Current recipe: see [`CURRENT.md`](CURRENT.md).** That file, not this page, is the source of truth.

**Launchers:** [`launchers/ds4-vision-tp2.sh <0|1>`](launchers/ds4-vision-tp2.sh) — ranks 1 (bluey) then 0 (asusi, head) · [`launchers/ds4-vision-tp4.sh <0|1|2|3>`](launchers/ds4-vision-tp4.sh) — ranks 3, 2, 1, then 0 (asusi, head). Both serve `:8888` as **`deepseek-v4-flash-dspark`**.

**Preflight — Patch 4 fails silently, so check it before trusting any number:** `./scripts/check-patch4.sh <head-container> <worker-container>`

Everything else on this page is reference; `archive/` (coming) is history.

## 🆕 2026-08-31 — now running **DeepSeek-V4-Flash-Vision-Exp**, with native vision

DeepSeek released
[**DeepSeek-V4-Flash-Vision-Exp**](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp)
today — the first multimodal model in the V4 family. **It is deployed here the same day,
on 2x DGX Spark at TP2, with real image input and DSpark speculative decoding intact.**

```
"Based on the image, the two colors are red and blue.
 Red is on the left side. Blue is on the right side."
```

Not a sidecar and not a VLM proxy — the model's own 32-block ViT and aligner running
inside vLLM. This needed a genuine port: vLLM's `DeepseekV4ForCausalLM` is the **text-only**
class, and the vision checkpoint reports the *same* architecture string while carrying 316
tensors vLLM has nowhere to put, so it fails to load at all. DeepSeek shipped only a
reference implementation, explicitly "rather than a production serving engine."

| | measured, 2x GB10, TP2, temperature 0 |
|---|---|
| **KV cache pool** | **2,904,519 tokens** (18.18 GiB) |
| **Context** | **1,500,000** per request · max concurrency **1.94x** |
| Vision, 336x336 + 26-token answer | **1.03 s** end to end |
| Image understanding | colour **and** position correct, both orientations |

Measured on the profile as it stood that day — `MAX_MODEL_LEN=1500000`, `MAX_NUM_SEQS=12`,
`GPU_MEMORY_UTILIZATION=0.85`, `MTP_NUM_TOKENS=3` — with the vision port on top. **The current
validated profile is `MAX_MODEL_LEN=1048576` (1M) and `MTP_NUM_TOKENS=5`; the KV/context figures
in the table above were taken at 1.5M and are not current — see [`CURRENT.md`](CURRENT.md).**
**Use `k=5`, same as the 0731 recipe, with Patch 4 mounted.** An earlier version of this branch said k=3: that A/B was measured without the Patch 4 `spec-dspark.py` mount, which silently halves the draft's acceptance (see `vision-exp/README.md`, [#48](../../issues/48)). With Patch 4 present, k=5 wins count-to-300 by a third and code is neutral. This release moved `num_nextn_predict_layers` from 1 to 3, which is why the drafter deserves a second look, but not at k=3.

**→ [Full guide, the twelve blockers, and the port: `vision-exp/`](vision-exp/README.md)**

Two deviations from the reference are documented there and are **not** yet fixed —
bidirectional attention inside image spans, and the new `bias_vl` modality-specific MoE
routing bias. Both affect image quality, neither affects text. Read that section before
quoting this against benchmarks.

Text-only `0731` is unchanged and still fully supported — every patch is guarded on
`vision_n_layers`, so the same files serve both checkpoints.

---

> Self-contained two-node DGX Spark recipe for serving DeepSeek-V4-Flash with vLLM
> TP=2, DSpark speculative decoding, and an experimental `nvfp4_ds_mla` KV cache —
> 1M-token calibrated context, clean under agent concurrency.
>
> **Current default: `DeepSeek-V4-Flash-Vision-Exp`** — the experimental **vision** variant
> (native image input) served on our 2× DGX Spark cluster. It reuses this repo's entire text
> recipe (TP=2, `nvfp4_ds_mla` KV, 1M context, DSpark `k=5`, Patch 3, Patch 4) and adds the
> vision-model files as read-only bind-mounts. **Start here →
> [Serving DeepSeek-V4-Flash-Vision-Exp](#serving-deepseek-v4-flash-vision-exp--current-default-2026-08-31)**
> and the full byte-for-byte command in
> [`VISION-EXP-DEFAULT-CONFIG.md`](VISION-EXP-DEFAULT-CONFIG.md).
>
> **Also covers the text checkpoints** (the tuning/troubleshooting base the vision recipe inherits):
> - **`deepseek-ai/DeepSeek-V4-Flash-0731`** (official release) — **78 tok/s peak, ~55 typical.**
>   Requires [Patch 4](patches/0004-dspark-shared-expert-gate-up-proj.patch); without it you get
>   roughly half speed at unchanged output quality. See
>   [Updating to the official 0731 release](#updating-to-the-official-deepseek-v4-flash-0731-release-2026-07-31).
> - **`fraserprice/DeepSeek-V4-Flash-DSpark`** (preview) — 84.3 tok/s peak. Everything in this
>   README below the 0731 section was measured on this checkpoint and still stands.

---

## Serving DeepSeek-V4-Flash-Vision-Exp — current default (2026-08-31)

This is what we run today: the experimental **vision** build of DeepSeek-V4-Flash (native image
input) on **2× DGX Spark** — head **asusi** (rank0) + worker **bluey** (rank1), TP=2, served on
`:8888`. It is the same recipe as the text model below, pointed at the vision checkpoint and with
the vision-model files added as read-only bind-mounts. **[Patch 4](#the-fix-that-must-not-be-dropped-on-the-vision-port)
is mandatory** — the vision port silently dropped it once and cost us roughly half our decode speed.

- **Model:** `DeepSeek-V4-Flash-Vision-Exp` (checkpoint pinned at commit
  `86f746b36186f0e567729a5c06a8c918caba82a9`).
- **Runtime image:** `vllm-dspark-runtime:mia-raf-pr1-nvfp4-probe-c-keys-concurrency-p2b`
  (vLLM `0.21.1rc1.dev339+g1967a5627bc3`, B12X MXFP4 MoE). This is the same probe-c image family the
  text recipe was captured on, so it predates the baked-in patches and needs them bind-mounted.
- **Serving config (verified live):** `--tensor-parallel-size 2`, `--kv-cache-dtype nvfp4_ds_mla`,
  `--block-size 256`, `--max-model-len 1048576`, `--gpu-memory-utilization 0.85`,
  `--max-num-seqs 12`, DSpark spec-decode `num_speculative_tokens: 5`. KV pool measured
  **2,790,000 tokens / 19.12 GiB** at `gpu_memory_utilization=0.85`.

### Mounts (this is the whole point)

The vision port runs the same launcher/compose as the text recipe, pointed at the Vision-Exp
weights, with these read-only bind-mounts. Stage each source file on **both** nodes at `/var/tmp/`
first (`start-*.sh` does not copy bind-mounted files to the worker):

```bash
# Patch 3 — cold-start garble root fix (source: recipe/overlay/vllm/v1/core/sched/scheduler.py)
-v /var/tmp/patch3-scheduler.py:/opt/env/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py:ro \
# Patch 4 — DSpark draft shared-expert loader fix (source: recipe/overlay/vllm/v1/spec_decode/dspark.py)
#   *** REQUIRED. Without this mount the draft's always-on shared expert loads uninitialised
#   *** and decode runs at ~half speed, failing SILENTLY (the drop is a logger.debug line).
-v /var/tmp/spec-dspark.py:/opt/env/lib/python3.12/site-packages/vllm/v1/spec_decode/dspark.py:ro \
# Vision-model files (native image support): ds4v_model.py / ds4v_vision.py / ds4v_mm.py / ds4v_registry.py
#   staged the same way onto the image's DeepSeek-V4 model + registry module paths.
-v /var/tmp/ds4v_model.py:.../ds4v_model.py:ro \
-v /var/tmp/ds4v_vision.py:.../ds4v_vision.py:ro \
-v /var/tmp/ds4v_mm.py:.../ds4v_mm.py:ro \
-v /var/tmp/ds4v_registry.py:.../ds4v_registry.py:ro \
```

Full byte-for-byte command, env, and node table: **[`VISION-EXP-DEFAULT-CONFIG.md`](VISION-EXP-DEFAULT-CONFIG.md)**.

### The fix that must not be dropped on the vision port

When we stood the vision serving port up, its run command carried the Patch 3 and vision mounts but
**silently dropped the Patch 4 `spec-dspark.py` mount**. The stock loader then took over: the DSpark
draft's shared expert (`shared_experts.gate_up_proj`) loaded **uninitialised** (12 tensors dropped),
draft acceptance collapsed, and decode ran at **roughly half speed** — with perfect output quality,
which is exactly what sends you looking in the wrong place. It fails **silently**: the dropped
tensors are reported at `logger.debug` ("Skipping unknown DSpark weight"), invisible at the default
INFO level, and the broken load "reports success". Full mechanism in
[`DSPARK-SHARED-EXPERT-FIX.md`](DSPARK-SHARED-EXPERT-FIX.md).

The fix is the one mount line above:

```bash
-v /var/tmp/spec-dspark.py:/opt/env/lib/python3.12/site-packages/vllm/v1/spec_decode/dspark.py:ro
```

**Verify it landed** (run against the head *and* worker container):

```bash
# Expect 6. The patched loader has the two shared-expert mapping rows plus their comment.
docker exec <container> grep -c shared_experts \
  /opt/env/lib/python3.12/site-packages/vllm/v1/spec_decode/dspark.py     # -> 6

# The stock/unpatched loader carries only the 2 attention rows and returns 0 here.
# Definitive: boot with VLLM_LOGGING_LEVEL=DEBUG and confirm there are NO
# "Skipping unknown DSpark weight" lines for shared_experts.w1 / shared_experts.w3:
docker logs <container> 2>&1 | grep "Skipping unknown DSpark weight.*shared_experts"   # -> (empty)
```

Or run [`scripts/check-patch4.sh <head-container> <worker-container>`](scripts/check-patch4.sh).

### Verified speed (2× DGX Spark, warmed, single-stream, with Patch 4)

Server-reported `completion_tokens` over wall time, temp 0, **warm** engine (see the warm-up
caveat below — warm it with a few long generations first):

| workload | with Patch 4 | without Patch 4 |
| --- | ---: | ---: |
| count to 100 | **80.1 tok/s** | 50.7 |
| code | **51.8** | 47.2 |
| prose | **33.2** | 30.4 |

KV pool **2.79M tokens / 19.12 GiB** at `gpu_memory_utilization=0.85`. (For reference,
MiaAI-Lab's comparable two-node recipe reports ~2.33M tokens and 62–83 tok/s single-stream.)

**Read these honestly.** The large jump on *count* is Patch 4 (the shared expert now computes
correctly) **plus** a warm-up effect — this image has a documented cold-start penalty (~58 → 83
tok/s), so a cold count number understates the warm one. Acceptance on prose-heavy traffic stays
modest (~25%), which is inherent to this vision variant, so **code and prose gained less than
count**. Patch 4 is a genuine correctness fix regardless of the exact per-workload attribution: the
draft's always-on shared expert was loading uninitialised, and now it isn't.

**`k` is 5 on this image — do not use `k=6`.** The DSpark drafter requires
`num_speculative_tokens` divisible by `n_predict=5`; `k=6` is rejected at boot with
`must be divisible by n_predict=5`. `k=5` is correct. (Same runtime property documented for the
text recipe: `k <= 5, or a multiple of 5`.)

---

## Updating to the official `DeepSeek-V4-Flash-0731` release (2026-07-31)

> Everything else in this README was measured on the **preview** checkpoint,
> `fraserprice/DeepSeek-V4-Flash-DSpark`, and still stands. This section is for
> people moving to DeepSeek's official `deepseek-ai/DeepSeek-V4-Flash-0731` release.

**If you swap in the 0731 weights and change nothing else, decode throughput roughly halves —
with no drop in output quality.** That combination is the tell, and it sends you looking in the
wrong place. It is not the weights and not a config regression: vLLM's DSpark draft weight loader
silently drops twelve tensors, and 0731 is far more sensitive to the loss than the preview was.

### What to change

## Patch 5 — stop strings must not fire inside the reasoning segment

If you serve this model with a harness that sends `stop` sequences (lm-evaluation-harness
sends `stop[:4]` on **every** request), you are almost certainly losing answers silently.

vLLM's v1 detokenizer matches client stop strings against the whole output stream. With
think-in-prompt templates, generation starts *inside* `<think>`, and chain-of-thought
naturally restates phrases like `Question:`. The stop fires mid-reasoning, `</think>`
never arrives, and the reasoning parser returns `content: null`. The request looks like a
model failure; it is a serving-layer one. Hosted deployments of the same checkpoint are
immune because they scope stops to content.

Apply **[Patch 5](patches/0005-suppress-stops-in-reasoning.patch)** — bind-mount, no rebuild:

```bash
-v /path/to/patched/detokenizer.py:/opt/env/lib/python3.12/site-packages/vllm/v1/engine/detokenizer.py:ro
```

> **Both nodes.** `start-deepseek-v4-flash-dspark.sh` syncs the compose and env files to the
> worker but **not** bind-mounted patch files. The file must exist at the same path on the
> worker too, or it silently runs unpatched and you get confusing half-fixed results.

Guard is per-request and needs no configuration: if the request's last prompt token is
`<think>`, stop strings stay dormant until `</think>` appears. EOS and `max_tokens` are
unaffected; non-thinking requests are untouched. Opt out with
`VLLM_SUPPRESS_STOPS_IN_REASONING=0`.

Measured on 2× DGX Spark (GB10), TP=2, k=5, `unsloth/DeepSeek-V4-Flash-0731`:

| metric | before | after |
|---|---|---|
| decapitation reproducer (seed + stop) | 43 tok, `content: null` | 344 tok, correct answer |
| stop honored on a non-thinking request | ✓ | ✓ (cuts at exact stop) |
| GSM8K n=50, temp 0.6 / top_p 0.95 | 8–15 nulls, 0.66–0.84 | **1 null, 0.98** |

The single residual null is mechanism (B) in [#18](../../issues/18) — a marginal-stability
reasoning runaway, unrelated to stop strings. Patch 5 does not address it.

## Patch 5 and issue #18 (B): the runaway becomes *more* visible, not less

Worth stating so nobody reads it as a regression. Mechanism (B) is a reasoning runaway in
which `</think>` never arrives. Because Patch 5 keeps stops dormant until the end marker
appears, a request in that state now has stops dormant for its whole life and runs to
`max_tokens` — where previously a client stop string could cut it short by accident.

That is correct by design: a stop string was never meant to bound reasoning, and a run
truncated by one was returning `content: null` anyway. But the practical effect is that
(B) shows up as a full-budget request after this patch instead of a short one, so a fleet
that applies Patch 5 may see *reported* token usage on those requests rise. The failure
rate does not change; only how long each failure takes to admit it.

If you are measuring (B), note that stop strings are no longer a confound in either
direction, which is the point.

Apply **[Patch 4](patches/0004-dspark-shared-expert-gate-up-proj.patch)** on top of your existing
setup. Two lines, no rebuild — bind-mount it read-only:

```bash
-v /path/to/patched/dspark.py:/opt/env/lib/python3.12/site-packages/vllm/v1/spec_decode/dspark.py:ro
```

Everything else in the recipe carries over unchanged: same runtime, same `k=5`, same
`nvfp4_ds_mla` KV, same 1M context, same Patch 3. Point `--model` at the 0731 directory and go.

### What it is

The draft's shared expert is a `DeepseekV4MLP` whose projections are `gate_up_proj` (fed by
checkpoint tensors `w1` and `w3`) and `down_proj` (fed by `w2`). The draft loader renames only
`w2`, and its stacked-parameter mapping carries only the two attention entries — so `w1`/`w3`
match nothing and hit `logger.debug("Skipping unknown DSpark weight")`, which is invisible at
INFO. Twelve tensors, across all three draft stages:

```
model.layers.{43,44,45}.ffn.shared_experts.gate_up_proj.{weight,weight_scale_inv}
```

`n_shared_experts: 1` and that expert is **always-on** — summed into every token unconditionally.
So each draft stage ran with its always-active expert uninitialised. Because the target model
still verifies every token, output stays correct; only acceptance collapses. The target's own
loader has the two rows the draft loader is missing
([`model.py:1952-1953`](DSPARK-SHARED-EXPERT-FIX.md)).

### Measured on 0731 (2× DGX Spark, TP=2, k=5, nvfp4 KV, 1M context)

| | accept | tok/step | steps/s | mean tok/s | peak tok/s |
|---|---|---|---|---|---|
| 0731, stock loader | 25.7% | 2.28 | 14.4 | 32.7 | 42.0 |
| **0731, Patch 4** | **60.2%** | **4.01** | 13.8 | **55.4** | **66.1** |

On the repo's long-standing peak-finder prompt shape (`"Count from 1 to 300, separated by
commas."`, temp 0, warm) the patched 0731 server reaches **78.4 tok/s at 98.9% acceptance —
5.95 accepted tokens per step out of a maximum 6.** That is what a correctly loaded DSpark
drafter looks like: it is accepting essentially every token it proposes. The same prompt
immediately after a cold start measures 56.8 tok/s, which is the documented cold-start penalty,
not a config difference.

Per-position acceptance `0.631/0.282/0.181/0.114/0.067` → `0.826/0.725/0.572/0.471/0.399`.
`steps/s` is unchanged — the entire deficit was draft acceptance. Pooled over ~35 min of real
agent traffic the patched server holds **56.1%** acceptance (mean accepted length 4.0–5.0 of 5).

By content, patched: structured/repetitive **78.3% / 66.1 tok/s**, code generation
**68.7% / 62.2**, prose reasoning **33.7% / 37.8**. Acceptance is content-driven, so a single
headline number without the content mix behind it is not meaningful.

**Scope of the bug.** The preview checkpoint carries the same twelve tensors and is therefore also
affected, but we have not measured preview with Patch 4 applied, so the size of its gain is
unknown. Note the preview measured 57.8% acceptance *unpatched* — close to where 0731 lands
*patched* — so whatever the mechanism, 0731 is hit much harder. That asymmetry is not yet explained.

Full write-up, evidence, and how to check whether you are affected:
**[`DSPARK-SHARED-EXPERT-FIX.md`](DSPARK-SHARED-EXPERT-FIX.md)**

### Two other things that changed with 0731

**Benchmark with `stream: false`.** Under speculative decoding vLLM emits at most one SSE chunk
per decode *step*, carrying every token accepted in that step. Counting streamed content deltas
therefore measures **steps/s, not tokens/s**, and under-reports by the acceptance length — 14.7
vs 60.1 tok/s on the identical request. Read `usage.completion_tokens`, or divide server-side
`vllm:generation_tokens_total` by wall time.

**`k` is still 5 on this runtime — and that is a property of the runtime, not the checkpoint.**
`dspark_block_size = 5` in the 0731 checkpoint exactly as in the preview, and DeepSeek's model
card recommends `num_speculative_tokens: 7`.

On **this recipe's image** (vLLM `0.21.1rc1.dev339+g1967a5627bc3` + B12X), k=7 does not work.
`SpeculativeConfig.hf_config_override` has a DSpark branch that sets `n_predict =
dspark_block_size = 5`, so the divisibility guard rejects it at boot. Patch that guard out and the
run crashes on the first generation with `The size of tensor a (7) must match the size of tensor b
(5)` — the draft model hardcodes its block width to `dspark_block_size`, so `propose()` returns 5
columns however large `k` is. `k=10` fails the same way. The accurate rule **on this image** is
`k <= 5`, or a multiple of 5.

On the **anemll 0.25.2 lineage** (`ghcr.io/anemll/dspark-vllm-gx10:0.1.1`, vLLM
`0.25.2.dev0+g752a3a504`) neither failure occurs. That build has no DSpark branch in
`hf_config_override`, so `n_predict` resolves to `num_nextn_predict_layers` = **1** and the guard
is inert; and its `DSparkSpeculator` sizes the draft block from `num_speculative_tokens` rather
than `dspark_block_size`, so a 7-wide draft is simply a wider single block. k=7 boots and
generates there — confirmed by @robotnurse in #22 with per-position acceptance for positions 0-6.
Corollary on that image: omit `num_speculative_tokens` and you get k=1, not k=5.

Deeper drafts are still not the missing speed on either runtime: positions 4-5 accept at
0.078/0.047 on hard content here, and #22 measured k=5 → k=7 buying +3.3% tokens/step for a 33%
wider verify batch.

### Ruled out by measurement, so you don't repeat it

| tried | result |
|---|---|
| `draft_sample_method` greedy vs probabilistic | no change — it is a **no-op** on the DSpark path (the proposer never populates draft probs unless `VLLM_DSPARK_EXPORT_DRAFT_PROBS=1`) |
| `fp8_ds_mla` vs `nvfp4_ds_mla` KV | no change to acceptance — pick for pool size |
| temperature 0 vs 0.7 | no change |
| B12X kernels off | worse (steps/s 14.4 → 10.7), then crashed |
| dedicated nodes, zero competing traffic | no change — contention was never involved |

If **drafted** throughput looks healthy while **accepted** throughput is low, that is an
acceptance problem: check Patch 4 before anything else.

---

## TL;DR

> **Full re-characterisation 2026-07-29.** Every number below was re-measured on this
> config, back to back, on a dedicated experiment lane — decode by content type,
> concurrency c1–c6, and prefill at depth. Headline: **84.3 tok/s peak, 67.6 mean,
> 197 tok/s aggregate at c6, 2639 tok/s prefill at 100K.** The newer vLLM 0.25.2 runtime
> was tested head-to-head on the same hardware and **lost** — see
> [`RUNTIME-BAKEOFF-2026-07-29.md`](RUNTIME-BAKEOFF-2026-07-29.md), which also contains the
> per-position acceptance data that explains why decode bursts and then drops.

> **Defaults fixed 2026-07-29 — re-pull if you cloned before this.** The shipped
> `.env`/compose defaults were still handing out `MTP_NUM_TOKENS=3`, a leftover from the
> 2026-07-03 greedy-draft garble fix, even though Patch 3 made `k=5` garble-safe and every
> doc here specifies 5. **`k=3` costs ~24% decode.** If you benchmarked this recipe with the
> old defaults you measured the slow path — set `MTP_NUM_TOKENS=5` (now the default) and
> re-run.


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
- `MTP_NUM_TOKENS=5` (with `draft_sample_method=probabilistic`; see the [garble fix](#garble-fix-2026-07-03))
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

> **⚠️ vLLM version note (runtime fixes ship in the image):** all DSpark source
> patches — including the concurrency-crash fix and the local scheduler-queue
> lifetime fix from issue #26 — are baked into the runtime image
> from `recipe/overlay/` at build time; there is **no runtime bind-mount** of
> `dspark_proposer.py`. (A mounted proposer from one vLLM version crashes another
> with `propose() got an unexpected keyword argument ...` — this took down a rig
> on 2026-07-08 after a same-tag image rebuild.) `start-deepseek-v4-flash-dspark.sh`
> now verifies the image against `recipe/overlay/` and rebuilds automatically when
> stale (skip with `SKIP_OVERLAY_CHECK=1`). If you run a **different prebuilt vLLM
> image** that still needs the non-uniform-batch guard, patch that image's own
> proposer instead:
> ```bash
> docker create --name t <your-image>
> docker cp t:/opt/env/lib/python3.12/site-packages/vllm/v1/spec_decode/dspark_proposer.py ./myproposer.py
> docker rm t
> python3 scripts/apply-nonuniform-guard.py ./myproposer.py   # version-independent
> # then bind-mount ./myproposer.py at that same container path (compose override)
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

## Reasoning / thinking mode

Reasoning is **off by default** in this recipe (`--default-chat-template-kwargs '{"thinking":false}'`).
Everything below was measured on this stack; several of these have cost people real time.

**The response field is `reasoning`, not `reasoning_content`.**
Non-streaming: `choices[0].message.reasoning`. Streaming: `choices[0].delta.reasoning`.
There is **no** `reasoning_content` key in a response on this runtime — it is deprecated and only
accepted on *input*. Clients reading `reasoning_content` see empty and conclude reasoning
extraction is broken. Credit @vinicius-symetrix (PR #13) for independently reporting the
streaming half of this.

**`<think>` is written into the prompt, not generated.**

```
thinking off →  ...<｜Assistant｜></think>
thinking on  →  ...<｜Assistant｜><think>
```

So in thinking mode the model's output *starts* with reasoning text and ends with `</think>`;
there is no opening tag in the completion. **A missing `<think>` in the output is correct
behaviour.** If you see `</think>` inside `content`, your server is missing
`--reasoning-parser deepseek_v4` and `--reasoning-config`.

**Enabling it.** Per request: `chat_template_kwargs: {"thinking": true}`, or a top-level
`reasoning_effort` of `low`/`high`/`max`. Server-wide:
`--default-chat-template-kwargs '{"thinking":true}'`.

**Never send `reasoning_effort: "none"` together with `thinking: true`.** `"none"` forces
chat-mode formatting while the reasoning parser stays armed for thinking; with no `</think>`
in the output the parser puts the *entire* response into `reasoning` and returns
`content: null`. Measured 4/4 — real `completion_tokens`, `finish_reason: stop`, empty content.
`reasoning_effort` on its own is fine.

**`reasoning_effort: "low"`, `"medium"` and `"high"` all produce *no* effort prefix.** They
normalise to an internal `"high"` that has no injection branch, so nothing is added to the
prompt. Only `"max"`/`"xhigh"` inject — and what they inject is the model's **high** text
(79 tokens), not its `max` text (96). **The model's `max` effort is unreachable on this
tokenizer mode**, and anyone selecting `"high"` here is running at the vendor's *low*.

An earlier revision of this section said `"low"` behaves as `"high"`. That was true of the
runtime's internal variable and **inverted as advice** — corrected after @Capicua25x measured
it (issue #25). Verify in three requests; `prompt_tokens` is the whole witness:

```bash
for E in low high max; do
  printf '%-5s ' "$E"
  curl -s http://127.0.0.1:8888/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"deepseek-v4-flash-dspark\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],
         \"max_tokens\":1,\"reasoning_effort\":\"$E\"}" \
  | python3 -c 'import json,sys; print("prompt_tokens =", json.load(sys.stdin)["usage"]["prompt_tokens"])'
done
# this stack:  low 5 | high 5 (identical — no prefix) | max 84 (+79)
# api.deepseek.com, same messages: low 681 | high 760 (+79) | max 773
```

PR #24 vendors the checkpoint's own three-level table and restores the distinction. **Note the
behaviour change when it lands:** `reasoning_effort:"max"` will then inject DeepSeek's real max
text instead of its high text, so existing callers of `"max"` get a materially stronger
instruction. `"low"` and omitted stay byte-identical.

**With `--tokenizer-mode deepseek_v4`, a `chat_template.jinja` in the model directory is
ignored** — prompt formatting comes from the checkpoint's built-in encoder, not a Jinja
template. This is why the HuggingFace discussion #26 workaround has no effect here. It is
stronger than that: **explicitly passing `--chat-template /path/to/chat_template.jinja` is also
ignored** — it shows up in the engine's `non-default args` and changes nothing (measured in
#25). There is no way to reach the template's three-way effort split without leaving
`tokenizer_mode=deepseek_v4`, which this recipe needs for DSpark.

**Thinking mode needs output budget.** Reasoning consumes `max_tokens` before any content is
produced, so a small cap yields `finish_reason: length` with empty `content`. If you benchmark
thinking mode at a low cap you will measure truncation, not the model.

## Benchmarks

### 2026-07-29 full re-characterisation (authoritative — start here)

Measured on the shipped default config (`MTP_NUM_TOKENS=5`,
`draft_sample_method: probabilistic`, `nvfp4_ds_mla`, `max_model_len=1048576`,
`max_num_seqs=6`, `gpu_memory_utilization=0.78`, TP=2) on a dedicated experiment lane, so
no production traffic contaminated the numbers. Harness:
[`benchmarks/bench_full.py`](benchmarks/bench_full.py).

**Decode by content type** — temp 0, warm, best-of-2, server-reported
`completion_tokens` over wall time:

| prompt | tokens | sec | tok/s |
| --- | ---: | ---: | ---: |
| count to 300 | 600 | 7.12 | **84.3** |
| 12x12 multiplication table | 900 | 11.56 | 77.9 |
| 60-object JSON array | 800 | 10.39 | 77.0 |
| binary search tree (code) | 600 | 9.34 | 64.2 |
| 200-word narrative | 251 | 7.25 | 34.6 |
| **peak / mean** | | | **84.3 / 67.6** |

**Concurrency** — same prompt, 400 tokens each, aggregate and per-stream:

| concurrency | aggregate tok/s | per-stream tok/s |
| ---: | ---: | ---: |
| 1 | 61.0 | 61.0 |
| 2 | 91.7 | 46.9 |
| 4 | 151.1 | 38.7 |
| 6 | **197.3** | 33.6 |

**Prefill** — TTFT method, 1 output token:

| prompt depth | prompt tokens | sec | tok/s |
| ---: | ---: | ---: | ---: |
| 8K | 6,234 | 4.1 | 1,513 |
| 32K | 24,900 | 10.9 | 2,284 |
| 100K | 77,790 | 29.5 | **2,639** |

KV pool at `gpu_memory_utilization=0.78`: **1,548,597 tokens** on this boot. Treat pool size
as a per-boot figure, not a fixed property — two boots of an identical config measured
1,385,765 and 1,533,940, an 11% swing, because available KV memory on GB10 varies with what
else has touched unified memory. Both are far above the 1M calibrated ceiling, which is what
matters.

**The 34 → 84 tok/s spread is one server on one config.** Decode speed here is
`steps/s x accepted-tokens-per-step`, and DSpark acceptance is content-driven, so any
single-prompt "tok/s" claim for this model — including ours — is a statement about the
prompt as much as the hardware. Quote the mean for planning and the peak for bragging.

### What an agent fleet actually gets (realistic mixed traffic)

Every number above is a benchmark: one prompt shape, temp 0, best-of-2. Useful for comparing
configurations, misleading for capacity planning. So here is the same server under traffic that
looks like actual agent work — tool calls, code refactors, structured JSON, multi-step
reasoning, explanatory prose and long-context summarisation, at mixed temperatures (0.0/0.3/0.7)
and mixed token budgets (300/500/800), 4 concurrent workers, via
[`benchmarks/soak.py`](benchmarks/soak.py):

| | benchmark (BST prompt, temp 0) | realistic mixed traffic |
| --- | ---: | ---: |
| per-stream tok/s @ c4 | 38.7 | **22.3** |
| aggregate tok/s @ c4 | 151.1 | **88.6** |

**Plan with ~88 aggregate / ~22 per-stream at c4, not 151.** The difference is not a
regression and not a config problem — it is DSpark acceptance responding to harder, more varied
content, exactly as the per-position acceptance data predicts. The benchmark number is real;
it just describes a prompt you will rarely send.

Full run: **553 requests / 212,974 generated tokens over 40 minutes at concurrency 4** —
**88.6 tok/s aggregate**, and **zero** soft-empty completions, zero degenerate outputs
(repetition loops, CJK drift, template/XML leakage) and zero errors. Per-stream throughput held
at 22.1-22.3 tok/s across all eight 5-minute windows, and neither container restarted.

That is a genuine stability result for the shipped config, but read it carefully as evidence for
[issue #6](https://github.com/tonyd2wild/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark/issues/6):
it is a **negative** result, and without knowing the reporters' daily request volume it cannot be
converted into "we would have expected N events." 553 clean requests is much better than the
5-prompt gate it replaces; it is still not proof of absence.

### Warm up before you benchmark — the first requests run ~30% slow

Found while verifying a rollback, and it invalidates a lot of casual benchmarking of this
recipe (including one of my own measurements below, which is why it is documented here rather
than quietly fixed).

Immediately after `Application startup complete`, with CUDA graphs already captured and 3
short warm-up calls sent, the `count300` prompt measured:

```
58.5 tok/s
```

The same prompt on the same container, after ~5 long generations had passed through it:

```
run 1   83.3        run 3   83.1
run 2   83.2        run 4   83.2
```

**58.5 → 83.3 tok/s. A 30% penalty that disappears after a few hundred tokens of real
traffic, and it is not visible in the boot log** — the server reports itself ready, graphs are
captured, and it answers correctly the whole time. It is simply slow for the first few
requests.

Practical consequences:

- **Send real traffic before you trust a number.** A handful of 100-token warm-up calls is
  *not* enough; it took gate-sized (500-700 token) generations to reach steady state. The
  harness in [`benchmarks/bench_full.py`](benchmarks/bench_full.py) warms with 4 short calls
  and takes best-of-2 per prompt, which is adequate on an engine that has been serving but
  **not** adequate on a freshly booted one.
- **This probably explains a chunk of the spread in reported numbers for this model.** If you
  benchmarked right after boot you measured the cold path.
- Steady state is genuinely stable once reached: four consecutive runs within 0.2 tok/s.

**It is not only a boot effect — the warm state decays when the server goes idle.** Confirmed on
the same container with no restart: after ~30 minutes idle following a 40-minute soak, `count300`
measured **60.4 tok/s**; heavy warm-up restored it to **83.5** minutes later. So the penalty will
hit an agent fleet after any quiet period, not just after a deploy. If you care about the first
response after idle, keep a trickle of traffic going — and never benchmark straight after a lull.

### Which prompt shapes actually reach the ceiling

The headline 84.3 tok/s comes from `Print the numbers 1 to 300`, which is a toy. The useful
question is whether any *real* output shape gets there. Measured on a warm engine (warm check
`count300 = 83.5` immediately before the run, so these are not warm-up artefacts):

| shape | tok/s | tokens | what it is |
| --- | ---: | ---: | --- |
| count to 300 | **83.1** | 600 | the toy baseline |
| **bulk SQL INSERTs** | **77.7** | 1200 | 60 rows, fixed template |
| 20 identical dataclasses | 75.5 | 814 | DTO/boilerplate generation |
| `.env` with 60 entries | 74.8 | 780 | config generation |
| CRUD endpoints, 8 resources | 72.2 | 1400 | FastAPI skeleton, 4 routes each |
| CSV → JSON | 72.0 | 305 | transformation |
| JSON fixtures, 60 objects | 69.7 | 1200 | test data |
| CSV → markdown table | 65.4 | 199 | transformation |
| add type hints to 5 functions | 59.1 | 134 | small edit |
| original prose | 31.1 | 292 | control |

**Nothing real reaches 84.** The practical ceiling for useful work is **~78**, and this is the
prompt that got there:

```
Generate 60 SQL INSERT statements for table users(id, email, created_at).
Use the exact form: INSERT INTO users (id, email, created_at) VALUES (N,
'user_N@example.com', '2026-01-01'); — ids 1 to 60, one per line. SQL only.
```

Two levers, and the second is the one people miss.

**1. Rigid repeated structure.** Give the template and say "the exact form". The draft model
locks onto the pattern after the second row and then predicts all `k=5` tokens ahead. Bulk
INSERTs, DTOs, config files and fixtures all land in the mid-to-high 70s.

**2. Ask for long output — short requests are mathematically capped.** `add-types` scored 59.1,
but it only produced **134 tokens**. Fixed per-request overhead is roughly 0.5 s, so:

```
134 tokens / 80 tok/s  +  0.5 s overhead  =  2.18 s   →  61 tok/s apparent
measured: 2.27 s                                      →  59.1 tok/s
```

That number is overhead amortisation, **not** poor acceptance. A 134-token request cannot post a
high tok/s no matter how predictable it is. The 1200-1400-token runs are where the real ceiling
shows up — so if you are benchmarking, use long generations, and if you are quoting a number,
say how many tokens it was.

Harness: [`benchmarks/realwork_peak.py`](benchmarks/realwork_peak.py).

**Keep this in proportion.** 78 tok/s is the good end of the distribution. Mixed agent traffic
averages **~22 tok/s per-stream / ~88 aggregate at c4** (see above) — that is the number to plan
capacity with. The 78 is what you get when the work happens to be templated bulk generation.

### Is there a faster runtime? (tested, no)

vLLM **0.25.2** (`ghcr.io/anemll/dspark-vllm-gx10:0.1.1`, torch 2.11/cu13) was booted on the
same two nodes with identical args and benchmarked identically. It lost on the axes that
matter: **9% down on peak decode, 8% on mean, 29% down at c6.** It won c2 concurrency
(+11%) and prefill at depth (+9% at 32K). Full data, plus the per-position acceptance
measurement that explains burst-then-drop, in
[`RUNTIME-BAKEOFF-2026-07-29.md`](RUNTIME-BAKEOFF-2026-07-29.md).

Verified warm on both sides: our stack has a ~30% cold-start penalty (see above), so the
0.25.2 figures were re-measured after heavy warming to be sure the comparison was fair —
76.7 warm vs 77.2 cold. **0.25.2 has no meaningful cold-start penalty**; it front-loads
autotune and warmup at boot, which the older runtime defers to first traffic. The gap is real.

Short version of why: both runtimes **saturate the draft** (0.25.2 logs 100% acceptance on
the fast prompt and still loses), so the gap is pure **step time**. Two things set it, and
this repo has both: the **B12X MoE kernels** (0.25.2 falls back to stock `DEEPGEMM_MXFP4`)
and a **working torch.compile path** (`torch.compile took 4.39 s` from the AOT cache — on
0.25.2 this model is simply unsupported by compile, in either direction). Both live on the
older vLLM, which is why this recipe stays there.

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
- `--max-cudagraph-capture-size 36` (derived: `max-num-seqs × (num_speculative_tokens + 1)` = 6 × 6)
- `--gpu-memory-utilization 0.80`
- `--enable-prefix-caching`
- `--async-scheduling`
- `--enable-chunked-prefill`
- `--generation-config vllm` (no `--override-generation-config`)
- `--speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}'`
  - **Use `num_speculative_tokens: 5` on this 1M / `max-num-seqs 6` profile.** It is worth roughly **+24%**
    over `k=3`. See [`DEFAULT-CONFIG.md`](DEFAULT-CONFIG.md) and
    [`SPEED-UPDATE-2026-07-29.md`](SPEED-UPDATE-2026-07-29.md) (83.4 tok/s peak, 74.1 mean).
  - Set `--max-cudagraph-capture-size` explicitly to `max-num-seqs × (k + 1)` = **36**. The shipped
    `docker-compose.dspark.yml` derives this for you.
  - Keep `draft_sample_method:probabilistic` — it beats greedy for DSpark's calibrated draft heads.

  > ⚠️ **Corrected 2026-08-05.** This section previously stated that the 1M / seqs-6 profile **MUST** use
  > `num_speculative_tokens: 3`, and that `k=5` would fail engine init with `No valid cudagraph sizes`
  > (verified 2026-07-04). **That is no longer true and the reasoning was incomplete.** The original claim
  > assumed `--max-cudagraph-capture-size` was pinned to `max-num-seqs` (6), leaving vLLM's `[1,2,4]` ladder
  > with nothing divisible by `k+1 = 6`. The fix is not to lower `k` — it is to set the capture size to
  > `seqs × (k+1) = 36`, which is what the compose file now does. `k=3` still boots, it is just the slow
  > path. Anyone who benchmarked this repo from the README before this date was measuring `k=3`; re-run
  > with `k=5` before publishing numbers anywhere.

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
`VLLM_USE_B12X_WO_PROJECTION=0`. The default `MTP_NUM_TOKENS=5` with
`draft_sample_method=probabilistic` (2026-07-03 garble fix) applies here too;
older runs used greedy-draft MTP5, which upstream Mia and Keys had validated but
which caused the cold-start concurrent garble in agent serving.

### Tuning to combine context + concurrency

- To combine DSpark concurrency with longer context, pick a lower context target
  first, then raise concurrency slowly while watching boot logs, KV allocation,
  acceptance, and request errors.
- The current validated agent-serving profile is `MAX_MODEL_LEN=1048576`,
  `MAX_NUM_SEQS=12`, `GPU_MEMORY_UTILIZATION=0.85`, `MTP_NUM_TOKENS=5` with
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

 ### Container will not start / model never loads — HF cache ownership

**Symptom.** The container starts and then dies (or hangs) before loading weights; no obvious
error about the model path itself.

**Cause.** The container runs as **uid 1000**. If the host HF cache directory is owned by root
(common if a download was run under `sudo`, or the dir was created by a root-run container),
the container cannot write its cache/lock files.

**Fix** (credit: [@AndreasKunar](https://github.com/AndreasKunar), issue #9 — this fixed startup
for him with no recipe edits at all):

```bash
sudo chown -R 1000:1000 /path/to/your/hf-cache
```

Check ownership before blaming the recipe:

```bash
ls -ld "${HF_CACHE:-$HOME/.cache/huggingface}"
```

### Empty `content` with real `completion_tokens` — classify before you report

A bare "null-content rate" now aggregates at least five unrelated causes, and they want
different fixes. Two fields settle which one you have. Do this before opening an issue or
quoting a rate:

| `finish_reason` | `</think>` in the raw output | what it is |
| --- | --- | --- |
| `stop` | no | a **client stop string fired inside reasoning** — the CoT restated it, generation was decapitated before `</think>`. lm-eval sends `stop[:4]` on every request. Fix: PR #21's reasoning-aware stop guard, or `until: []` client-side. |
| `length` | no | **budget exceeded**, not a hang. Reasoning is heavy-tailed even on trivial prompts (48–440 tokens measured on `"What's 1 + 1?"`). Raise `max_tokens` and re-measure; if the rate moves with the budget it was never a non-termination. |
| `length` | no, *and* the rate does not move with budget | **genuine non-termination** — a repetition loop. Detector that works: 3 consecutive 4,000-char windows below 2% novel word-8-grams. Block-level uniqueness reads *high* on plainly looping text; do not use it. Tracked in issue #18 (B). Sampling at the checkpoint's specified `temperature 1.0` measured 18/18 terminating vs 14/36 at 0.6. |
| `stop` | n/a | you sent **`reasoning_effort:"none"` with `thinking:true`** — chat-mode prompt, thinking-armed parser. See above in *Reasoning / thinking mode*. |
| `stop` | yes, and the answer is missing at the client only | the client is reading **`reasoning_content`**; the response key is **`reasoning`**. |

A sixth, rarer one: the model occasionally emits a pseudo-tag scaffold
(`<STORE_AND_RETURN> 570 </STORE_AND_RETURN>`, `<STDERR> final</STDERR>630`) as its *entire*
output and never closes `</think>`, so the parser files everything as reasoning. Measured
5/60 → 0/60 with a marker-specific fallback, ~0.8% residual on a different tag; tag-matching is
whack-a-mole, so this is documented rather than patched. Non-streaming only in the reported
measurements. Credit @robotnurse (issue #6).

Classifying costs nothing and it is what made issue #18 tractable.

### Sharing the HF cache over NFS between the nodes — seven JIT caches, three failures

**Symptom.** Any of, usually in this order as you fix each one:

* `torch.compile` dies at startup with a `FileExistsError` / `makedirs` race
* DeepGEMM asserts `runtime != nullptr` — stale or half-written cubins read over NFS
* the head's first engine attempt dies every boot on an **ABI-mismatched FlashInfer
  `sampling.so`**, compiled by one node and silently loaded by the other. Silent because this
  stack runs `FLASHINFER_DISABLE_VERSION_CHECK=1`. With the worker entrypoint not retrying
  after a process death, this also orphans the worker on each boot.

None of those messages mention the cache. They read like broken kernels or a broken build.

**Cause.** Downloading the 167 GB checkpoint once and serving it to both nodes over NFS is the
obvious move — but seven JIT/workspace caches default to (or historically sat under) the same
tree, and then **both ranks JIT into the same directories concurrently**.

**Fix.** `docker-compose.dspark.yml` now mounts a **separate, node-local** volume at
`/vllm-cache` and points all seven at it, independent of where `HF_CACHE` lives:

| variable | value |
| --- | --- |
| `VLLM_CACHE_ROOT` | `/vllm-cache` |
| `DG_JIT_CACHE_DIR` | `/vllm-cache/deepgemm-cache` |
| `FLASHINFER_WORKSPACE_BASE` | `/vllm-cache/flashinfer` |
| `TILELANG_CACHE_DIR` | `/vllm-cache/tilelang` |
| `TORCHINDUCTOR_CACHE_DIR` | `/vllm-cache/torchinductor-cache` |
| `TRITON_CACHE_DIR` | `/vllm-cache/triton-cache` |
| `TORCH_EXTENSIONS_DIR` | `/vllm-cache/torch_extensions` |

The host path is `JIT_CACHE_DIR` (default `${HOME}/.cache/vllm-dspark`) — **keep it on local
disk on every node**. If you already ran with a shared cache, purge the poisoned directories
once; a stale `sampling.so` survives the config change.

Credit [@antoniohlc](https://github.com/antoniohlc), issue #27, who found the whole set one
crash at a time and reproduced this repo's numbers (69.8–73.8 tok/s warm single-stream) once it
was fixed. The **model weights** in `HF_CACHE` are fine on NFS; it is only the JIT tree that
must be per-node.

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
| [`CURRENT.md`](CURRENT.md) | **start here** — the recipe we actually run today, one section per topology (TP2, TP4) |
| [`launchers/`](launchers/) | the two runnable launchers: `ds4-vision-tp2.sh <0\|1>` and `ds4-vision-tp4.sh <0\|1\|2\|3>`. One launcher per recipe, canonical path. `vision-exp/ds4-vision-tp2.sh` is a symlink here (older PRs/issues cite it) |
| [`vision-exp/`](vision-exp/README.md) | the vision port itself — the twelve blockers, `port/*.py`, `build-ds4v-files.sh` (stages the four bind-mounted files on every node) |
| [`sparkrun/`](sparkrun/README.md) | self-contained sparkrun recipes for the Vision-Exp and text 0731 checkpoints |
| [`parity/`](parity/) | reproducible serving-fidelity bench and the frozen hosted reference card |
| `VISION-EXP-DEFAULT-CONFIG.md` | long-form explanation of the TP2 launcher's flags, mounts, benchmarks |
| `DSPARK-SHARED-EXPERT-FIX.md` | Patch 4 write-up (incl. the vision-port dropped-mount incident) |
| `scripts/check-patch4.sh` | fail-closed preflight that Patch 4 (`spec-dspark.py`) is mounted on every node |
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
