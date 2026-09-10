# Patch 1, Patch 2 & Patch 3 — detailed reference

Patches 1/2/2b live in the DSpark vLLM overlay and together make `--max-num-seqs > 1`
**correct** under vLLM-v1 continuous batching. Single-stream and uniform-static
batches keep the original code path (byte-identical). **Patch 3 fixes the cold-start
garble on long resumed conversations** (root cause — see below; credit roady001, issue #3).

Files changed:

| file | + | − | role |
|---|---:|---:|---|
| `vllm/v1/spec_decode/dspark_proposer.py` | 158 | 10 | draft loop, slot map, ragged context (Patch 1+2+2b) |
| `vllm/models/deepseek_v4/nvidia/dspark.py` | 110 | 12 | persistent KV store (`store_main_kv`), `prefill_main` |
| `vllm/v1/worker/gpu_model_runner.py` | 10 | 0 | thread `req_ids` into `propose()` |
| `vllm/v1/core/sched/scheduler.py` | 4 | 0 | guard spec-placeholder resize (Patch 3) |

---

## Patch 1 — request-stable KV slot

### Symptom
At `max_num_seqs>1`, draft acceptance collapsed toward 0 (garbage drafts), even
though nothing crashed — the engine silently degraded to single-stream quality.

### Root cause
DSpark's draft keeps one persistent cross-step tensor per attention module —
`DeepSeekV4DSparkAttention.main_kv_cache`, shape `[max_num_seqs, window, head_dim]`
— a per-row **ring buffer** holding each sequence's sliding-window KV history. It
was read/written by **batch-row position** (`main_kv_cache[:batch_size]`). The
draft proposer carried **no request identity**.

Under vLLM-v1 continuous batching the running set is *condensed* whenever a request
finishes (a later request is moved into the freed row). The model's persistent
`main_kv_cache` row is **not** moved with it, so after a condense a request reads a
ring buffer that belongs to a **different** request → corrupted draft context →
acceptance collapse. (Single-stream never condenses row 0, which is why it worked.)

### Fix
Key the persistent cache by a **stable per-request slot** instead of batch row:

- `dspark_proposer.py`: add `self._req_id_to_slot: dict[str,int]` and
  `self._free_slots`. `_row_to_slot(req_ids)` reclaims slots of finished requests,
  assigns a free slot (lowest-first) to new ones, and returns the slot per row in
  `req_ids` order. A persistent, cudagraph-captured `_draft_slot_index_buffer`
  carries the slots into the graphed draft read path.
- `dspark.py`: `store_main_kv` and `forward_dspark` index the cache by
  `slot_index` (gather `index_select` on read, scatter `index_copy_` on write)
  instead of `[:batch_size]`.
- `gpu_model_runner.py`: pass `req_ids=self.input_batch.req_ids` into `propose()`
  (only for the DSpark proposer).

### Why it's safe
The math is unchanged — it only re-routes which physical row a request uses. When
the computed permutation is identity (a genuine single-request-at-a-time server
always gets slot 0), the code takes the **original in-place write path,
byte-for-byte**. Gating is on the *permutation identity*, not on `batch==1`, so the
"batch condenses to one surviving request holding a non-zero slot" case stays
correct.

---

## Patch 2 — ragged context path

### Symptom
Under real (independent / staggered) arrivals at `max_num_seqs>1`, the server
returned HTTP 500:

```
ValueError: DSpark currently requires uniform flattened per-request inputs;
got 41 rows for batch_size=2.   (dspark_proposer.py: _view_by_request)
```

### Root cause
`prepare_context` reshaped the flat target hidden states into a **rectangular**
`[batch, seq, H]` via `_view_by_request` / `_positions_by_request`, asserting every
request contributed the **same** number of rows. With chunked prefill (required —
disabling it needs `max_num_batched_tokens >= max_model_len`, infeasible at long
context) a single step **mixes prefill and decode** rows, so per-request row counts
differ (e.g. "41 rows for batch_size=2" = one request prefilling alongside one
decoding). Rectangular reshape is impossible → crash. The static benchmark passed
only because all prompts were identical length (uniform).

### Fix
Make the context path **ragged** using `query_start_loc` (per-request segment
offsets) — the same mechanism `_trim_rejected_target_context` already used:

- `dspark_proposer.py` `prepare_context`: detect non-uniform segment lengths
  (`ragged = len(set(seg_lengths)) != 1`). In the ragged branch, skip the
  rectangular view; compute each request's draft anchor with a flat index
  `anchor_idx = starts + clamp(len - rejected - 1, 0, len-1)` and
  `index_select` the per-request last hidden/positions. Pass the flat hidden +
  `query_start_loc` + `slot_index` to `prefill_main`.
- `dspark.py`: `store_main_kv(..., query_start_loc=...)` dispatches to a new
  `_store_main_kv_ragged` that loops requests via `query_start_loc`, truncates each
  segment to the last `window_size` rows, computes `slots = positions % window`,
  applies the rejected-suffix mask, and `index_copy_`s into that request's slot.
  `prefill_main` threads `query_start_loc` through and skips the rectangular view in
  ragged mode.

### Why it's safe
Storage is **position-addressed** (`positions % window`), so it never needed
uniform lengths — only the intermediate rectangular view did. When lengths are
uniform (`query_start_loc is None` / static / single-stream) the original
rectangular fast-path runs unchanged. Ragged/mixed steps run **eager** (mixed steps
are never cudagraph-captured), so dynamic Python loops / variable shapes are safe;
the uniform decode-only graphed path is untouched.

### Scope
Only the `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1` path was made ragged (the path
used in serving). The legacy `_trim_rejected_target_context` path still assumes
uniform. **Run with `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`.**

---

## Patch 2b — ragged detection independent of rejection

### Symptom (found by the GSM8K quality eval)
After Patch 2, a prefill-heavy step with **no rejection** still 500'd:
`ValueError: ... got 166 rows for batch_size=3` at `_view_by_request`. Earlier
staggered tests (uniform-ish prompts) missed it; GSM8K's varied prompt lengths hit
it.

### Root cause
Patch 2 computed `ragged` **only inside** `if gpu_mask and num_rejected_tokens_gpu
is not None`. On steps with no rejection (`num_rejected=None`, e.g. fresh requests
prefilling), detection was skipped and the code fell through to the rectangular
`_view_by_request` → crash. Raggedness depends on `query_start_loc` segment lengths,
**not** on rejection.

### Fix
- Enter the detection/ragged branch whenever `_gpu_rejected_context_mask` is on,
  **regardless of `num_rejected_tokens_gpu`** (which may be `None`).
- In the ragged anchor, default `rejected` to zeros when `num_rejected_tokens_gpu is
  None`. `_store_main_kv_ragged` already handled `None` (no masking).

### Validation
GSM8K N=8 (200 Q) — the load that crashed pre-fix — now completes with **0 errors**,
93.5% accuracy vs 95.0% sequential, **97.5% per-question agreement** (quality-neutral
within batch FP-nondeterminism).

---

## Patch 3 — no spec placeholders on prefill chunks (cold-start garble fix)

**Credit: roady001 (issue #3).** Confirmed to fix the garble on its own, with none
of the launch/config changes applied — i.e. this is the actual root cause, not a
symptom reducer.

### Symptom
Resuming an existing long conversation **cold** (server restarted, or the prompt
fell out of the prefix cache) produced garbage at the **start** of the reply —
prompt echo, leaked tool/schema text, "your message was cut off"-style replies —
then generation recovered. Warm continuations of the same conversation were
clean. Reproduced deterministically with a ~98K-token prompt at `temperature 0`:
cold run echoed prompt content from ~=`prompt_len - 8192` (the last chunk
boundary) and looped; the immediately repeated (warm) run answered correctly.

### Root cause
The DSpark async-scheduling addition in `Scheduler.update_from_output` resizes
the `[-1]` spec-token placeholder list to DSpark's confidence-scheduled draft
length. It ran for **every running, non-stopped request in the batch — including
requests still mid-way through a chunked prefill**. Upstream never installs spec
placeholders on prefill chunks: `AsyncScheduler._update_after_schedule` skips
`request.is_prefill_chunk`, and `Scheduler.update_draft_token_ids` explicitly
clears drafts for prefill chunks.

With a long cold prompt (multiple `max_num_batched_tokens=8192` chunks), the
illegal placeholders made the scheduler attach `num_speculative_tokens` spec
tokens to the request's **final prompt chunk**. Two corruptions follow:

1. The drafts verified there were proposed from the **truncated** prompt
   (mid-prefill DSpark drafts — prompt-continuation predictions).
2. The request was in the previous step's **discard set** (mid-prefill), so the
   worker excludes it from `prev_req_id_to_index`; `_prepare_input_ids` then
   never scatters real draft ids over the `-1` placeholders, and the target
   forward for the final chunk sees invalid token ids at the spec positions.

Either way the first sampled tokens of the reply are conditioned on a corrupted
prompt tail — visible as prompt echo / garble at the start of a cold resume,
recovering once pure decode steps take over. (Earlier this corrupted whole
conversations; Patches 1/2/2b fixed the persistent-KV side, leaving only the
cold-start window.)

### Fix
`vllm/v1/core/sched/scheduler.py` (`update_from_output`): only resize spec
placeholders for requests the AsyncScheduler itself would give placeholders —
`new_token_ids` non-empty, `not request.is_prefill_chunk`, and
`request.status == RequestStatus.RUNNING` (a preempted request must keep its
cleared spec list).

### Validation
Same ~98K-token cold-resume prompt at `temperature 0` after the fix: the cold
first reply is correct (no echo, no cut-off complaint), and matches the warm
rerun. Short direct prompts and the deterministic `NVFP4 DSPARK OK` check are
unchanged.

### Notes (independently confirmed)
- `repetition_penalty=1.05` is a separate DSpark crash risk; drop it first if an
  illegal-memory / IMA crash appears.
- `draft_sample_method=probabilistic` is a valid tuning option but is **not**
  needed to fix this garble once the scheduler guard is in place.

---

## Patch 6 — preserve the local scheduler queue across long model loads

### Symptom

On the two-node SparkRun profile, replacing the stock checkpoint with the
documented 155 GiB abliterated drop-in could load all 48 shards and then fail
before opening the API:

```text
AttributeError: 'ShmRingBuffer' object has no attribute 'shared_memory'
```

The later TCPStore and NCCL broken-pipe messages on the peer rank were secondary.

### Root cause

The executor creates its scheduler-output `MessageQueue` before spawning
`WorkerProc`, but the worker normally opens the local reader only after
`init_device()` and `load_model()`. During that multi-minute gap, another process
lifecycle can unlink the queue's POSIX SHM name. The existing mapping remains
valid in the creator, but a late reader can no longer open the name.

`ShmRingBuffer` also suppressed that `FileNotFoundError`, assuming the object had
been deserialized on another node. That left a half-constructed object and hid
the useful failure until its first dequeue.

### Fix

- `WorkerProc.__init__` pre-opens only readers whose rank is local, before worker
  initialization or model loading.
- `MessageQueue` caches that live mapping by `(rank, buffer_name)` and the normal
  post-initialization `create_from_handle()` call consumes and reuses it.
- Remote readers are unchanged; they still initialize through the distributed
  process group after `init_device()`.
- A missing local segment now raises immediately with its SHM name and reader
  rank.
- Both vLLM files are copied from `recipe/overlay/` into the base overlay image,
  so the fix reaches every Stage-C node without a head-only bind mount.

### Validation

The overlay image build includes a CPU-only lifetime regression: attach the local
reader, unlink the POSIX name, verify the normal late-open path reuses the live
mapping, then verify a second uncached open fails with the name/rank diagnostic.

Before upstreaming, issue #26's reporter validated the same patch on the affected
two-node deployment: correct 1M model metadata, a real completion, six concurrent
completions, and no CUDA, NCCL, EngineDead, or request errors. That live system was
not reused for PR testing; maintainers should retain their requested two-node
reproduction gate before merge.
