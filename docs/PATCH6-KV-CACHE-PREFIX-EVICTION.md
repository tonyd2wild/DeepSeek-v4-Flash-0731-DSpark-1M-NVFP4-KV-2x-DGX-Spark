# Patch 6 — stop long prefills from evicting every other cached prefix (DSV4 hybrid KV)

Target: `vllm/v1/core/single_type_kv_cache_manager.py` in the DSpark runtime
(`vllm-dspark-runtime:mia-raf-pr1-nvfp4-probe-c-keys-concurrency-p2b`,
vLLM `0.21.1rc1.dev339+g1967a5627bc3`). Unified diff: `patch6-single_type_kv_cache_manager.diff`
(91 changed lines, single file). Deploy like the other patches: stage at `/var/tmp/` on both
nodes and bind-mount read-only over the module path; launcher `ds4-vision-tp2-spark-recycle.sh`.

## Root cause (two parts, both in this file)

1. **Unbounded prompt-block protection.** `MLAAttentionManager.cache_blocks` (opted in for
   `model_version == "deepseek_v4"`) and `SlidingWindowMLAManager.cache_blocks` call
   `_protect_prompt_blocks`, which `touch()`es prompt blocks — an extra `ref_cnt` that survives
   the request. The only cap, `2 * max_model_len / block_size`, is 8,192–524,288 pages per
   group at `max_model_len=1M`, i.e. larger than the pool (12,621 pages). The protected set
   therefore grows by ~30–60 pages on *every* request (+0.45–0.79 pp of the pool), never shrinks
   while idle, and is released only FIFO under allocation pressure — evicting the oldest
   conversation's prefix first.
2. **Sliding-window churn through the shared LRU.** DSV4's compressor/indexer groups use pages of
   a few tokens. `remove_skipped_blocks` frees each skipped page to the global free queue and the
   next `allocate_new_blocks` pops fresh pages from the queue *head*. One 354K-token prefill
   cycles the whole pool several times, evicting every other request's cached pages — with any
   amount of free space. The protection in (1) was the fork's shield against (2).

Observed effect for an agent: the first call of a user turn (which cannot hit at the previous
prompt boundary once the prior round's reasoning is stripped) re-prefills ~400K tokens
(235 s) instead of 1–4 s, and any large intervening request (e.g. a background review)
destroys the conversation's prefix.

## Fix

- Cap the protected set to a fraction of the pool, shared across the KV groups
  (`VLLM_PROTECTED_PROMPT_BLOCKS_FRACTION`, default 0.30; managers count themselves).
- Sliding-window groups keep pages they skip in a per-request recycle list (ref_cnt kept,
  hash evicted) and reuse them for that request's next allocations; leftovers are freed at
  `free()`. Shared or protected pages (`ref_cnt != 1`) are freed as before.
  `VLLM_SWA_RECYCLE_SKIPPED_BLOCKS=0` disables it. Admission accounting is unchanged
  (conservative: it ignores recyclable pages).

## Validation (2× DGX Spark, TP=2, DeepSeek-V4-Flash-Vision-Exp, DSpark on)

```
                                        stock build          Patch 5 (cap only)   Patch 5+6
354K FULL, resend                       100%  1.3 s          100%  1.3 s          100%  1.2 s
354K ALT, then FULL again               0%    236 s          0%    229 s          100%  1.2 s
ALT again                               0%    235 s          0%    230 s          100%  1.3 s
THIRD tree, then FULL / ALT             —                    —                    100% / 100%
FULL + 300-token decode, then resend    —                    —                    100% (6.8 s decode)
idle usage after 12 small requests      +0.47 pp each        +0.46 pp each        +0.48 pp each (capped later)
idle usage after 80 small requests      ~37% (extrapolated)  15.2%                —
greedy output vs stock                  reference            identical            identical
1200-token generation (thinking off)    —                    —                    coherent, 35.6 tok/s
reasoning_effort=max                    —                    —                    reasoning + answer returned
```
Small requests still pin pages until the cap engages (by design of the fork's protection);
with the cap the pool stabilises (15% after 80 requests here) instead of growing forever.
