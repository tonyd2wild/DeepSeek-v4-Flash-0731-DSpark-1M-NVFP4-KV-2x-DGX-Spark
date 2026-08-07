# Running this recipe on hardware other than the author's

Notes from a clean-room bring-up on 2× DGX Spark (GB10 sm_121a, 200G CX7) that is
not the machine this recipe was developed on. The recipe itself is correct — these
are the places where a different host trips over an assumption. Numbers below were
measured on the `deepseek-ai/DeepSeek-V4-Flash-0731` checkpoint with the Stage-C
runtime + the nvfp4 chain, DSpark k=5 probabilistic, 1M context.

## 1. `nvfp4_ds_mla` lives in the three-stage image, not the overlay

Building only `recipe/Dockerfile.dspark-runtime-overlay` gives an image whose vLLM
rejects the KV dtype:

```
vllm serve: error: argument --kv-cache-dtype: invalid choice: 'nvfp4_ds_mla'
  (choose from auto, bfloat16, float16, fp8, fp8_ds_mla, ...)
```

The dtype comes from `recipe/nvfp4/Dockerfile.stage-{a,b,c}`, chained on top of the
overlay. A one-line note near the build instructions would save the boot cycle.

## 2. The launcher uses the root compose file

`start-deepseek-v4-flash-dspark.sh` defaults to `./docker-compose.dspark.yml`.
Edits made to `verified-deployed-2026-07-04/docker-compose.dspark.yml` are silently
ignored — easy to lose time on when both files exist and look equivalent.

## 3. `GLOO_SOCKET_IFNAME` / `TP_SOCKET_IFNAME` are baked into the base image

The base image ships these pointing at the author's NIC. On another host that
interface is down or absent and rank init dies in:

```
RuntimeError: [enforce fail at /pytorch/third_party/gloo/gloo/transport/tcp/device.cc]
```

The compose file passes `NCCL_SOCKET_IFNAME` through but not these two, so a correct
`.env.dspark` still boots into the failure. **This PR defaults both to
`NCCL_SOCKET_IFNAME`**, which makes one value in `.env.dspark` cover all three.

## 4. `DSPARK_MODEL` and symlinks

Serving weights already on disk needs a bind mount (the compose assumes the HF cache
path). Once mounted, `DSPARK_MODEL` must be a real directory inside the mount — a
symlink whose target is a host path outside it resolves to nothing the container can
see, and vLLM falls back to treating the value as a repo id:

```
huggingface_hub.errors.HFValidationError: Repo id must be in the form ...
```

This PR documents the override-file pattern for the mount rather than adding an
unconditional volume, so nothing changes for HF-cache users.

## 5. systemd needs `HOME`

`.env.dspark` expands `${HOME}` (for `HF_CACHE`). Under a systemd unit `HOME` is
unset and the launcher aborts with `HOME: unbound variable`. Adding
`Environment=HOME=/root` to the unit fixes it.

## 6. GB10 power state after a reboot (not a recipe bug, but it looks like one)

Worth flagging because the symptom mimics a bad config. After one of our nodes
crashed and rebooted, it sat at ~22 W / 2086 MHz under load while the healthy node
ran ~42 W / 2502 MHz. Because TP=2 is lockstep, the pair ran at the slow node's pace:

| | peak decode | step latency | DSpark acceptance |
|---|---|---|---|
| degraded node in the pair | 42–43 tok/s | ~140 ms | 6.02 tok/step (already optimal) |
| after `nvidia-smi -lgc 3003` on both | **83 tok/s** | ~72 ms | unchanged |

Acceptance was perfect the whole time, which is what makes this confusing — the
drafter and the config are fine; only the clock is wrong. A quick check under load:

```bash
nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader
# asymmetry between the two nodes => this
```

## 7. `--default-chat-template-kwargs '{"thinking":false}'`

Not a bug — the recipe optimises for throughput. But it is worth stating loudly,
because the checkpoint has no Jinja chat template (it ships `encoding/` scripts), so
the only thing that turns reasoning back on is:

```json
"chat_template_kwargs": {"thinking": true, "reasoning_effort": "high"}
```

Top-level `reasoning_effort` is ignored. Levels are `low` (default) / `high` / `max`.

Measured on our own execution-graded harness (LiveCodeBench-style, 20 frozen
problems, 3 public + up to 40 private tests per problem, a problem counts only when
every private test passes; plus a procedural seed-generated suite of 48 cases):

| | procedural suite | LCB, one-shot | LCB, after a 32k-token retry of the failures |
|---|---|---|---|
| thinking off (recipe default) | 0.875 | 12/20 | 13/20 |
| **thinking on, effort high** | **0.979** | 11/20 | **20/20** |

The one-shot number goes *down* when reasoning is enabled — with an 8k cap the model
spends the budget thinking and gets truncated. Every one of the nine failures was
`finish_reason=length`, and all nine passed once retried with a 32k budget. Anyone
benchmarking this checkpoint should report the thinking setting and retry
length-capped failures, or the result measures the cap rather than the model.

## 8. Streamed reasoning field name

The runtime emits `delta.reasoning`; OpenAI-compatible clients expect
`delta.reasoning_content`. Clients that render a reasoning panel sit on "Thinking…"
until the whole response lands. A small translating proxy in front of the server is
enough; noting it in the README would spare people the debugging.
