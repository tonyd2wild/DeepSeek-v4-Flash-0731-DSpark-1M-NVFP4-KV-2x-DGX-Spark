#!/usr/bin/env bash
# ds4-vision-tp2.sh <0|1>
# DeepSeek-V4-Flash-Vision-Exp, TP2 across asusi (rank0/head) + bluey (rank1).
# Config is byte-for-byte from tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark
# DEFAULT-CONFIG.md "Exact vLLM command (as running)", with only these adaptations:
#   * model path  -> the locally downloaded Vision-Exp checkpoint
#   * rank map    -> asusi 192.168.192.3 (head) / bluey 192.168.192.1 (worker)
#   * NCCL_IB_HCA / SOCKET_IFNAME -> this fleet's actual devices (the repo ships placeholders)
#   * single NIC  -> plane B (roceP2p1s0f0) is NOT on a common subnet between these two nodes
#                    (asusi 169.254.61.52 link-local vs bluey 192.168.193.1), so MERGE_NICS
#                    would try to bring up RC QPs across mismatched subnets. Plane A only.
set -euo pipefail
NODE_RANK="${1:?usage: ds4-vision-tp2.sh <0|1>}"

IMAGE="vllm-dspark-runtime:mia-raf-pr1-nvfp4-probe-c-keys-concurrency-p2b"
NAME="vllm_ds4_vision"
MODEL_DIR="${MODEL_DIR:-DeepSeek-V4-Flash-Vision-Exp}"   # uncensored drop-in: MODEL_DIR=keys-DeepSeekV4Flash-Vision-EXP-ablit (same shards, same tokenizer)
MODEL_IN_CONTAINER="/models/$MODEL_DIR"
MASTER_ADDR="192.168.192.3"     # asusi
MASTER_PORT="25440"
PORT="8888"

case "$NODE_RANK" in
  0) HOST_IP=192.168.192.3; HEADLESS=""           # asusi = head, reads model over NFS from bluey
     MODELS_HOST="/mnt/bluey-models" ;;
  1) HOST_IP=192.168.192.1; HEADLESS="--headless" # bluey = worker, model is local
     MODELS_HOST="/var/tmp/models" ;;
  *) echo "rank must be 0 or 1" >&2; exit 2 ;;
esac

test -d "$MODELS_HOST/$MODEL_DIR" || {
  echo "MODEL MISSING at $MODELS_HOST/$MODEL_DIR" >&2; exit 3; }
test -f /var/tmp/patch3-scheduler.py || { echo "patch3-scheduler.py MISSING at /var/tmp" >&2; exit 4; }
test -f /var/tmp/spec-dspark.py || { echo "spec-dspark.py (Patch 4, DSpark shared-expert loader fix) MISSING at /var/tmp — source: recipe/overlay/vllm/v1/spec_decode/dspark.py on main" >&2; exit 4; }

mkdir -p "$HOME/.cache/vllm-dspark" "$HOME/.cache/huggingface"
docker rm -f "$NAME" 2>/dev/null || true

# k=5 matches the main default recipe. The earlier k=3 default came from an A/B run without the Patch 4 mount (issue #48).
SPEC='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}'
REASON='{"reasoning_parser":"deepseek_v4","reasoning_start_str":"<think>","reasoning_end_str":"</think>"}'

docker run -d --name "$NAME" --restart no \
  --network host --ipc host --shm-size 64g \
  --ulimit memlock=-1:-1 --ulimit stack=67108864 \
  --gpus all --device /dev/infiniband:/dev/infiniband \
  -v "$MODELS_HOST:/models:ro" \
  -v "$HOME/.cache/huggingface:/cache/huggingface" \
  -v "$HOME/.cache/vllm-dspark:/vllm-cache" \
  -v /var/tmp/patch3-scheduler.py:/opt/env/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py:ro \
  `# Patch 4: DSpark draft shared-expert loader fix. Without it the always-on shared expert loads uninitialised and the draft runs at ~half speed, silently (DSPARK-SHARED-EXPERT-FIX.md). Verify: scripts/check-patch4.sh` \
  -v /var/tmp/spec-dspark.py:/opt/env/lib/python3.12/site-packages/vllm/v1/spec_decode/dspark.py:ro \
  `# Vision-Exp port: DeepseekV4ForCausalLM has no vision tower/aligner, so the` \
  `# stock class rejects the checkpoint with "no module or parameter named aligner".` \
  `# ds4v_model.py = image's model.py + patch_vision.py; ds4v_vision.py = ported ViT+Aligner.` \
  -v /var/tmp/ds4v_model.py:/opt/env/lib/python3.12/site-packages/vllm/models/deepseek_v4/nvidia/model.py:ro \
  -v /var/tmp/ds4v_vision.py:/opt/env/lib/python3.12/site-packages/vllm/models/deepseek_v4/nvidia/ds4v_vision.py:ro \
  -v /var/tmp/ds4v_mm.py:/opt/env/lib/python3.12/site-packages/vllm/models/deepseek_v4/nvidia/ds4v_mm.py:ro \
  `# vLLM decides is_multimodal_model from a STATIC arch-name table, not from the` \
  `# class. This registry adds a multimodal alias pointing at the same class;` \
  `# --hf-overrides below selects it. Without this: "is not a multimodal model".` \
  -v /var/tmp/ds4v_registry.py:/opt/env/lib/python3.12/site-packages/vllm/model_executor/models/registry.py:ro \
  -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_DISABLE_XET=1 \
  -e VLLM_CACHE_ROOT=/vllm-cache \
  -e DG_JIT_CACHE_DIR=/vllm-cache/deepgemm-cache \
  -e FLASHINFER_WORKSPACE_BASE=/vllm-cache/flashinfer \
  -e TILELANG_CACHE_DIR=/vllm-cache/tilelang \
  -e TORCHINDUCTOR_CACHE_DIR=/vllm-cache/torchinductor-cache \
  -e TRITON_CACHE_DIR=/vllm-cache/triton-cache \
  -e TORCH_EXTENSIONS_DIR=/vllm-cache/torch_extensions \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  `# Inference-time RPC deadline -- ENGINE_READY_TIMEOUT_S above covers boot only.` \
  `# vLLM default is 300s, shorter than a cold JIT triggered mid-inference by an` \
  `# MoE/batch shape warmup missed; the worker blocks and the engine is shut down (#8).` \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}" \
  -e DSPARK_SLOT_CLAMP=1 \
  -e VLLM_HOST_IP="$HOST_IP" \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_TRITON_MLA_SPARSE=1 \
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  -e VLLM_SKIP_INIT_MEMORY_CHECK=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_USE_B12X_MOE=1 -e VLLM_USE_B12X_WO_PROJECTION=1 \
  -e VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM=0 -e VLLM_B12X_W4A16_FORCE_BLOCKS_MAX_M=16 \
  -e B12X_W4A16_TC_DECODE=0 \
  -e VLLM_DSPARK_CONFIDENCE_THRESHOLD=0.0 -e VLLM_DSPARK_CONFIDENCE_SCHEDULER=off \
  -e VLLM_DSPARK_LOCAL_ARGMAX=1 -e VLLM_DSPARK_REPLICATE_MARKOV_W1=1 \
  -e VLLM_DSPARK_FUSED_MARKOV_ARGMAX=0 -e VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1 \
  -e VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT=0 -e VLLM_DSPARK_HARDWARE_SCHEDULER_EARLY_STOP=1 \
  -e VLLM_DSV4_B12X_COMPRESSED_MLA=0 \
  -e VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE=0 -e VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE_EXACT=0 \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e TILELANG_CLEANUP_TEMP_FILES=1 -e DG_JIT_USE_NVRTC=0 -e DG_JIT_NVCC_COMPILER=/opt/env/bin/nvcc \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=rocep1s0f0 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 -e TP_SOCKET_IFNAME=enp1s0f0np0 \
  -e NCCL_IB_GID_INDEX=3 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN -e NCCL_NVLS_ENABLE=0 \
  "$IMAGE" \
  -lc "
    export PATH=\"/opt/env/bin:/opt/env/nvvm/bin:/opt/env/targets/sbsa-linux/nvvm/bin:\${PATH:-}\";
    export CUDA_HOME=\"\${CUDA_HOME:-/opt/env/targets/sbsa-linux}\";
    export CUDA_PATH=\"\${CUDA_PATH:-\${CUDA_HOME}}\";
    export CUDAToolkit_ROOT=\"\${CUDAToolkit_ROOT:-\${CUDA_HOME}}\";
    export LD_LIBRARY_PATH=\"/opt/env/lib:/opt/env/targets/sbsa-linux/lib:\${LD_LIBRARY_PATH:-}\";
    exec /opt/env/bin/vllm serve $MODEL_IN_CONTAINER \
      --hf-overrides '{\"architectures\":[\"DeepseekV4VForConditionalGeneration\"]}' \
      --served-model-name deepseek-v4-flash-dspark \
      --host 0.0.0.0 --port $PORT \
      --trust-remote-code \
      --tensor-parallel-size 2 --pipeline-parallel-size 1 \
      --kv-cache-dtype nvfp4_ds_mla \
      --block-size 256 \
      --max-model-len 1048576 \
      --max-num-seqs 12 \
      --max-num-batched-tokens 8192 \
      --max-cudagraph-capture-size 12 \
      --gpu-memory-utilization 0.85 \
      --enable-prefix-caching \
      --async-scheduling \
      --enable-chunked-prefill \
      --speculative-config '$SPEC' \
      --tokenizer-mode deepseek_v4 \
      --distributed-executor-backend mp \
      --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
      --reasoning-parser deepseek_v4 \
      --reasoning-config '$REASON' \
      --default-chat-template-kwargs '{\"thinking\":false}' \
      --generation-config vllm \
      --enable-flashinfer-autotune \
      --nnodes 2 --node-rank $NODE_RANK --master-addr $MASTER_ADDR --master-port $MASTER_PORT $HEADLESS
  "

echo "launched $NAME rank=$NODE_RANK host=$HOST_IP models=$MODELS_HOST"
sleep 3
docker ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || { echo "$NAME exited immediately" >&2; exit 1; }
