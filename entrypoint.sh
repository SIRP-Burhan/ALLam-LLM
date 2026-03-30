#!/usr/bin/env bash
# =============================================================================
# entrypoint.sh — vLLM server startup for serverless containers
# =============================================================================
set -e

echo "=============================================="
echo " vLLM Serverless Container"
echo " Model : ${MODEL_ID}"
echo " Host  : ${HOST}:${PORT}"
echo " DTYPE : ${DTYPE}"
echo " GPU Util: ${GPU_MEMORY_UTILIZATION}"
echo " Max Len: ${MAX_MODEL_LEN}"
echo "=============================================="

# Validate required env
if [ -z "${HUGGING_FACE_HUB_TOKEN}" ]; then
  echo "⚠️  WARNING: HUGGING_FACE_HUB_TOKEN is not set. Private models will fail to download."
fi

# GPU sanity check
if command -v nvidia-smi &> /dev/null; then
  echo "🖥️  GPU detected:"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
  echo "⚠️  WARNING: nvidia-smi not found. Running CPU-only (not recommended)."
fi

echo "🚀 Starting vLLM server..."

exec vllm serve "${MODEL_ID}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype "${DTYPE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --override-generation-config='{"attn_temperature_tuning": true}' \
  --trust-remote-code
