#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
MODEL="humain-ai/ALLaM-7B-Instruct-preview"
PORT=8000
SESSION="LLM"
WORK_DIR="/home/ubuntu"
VENV_DIR="${WORK_DIR}/.venv"
LOG_FILE="${WORK_DIR}/vllm.log"
HF_CACHE="/lambda/nfs/llm-file-system/.cache/huggingface"

# Load token from environment or fallback (prefer env var over hardcoding)
: "${HUGGING_FACE_HUB_TOKEN:?Set HUGGING_FACE_HUB_TOKEN before running this script}"

# ── System dependencies (skip if already installed) ──────────────────────────
install_if_missing() {
  for pkg in "$@"; do
    dpkg -s "$pkg" &>/dev/null || NEED_INSTALL+=("$pkg")
  done
}
NEED_INSTALL=()
install_if_missing screen lsof pciutils
if [ ${#NEED_INSTALL[@]} -gt 0 ]; then
  echo "Installing: ${NEED_INSTALL[*]}"
  apt-get update -qq && apt-get install -y -qq "${NEED_INSTALL[@]}"
fi

# Install uv if not present
if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="/root/.local/bin:$PATH"
fi

# ── Kill existing GPU / vLLM / port processes ────────────────────────────────
cleanup() {
  echo "Cleaning up existing processes..."

  # Kill GPU processes
  if command -v nvidia-smi &>/dev/null; then
    local gpu_pids
    gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
    [ -n "$gpu_pids" ] && kill -9 $gpu_pids 2>/dev/null || true
  fi

  # Kill vLLM and free port
  pkill -9 -f "vllm serve" 2>/dev/null || true
  lsof -ti:${PORT} | xargs -r kill -9 2>/dev/null || true

  # Kill existing screen session
  screen -S "$SESSION" -X quit 2>/dev/null || true

  # Brief pause for GPU memory to release
  sleep 2
}
cleanup

# ── Virtual environment (reuse if intact, rebuild otherwise) ─────────────────
setup_venv() {
  export PATH="/root/.local/bin:$PATH"

  # Rebuild only if venv is missing or broken
  if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    echo "Creating virtual environment..."
    rm -rf "$VENV_DIR"
    uv venv --python 3.10.11
  fi

  source "${VENV_DIR}/bin/activate"

  # Install/upgrade only if vllm isn't already installed at desired version
  if ! python -c "import vllm" 2>/dev/null; then
    echo "Installing Python packages..."
    uv pip install vllm transformers accelerate pillow
  fi
}
cd "${WORK_DIR}"
setup_venv

# ── Launch vLLM in screen ───────────────────────────────────────────────────
echo "Starting vLLM in screen session '${SESSION}'..."

screen -dmS "${SESSION}" bash -c "
  cd ${WORK_DIR}
  export PATH='/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/root/.local/bin:\$PATH'
  source ${VENV_DIR}/bin/activate
  export HF_HOME='${HF_CACHE}'
  export HUGGING_FACE_HUB_TOKEN='${HUGGING_FACE_HUB_TOKEN}'

  vllm serve '${MODEL}' \\
    --port ${PORT} \\
    --host 0.0.0.0 \\
    --enforce-eager \\
    --override-generation-config='{\"attn_temperature_tuning\": true}' \\
    2>&1 | tee ${LOG_FILE}

  echo
  echo 'vLLM exited. Review: ${LOG_FILE}'
  exec bash
"

# ── Wait for server to be ready ──────────────────────────────────────────────
echo "Waiting for vLLM to start on port ${PORT}..."
for i in $(seq 1 120); do
  if curl -sf "http://localhost:${PORT}/health" &>/dev/null; then
    echo "vLLM is ready! (took ~${i}s)"
    echo "  Model:   ${MODEL}"
    echo "  API:     http://0.0.0.0:${PORT}"
    echo "  Logs:    screen -r ${SESSION}"
    exit 0
  fi
  sleep 1
done

echo "vLLM failed to start within 120s. Check: screen -r ${SESSION}"
exit 1
