# =============================================================================
# vLLM Docker Image — humain-ai/ALLaM-7B-Instruct-preview
# Target: Serverless GPU deployment (RunPod, Modal, AWS Lambda w/ GPU, etc.)
# Base: NVIDIA CUDA 12.1 + Ubuntu 22.04
# =============================================================================

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# -----------------------------------------------------------------------------
# Build arguments — override at build time via --build-arg
# -----------------------------------------------------------------------------
ARG HF_TOKEN
ARG MODEL_ID=humain-ai/ALLaM-7B-Instruct-preview
ARG VLLM_VERSION=0.9.0
ARG PYTHON_VERSION=3.10

# -----------------------------------------------------------------------------
# Environment variables
# -----------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # HuggingFace config
    HF_HOME=/app/model_cache \
    HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} \
    # vLLM / model config — can be overridden at container runtime
    MODEL_ID=${MODEL_ID} \
    HOST=0.0.0.0 \
    PORT=8000 \
    MAX_MODEL_LEN=4096 \
    GPU_MEMORY_UTILIZATION=0.90 \
    TENSOR_PARALLEL_SIZE=1 \
    DTYPE=auto \
    # Disable tokenizer parallelism warnings
    TOKENIZERS_PARALLELISM=false \
    # CUDA settings
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    NCCL_DEBUG=WARN

# -----------------------------------------------------------------------------
# System dependencies
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y  \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    python3-venv \
    curl \
    git \
    wget \
    lsof \
    pciutils \
    libgomp1 \
    # Required for some CUDA ops
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

# -----------------------------------------------------------------------------
# Python dependencies
# -----------------------------------------------------------------------------
RUN pip install --no-cache-dir --upgrade pip

RUN pip install -U vllm transformers accelerate pillow

# -----------------------------------------------------------------------------
# App directory setup
# -----------------------------------------------------------------------------
WORKDIR /app

# Create cache directory for model weights
RUN mkdir -p /app/model_cache

# Copy the entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# -----------------------------------------------------------------------------
# Optional: Pre-download model weights at build time
# Uncomment the block below if you want weights baked into the image.
# Note: this will significantly increase image size (~14GB+).
# For serverless, it's usually better to mount a volume or use a model cache.
# -----------------------------------------------------------------------------
# RUN --mount=type=secret,id=hf_token \
#     HF_TOKEN=$(cat /run/secrets/hf_token) \
#     python -c "
# from huggingface_hub import snapshot_download
# snapshot_download(
#     repo_id='${MODEL_ID}',
#     token='${HF_TOKEN}',
#     local_dir='/app/model_cache/${MODEL_ID}',
#     ignore_patterns=['*.msgpack', '*.h5', 'flax_model*', 'tf_model*']
# )
# print('Model downloaded successfully.')
# "

# -----------------------------------------------------------------------------
# Expose port
# -----------------------------------------------------------------------------
EXPOSE 8000

# -----------------------------------------------------------------------------
# Health check — polls the vLLM /health endpoint
# Adjust intervals to taste; serverless platforms often have their own probes
# -----------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
ENTRYPOINT ["/app/entrypoint.sh"]
