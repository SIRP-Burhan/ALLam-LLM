FROM vllm/vllm-openai:latest

# Build arguments
ARG HF_TOKEN
ARG MODEL_ID=humain-ai/ALLaM-7B-Instruct-preview

# Environment variables
ENV HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} \
    HF_HOME=/app/model_cache \
    MODEL_ID=${MODEL_ID} \
    HOST=0.0.0.0 \
    PORT=8000 \
    MAX_MODEL_LEN=4096 \
    GPU_MEMORY_UTILIZATION=0.90 \
    TENSOR_PARALLEL_SIZE=1 \
    DTYPE=auto \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app
RUN mkdir -p /app/model_cache

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]