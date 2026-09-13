# ============================================================
# ForgeAI - Production Docker Image (NVIDIA vLLM-only)
# Base: Official vLLM OpenAI image (v0.29.0)
# Note: For production deployments, operators should resolve and pin
# VLLM_IMAGE to an immutable RepoDigest (e.g. vllm/vllm-openai@sha256:...)
# ============================================================

ARG VLLM_IMAGE=vllm/vllm-openai:v0.29.0
FROM ${VLLM_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FORGEAI_TELEMETRY_ENABLED=false \
    HF_HOME=/root/.cache/huggingface \
    FORGEAI_HOME=/root/.forgeai

WORKDIR /workspace

# Copy ForgeAI packaging and source files
COPY pyproject.toml README.md ./
COPY src/ src/

# Install ForgeAI base package and runtime dependencies. vLLM 0.29.0 is
# supplied by the upstream image and enforced again by ForgeAI at startup.
RUN pip install --no-build-isolation .

EXPOSE 11434

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "import httpx; r = httpx.get('http://localhost:11434/healthz'); assert r.status_code == 200"

ENTRYPOINT ["forgeai", "serve"]
CMD ["--host", "0.0.0.0", "--port", "11434", "--auth"]
