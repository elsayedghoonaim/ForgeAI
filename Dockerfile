# ============================================================
# ForgeAI - Production Docker Image (NVIDIA vLLM-only)
# Base: Official vLLM OpenAI image (v0.30.0)
# Note: For production deployments, operators should resolve and pin
# VLLM_IMAGE to an immutable RepoDigest (e.g. vllm/vllm-openai@sha256:...)
# ============================================================

ARG VLLM_IMAGE=vllm/vllm-openai:v0.30.0
FROM ${VLLM_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FORGEAI_TELEMETRY_ENABLED=false \
    HOME=/data \
    HF_HOME=/data/huggingface \
    FORGEAI_HOME=/data/forgeai \
    XDG_CACHE_HOME=/data/.cache
# Auth is enabled via the default CMD (--auth); `forgeai serve` also refuses to bind
# a non-loopback host without auth. Provide FORGEAI_AUTH_SECRET_KEY (>= 32 bytes) and
# FORGEAI_BOOTSTRAP_API_KEY at runtime (env or orchestrator secret), or the server refuses to start.

# Run as an unprivileged user; all writable state lives under /data.
RUN groupadd --gid 10001 forgeai \
    && useradd --uid 10001 --gid 10001 --home-dir /data --no-create-home --shell /usr/sbin/nologin forgeai \
    && mkdir -p /data/huggingface /data/forgeai /data/.cache \
    && chown -R 10001:10001 /data

WORKDIR /workspace

# Copy ForgeAI packaging and source files
COPY pyproject.toml README.md ./
COPY src/ src/

# Install ForgeAI base package and runtime dependencies. vLLM 0.30.0 is
# supplied by the upstream image and enforced again by ForgeAI at startup.
RUN pip install --no-build-isolation .

USER 10001:10001
VOLUME ["/data"]

EXPOSE 11434

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "import httpx; r = httpx.get('http://127.0.0.1:11434/healthz'); assert r.status_code == 200"

ENTRYPOINT ["forgeai", "serve"]
CMD ["--host", "0.0.0.0", "--port", "11434", "--auth"]
