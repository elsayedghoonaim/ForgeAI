# ============================================================
# ForgeAI - Production Docker Image (dual-backend)
# Base: NVIDIA CUDA 12.4.0 Runtime (Ubuntu 22.04)
# ============================================================

ARG BACKEND=all
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip python3.11-venv git \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY requirements/ requirements/
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements/base.txt

ARG BACKEND
RUN if [ "$BACKEND" = "vllm" ] || [ "$BACKEND" = "all" ]; then \
        pip install 'vllm>=0.14.0'; \
    fi \
    && if [ "$BACKEND" = "llamacpp" ] || [ "$BACKEND" = "all" ]; then \
        CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python; \
    fi

COPY . .
RUN pip install --no-deps .

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV FORGEAI_TELEMETRY_ENABLED=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN useradd -m -s /bin/bash forgeai
USER forgeai
WORKDIR /home/forgeai

ENV HF_HOME=/home/forgeai/.cache/huggingface

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3.11 -c "import httpx; r = httpx.get('http://localhost:8000/healthz'); assert r.status_code == 200"

ENTRYPOINT ["forgeai"]
CMD ["--help"]
