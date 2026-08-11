# ForgeAI

`forgeai` (v2.0.0) is a high-performance, vLLM-only local model runtime and daemon providing **Ollama CLI and API compatibility** alongside **OpenAI-compatible endpoints**.

- **Single Process Daemon**: `forgeai serve` runs on default local address `http://127.0.0.1:11434` (container images bind `0.0.0.0:11434`).
- **Thin Client Architecture**: Commands like `pull`, `run`, `ls`, `ps`, `show`, `stop`, and `rm` act as thin clients communicating with the running daemon. `forgeai create` validates and registers manifests locally in the shared registry without requiring a running daemon or loading an engine.
- **Warm-Engine Reuse**: Manages warm vLLM engine instances in memory with bounded model capacity and `keep_alive` idle eviction. No per-request serving process creation.
- **Dual API Surface**: Exposes all 9 implemented Ollama-compatible `/api/*` endpoints (streaming via NDJSON) and OpenAI-compatible `/v1/*` endpoints (streaming via SSE).
- **Pinned vLLM Runtime**: Pinned to exact `vllm==0.22.1` for maximum stability and native TurboQuant KV-cache support.

---

## Quick Start Workflow

`pull`, `run`, and daemon-backed management commands require an already running `forgeai serve` daemon.

### 1. Start the ForgeAI Daemon

In your main server shell or service manager:

```bash
forgeai serve
```

### 2. Pull a Model (in another terminal)

```bash
forgeai pull Qwen/Qwen2.5-7B-Instruct
```

### 3. Run Inference

```bash
forgeai run Qwen/Qwen2.5-7B-Instruct "Explain quantum computing in 3 sentences."
```

---

## Architecture & System Stance

| Component | Policy & Stance |
|-----------|-----------------|
| **Engine** | Strictly **vLLM-only** (`vllm==0.22.1`). Alternative backends (llama.cpp) are removed. |
| **Hardware** | Requires a supported GPU vLLM runtime (`vllm==0.22.1`). The primary bundled image and WSL path target **NVIDIA CUDA**. |
| **ROCm Support** | Requires separate official vLLM ROCm wheels/images. TurboQuant is **unavailable/deferred on ROCm** (ROCm users must explicitly select supported non-TurboQuant KV dtypes: `auto` or `fp8`). `auto` preserves the model dtype (commonly BF16). |
| **CPU Support** | CPU inference is **unsupported** (no CPU fallback exists). |
| **Model Formats** | **Hugging Face repositories** and local **safetensors** directories are supported. **GGUF models are explicitly rejected** at admission time. |
| **Manifest System** | Replaces Ollama Modelfile DSL with declarative **YAML manifests** (`ForgeAIManifest`). |
| **Process Model** | Single persistent daemon process (`127.0.0.1:11434`) with warm-engine reuse. No per-request process spawning. |

---

## Installation

ForgeAI 2.0.0 requires **Python `>=3.12,<3.13`**.

### GPU Runtime Installation (vLLM)

For GPU inference on NVIDIA CUDA / WSL2:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-build-isolation -e ".[gpu]"
```

### Management / Development Install (Non-Inference)

For local development, unit tests, and management commands without model loading:

```bash
python -m pip install --no-build-isolation -e ".[dev]"
```

> [!WARNING]
> The base development installation without vLLM is provided **only** for management and unit testing. It **cannot perform model inference** and does **not** act as a CPU fallback.

### WSL2 Bootstrap

For WSL2 checkouts (including `/mnt/...` DrvFs mounts):

```bash
chmod +x scripts/bootstrap_wsl.sh
./scripts/bootstrap_wsl.sh
source .venv/bin/activate
```

---

## Supported Models & YAML Manifests

### Supported Model Sources
- Hugging Face model IDs (e.g., `Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Meta-Llama-3-8B-Instruct`)
- Local directories containing valid Hugging Face safetensors weights and `config.json`

### Unsupported Formats
- GGUF files (`.gguf`) — rejected with explicit admission error
- Ollama Modelfile DSL syntax
- llama.cpp model weights or options
- CPU inference execution

### ForgeAI YAML Manifests (`src/forgeai/models/manifest.py`)
ForgeAI replaces the legacy Ollama Modelfile DSL with declarative YAML manifests matching the `ForgeAIManifest` schema:

```yaml
schema_version: "2.0"
source_kind: "huggingface"
name: "qwen-custom"
model: "Qwen/Qwen2.5-7B-Instruct"
revision: "main"
tokenizer_override: null
system_prompt: "You are a helpful, expert AI pair programming assistant."
chat_template: "jinja"
parameters:
  temperature: 0.7
  top_p: 0.95
  top_k: 40
  max_tokens: 2048
  stop: []
engine_settings:
  tensor_parallel_size: 1
  pipeline_parallel_size: 1
  gpu_memory_utilization: 0.85
  enforce_eager: false
  weight_quantization: "fp8"          # Model-weight quantization (none, awq, gptq, fp8, bitsandbytes)
  trust_remote_code: false
kv_cache:
  dtype: "turboquant_4bit_nc"         # KV-cache quantization (auto, fp8, turboquant_k8v4, turboquant_4bit_nc)
```

---

## Quantization: Model-Weight vs KV-Cache

ForgeAI explicitly distinguishes between model-weight quantization and KV-cache quantization:

1. **Model-Weight Quantization** (`engine_settings.weight_quantization`): Controls weight precision loaded into VRAM. Supported options depend on model weights and vLLM capabilities: `none`, `awq`, `gptq`, `fp8`, `bitsandbytes`.
2. **KV-Cache Quantization** (`kv_cache.dtype`): Controls precision of the key-value attention cache in vLLM memory:
   - `auto`: Preserves the model's native execution precision (commonly BF16)
   - `fp8`: 8-bit floating point KV cache
   - `turboquant_4bit_nc` & `turboquant_k8v4`: Native TurboQuant presets exposed by pinned `vllm==0.22.1`.

> [!IMPORTANT]
> `turboquant_4bit_nc` and `turboquant_k8v4` are native in pinned `vllm==0.22.1` ([vLLM TurboQuant Docs](https://docs.vllm.ai/en/v0.22.1/api/vllm/model_executor/layers/quantization/turboquant/)), but remain **experimental and POC-gated** in ForgeAI until Task 7 hardware validation. No GPU runtime validation was performed on this machine.

---

## Ollama-Compatible CLI Commands

Daemon-backed CLI commands (`pull`, `run`, `ps`, `stop`, etc.) communicate with an active `forgeai serve` daemon. `forgeai create` operates directly on the local manifest registry without loading an engine.

### `serve`
Start the single-daemon model server (default local bind: `127.0.0.1:11434`). `serve` takes options only and has no positional model argument.

```bash
forgeai serve [--host 127.0.0.1] [--port 11434]
```

### `pull`
Download and register model weights into local Hugging Face cache via the daemon.

```bash
forgeai pull Qwen/Qwen2.5-7B-Instruct
```

### `run`
Run one-shot inference or start interactive terminal chat via the daemon.

```bash
forgeai run Qwen/Qwen2.5-7B-Instruct "Explain black holes in one paragraph."
```

### `create`
Register a model tag from a local YAML manifest file (`-f` / `--file` required). Validates and writes the manifest through the local `ModelRegistry` in the shared ForgeAI home/registry; does not require a running daemon or load a vLLM engine.

```bash
forgeai create my-model -f manifest.yaml
```

### `ls`
List registered model tags and manifests.

```bash
forgeai ls
```

### `ps`
Inspect currently running warm model engines and GPU utilization.

```bash
forgeai ps
```

### `show`
Inspect model details, manifest parameters, and engine settings.

```bash
forgeai show Qwen/Qwen2.5-7B-Instruct
```

### `stop`
Evict a warm model engine from GPU memory.

```bash
forgeai stop Qwen/Qwen2.5-7B-Instruct
```

### `rm`
Remove a model manifest or local model registration.

```bash
forgeai rm Qwen/Qwen2.5-7B-Instruct
```

---

## Deprecated & Rejected Flags (Migration Matrix)

ForgeAI 2.0.0 is strictly vLLM-only. Legacy llama.cpp flags and dual-backend selectors are rejected at CLI admission:

| Deprecated Command / Option | Status in 2.0.0 | Migration Action |
|-----------------------------|-----------------|------------------|
| `--backend` / `-b` | **REJECTED** | Removed. Engine is strictly vLLM. |
| `--n-gpu-layers` | **REJECTED** | Removed. vLLM offloads all layers to GPU. |
| `--n-ctx` | **REJECTED** | Use daemon `--max-model-len` or `FORGEAI_MAX_MODEL_LEN` (vLLM model-context setting). Note that `parameters.max_tokens` is a generation token limit, not model context length. |
| `.gguf` file arguments | **REJECTED** | Use HF safetensors model repositories or local directories. |
| `Modelfile` syntax | **REJECTED** | Use ForgeAI YAML manifests (`forgeai create TAG -f manifest.yaml`). |
| CPU execution | **REJECTED** | Supported GPU vLLM runtime required. CPU fallback is unsupported. |
| `serve MODEL` positional | **REJECTED** | `serve` accepts host/port flags; models are selected via request/run. |

---

## API Reference

Default server address: `http://127.0.0.1:11434`

### Implemented Ollama-Compatible Endpoints (`/api/*`)
Streaming on Ollama endpoints uses **Newline-Delimited JSON (NDJSON)** (`"stream": true`).

1. `POST /api/generate` — Text generation completion (NDJSON streaming)
2. `POST /api/chat` — Chat completion (NDJSON streaming)
3. `POST /api/embed` — Compute text embeddings
4. `GET /api/tags` — List registered model tags (`ls`)
5. `GET /api/ps` — List running warm model engines (`ps`)
6. `POST /api/show` — Show model manifest details
7. `POST /api/pull` — Pull model weights
8. `DELETE /api/delete` — Delete model registration
9. `GET /api/version` — Get server version string

### OpenAI-Compatible Endpoints (`/v1/*`)
Streaming on OpenAI endpoints uses **Server-Sent Events (SSE)** (`"stream": true`).

- `GET /v1/models` — List served models
- `GET /v1/models/{model_id}` — Inspect specific model
- `POST /v1/chat/completions` — OpenAI-compatible chat completions (SSE streaming)

### Service Probes & Diagnostics
- `GET /healthz` — Liveness probe
- `GET /readyz` — Readiness probe
- `GET /metrics` — Prometheus metrics

---

## Docker & Container Deployment

### Base Image Strategy
ForgeAI 2.0.0 uses an NVIDIA vLLM base image (`vllm/vllm-openai:v0.22.1`).

> [!NOTE]
> Production operators should resolve and pin the official base image by immutable `RepoDigest` (e.g., `vllm/vllm-openai@sha256:...`) after pulling.

### Docker Build & Run

```bash
docker build -t forgeai:2.0.0 .
docker run --gpus all -p 11434:11434 forgeai:2.0.0
```

### Docker Compose

```bash
docker compose up -d
```

### Kubernetes

```bash
kubectl apply -f k8s/deployment.yaml
```

---

## Official Reference Links

- [Official Ollama Documentation](https://docs.ollama.com/)
- [Official vLLM Documentation (v0.22.1)](https://docs.vllm.ai/en/v0.22.1/)
- [vLLM TurboQuant Quantization API](https://docs.vllm.ai/en/v0.22.1/api/vllm/model_executor/layers/quantization/turboquant/)
- [vLLM GPU Installation Guide](https://docs.vllm.ai/en/v0.22.1/getting_started/installation/gpu/)
- [ForgeAI Self-Contained Execution Plan](docs/plans/ollama-vllm-turboquant.md) — Full compatibility and resource allocation matrix.

---

## Unexecuted Deferred Validation Commands

The following verification commands were deferred as required by the action safety policy:

```bash
# Code quality and unit tests (CPU/no-GPU)
python -m pytest tests -v --tb=short
python -m ruff check src tests
python -m mypy src

# Container & deployment dry-runs
docker build -t forgeai:2.0.0 .
docker compose config
kubectl apply --dry-run=client -f k8s/deployment.yaml

# API smoke verification (against a running daemon)
./scripts/smoke_api.sh
```

---

## License

Apache License 2.0
