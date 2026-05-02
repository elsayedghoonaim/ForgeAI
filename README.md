# ForgeAI

`forgeai` is a unified, dual-backend CLI and API server for local LLMs. It wraps both **vLLM** and **llama.cpp** under a single interface:

- a local CLI for pulling models, one-shot inference, interactive chat, diagnostics, batch jobs, and benchmarking
- a FastAPI service with health probes, Prometheus metrics, auth hooks, and OpenAI-style chat/model endpoints
- automatic backend selection — pass a `.gguf` file and it picks llama.cpp; pass a HuggingFace repo and it picks vLLM
- smart GGUF discovery on HuggingFace when you want to use llama.cpp with a non-GGUF model
- persistent local config, reusable deployment profiles, and WSL-friendly bootstrap

## What This Project Is For

Use ForgeAI when you want a practical operational layer for:

- local terminal workflows (chat, run, batch)
- simple model-serving and smoke testing
- lightweight API deployments with auth, rate limiting, and metrics
- repeatable environment checks (`doctor`)
- reproducible profiles for common deployment settings

## Supported Backends

| Backend | Engine | Model Format | Hardware | Install Extra |
|---------|--------|-------------|----------|---------------|
| **vllm** | `vllm.LLM` / `AsyncLLM` | HuggingFace safetensors | GPU only | `pip install 'forgeai[vllm]'` |
| **llama_cpp** | `llama_cpp.Llama` | GGUF | CPU + optional GPU | `pip install 'forgeai[llamacpp]'` |

Backend is selected per-run:

```bash
# Explicit
forgeai run google/gemma-4-E2B-it --prompt "Hello" --backend vllm
forgeai run ./model.gguf --prompt "Hello" --backend llama_cpp

# Auto-detect (default)
forgeai run google/gemma-4-E2B-it --prompt "Hello"        # → vllm
forgeai run ./model.gguf --prompt "Hello"                  # → llama_cpp
```

### Resolution order (top wins)

1. `--backend` CLI flag
2. `FORGEAI_BACKEND` env var
3. `~/.forgeai/config.yaml` → `default_backend`
4. **Auto-detect** from model: `.gguf` file → `llama_cpp`, HF repo ID → `vllm`

## Installation

### Base Development Install

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-build-isolation -e ".[dev]"
```

### GPU Runtime (vLLM)

```bash
python -m pip install --no-build-isolation -e ".[vllm,dev]"
```

### llama.cpp Runtime

```bash
python -m pip install --no-build-isolation -e ".[llamacpp,dev]"
```

### Full Install (all backends)

```bash
python -m pip install --no-build-isolation -e ".[all,dev]"
```

### WSL Bootstrap

```bash
chmod +x scripts/bootstrap_wsl.sh
./scripts/bootstrap_wsl.sh
source .venv/bin/activate
```

See [docs/WSL.md](docs/WSL.md) for the full WSL path.

## Quick Start with `google/gemma-4-E2B-it`

### 1. Activate the Environment

```bash
# Linux / macOS / WSL
source .venv/bin/activate

# Windows PowerShell (if you created the venv on Windows)
# .\.venv\Scripts\activate
```

### 2. Optional: Store a Hugging Face Token

```bash
forgeai config login YOUR_HF_TOKEN
```

### 3. Pull the Model

```bash
forgeai pull google/gemma-4-E2B-it
```

### 4. Run One-Shot Inference

```bash
forgeai run google/gemma-4-E2B-it --prompt "Explain black holes simply in 5 bullet points."
```

### 5. Start Interactive Chat

```bash
forgeai chat google/gemma-4-E2B-it
```

### 6. Start the API Server

```bash
forgeai serve google/gemma-4-E2B-it
```

### 7. Smoke Test the API

```bash
./scripts/smoke_api.sh
```

## Command Entry Point

```bash
forgeai --help
```

Global options:

- `--version`, `-V`: show the package version
- `--verbose`, `-v`: enable debug logging

Fallback:

```bash
python -m forgeai --help
```

## Command Reference

### `pull`

Download and cache a model from HuggingFace.

```bash
forgeai pull MODEL [OPTIONS]
```

Options:

- `--cache-dir TEXT`: custom HuggingFace cache directory
- `--revision TEXT`: model branch, tag, or revision
- `--token TEXT`: HuggingFace token override
- `--skip-scan`: skip post-download safety scan

Examples:

```bash
forgeai pull google/gemma-4-E2B-it
forgeai pull google/gemma-4-E2B-it --revision main
```

### `run`

One-shot inference from the terminal.

```bash
forgeai run MODEL --prompt TEXT [OPTIONS]
```

Options:

- `--prompt`, `-p TEXT`: required input prompt
- `--max-tokens INTEGER`: max tokens, default `512`
- `--temperature`, `-t FLOAT`: sampling temperature, default `0.7`
- `--top-p FLOAT`: top-p sampling, default `0.95`
- `--backend`, `-b TEXT`: `auto`, `vllm`, or `llama_cpp`, default `auto`
- `--n-gpu-layers INTEGER`: GPU layers for llama.cpp (`-1` = all), default `0`
- `--n-ctx INTEGER`: context window for llama.cpp, default `4096`
- `--auto-optimize`: auto-tune tensor parallel size
- `--dry-run`: validate and estimate memory without loading weights
- `--gpu-util FLOAT`: target GPU utilization (vLLM only, auto-tuned when omitted)
- `--tp INTEGER`: tensor parallel size (vLLM only)
- `--stream/--no-stream`: stream tokens or wait, default `--stream`
- `--startup-logs`: show raw startup logs

Examples:

```bash
forgeai run google/gemma-4-E2B-it --prompt "Hello"
forgeai run google/gemma-4-E2B-it --prompt "Summarize this" --no-stream
forgeai run google/gemma-4-E2B-it --prompt "Explain CUDA" --dry-run
forgeai run ./model.gguf --prompt "Hello" --n-gpu-layers -1
```

### `chat`

Interactive terminal chat session.

```bash
forgeai chat MODEL [OPTIONS]
```

Options:

- `--system TEXT`: optional system prompt
- `--max-tokens INTEGER`: max tokens per response, default `512`
- `--temperature`, `-t FLOAT`: sampling temperature, default `0.7`
- `--top-p FLOAT`: top-p sampling, default `0.95`
- `--backend`, `-b TEXT`: `auto`, `vllm`, or `llama_cpp`
- `--n-gpu-layers INTEGER`: GPU layers for llama.cpp
- `--n-ctx INTEGER`: context window for llama.cpp
- `--auto-optimize`: auto-tune TP size
- `--gpu-util FLOAT`: GPU utilization (vLLM only)
- `--tp INTEGER`: tensor parallel size (vLLM only)
- `--stream/--no-stream`: default `--stream`
- `--startup-logs`: show raw startup logs

Slash commands: `/exit`, `/quit`, `/clear`

```bash
forgeai chat google/gemma-4-E2B-it
forgeai chat google/gemma-4-E2B-it --system "You are a concise assistant."
forgeai chat ./model.gguf --n-gpu-layers -1
```

### `serve`

Start the FastAPI production API server.

```bash
forgeai serve [MODEL] [OPTIONS]
```

If `MODEL` is omitted, uses `FORGEAI_MODEL_NAME`.

Options:

- `--host TEXT`: bind address, default `0.0.0.0`
- `--port INTEGER`: bind port, default `8000`
- `--backend`, `-b TEXT`: `auto`, `vllm`, or `llama_cpp`
- `--n-gpu-layers INTEGER`: GPU layers for llama.cpp
- `--n-ctx INTEGER`: context window for llama.cpp
- `--gpu-util FLOAT`: GPU utilization (auto-tuned when omitted)
- `--tp INTEGER`: tensor parallel size
- `--auto-optimize`: auto-tune TP size
- `--auth`: enable authentication
- `--workers INTEGER`: Uvicorn workers, must be `1`
- `--log-level TEXT`: log level, default `info`

```bash
forgeai serve google/gemma-4-E2B-it
forgeai serve google/gemma-4-E2B-it --host 127.0.0.1 --port 8000
forgeai serve ./model.gguf --backend llama_cpp --n-gpu-layers -1
```

#### Auth for `serve`

```bash
export FORGEAI_AUTH_ENABLED=true
export FORGEAI_AUTH_SECRET_KEY=replace-me-with-a-real-secret
export FORGEAI_BOOTSTRAP_API_KEY=replace-with-a-long-random-value
export FORGEAI_BOOTSTRAP_API_KEY_NAME=bootstrap
export FORGEAI_BOOTSTRAP_API_KEY_ROLE=admin
forgeai serve google/gemma-4-E2B-it --auth
```

### `doctor`

Validate the environment and print a deployment audit summary.

```bash
forgeai doctor [--full]
```

Checks: Python version, vLLM availability, llama-cpp-python availability, GPU detection, CUDA, runtime deps, security.

### `ps`

Inspect detected GPUs and running ForgeAI processes.

```bash
forgeai ps
```

### `batch`

Offline JSONL processing.

```bash
forgeai batch MODEL --input FILE [OPTIONS]
```

Options:

- `--input`, `-i TEXT`: required input JSONL path
- `--output`, `-o TEXT`: output JSONL path, default `output.jsonl`
- `--max-tokens INTEGER`: max tokens per request, default `512`
- `--temperature FLOAT`: default `0.0`
- `--batch-size INTEGER`: logical batch size, default `32`
- `--prompt-field TEXT`: JSON field, default `prompt`
- `--backend`, `-b TEXT`: `auto`, `vllm`, or `llama_cpp`
- `--n-gpu-layers INTEGER`: GPU layers for llama.cpp
- `--n-ctx INTEGER`: context window for llama.cpp

```bash
forgeai batch google/gemma-4-E2B-it --input prompts.jsonl
forgeai batch google/gemma-4-E2B-it --input prompts.jsonl --output results.jsonl
```

### `benchmark`

Repeatable performance measurement.

```bash
forgeai benchmark MODEL [OPTIONS]
```

Options:

- `--iterations`, `-n INTEGER`: measured runs, default `5`
- `--max-tokens INTEGER`: max tokens per run, default `256`
- `--warmup INTEGER`: warmup iterations, default `1`
- `--prompt TEXT`: custom benchmark prompt
- `--backend`, `-b TEXT`: `auto`, `vllm`, or `llama_cpp`
- `--n-gpu-layers INTEGER`: GPU layers for llama.cpp
- `--n-ctx INTEGER`: context window for llama.cpp

```bash
forgeai benchmark google/gemma-4-E2B-it
forgeai benchmark google/gemma-4-E2B-it --iterations 10 --warmup 2
```

### `config`

Manage persistent local settings.

```bash
forgeai config set KEY VALUE
forgeai config get KEY
forgeai config list
forgeai config delete KEY
forgeai config login TOKEN
```

Config stored at `~/.forgeai/config.yaml`.

### `profile`

Manage reusable deployment profiles.

```bash
forgeai profile save NAME [OPTIONS]
forgeai profile load NAME
forgeai profile list
forgeai profile delete NAME
```

Profiles stored under `~/.forgeai/profiles/`.

## API Reference

When the server is running: `http://HOST:PORT/docs`

Endpoints:

- `GET /healthz` — liveness probe
- `GET /readyz` — readiness probe
- `GET /metrics` — Prometheus metrics
- `GET /v1/models` — list served models
- `GET /v1/models/{model_id}` — model details
- `POST /v1/chat/completions` — OpenAI-compatible chat completion (streaming and non-streaming)

### `POST /v1/chat/completions`

Supported fields: `model`, `messages`, `temperature`, `top_p`, `max_tokens`, `stream`, `stop`.

Non-streaming:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-4-E2B-it",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

Streaming (SSE):

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-4-E2B-it",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

> **Note:** Streaming is supported when the backend supports it (llama.cpp always does; vLLM requires `--stream` at startup).

## Runtime Behavior

### Dynamic Runtime Tuning

When `--gpu-util` is omitted, the CLI derives a hardware-aware target from detected GPU topology and free VRAM.

- `chat` auto-tunes `max_num_seqs`, `max_model_len`, eager execution
- `run` auto-tunes `max_num_seqs`, `max_model_len`, `max_num_batched_tokens`, eager execution
- `serve` auto-tunes GPU utilization, TP only with `--auto-optimize`

Environment overrides:

- `FORGEAI_MAX_NUM_SEQS`, `FORGEAI_MAX_MODEL_LEN`, `FORGEAI_ENFORCE_EAGER`
- `FORGEAI_RUN_MAX_NUM_SEQS`, `FORGEAI_RUN_MAX_MODEL_LEN`, `FORGEAI_RUN_ENFORCE_EAGER`, `FORGEAI_RUN_MAX_NUM_BATCHED_TOKENS`

### Quiet Startup vs Raw Logs

`chat` and `run` default to quiet startup. Use `--startup-logs` for raw engine output.

## Environment Variables

All settings use the `FORGEAI_` prefix. Key variables:

| Variable | Description |
|----------|-------------|
| `FORGEAI_MODEL_NAME` | Default model name or HF repo ID |
| `FORGEAI_BACKEND` | Backend selector: `auto`, `vllm`, `llama_cpp` |
| `FORGEAI_MODEL_PATH` | Local model path |
| `FORGEAI_MAX_MODEL_LEN` | Override model context limit |
| `FORGEAI_TENSOR_PARALLEL_SIZE` | Tensor parallel size |
| `FORGEAI_GPU_MEMORY_UTILIZATION` | Target GPU memory utilization |
| `FORGEAI_N_GPU_LAYERS` | llama.cpp GPU layer offload |
| `FORGEAI_N_CTX` | llama.cpp context window size |
| `FORGEAI_HOST` | Server bind host |
| `FORGEAI_PORT` | Server bind port |
| `FORGEAI_AUTH_ENABLED` | Enable API auth |
| `FORGEAI_AUTH_SECRET_KEY` | JWT signing secret |
| `FORGEAI_TELEMETRY_ENABLED` | Opt-in telemetry |

Non-prefixed: `HF_TOKEN`, `HF_HOME`, `CUDA_HOME`/`CUDA_PATH`.

## Docker

Build for a specific backend:

```bash
# All backends (default)
docker build -t forgeai .

# vLLM only
docker build --build-arg BACKEND=vllm -t forgeai-vllm .

# llama.cpp only
docker build --build-arg BACKEND=llamacpp -t forgeai-llamacpp .
```

Run:

```bash
docker run --gpus all -p 8000:8000 forgeai serve google/gemma-4-E2B-it
```

## Python SDK

See [sdk/python/README.md](sdk/python/README.md).

## Diagnostics and Testing

```bash
python -m pytest tests -v --tb=short
python -m ruff check src tests
forgeai doctor --full
./scripts/smoke_api.sh
```

## Project Layout

```text
src/forgeai/
|-- api/           FastAPI app and HTTP routes
|-- cli/           Typer commands and runtime tuning helpers
|-- core/          Settings, engine lifecycle, backend abstraction
|   `-- backends/  BaseBackend, VLLMBackend, LlamaCppBackend, factory
|-- models/        Model download, metadata, GGUF discovery
|-- monitoring/    Logging and metrics
|-- security/      Auth, middleware, rate limiting, compliance
`-- utils/         GPU inspection, helpers, memory estimation

tests/             Automated regression coverage
docs/              Supplemental documentation
sdk/python/        Async Python SDK
```

## Security and Compliance

- See [SECURITY.md](SECURITY.md) for disclosure policy and security architecture
- See [src/forgeai/security/compliance/soc2_requirements.md](src/forgeai/security/compliance/soc2_requirements.md) for SOC2 control mapping

## License

Apache License 2.0
