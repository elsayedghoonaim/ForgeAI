# WSL Setup

This project is designed for high-performance inference on Linux and WSL2 using NVIDIA CUDA GPUs. If your repository lives under `/mnt/...`, use the WSL bootstrap script to set up a virtual environment on the native Linux filesystem.

## What This Document Covers

Use this guide when you want to:

- create a working Python 3.12 `.venv` from WSL
- install ForgeAI 2.0.0 with pinned `vllm==0.22.1` GPU extras
- avoid editable-install failures on DrvFs mounts
- verify CUDA and the local CLI
- run the single-daemon server on port 11434 and smoke-test it

## Prerequisites

- WSL2
- Ubuntu or another Linux distro with **Python 3.12**
- `git` and `build-essential`
- NVIDIA Windows driver with CUDA support exposed to WSL2 for GPU inference

For GPU inference:
- install the NVIDIA driver on Windows (it exposes CUDA to WSL2 automatically)
- do not install Linux display drivers inside WSL2

> **Note on ROCm:** ROCm packaging requires separate official vLLM wheels/images. ForgeAI TurboQuant is unavailable and deferred on ROCm; ROCm users must explicitly select a supported non-TurboQuant KV cache dtype (`auto` or `fp8`). The `auto` setting preserves the model dtype (commonly BF16). CPU inference is unsupported (no CPU fallback exists).

## Bootstrap

From the repository root:

```bash
chmod +x scripts/bootstrap_wsl.sh
./scripts/bootstrap_wsl.sh
source .venv/bin/activate
```

What the script does:
- requires Python 3.12
- creates a virtual environment on the Linux filesystem if under `/mnt/...`
- installs the project in editable mode with base dependencies and optional GPU extras (`vllm==0.22.1`)

## Development / Non-GPU Install

If you only want a lightweight management environment for unit tests and code inspection without model inference:

```bash
INSTALL_GPU=0 ./scripts/bootstrap_wsl.sh
```

> **Warning:** Without `vllm==0.22.1`, model inference is **UNAVAILABLE**. CPU fallback is not supported.

## Manual Install

If you prefer to set up manually:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

For GPU inference (required for running models):

```bash
python -m pip install --no-build-isolation -e ".[gpu,dev]"
```

For management/dev only (no inference):

```bash
python -m pip install --no-build-isolation -e ".[dev]"
```

## CUDA and GPU Verification

Inside the activated virtual environment:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('Device name:', torch.cuda.get_device_name(0))"
forgeai doctor --full
```

Optional CUDA environment variables:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

## Starting the Daemon Server (`serve`)

`pull`, `run`, `ps`, and other CLI commands are thin clients of an already running `forgeai serve` daemon.

Start the daemon first (defaults to `127.0.0.1:11434`):

```bash
forgeai serve
```

Specify host or port explicitly if needed:

```bash
forgeai serve --host 127.0.0.1 --port 11434
```

With API Authentication:

```bash
export FORGEAI_AUTH_ENABLED=true
export FORGEAI_AUTH_SECRET_KEY=replace-me-with-a-real-secret
export FORGEAI_BOOTSTRAP_API_KEY=replace-with-a-long-random-value
forgeai serve --auth
```

> **Note:** Keep `forgeai serve` running in its own terminal window or background service.

## Local CLI Usage (Thin Client Commands)

With `forgeai serve` running in another shell:

### 1. Store Hugging Face Token (if pulling gated models)

```bash
forgeai config login YOUR_HF_TOKEN
```

### 2. Pull a Model

Pull a model repository into local cache via the running daemon:

```bash
forgeai pull Qwen/Qwen2.5-7B-Instruct
```

### 3. Register a Model Manifest (`create`)

Register a model tag using a local YAML manifest (`-f` / `--file` required):

```bash
forgeai create my-model -f manifest.yaml
```

### 4. One-Shot & Interactive Inference (`run`)

Send an inference request to the warm engine daemon:

```bash
forgeai run Qwen/Qwen2.5-7B-Instruct "Explain black holes simply."
```

### 5. Inspect Running Models & System GPUs (`ps`, `ls`, `show`)

```bash
forgeai ls
forgeai ps
forgeai show Qwen/Qwen2.5-7B-Instruct
```

### 6. Stop / Remove Models (`stop`, `rm`)

```bash
forgeai stop Qwen/Qwen2.5-7B-Instruct
forgeai rm Qwen/Qwen2.5-7B-Instruct
```

## Smoke Test

In another shell while the server is running:

```bash
./scripts/smoke_api.sh
```

With auth:

```bash
API_KEY=replace-with-a-long-random-value ./scripts/smoke_api.sh
```

## WSL-Specific Runtime Notes

- **Multiprocessing**: vLLM may use the `spawn` start method on WSL2 for correctness.
- **Warm-Engine Reuse & Eviction**: The daemon reuses warm vLLM engine instances in memory; eviction occurs based on `keep_alive` idle timers or model limits.
- **Single Process**: `serve` runs a single daemon process on port 11434 without positional model arguments (`serve` has no positional model parameter). `--workers` must remain `1`.
- **GGUF & CPU Rejection**: GGUF formats, llama.cpp, and CPU inference are unsupported. Passing `.gguf` files triggers immediate admission rejection.
