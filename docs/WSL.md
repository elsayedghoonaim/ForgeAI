# WSL Setup

This project is designed to work well from Linux and WSL. If your repository lives under `/mnt/...`, use the WSL bootstrap path documented here instead of trying to treat the checkout like a native Linux filesystem.

## What This Document Covers

Use this guide when you want to:

- create a working `.venv` from WSL
- install the project with or without GPU extras
- avoid editable-install failures on DrvFs mounts
- verify CUDA and the local CLI
- run the API and smoke-test it from WSL

## Prerequisites

- WSL2
- Ubuntu or another Linux distro with Python 3.10+
- `git`
- `build-essential`
- NVIDIA Windows driver plus CUDA support exposed to WSL if you want real GPU inference

For GPU inference, the important distinction is:

- install the NVIDIA driver on Windows
- install the CUDA toolkit in WSL if needed
- do not install Linux display drivers inside WSL

## Bootstrap

From the repository root:

```bash
chmod +x scripts/bootstrap_wsl.sh
./scripts/bootstrap_wsl.sh
source .venv/bin/activate
```

What the script does:

- creates a virtual environment
- installs the project in editable mode
- installs development dependencies
- optionally installs the GPU extras

When the repository is under `/mnt/...`, the script creates the real virtual environment on the Linux filesystem and links it back to `.venv`. This avoids common `pip` and editable-install issues on DrvFs-mounted paths.

## Skip GPU Extras

If you only want a lightweight install:

```bash
INSTALL_GPU=0 ./scripts/bootstrap_wsl.sh
```

## Manual Install

If you do not want to use the bootstrap script:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-build-isolation -e ".[dev]"
```

For GPU inference:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-build-isolation -e ".[gpu,dev]"
```

## CUDA and GPU Verification

Inside the activated virtual environment:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"
forgeai doctor --full
```

Optional CUDA environment variables for the current shell:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

## First Model Pull

If the model requires Hugging Face access, store a token first:

```bash
forgeai config login YOUR_HF_TOKEN
```

Then pull a model:

```bash
forgeai pull google/gemma-4-E2B-it
```

## Local CLI Usage

One-shot inference:

```bash
forgeai run google/gemma-4-E2B-it --prompt "Explain WSL in one paragraph."
```

Interactive chat:

```bash
forgeai chat google/gemma-4-E2B-it
```

Useful chat commands:

- `/exit`
- `/quit`
- `/clear`

## API Startup

Without auth:

```bash
export FORGEAI_MODEL_NAME=google/gemma-4-E2B-it
forgeai serve
```

With auth:

```bash
export FORGEAI_MODEL_NAME=google/gemma-4-E2B-it
export FORGEAI_AUTH_ENABLED=true
export FORGEAI_AUTH_SECRET_KEY=replace-me-with-a-real-secret
export FORGEAI_BOOTSTRAP_API_KEY=replace-with-a-long-random-value
forgeai serve --auth
```

## Smoke Test

In another shell:

```bash
API_KEY=replace-with-a-long-random-value ./scripts/smoke_api.sh
```

Without auth:

```bash
./scripts/smoke_api.sh
```

## WSL-Specific Runtime Notes

- vLLM may switch to the `spawn` multiprocessing start method on WSL. That is expected and trades startup speed for correctness.
- First startup can be noticeably slower than later runs because of downloads, compilation, and cache warmup.
- CLI commands `run` and `chat` default to quieter startup output; use `--startup-logs` when you want raw engine logs.
- `chat` and `run` auto-tune several runtime knobs when you omit `--gpu-util`.
- `serve` remains conservative and single-process; `--workers` must stay `1`.

## Troubleshooting

### Editable install fails under `/mnt/...`

Use the bootstrap script or keep the `.venv` target on the Linux filesystem rather than inside the mounted Windows path.

### `forgeai: command not found`

Make sure the environment is activated:

```bash
source .venv/bin/activate
which forgeai
```

If the script is still missing:

```bash
python -m pip install --no-build-isolation -e ".[gpu,dev]"
hash -r
```

### CUDA works in one shell but not another

Check that you are using the same interpreter:

```bash
which python
echo "$VIRTUAL_ENV"
```

For this project, your interactive WSL shell is the source of truth for GPU availability.

### Model startup is slow

This is usually caused by some combination of:

- first-time model download
- first-time CUDA/Triton/Inductor compilation
- limited VRAM
- WSL process startup overhead

For diagnosis:

```bash
forgeai chat google/gemma-4-E2B-it --startup-logs
watch -n 1 nvidia-smi
```

### HTTP streaming does not work

That is expected in the current build. The CLI supports streaming for `chat` and `run`, but the API endpoint `/v1/chat/completions` is still non-streaming.
