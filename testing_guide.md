# ForgeAI Testing Guide — `google/gemma-4-E2B-it`

This guide walks through **every feature** of ForgeAI using `google/gemma-4-E2B-it` as the test model. Follow each section in order.

---

## Prerequisites

> [!IMPORTANT]
> `google/gemma-4-E2B-it` is a GPU-accelerated model requiring vLLM and a compatible NVIDIA GPU (≥24 GB VRAM recommended). If you want to test with llama.cpp instead, see the [llama.cpp Alternative](#llamacpp-alternative) section at the bottom.

### Hardware
- NVIDIA GPU with ≥24 GB VRAM (e.g., RTX 4090, A100, L40)
- CUDA 12.x driver installed

### Software
- Python ≥ 3.10
- `pip`, `setuptools`, `wheel` up to date
- A HuggingFace account with access to `google/gemma-4-E2B-it` (if gated)

---

## Step 0 — Install ForgeAI

```bash
# Clone the repo (if not done already)
cd /path/to/vLLM

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate          # Windows

# Install with vLLM backend + dev tools
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-build-isolation -e ".[vllm,dev]"
```

Verify:
```bash
forgeai --version
# Expected: forgeai v1.1.0
```

---

## Step 1 — `config login` (Store HF Token)

```bash
forgeai config login YOUR_HF_TOKEN
```

Verify:
```bash
forgeai config get hf_token
# Expected: shows your token
```

Other config commands:
```bash
forgeai config list
forgeai config set cache_dir /my/custom/cache
forgeai config delete cache_dir
```

✅ **Pass Criteria:** Token stored in `~/.forgeai/config.yaml`.

---

## Step 2 — `doctor` (Environment Diagnostics)

```bash
forgeai doctor
```

**Expected Output:**
| Check | Expected |
|-------|----------|
| Python ≥ 3.10 | ✓ |
| vLLM ≥ 0.14.0 | ✓ |
| llama-cpp-python | ✓ or ✗ (optional) |
| GPU available | ✓ (1+ GPUs) |
| CUDA configured | ✓ |
| Dependencies | All ✓ |

Full audit:
```bash
forgeai doctor --full
```

✅ **Pass Criteria:** Score ≥ 80/100. All critical checks green.

---

## Step 3 — `pull` (Download Model)

```bash
forgeai pull google/gemma-4-E2B-it
```

**Expected:** Model files downloaded to the HuggingFace cache. Progress bar shown during download.

Variations:
```bash
forgeai pull google/gemma-4-E2B-it --revision main
forgeai pull google/gemma-4-E2B-it --skip-scan
```

✅ **Pass Criteria:** Command completes without errors. Model cached locally.

---

## Step 4 — `run` (One-Shot Inference)

### 4a — Streaming (default)

```bash
forgeai run google/gemma-4-E2B-it --prompt "Explain black holes simply in 5 bullet points."
```

**Expected:** Tokens appear incrementally in the terminal. Ends with stats: `Tokens: X | Speed: X tok/s | Time: Xs`.

### 4b — Non-Streaming

```bash
forgeai run google/gemma-4-E2B-it --prompt "What is quantum computing?" --no-stream
```

**Expected:** Full response appears inside a Rich panel after generation completes.

### 4c — Dry Run

```bash
forgeai run google/gemma-4-E2B-it --prompt "Test" --dry-run
```

**Expected:** VRAM estimation printed. No model weights loaded.

### 4d — Custom Parameters

```bash
forgeai run google/gemma-4-E2B-it \
  --prompt "Write a haiku about AI" \
  --max-tokens 64 \
  --temperature 1.0 \
  --top-p 0.9 \
  --gpu-util 0.80
```

✅ **Pass Criteria:** All four variants complete without errors.

---

## Step 5 — `chat` (Interactive Session)

```bash
forgeai chat google/gemma-4-E2B-it
```

**Once loaded, test these interactions:**

```
You: Hello! Who are you?
# → Assistant responds

You: What is the capital of France?
# → Assistant responds

You: /clear
# → "Chat history cleared."

You: What did I just ask you?
# → Should NOT remember the previous question

You: /exit
# → "Chat ended."
```

### With System Prompt

```bash
forgeai chat google/gemma-4-E2B-it --system "You always respond in haiku."
```

### Non-Streaming Chat

```bash
forgeai chat google/gemma-4-E2B-it --no-stream
```

**Expected:** Responses appear in Rich panels instead of streaming.

✅ **Pass Criteria:** Chat works interactively. `/clear` resets history. `/exit` quits cleanly.

---

## Step 6 — `serve` (API Server)

```bash
forgeai serve google/gemma-4-E2B-it --host 127.0.0.1 --port 8000
```

**Expected:** Server starts and prints:
```
ForgeAI Server
  Model:    google/gemma-4-E2B-it
  Address:  http://127.0.0.1:8000
  Docs:     http://127.0.0.1:8000/docs
```

### 6a — Health Probes (in another terminal)

```bash
curl http://127.0.0.1:8000/healthz
# Expected: {"status":"ok"}

curl http://127.0.0.1:8000/readyz
# Expected: {"status":"ready","model":"google/gemma-4-E2B-it"}
```

### 6b — List Models

```bash
curl http://127.0.0.1:8000/v1/models
# Expected: JSON with model "google/gemma-4-E2B-it"
```

### 6c — Chat Completion (Non-Streaming)

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-4-E2B-it",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Say hello in one sentence."}
    ],
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

**Expected:** JSON response with `choices[0].message.content`.

### 6d — Chat Completion (Streaming)

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-4-E2B-it",
    "messages": [{"role": "user", "content": "Count from 1 to 10."}],
    "stream": true
  }'
```

**Expected:** If the engine was started with streaming support, SSE chunks appear:
```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk",...}
data: {"id":"chatcmpl-...","object":"chat.completion.chunk",...}
data: [DONE]
```

> [!NOTE]
> If the engine does not support streaming (default vLLM non-streaming init), the server returns `501`.

### 6e — Metrics

```bash
curl http://127.0.0.1:8000/metrics
# Expected: Prometheus text format with forgeai_requests_total, etc.
```

### 6f — API Docs

Open in browser: http://127.0.0.1:8000/docs

**Expected:** Interactive Swagger UI.

✅ **Pass Criteria:** All endpoints respond correctly. Stop the server with `Ctrl+C`.

---

## Step 7 — `batch` (Offline Processing)

Create a test JSONL file:

```bash
echo '{"prompt": "What is Python?"}' > test_prompts.jsonl
echo '{"prompt": "Explain transformers."}' >> test_prompts.jsonl
echo '{"prompt": "What is CUDA?"}' >> test_prompts.jsonl
```

Run:
```bash
forgeai batch google/gemma-4-E2B-it --input test_prompts.jsonl --output test_results.jsonl
```

**Expected:**
```
✓ Batch complete
  Processed: 3 prompts
  Tokens:    X
  Time:      X.Xs
  Output:    test_results.jsonl
```

Verify output:
```bash
cat test_results.jsonl
# Each line is JSON with "prompt", "output", "tokens", "finish_reason"
```

✅ **Pass Criteria:** All 3 prompts processed. Output file contains valid JSONL.

---

## Step 8 — `benchmark` (Performance Profiling)

```bash
forgeai benchmark google/gemma-4-E2B-it --iterations 3 --warmup 1
```

**Expected:** A Rich table with:
- Avg Latency
- P50 Latency
- Avg Tokens/s
- Total Prompt/Completion Tokens
- Total Time

Custom prompt:
```bash
forgeai benchmark google/gemma-4-E2B-it --iterations 5 --prompt "Explain neural networks."
```

✅ **Pass Criteria:** Benchmark completes. Table printed with valid metrics.

---

## Step 9 — `ps` (Process Listing)

```bash
forgeai ps
```

**Expected:** Shows GPU topology and any running `forgeai`/`vllm` processes.

✅ **Pass Criteria:** GPU info displayed (if available).

---

## Step 10 — `profile` (Deployment Profiles)

### Save a Profile

```bash
forgeai profile save gemma-dev \
  --model google/gemma-4-E2B-it \
  --gpu-util 0.85 \
  --max-model-len 8192 \
  --tp 1
```

### List Profiles

```bash
forgeai profile list
# Expected: gemma-dev listed
```

### Load a Profile

```bash
forgeai profile load gemma-dev
```

### Delete a Profile

```bash
forgeai profile delete gemma-dev
```

✅ **Pass Criteria:** Profile CRUD works. Files appear in `~/.forgeai/profiles/`.

---

## Step 11 — Auth-Enabled Server

```bash
export FORGEAI_AUTH_ENABLED=true
export FORGEAI_AUTH_SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
export FORGEAI_BOOTSTRAP_API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
export FORGEAI_BOOTSTRAP_API_KEY_NAME=bootstrap
export FORGEAI_BOOTSTRAP_API_KEY_ROLE=admin

forgeai serve google/gemma-4-E2B-it --auth --host 127.0.0.1
```

### Test without credentials:
```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hi"}]}'
# Expected: 401 Unauthorized
```

### Test with API key:
```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $FORGEAI_BOOTSTRAP_API_KEY" \
  -d '{"messages": [{"role": "user", "content": "Hi"}]}'
# Expected: 200 OK with response
```

✅ **Pass Criteria:** Unauthenticated requests blocked. Admin key grants access.

---

## Step 12 — Unit Tests

```bash
python -m pytest tests/ -v --tb=short
```

**Expected:** All 25 tests pass.

```bash
python -m ruff check src tests
```

**Expected:** No lint errors.

✅ **Pass Criteria:** 25/25 tests pass. 0 lint errors.

---

## Step 13 — Docker

```bash
# Build
docker build -t forgeai .

# Run
docker run --gpus all -p 8000:8000 forgeai serve google/gemma-4-E2B-it

# Test (from host)
curl http://127.0.0.1:8000/healthz
```

Build with specific backend:
```bash
docker build --build-arg BACKEND=vllm -t forgeai-vllm .
docker build --build-arg BACKEND=llamacpp -t forgeai-llamacpp .
```

✅ **Pass Criteria:** Container starts, healthz returns `{"status":"ok"}`.

---

## llama.cpp Alternative

If you don't have a GPU or want to test the llama.cpp backend:

```bash
# Install llamacpp extra
pip install -e ".[llamacpp,dev]"

# Run with auto GGUF discovery
forgeai run google/gemma-4-E2B-it --prompt "Hello" --backend llama_cpp

# Or download a GGUF directly
forgeai run ./path/to/model.gguf --prompt "Hello" --n-gpu-layers -1

# Chat
forgeai chat google/gemma-4-E2B-it --backend llama_cpp --n-gpu-layers -1

# Serve
forgeai serve google/gemma-4-E2B-it --backend llama_cpp --n-gpu-layers -1
```

> [!NOTE]
> When using `--backend llama_cpp` with a non-GGUF model, ForgeAI automatically searches HuggingFace for GGUF variants (e.g., `bartowski/gemma-4-E2B-it-GGUF`) and downloads the best quantization.

---

## Summary Checklist

| # | Feature | Command | Status |
|---|---------|---------|--------|
| 0 | Install | `pip install -e ".[vllm,dev]"` | ☐ |
| 1 | Config | `forgeai config login` | ☐ |
| 2 | Doctor | `forgeai doctor --full` | ☐ |
| 3 | Pull | `forgeai pull google/gemma-4-E2B-it` | ☐ |
| 4 | Run (stream) | `forgeai run ... --prompt "..."` | ☐ |
| 4 | Run (no-stream) | `forgeai run ... --no-stream` | ☐ |
| 4 | Run (dry-run) | `forgeai run ... --dry-run` | ☐ |
| 5 | Chat | `forgeai chat ...` | ☐ |
| 5 | Chat (system) | `forgeai chat ... --system "..."` | ☐ |
| 6 | Serve | `forgeai serve ...` | ☐ |
| 6 | API /healthz | `curl .../healthz` | ☐ |
| 6 | API /readyz | `curl .../readyz` | ☐ |
| 6 | API /v1/models | `curl .../v1/models` | ☐ |
| 6 | API chat | `curl POST .../v1/chat/completions` | ☐ |
| 6 | API streaming | `curl POST ... stream: true` | ☐ |
| 6 | API metrics | `curl .../metrics` | ☐ |
| 6 | API docs | Browser: `.../docs` | ☐ |
| 7 | Batch | `forgeai batch ... --input ...` | ☐ |
| 8 | Benchmark | `forgeai benchmark ...` | ☐ |
| 9 | PS | `forgeai ps` | ☐ |
| 10 | Profile | `forgeai profile save/load/list/delete` | ☐ |
| 11 | Auth | `forgeai serve ... --auth` | ☐ |
| 12 | Tests | `pytest tests/ -v` | ☐ |
| 13 | Docker | `docker build && docker run` | ☐ |
