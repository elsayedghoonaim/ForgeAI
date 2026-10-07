# ForgeAI Review and Remediation Plan

**Date:** 2026-10-07
**Scope:** all of `src/`, `tests/`, `scripts/`, the packaging, CI, Docker and k8s files.
**How the review was done:**
- Read the code across the whole repo.
- Ran the test suite, `ruff` and `mypy` on Python 3.12 with current dependency versions.

---

## 1. Current state

| Check | Result |
|---|---|
| `pytest` | Fails. **2 test files don't load** (`test_ollama_cli.py`, `test_run.py`). Of the rest, **7 fail and 190 pass**. |
| `ruff check` | **65 errors** (38 can be fixed automatically). CI runs this step, so **CI is red**. |
| `mypy` (`make typecheck`) | **52 errors** in 9 files. Several are real `None` / division-by-zero bugs. |
| CLI import time | About 0.39 s for `import forgeai.cli.main` and about 0.5 s for `forgeai ls`. |

The architecture is sound: a thin-client CLI, one daemon, and warm vLLM engines. Most problems fall into four groups:

1. **Security defaults.** The server runs without auth on `0.0.0.0`, as root.
2. **Engine lifecycle bugs.** One unload breaks the others, there are races, and resources leak.
3. **Hot-path inefficiency.** Disk scans run on every request, and one global lock serializes all generation.
4. **Test and CI drift.**

---

## 2. Root causes of the failing tests

| Test | Cause | Fix |
|---|---|---|
| `test_ollama_cli.py`, `test_run.py` (don't load) | `CliRunner(mix_stderr=False)` was removed in Click 8.2. Typer 0.27 bundles its own Click, so pinning `click` alone won't help. | Use `CliRunner()` (stderr is already captured separately). Add upper bounds for `typer`, `pytest` and `click` in the dev extras. |
| `test_devtool_engine_lock_serialization` | `BaseBackend.generate()` gained a `top_k` parameter. The test's `DummyBackend` still has the old signature, and the engine passes arguments by position. | Add `top_k` to the test double. Pass arguments **by keyword** in `engine.py:1089,1121`. |
| `SecurityRepoIdTests` (3) | `validate_repo_id_string` (`manifest.py:19-55`) raises different error messages than the tests expect, and it has **no length limit**: a 201-character ID is accepted. | Use one error-message prefix, enforce a maximum length, and reject `--` and surrounding whitespace (Phase 1). |
| `test_benchmark_evaluate_handles_pass_and_non_pass_exit_codes` | The test imports schema names that were renamed (`BenchmarkGateOutcome`, `BenchmarkHardwareSpec`). | Rewrite it against `GateOutcome` and `HardwareMetadata`, using a fully measured artifact. |
| `test_blank_template_artifact_defaults`, `test_absent_hardware_evidence_unvalidated_template` | `turboquant.py:688-694` marks an `auto` profile with no measurements as `INCOMPLETE` instead of `NOT_RUN`. | Use `STATUS_NOT_RUN` there, as the TurboQuant path already does at line 785. |

---

## 3. Findings by priority

### Critical

- **C1. No auth on a public port.**
  - Where: `config.py:158`, `serve.py:82`, Dockerfile, `docker-compose.yml`, `k8s/deployment.yaml`.
  - Problem: `auth_enabled` defaults to `False`, and every container file binds `0.0.0.0`. Anyone who can reach the port can call `/api/pull` (fill the disk with arbitrary Hugging Face repos), `/api/delete`, unload engines, and read `/docs`.
  - Fix:
    - Refuse to start when the host isn't loopback and auth is off, unless `--insecure-no-auth` is passed.
    - Turn auth on in the image and manifests, with secrets coming from a k8s Secret.
- **C2. Unloading one model shuts down Ray for every engine.**
  - Where: `vllm_backend.py:476-479`.
  - Problem: `VLLMBackend.shutdown()` calls the global `ray.shutdown()` on every eviction or TTL expiry. With `max_loaded_models > 1`, this breaks every other tensor-parallel engine.
  - Fix: call `ray.shutdown()` only from the manager's final shutdown.
- **C3. The manifest is re-read and the snapshot re-scanned on every request.**
  - Where: `registry.py:210-279`, `manifest.py:144`, `api/runtime.py:46`.
  - Problem: each request parses the YAML about 3 times, runs `rglob("*.safetensors")`, and stats every file in the snapshot. All of this is synchronous, inside async handlers, so it blocks the event loop.
  - Fix:
    - Cache `(manifest, digest, snapshot_path)` keyed by the file's mtime.
    - Compute sizes lazily, only for listing endpoints, using `asyncio.to_thread`.
- **C4. Division by zero in benchmark evaluation.**
  - Where: `turboquant.py:973-976`.
  - Problem: a baseline value of 0 crashes `benchmark --mode evaluate`. This was reproduced.
  - Fix: add guards like the ones in the TurboQuant path, or reject values ≤ 0 during validation.

### High

- **H1. One lock serializes all generation.**
  - Where: `engine.py:1088,1120`.
  - Problem: this defeats vLLM's continuous batching. The streaming path holds the lock across every `yield`, so one stalled client blocks everyone. `_last_result` is shared, so concurrent requests can see each other's stats.
  - Fix: lock only the synchronous `LLM` path, and return stats per request.
- **H2. Streaming leaks leases and keeps generating after disconnect.**
  - Where: `chat.py:77-99`, `ollama.py:165,298`.
  - Problem: when the client disconnects, the generator isn't closed, so vLLM keeps decoding. A failed `response.start` leaves the reference count above 0, so the model is never evicted.
  - Fix: wrap the stream in `try/finally: await gen.aclose()`, and abort the engine request on `CancelledError`.
- **H3. The wrong model revision can be served.**
  - Where: `loader.py:129-134`.
  - Problem: if the requested revision is missing but exactly one snapshot exists, that snapshot is returned. The download is then skipped.
  - Fix: remove the single-snapshot fallback.
- **H4. The safety scanner deletes harmless models and doubles downloads.**
  - Where: `loader.py:171-177`, `safety_scanner.py:84,229`.
  - Problems:
    - The download doesn't exclude `*.bin`, so repos that ship both formats download both.
    - Each `.bin` costs 15 points, so 4 shards get the model deleted.
    - Deletion removes only the symlinks, not the blobs.
    - The API pull path never runs the scan.
    - A pickle that fails to parse is treated as **safe**.
  - Fix:
    - Download with `allow_patterns` for safetensors, JSON and tokenizer files only.
    - Delete the blobs as well as the symlinks.
    - Run the scan in the API pull path too.
    - Treat a parse error as unsafe.
- **H5. Concurrent engine loads race on process-wide state.**
  - Where: `vllm_backend.py:156-206`.
  - Problem: `os.environ` and the `warnings` filters are changed inside a worker thread, so parallel loads interfere with each other.
  - Fix: guard this with a module-level `threading.Lock`, or set the values once at startup.
- **H6. Shutdown can leak engines.**
  - Where: `engine.py:872-932, 1007`.
  - Problems:
    - If cleanup is interrupted, the entry is left in `UNLOADING`, and `stop()` skips it.
    - `stop()` unloads engines that still have requests in flight.
    - Shutdown waits on whole request tasks and can cancel them.
  - Fix:
    - Run cleanup under `asyncio.shield` and await cleanups already in progress.
    - Track only the load tasks.
- **H7. JWT and secret handling.**
  - Where: `auth.py:61,130`, `serve.py:106`.
  - Problems:
    - The default secret is `"change-me"`, but `serve` only rejects a different placeholder, and there's no minimum length.
    - The algorithm isn't checked.
    - The `permissions` claim is trusted as-is.
    - Revoked keys keep working through JWTs that were already issued.
  - Fix:
    - Require a secret of at least 32 bytes and reject both placeholders.
    - Allow only HS256, HS384 and HS512, and require `exp`, `sub` and `role`.
    - Derive permissions from the role, and check that `sub` is still an active key.
- **H8. The container runs as root.**
  - Where: Dockerfile, compose, k8s.
  - Problem: there's no `USER`, no `securityContext`, and compose uses `ipc: host`.
  - Fix:
    - Add `USER 10001` and move the caches to `/data`.
    - Set `runAsNonRoot`, drop all capabilities, and use `readOnlyRootFilesystem`.
- **H9. The audit logger blocks the event loop.**
  - Where: `audit_logger.py:56-75`.
  - Problems:
    - It does a synchronous file write under a lock on every request.
    - Unauthenticated 401s are logged with no limit, so anyone can grow the disk.
    - `verify_chain` fails on every daily file after the first.
  - Fix:
    - Write through a queue on a background thread.
    - Sample or rate-limit denied-auth events.
    - Carry the previous hash across days.
- **H10. The rate limiter does nothing when auth is off, and never limits failed auth.**
  - Where: `middleware.py:374`.
  - Fix: make it a separate pure-ASGI middleware, keyed by IP before auth and by actor after.
- **H11. CLI timeouts.**
  - Where: `cli/runtime.py:205,234,277`.
  - Problem: a 30 s read timeout applies to pulls, cold loads and non-streaming runs. Every timeout is reported as "is the server running?".
  - Fix:
    - Catch `httpx.TimeoutException` separately with its own message.
    - Use `read=None` for streaming, pull and generate.
- **H12. Benchmark teardown leaves orphan processes.**
  - Where: `runner.py:964-993`.
  - Problem: only the direct child is killed. The vLLM EngineCore and worker processes keep holding VRAM, which produces false leak failures.
  - Fix: start the server with `start_new_session=True` and kill the group with `os.killpg`.
- **H13. Flat provenance is ignored.**
  - Where: `runner.py:263-306`.
  - Fix: `prof_prov = prof_info.get("provenance") or prof_info`.

### Medium

- **M1. Middleware.**
  - Problem:
    - Both layers are `BaseHTTPMiddleware`, which adds per-chunk overhead and breaks disconnect handling.
    - Auth runs outermost, so 401/403/429 responses have no CORS headers and skip logging and metrics.
    - Latency is measured to the headers rather than to the end of the stream.
  - Fix:
    - Rewrite both as pure ASGI middleware, with CORS outermost.
    - Finish metrics on the last `send`.
- **M2. Admission and keys.**
  - Admission evicts at most one idle engine (`engine.py:612`). It should keep evicting while idle engines remain.
  - The cache key depends on `size_bytes > 0` (`runtime.py:58`), so the same model loads twice.
  - `EngineKey` identity fields are never applied, and `device_runtime_id` is hard-coded to `cuda:0`.
- **M3. Memory preflight.**
  - It ignores `pipeline_parallel_size` (it needs `tp × pp` GPUs).
  - It runs `detect_gpus()` twice. Detect once and pass the result to both places.
- **M4. Cache accounting.**
  - `get_cached_models` follows symlinks, so blob sizes are counted twice.
  - `garbage_collect_unreferenced` never deletes the blobs.
- **M5. API correctness.**
  - Error codes:
    - A `ValueError` from `parse_tag` turns into a 500; it should be a 400.
    - Embed failures return 400; they should be 500.
  - SSE:
    - The first chunk is missing `role`, and the last is missing `finish_reason`.
    - Errors that happen mid-stream are dropped silently.
  - Ollama unload: `keep_alive:0` with the default `stream=true` **loads** the model instead of unloading it (`ollama.py:86`).
  - OpenAI compatibility:
    - `content` should also accept a list of parts or `null`.
    - The route should be `/v1/models/{model_id:path}`.
- **M6. CLI startup time.**
  - About 144 ms comes from `create.py`, through the eager imports in `forgeai.models/__init__`.
  - Move imports inside the command functions and slim down the package `__init__` files.
  - Target: under 150 ms for `forgeai ls`.
- **M7. CLI behavior.**
  - `stop` is a generate request, so it can load a model and it always prints "stopped". Add a real unload endpoint.
  - `chat` and `batch` start an engine in-process, which contradicts the thin-client design.
  - `batch --batch-size` does nothing: it calls `asyncio.run` once per prompt, one at a time.
- **M8. Silent failures.**
  - `runner.py:951` and `runtime.py:438` swallow errors with `except: pass`.
  - `nvidia-smi` is called without a timeout.
  - `gpu.py` skips `nvmlShutdown` when an exception is raised.
- **M9. `max_tokens=0` becomes 512.**
  - Where: `vllm_backend.py:296,303,418`.
  - Fix: test `is None` instead of using `or`.

### Low and cleanup

- **Dead code.**
  - Remove: `AdapterManager` (it calls vLLM methods that don't exist), `VisionPipeline` (unused and an SSRF risk), `detect_quantization`, `garbage_collect_unreferenced` (or fix it), `select_resource_profile`, and `scripts/benchmark_turboquant.py` (it duplicates the CLI and has drifted).
  - Fix: `PRIMARY_PROFILES` lists `turboquant_3bit_nc`, which the config rejects.
- **Duplication.**
  - Make one helper each for:
    - the GGUF rejection (in 5 places)
    - the KV-dtype allowlist (4 places)
    - the vLLM version check (3 places)
    - the leak-threshold formula
  - Merge `resolve_backend` into `create_backend`.
  - Generate `from_dict` with `dataclasses.fields()`.
- **Small leaks.**
  - Rate-limit keys are never pruned.
  - `_timer_tasks` and `_key_to_tag` grow without bound.
  - The telemetry buffer is only flushed in `__del__`.
- **Small fixes.**
  - Validate the `X-Request-ID` header.
  - Make CORS origins configurable, empty by default.
  - Use `CONTENT_TYPE_LATEST` for `/metrics`.
  - Parse bare IPv6 addresses correctly in `resolve_base_url`.
  - Handle a null `size_vram` in `ps`.
  - Use `time.monotonic()` in the rate limiter.
- **Packaging and CI.**
  - Make `gpu` / `vllm` / `all` a single extra.
  - Change the "Production/Stable" classifier to Beta.
  - Add `mypy` to CI.
  - Drop `continue-on-error` from the safety scan.
  - Pin GitHub Actions by SHA.
  - Add an image scan with Trivy.
  - Fix `.PHONY`.
  - Remove the `sys.path.insert` hacks from the tests.

---

## 4. Execution plan

### Phase 0: Get CI green (about 1 day)

1. Fix the 9 broken tests as described in §2.
2. Run `ruff check --fix`, then fix the remaining 27 issues by hand.
3. Add upper bounds for `typer`, `click` and `pytest` in `[dev]`. Add a `uv.lock` or `requirements-dev.lock`.
4. Fix the real `mypy` bugs (C4, H13, `config.py:214/223`, `safety_scanner.py:173`). Add `mypy` to CI, starting with a baseline and ratcheting it down.

**Done when:** `make lint test typecheck` all pass locally and in CI.

### Phase 1: Security hardening (2–3 days)

C1, H7, H8, H9, H10, the repo-ID validation from §2, and dropping `continue-on-error` from the CI safety scan.
- Add tests:
  - Startup is refused when the host is public and auth is off.
  - A revoked key's JWT is rejected.
  - Rate limiting works with auth off.
  - Repo IDs containing `--` or over the length limit are rejected.

### Phase 2: Engine correctness (3–4 days)

C2, H1, H2, H3, H5, H6, M2, M3, M9.
- Add tests:
  - With two engines loaded, unloading one leaves the other working.
  - A client disconnect frees the lease and aborts the request.
  - Concurrent requests run in parallel (not serialized) on a fake async backend.
  - Shutdown cancelled partway through doesn't leak an engine.

### Phase 3: Performance (2–3 days)

C3, M1, M4, M6, H4 (download filter).
- Benchmarks to record before and after:
  - p50/p95 time-to-first-token for a warm model under 1, 8 and 32 concurrent clients (expect a large gain from H1 and C3).
  - `/api/tags` latency with 20 models in the cache.
  - `forgeai ls` cold start (target under 150 ms).
  - Event-loop blocking, checked with `asyncio` debug mode (`slow_callback_duration=0.05`).

### Phase 4: API and CLI correctness (2–3 days)

C4, H11, H12, M5, M7, M8, H4 (scanner logic), and the unload endpoint.
- Add contract tests against the official OpenAI and Ollama client libraries.

### Phase 5: Cleanup (1–2 days)

Everything under "Low and cleanup": dead code, duplication, packaging. Expected result: about 1,000 fewer lines and a smaller attack surface.

**Order:** Phase 0 must come first. Phases 1 and 2 can run in parallel. Phase 3 depends on the H1 change from Phase 2.

---

## 5. Quick wins (under an hour each)

- `CliRunner()` instead of `CliRunner(mix_stderr=False)` (makes 2 test files load again).
- Add `top_k` to the test backend, and pass keyword arguments in `engine.py`.
- Remove the `ray.shutdown()` call from `VLLMBackend.shutdown()` (C2).
- Delete the single-snapshot fallback in `loader.py:132` (H3).
- `max_tokens if max_tokens is not None else 512` (M9).
- `prof_info.get("provenance") or prof_info` (H13).
- `ruff check --fix`.
