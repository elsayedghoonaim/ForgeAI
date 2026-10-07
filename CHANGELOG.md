# Changelog

## Unreleased

### Breaking changes

Security
- Authentication is required when binding a non-loopback host; `forgeai serve` refuses to start without it unless `--insecure-no-auth` is passed.
- `FORGEAI_AUTH_SECRET_KEY` must be at least 32 bytes. Only HMAC algorithms (HS256/HS384/HS512) are accepted.
- JWTs must carry `exp`, `sub` and `role`; tokens issued for revoked API keys are rejected.
- The default host is `127.0.0.1`.
- Interactive API docs (`/docs`, `/redoc`, `/openapi.json`) are disabled when auth is enabled.
- CORS is off by default; allowed origins must be configured explicitly.
- Rate limiting applies even when auth is off.
- `X-Request-ID` values are validated; invalid values are replaced.
- The container runs as non-root (uid 10001). All writable state lives under `/data` (`HOME`, `HF_HOME=/data/huggingface`, `FORGEAI_HOME=/data/forgeai`). Existing volumes mounted at root-owned paths must be re-mounted at `/data` and chowned to 10001.

Model downloads
- `*.py` files are downloaded only with `trust_remote_code`.
- Repositories that only ship `.bin` weights are no longer downloaded (safetensors only).
- Repo IDs containing `--` or exceeding the length limits are rejected.

CLI and API
- `forgeai chat` and `forgeai batch` are thin clients of the running daemon (`forgeai serve`); they no longer load an engine in-process.
- Removed `forgeai chat` flags: `--auto-optimize`, `--gpu-util`, `--tp`, `--startup-logs`.
- `forgeai batch` validates its input file (JSONL, prompt field, `--batch-size >= 1`) before sending requests; `--batch-size` is now the maximum number of concurrent requests.
- New `POST /api/unload` endpoint (used by `forgeai stop`).
- `forgeai run` interactive mode uses `/api/chat`.
- `forgeai benchmark` replaces `scripts/benchmark_turboquant.py`, which is removed.
- Benchmark evaluation derives the required soak duration from `--soak-duration` instead of a fixed 1800 s.

Packaging
- The `gpu`, `vllm` and `all` extras are one `vllm` extra; `gpu` and `all` remain as aliases (`pip install ".[gpu]"` still works).
- Development status classifier changed to Beta.

### Removed (unused code)
- `AdapterManager`, `VisionPipeline`, `forgeai.models.quantization` (`detect_quantization`), `select_resource_profile`, `CacheManager.garbage_collect_unreferenced`, `resolve_backend` (use `create_backend`).
- `PRIMARY_PROFILES` now matches the KV-cache dtypes the config accepts (`turboquant_3bit_nc` is no longer listed).
