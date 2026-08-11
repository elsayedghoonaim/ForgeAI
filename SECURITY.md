# Security Policy

This document describes how security issues are reported, what the current runtime protects against, and which limits you should keep in mind when deploying `forgeai`.

## Supported Versions

The current maintained line is the `2.0.x` series.

| Version | Supported |
|---------|-----------|
| `2.0.x` | Yes |
| `<2.0`  | No  |

## Reporting a Vulnerability

If you discover a security issue:

1. Do not open a public issue with exploit details.
2. Email `security@forgeai.dev`.
3. Include:
   - a description of the issue
   - affected version or commit
   - reproduction steps
   - expected impact
   - any mitigations you already identified
4. You should receive an acknowledgment within `48 hours`.
5. The goal is to ship a confirmed fix within `7 business days`, subject to severity and reproduction quality.

## Current Security Model

The project provides a security and operational wrapper around the pinned vLLM runtime (`vllm==0.22.1`).

### Runtime Baseline

- Exact pinned `vllm==0.22.1` dependency requirement for GPU inference
- Environment validation before startup
- Path and runtime safety checks in the security layer
- Mandatory rejection of GGUF model files at admission
- Optional post-download safety scan for pulled models

### API Protection

- Dual API surface: Ollama-compatible `/api/*` and OpenAI-compatible `/v1/*` endpoints on local default `127.0.0.1:11434` (container images bind `0.0.0.0:11434`)
- Optional API key authentication and JWT-based auth flows
- Route capability checks and RBAC
- Protected `/metrics` endpoint when auth is enabled
- Structured JSON error responses and request ID propagation
- Append-only audit logging hooks
- In-memory rate limiting
- Streaming support via Newline-Delimited JSON (NDJSON) for Ollama `/api/*` endpoints and Server-Sent Events (SSE) for OpenAI `/v1/*` endpoints

### Operational Controls

- Dependency vulnerability scanning in CI
- Environment diagnostics through `forgeai doctor --full`
- Bootstrap API key registration during authenticated server startup

## Known Security Constraints & System Boundaries

These are deliberate architecture boundaries and operational limits:

- **vLLM-Only Engine**: The project is strictly vLLM-only; alternative backends (such as llama.cpp) have been removed.
- **Supported GPU Runtime Required**: Inference requires a supported GPU vLLM runtime (`vllm==0.22.1`). The primary bundled image and WSL setup target NVIDIA CUDA. ROCm environments require separate official vLLM packages/images and supported non-TurboQuant KV cache dtypes (`auto`, `fp8`); TurboQuant remains unavailable/deferred on ROCm. CPU inference is intentionally unsupported (no CPU fallback exists).
- **GGUF Admission Rejection**: Attempting to load `.gguf` files or Ollama Modelfile DSL returns an explicit admission rejection error.
- **Single Process Daemon**: The server runs as a single process daemon with warm-engine reuse, bounded model count, and `keep_alive` idle eviction.
- **Post-Download Scan**: The safety scan is a lightweight sanity check, not a comprehensive malware or poisoned-weight guarantee.

## Dependency & CVE Mitigation Policy

- `vllm==0.22.1` is strictly pinned for GPU inference.
- GPU monitoring uses `nvidia-ml-py>=12.0.0`.
- Python `>=3.12,<3.13` environment isolation.
- Continuous vulnerability auditing via `pip-audit` and `safety` in CI workflows.

## Security Architecture

| Layer | Component | Purpose |
|-------|-----------|---------|
| Policy | `SECURITY.md` | Disclosure process and supported security stance |
| Runtime | `src/forgeai/core/security.py` | Version gating, environment validation, GGUF admission rejection |
| CLI diagnostics | `src/forgeai/cli/commands/doctor.py` | Deployment audit and remediation hints |
| Model handling | `src/forgeai/models/safety_scanner.py` | Lightweight post-download scan |
| API auth | `src/forgeai/security/auth.py` | API key, JWT, role handling |
| API middleware | `src/forgeai/security/middleware.py` | Request enforcement and capability checks |
| Rate limiting | `src/forgeai/security/rate_limit.py` | In-memory throttling |
| Audit logging | `src/forgeai/security/compliance/audit_logger.py` | Append-only audit events |
| CI/CD | `.github/workflows/security.yml` | Automated dependency scanning |

## Deployment Recommendations

For production-like use:

- Set a strong `FORGEAI_AUTH_SECRET_KEY` and `FORGEAI_BOOTSTRAP_API_KEY`.
- Provide an immutable base image reference by RepoDigest (e.g. `vllm/vllm-openai@sha256:...`).
- Enable authentication for network-exposed endpoints.
- Restrict network access in front of default port `11434`.
- Run `forgeai doctor --full` as part of environment validation.
- Review audit and request logs regularly.

## Related Documentation

- [README.md](README.md)
- [docs/WSL.md](docs/WSL.md)
- [src/forgeai/security/compliance/soc2_requirements.md](src/forgeai/security/compliance/soc2_requirements.md)
