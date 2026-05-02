# Security Policy

This document describes how security issues are reported, what the current runtime protects against, and which limits you should keep in mind when deploying `forgeai`.

## Supported Versions

The current maintained line is the `1.1.x` series.

| Version | Supported |
|---------|-----------|
| `1.1.x` | Yes |
| `<1.1`  | No  |

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

The project currently provides a layered security surface around a vLLM runtime.

### Runtime Baseline

- minimum safe `vllm` version enforced at runtime
- environment validation before use
- path and runtime checks in the core security layer
- optional post-download safety scan for pulled models

### API Protection

- optional API key authentication
- optional JWT-based auth flows
- route capability checks and RBAC
- protected `/metrics` when auth is enabled
- structured JSON error responses
- request ID propagation
- append-only audit logging hooks
- in-memory rate limiting

### Operational Controls

- dependency scanning in CI
- deployment diagnostics through `forgeai doctor --full`
- bootstrap API key registration during authenticated server startup

## Known Security Constraints

These are important current limits, not bugs in the policy text:

- the project is vLLM-only; changing backends is not a supported runtime path
- GGUF models are rejected
- the safety scan is a lightweight sanity check, not a comprehensive malware or poisoned-weight guarantee
- the API is single-process and one-engine-per-process
- the current built-in HTTP chat endpoint is non-streaming

## Current CVE Mitigation

### CVE-2026-22807 — vLLM Remote Code Execution

- affected versions: `vllm < 0.14.0`
- mitigation: runtime version enforcement in the security layer blocks startup with vulnerable versions
- operational backstop: dependency scans in CI

## Security Architecture

| Layer | Component | Purpose |
|-------|-----------|---------|
| Policy | `SECURITY.md` | Disclosure process and supported security stance |
| Runtime | `src/forgeai/core/security.py` | Version gating and environment validation |
| CLI diagnostics | `src/forgeai/cli/commands/doctor.py` | Deployment audit and remediation hints |
| Model handling | `src/forgeai/models/safety_scanner.py` | Lightweight post-download scan |
| API auth | `src/forgeai/security/auth.py` | API key, JWT, role handling |
| API middleware | `src/forgeai/security/middleware.py` | Request enforcement and capability checks |
| Rate limiting | `src/forgeai/security/rate_limit.py` | In-memory throttling |
| Audit logging | `src/forgeai/security/compliance/audit_logger.py` | Append-only audit events |
| CI/CD | `.github/workflows/security.yml` | Automated dependency scanning |

## Deployment Recommendations

For production-like use:

- set a real `FORGEAI_AUTH_SECRET_KEY`
- provide a strong `FORGEAI_BOOTSTRAP_API_KEY`
- enable auth explicitly for the API surface you expose
- restrict network access in front of the service
- keep `vllm` updated to supported versions
- run `forgeai doctor --full` as part of environment validation
- review audit and request logs regularly

## Dependency Policy

- runtime dependencies are pinned to minimum safe versions where appropriate
- `vllm>=0.14.0` is mandatory for a supported GPU runtime
- the project now uses `nvidia-ml-py` instead of deprecated `pynvml`
- transitive dependencies should still be validated in your own deployment pipeline

## Related Documentation

- [README.md](README.md)
- [docs/WSL.md](docs/WSL.md)
- [src/forgeai/security/compliance/soc2_requirements.md](src/forgeai/security/compliance/soc2_requirements.md)
