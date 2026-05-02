# SOC2 Compliance Requirements

This document maps the current `forgeai` implementation to the SOC2 Trust Services Criteria. It is a control mapping and evidence guide, not a certification claim by itself.

## Scope

The current mapping assumes the present runtime and deployment shape:

- vLLM-only backend
- optional authenticated FastAPI API surface
- request logging, metrics, rate limiting, and audit hooks
- CI-based dependency scanning
- deployment diagnostics through `doctor`

## Trust Services Criteria Mapping

## CC1: Control Environment

| Control | Current Implementation | Status |
|---------|------------------------|--------|
| CC1.1 Leadership commitment | `SECURITY.md`, disclosure process, version support policy | Implemented |
| CC1.2 Oversight and review | Git-based review and CI gates | Implemented |
| CC1.3 Roles and responsibility | API roles and route capability enforcement | Implemented |

## CC2: Communication and Information

| Control | Current Implementation | Status |
|---------|------------------------|--------|
| CC2.1 Internal communication | Structured logging, diagnostics, audit events | Implemented |
| CC2.2 External communication | FastAPI docs, README, API responses | Implemented |

## CC3: Risk Assessment

| Control | Current Implementation | Status |
|---------|------------------------|--------|
| CC3.1 Risk identification | `forgeai doctor --full`, security validation, runtime checks | Implemented |
| CC3.2 Supply-chain and model risk | Post-download safety scan, dependency scanning | Partially implemented |
| CC3.3 Change management | Versioned codebase, profiles, CI validation | Implemented |

Notes:

- the current safety scan is useful but limited; it should not be treated as a complete model trust framework

## CC6: Logical and Physical Access

| Control | Current Implementation | Status |
|---------|------------------------|--------|
| CC6.1 Access controls | API key and JWT support | Implemented |
| CC6.2 Authentication | `AuthManager`, bootstrap API key registration | Implemented |
| CC6.3 Authorization | route capability checks and roles | Implemented |
| CC6.6 Threat management | minimum vLLM version checks, dependency scans | Implemented |

## CC7: System Operations

| Control | Current Implementation | Status |
|---------|------------------------|--------|
| CC7.1 Monitoring | GPU inspection, health probes, metrics endpoint | Implemented |
| CC7.2 Incident support | audit logs, structured request logging, diagnostics | Implemented |
| CC7.3 Recovery support | deployment profiles, container/Kubernetes artifacts | Partially implemented |

Notes:

- recovery procedures depend on the surrounding deployment platform and are not fully automated by this project alone

## CC8: Change Management

| Control | Current Implementation | Status |
|---------|------------------------|--------|
| CC8.1 Change authorization | PR workflow, test suite, linting, CI | Implemented |

## CC9: Risk Mitigation

| Control | Current Implementation | Status |
|---------|------------------------|--------|
| CC9.1 Vendor management | pinned minimum versions, dependency policy | Implemented |
| CC9.2 Vendor risk review | automated dependency scanning in CI | Implemented |

## Evidence Collection Guide

For an audit or readiness review, collect evidence from:

1. CI/CD logs for dependency and security scans
2. `doctor --full` output from the target environment
3. audit log output written by `audit_logger.py`
4. API request logs with request IDs and auth context
5. configuration snapshots and deployment profiles
6. change history from Git commits and PR reviews
7. operational runbooks covering secrets, bootstrap keys, and environment setup

## Current Evidence Sources

Primary code and operational locations:

- `SECURITY.md`
- `src/forgeai/core/security.py`
- `src/forgeai/security/auth.py`
- `src/forgeai/security/middleware.py`
- `src/forgeai/security/rate_limit.py`
- `src/forgeai/security/compliance/audit_logger.py`
- `src/forgeai/cli/commands/doctor.py`
- `.github/workflows/`

## Gaps and Cautions

This project can support a SOC2-oriented control environment, but the following should be treated honestly:

- certification requires organizational process evidence, not just code
- the model safety scan is limited
- secret handling quality depends on your deployment environment
- single-process API design simplifies controls but also constrains scaling
- HTTP chat streaming is not currently part of the exposed API behavior

## Suggested Audit Timeline

| Phase | Target | Scope |
|-------|--------|-------|
| Gap assessment | Month 1 | Confirm technical and process gaps |
| Remediation | Month 2-3 | Close environment and process gaps |
| Type I audit | Month 4 | Point-in-time control review |
| Observation period | Month 5-10 | Evidence collection |
| Type II audit | Month 11 | Operating effectiveness review |
