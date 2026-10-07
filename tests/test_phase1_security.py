"""Phase 1 security hardening tests."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import jwt
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from forgeai.api.server import create_app
from forgeai.cli.main import app as cli_app
from forgeai.core.config import DevToolSettings
from forgeai.security.auth import AuthManager, Role
from forgeai.security.compliance.audit_logger import AuditLogger
from forgeai.security.rate_limit import MemoryRateLimiter

SECRET = "s" * 40
runner = CliRunner()


def client_for(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


def make_auth() -> tuple[AuthManager, str]:
    mgr = AuthManager(secret_key=SECRET)
    mgr.register_api_key("viewer-raw", "v", Role.VIEWER)
    return mgr, "viewer-raw"


# --- C1 -------------------------------------------------------------------


@pytest.fixture
def serve_env(monkeypatch):
    for var in ("FORGEAI_AUTH_ENABLED", "FORGEAI_INSECURE_NO_AUTH", "FORGEAI_AUTH_SECRET_KEY",
                "FORGEAI_BOOTSTRAP_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with (
        patch("uvicorn.run") as uv,
        patch("forgeai.api.server.create_app"),
        patch("forgeai.api.runtime.SharedRuntimeAdapter"),
        patch("forgeai.core.engine.EngineManager"),
        patch("forgeai.models.loader.CacheManager"),
        patch("forgeai.models.registry.ModelRegistry"),
        patch("forgeai.security.compliance.audit_logger.AuditLogger"),
    ):
        yield uv


def test_public_host_without_auth_refused(serve_env):
    res = runner.invoke(cli_app, ["serve", "--host", "0.0.0.0"])
    assert res.exit_code == 1
    assert "--insecure-no-auth" in res.stderr
    serve_env.assert_not_called()


@pytest.mark.parametrize("host", ["127.0.0.1", "127.5.0.1", "::1", "localhost"])
def test_loopback_without_auth_ok(serve_env, host):
    res = runner.invoke(cli_app, ["serve", "--host", host])
    assert res.exit_code == 0, res.output
    serve_env.assert_called_once()


def test_insecure_flag_and_env_allow_public(serve_env, monkeypatch):
    assert runner.invoke(cli_app, ["serve", "--host", "0.0.0.0", "--insecure-no-auth"]).exit_code == 0
    monkeypatch.setenv("FORGEAI_INSECURE_NO_AUTH", "true")
    assert runner.invoke(cli_app, ["serve", "--host", "0.0.0.0"]).exit_code == 0


@pytest.mark.parametrize("secret", ["", "change-me", "change-me-in-production", "short"])
def test_serve_rejects_bad_secret(serve_env, monkeypatch, secret):
    monkeypatch.setenv("FORGEAI_AUTH_SECRET_KEY", secret)
    monkeypatch.setenv("FORGEAI_BOOTSTRAP_API_KEY", "k")
    res = runner.invoke(cli_app, ["serve", "--auth", "--host", "0.0.0.0"])
    assert res.exit_code == 1
    serve_env.assert_not_called()


def test_serve_auth_with_good_secret_public_host_ok(serve_env, monkeypatch):
    monkeypatch.setenv("FORGEAI_AUTH_SECRET_KEY", SECRET)
    monkeypatch.setenv("FORGEAI_BOOTSTRAP_API_KEY", "bootstrap-key")
    res = runner.invoke(cli_app, ["serve", "--auth", "--host", "0.0.0.0"])
    assert res.exit_code == 0, res.output


@pytest.mark.asyncio
async def test_docs_disabled_when_auth_enabled():
    mgr, _ = make_auth()
    async with client_for(create_app(enable_auth=True, auth_manager=mgr)) as c:
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert (await c.get(path)).status_code == 404
    async with client_for(create_app()) as c:
        assert (await c.get("/docs")).status_code == 200


# --- H7 -------------------------------------------------------------------


@pytest.mark.parametrize("secret", ["", "change-me", "change-me-in-production", "x" * 31])
def test_auth_manager_rejects_bad_secret(secret):
    with pytest.raises(ValueError):
        AuthManager(secret_key=secret)


def test_bad_algorithm_rejected():
    with pytest.raises(ValueError):
        AuthManager(secret_key=SECRET, algorithm="none")
    with pytest.raises(ValidationError):
        DevToolSettings(auth_algorithm="RS256")
    assert DevToolSettings(auth_algorithm="HS512").auth_algorithm == "HS512"


def test_forged_permissions_claim_ignored():
    mgr, raw = make_auth()
    key = mgr.validate_api_key(raw)
    token = jwt.encode(
        {"sub": key.key_id, "role": "viewer", "permissions": ["admin"], "exp": time.time() + 60},
        SECRET,
        algorithm="HS256",
    )
    payload = mgr.verify_token(token)
    assert payload is not None
    assert payload.permissions == {"monitoring"}


def test_token_missing_required_claims_rejected():
    mgr, raw = make_auth()
    key = mgr.validate_api_key(raw)
    for claims in (
        {"sub": key.key_id, "role": "viewer"},
        {"role": "viewer", "exp": time.time() + 60},
        {"sub": key.key_id, "exp": time.time() + 60},
    ):
        assert mgr.verify_token(jwt.encode(claims, SECRET, algorithm="HS256")) is None


def test_revoked_key_jwt_rejected():
    mgr, raw = make_auth()
    key = mgr.validate_api_key(raw)
    token = mgr.create_token(key.key_id, key.role)
    assert mgr.verify_token(token) is not None
    mgr.revoke_key(key.key_id)
    assert mgr.verify_token(token) is None
    assert mgr.validate_api_key(raw) is None


def test_unknown_sub_jwt_rejected():
    mgr, _ = make_auth()
    assert mgr.verify_token(mgr.create_token("nope", Role.ADMIN)) is None


# --- H9 -------------------------------------------------------------------


def test_audit_log_written_by_background_thread(tmp_path):
    logger = AuditLogger(log_dir=str(tmp_path))
    gate = threading.Event()
    writers: list[str] = []
    real_open = open

    def slow_open(path, *args, **kwargs):
        if str(path).endswith(".jsonl") and "a" in args:
            writers.append(threading.current_thread().name)
            gate.wait(5)
        return real_open(path, *args, **kwargs)

    with patch("forgeai.security.compliance.audit_logger.open", slow_open, create=True):
        start = time.monotonic()
        logger.log("access", "a", "GET", "/x")  # must not block on the gated writer
        assert time.monotonic() - start < 1
        gate.set()
        logger.flush()
    assert writers == ["forgeai-audit-writer"]
    ok, count = logger.verify_chain(str(next(tmp_path.glob("audit_*.jsonl"))))
    assert ok and count == 1
    logger.close()


def test_verify_chain_across_two_days(tmp_path):
    logger = AuditLogger(log_dir=str(tmp_path))
    with patch("forgeai.security.compliance.audit_logger.datetime") as dt:
        from datetime import UTC, datetime

        dt.now.return_value = datetime(2026, 1, 1, 12, tzinfo=UTC)
        logger.log("access", "a", "GET", "/1")
        logger.log("access", "a", "GET", "/2")
        dt.now.return_value = datetime(2026, 1, 2, 12, tzinfo=UTC)
        logger.log("access", "a", "GET", "/3")
    logger.flush()
    files = sorted(tmp_path.glob("audit_*.jsonl"))
    assert [f.name for f in files] == ["audit_20260101.jsonl", "audit_20260102.jsonl"]
    assert logger.verify_chain(str(files[0])) == (True, 2)
    assert logger.verify_chain(str(files[1])) == (True, 1)
    # a fresh logger resumes the chain from disk too
    logger.close()
    logger2 = AuditLogger(log_dir=str(tmp_path))
    logger2.log("access", "a", "GET", "/4")
    logger2.flush()
    assert logger2.verify_chain(str(files[1]))[0]
    logger2.close()


@pytest.mark.asyncio
async def test_denied_auth_audit_is_throttled_per_ip():
    mgr, _ = make_auth()
    audit = MagicMock()
    app = create_app(enable_auth=True, auth_manager=mgr, audit_logger=audit)
    async with client_for(app) as c:
        for _ in range(30):
            assert (await c.get("/api/tags")).status_code == 401
    assert 0 < audit.log.call_count <= 10


# --- H10 ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limit_applies_with_auth_off():
    limiter = MemoryRateLimiter(limit=3, window_seconds=60)
    async with client_for(create_app(rate_limiter=limiter)) as c:
        codes = [(await c.get("/metrics")).status_code for _ in range(5)]
        health = [(await c.get("/healthz")).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    assert 429 not in health


@pytest.mark.asyncio
async def test_failed_auth_is_rate_limited():
    mgr, _ = make_auth()
    limiter = MemoryRateLimiter(limit=3, window_seconds=60)
    app = create_app(enable_auth=True, auth_manager=mgr, rate_limiter=limiter)
    async with client_for(app) as c:
        codes = [
            (await c.get("/api/tags", headers={"X-API-Key": "wrong"})).status_code
            for _ in range(5)
        ]
    assert codes == [401, 401, 401, 429, 429]


def test_rate_limiter_prunes_and_uses_monotonic():
    limiter = MemoryRateLimiter(limit=1, window_seconds=10)
    limiter.check("a", now=0.0)
    limiter.check("b", now=1.0)
    limiter._last_sweep = 0.0
    limiter.check("c", now=100.0)
    assert set(limiter._events) == {"c"}
    with patch("forgeai.security.rate_limit.time.monotonic", return_value=5.0) as mono:
        limiter.check("d")
        assert mono.called


# --- Small fixes ----------------------------------------------------------


@pytest.mark.asyncio
async def test_bad_request_id_replaced_and_good_kept():
    async with client_for(create_app()) as c:
        bad = await c.get("/healthz", headers={"X-Request-ID": "bad id\twith spaces!"})
        good = await c.get("/healthz", headers={"X-Request-ID": "abc-123_.X"})
        long = await c.get("/healthz", headers={"X-Request-ID": "a" * 65})
    assert bad.headers["X-Request-ID"] != "bad id\twith spaces!"
    assert len(bad.headers["X-Request-ID"]) == 32
    assert good.headers["X-Request-ID"] == "abc-123_.X"
    assert long.headers["X-Request-ID"] != "a" * 65


@pytest.mark.asyncio
async def test_cors_default_off_and_configurable():
    hdr = {"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"}
    async with client_for(create_app(settings=DevToolSettings())) as c:
        r = await c.options("/healthz", headers=hdr)
        assert "access-control-allow-origin" not in r.headers
    s = DevToolSettings(cors_allow_origins=["https://ok.example"])
    async with client_for(create_app(settings=s)) as c:
        ok = await c.options("/healthz", headers={**hdr, "Origin": "https://ok.example"})
        assert ok.headers["access-control-allow-origin"] == "https://ok.example"
        bad = await c.options("/healthz", headers=hdr)
        assert "access-control-allow-origin" not in bad.headers


@pytest.mark.asyncio
async def test_metrics_content_type():
    from prometheus_client import CONTENT_TYPE_LATEST

    async with client_for(create_app()) as c:
        r = await c.get("/metrics")
    assert r.headers["content-type"] == CONTENT_TYPE_LATEST


def test_deploy_manifests_are_valid_and_hardened():
    import yaml

    root = Path(__file__).resolve().parents[1]
    docs = list(yaml.safe_load_all((root / "k8s/deployment.yaml").read_text()))
    dep = next(d for d in docs if d["kind"] == "Deployment")
    pod = dep["spec"]["template"]["spec"]
    ctr = pod["containers"][0]
    sc = ctr["securityContext"]
    assert sc["runAsNonRoot"] and sc["runAsUser"] == 10001
    assert sc["allowPrivilegeEscalation"] is False
    assert sc["capabilities"]["drop"] == ["ALL"]
    assert sc["readOnlyRootFilesystem"] is True
    env = {e["name"]: e for e in ctr["env"]}
    assert env["FORGEAI_AUTH_ENABLED"]["value"] == "true"
    assert "secretKeyRef" in env["FORGEAI_AUTH_SECRET_KEY"]["valueFrom"]
    assert "secretKeyRef" in env["FORGEAI_BOOTSTRAP_API_KEY"]["valueFrom"]
    compose = yaml.safe_load((root / "docker-compose.yml").read_text())
    svc = compose["services"]["forgeai"]
    assert "ipc" not in svc and svc["shm_size"]
    assert json.dumps(svc["environment"]).count("FORGEAI_AUTH_ENABLED=true") == 1
