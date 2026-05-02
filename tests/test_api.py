from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forgeai.api.server import create_app


class FakeEngine:
    def __init__(self, running: bool = True) -> None:
        self.is_running = running
        self.supports_streaming = False
        self.settings = SimpleNamespace(model_name="test-model")
        self.last_messages = None
        self.last_prompt = None

    def build_prompt(self, messages):
        self.last_messages = messages
        return "<templated-prompt>"

    def generate(self, prompt, max_tokens, temperature, top_p, stop):
        self.last_prompt = prompt
        return SimpleNamespace(
            text="hello from forgeai",
            prompt_tokens=4,
            completion_tokens=3,
            total_tokens=7,
            finish_reason="stop",
        )


class FakeRole:
    def __init__(self, value: str) -> None:
        self.value = value


class FakeAuthManager:
    def __init__(self) -> None:
        self.keys = {
            "admin-key": ("key-admin", FakeRole("admin"), {"admin", "config", "inference", "models", "monitoring"}),
            "operator-key": ("key-operator", FakeRole("operator"), {"inference", "models", "monitoring"}),
            "viewer-key": ("key-viewer", FakeRole("viewer"), {"monitoring"}),
        }

    def validate_api_key(self, raw_key: str):
        if raw_key not in self.keys:
            return None
        key_id, role, _ = self.keys[raw_key]
        return SimpleNamespace(key_id=key_id, role=role)

    def verify_token(self, token: str):
        return None

    def permissions_for_role(self, role: FakeRole) -> set[str]:
        for _, known_role, permissions in self.keys.values():
            if known_role.value == role.value:
                return set(permissions)
        return set()


class FakeAuditLogger:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def log(self, **kwargs):
        self.events.append(kwargs)
        return kwargs


class FakeRateLimiter:
    def __init__(self, allow: bool = True, retry_after: int = 0) -> None:
        self.allow = allow
        self.retry_after = retry_after
        self.keys: list[str] = []

    def check(self, key: str):
        self.keys.append(key)
        return self.allow, self.retry_after


class ApiTests(unittest.IsolatedAsyncioTestCase):
    def async_client(self, app):
        transport = httpx.ASGITransport(app=app)
        return httpx.AsyncClient(transport=transport, base_url="http://testserver")

    def test_create_app_requires_auth_manager(self) -> None:
        with self.assertRaises(ValueError):
            create_app(enable_auth=True)

    async def test_readyz_reports_missing_engine_with_request_id(self) -> None:
        async with self.async_client(create_app()) as client:
            response = await client.get("/readyz")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["status"], "not_ready")
        self.assertIn("X-Request-ID", response.headers)

    async def test_chat_completion_uses_prompt_builder_and_records_metrics(self) -> None:
        engine = FakeEngine()
        async with self.async_client(create_app(engine=engine)) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "Say hello."},
                    ]
                },
            )
            metrics = await client.get("/metrics")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(engine.last_prompt, "<templated-prompt>")
        self.assertEqual(
            response.json()["choices"][0]["message"]["content"],
            "hello from forgeai",
        )
        self.assertIn("X-Request-ID", response.headers)
        self.assertEqual(metrics.status_code, 200)
        self.assertIn('forgeai_requests_total{method="POST",status="200"}', metrics.text)
        self.assertIn("forgeai_prompt_tokens_total", metrics.text)

    async def test_stream_requests_return_structured_error(self) -> None:
        """Streaming returns 501 when the engine does not support streaming."""
        async with self.async_client(create_app(engine=FakeEngine())) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                },
            )

        self.assertEqual(response.status_code, 501)
        body = response.json()
        self.assertEqual(body["error"]["type"], "http_error")
        self.assertIn("Streaming not supported", body["error"]["message"])
        self.assertEqual(body["error"]["request_id"], response.headers["X-Request-ID"])

    async def test_validation_errors_return_structured_response(self) -> None:
        async with self.async_client(create_app(engine=FakeEngine())) as client:
            response = await client.post("/v1/chat/completions", json={"messages": "wrong"})

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["type"], "validation_error")

    async def test_auth_blocks_requests_without_credentials_and_audits(self) -> None:
        audit_logger = FakeAuditLogger()
        app = create_app(
            engine=FakeEngine(),
            enable_auth=True,
            auth_manager=FakeAuthManager(),
            audit_logger=audit_logger,
        )
        async with self.async_client(app) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Hi"}]},
            )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["status_code"], 401)
        self.assertEqual(audit_logger.events[-1]["outcome"], "denied")

    async def test_viewer_can_access_metrics_but_not_inference(self) -> None:
        app = create_app(
            engine=FakeEngine(),
            enable_auth=True,
            auth_manager=FakeAuthManager(),
        )
        async with self.async_client(app) as client:
            metrics = await client.get("/metrics", headers={"X-API-Key": "viewer-key"})
            chat = await client.post(
                "/v1/chat/completions",
                headers={"X-API-Key": "viewer-key"},
                json={"messages": [{"role": "user", "content": "Hi"}]},
            )

        self.assertEqual(metrics.status_code, 200)
        self.assertEqual(chat.status_code, 403)
        self.assertIn("Permission 'inference'", chat.json()["error"]["message"])

    async def test_auth_accepts_admin_api_key(self) -> None:
        app = create_app(
            engine=FakeEngine(),
            enable_auth=True,
            auth_manager=FakeAuthManager(),
        )
        async with self.async_client(app) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={"X-API-Key": "admin-key"},
                json={"messages": [{"role": "user", "content": "Hi"}]},
            )

        self.assertEqual(response.status_code, 200)

    async def test_rate_limit_returns_429_and_retry_after(self) -> None:
        rate_limiter = FakeRateLimiter(allow=False, retry_after=7)
        app = create_app(
            engine=FakeEngine(),
            enable_auth=True,
            auth_manager=FakeAuthManager(),
            rate_limiter=rate_limiter,
        )
        async with self.async_client(app) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={"X-API-Key": "admin-key"},
                json={"messages": [{"role": "user", "content": "Hi"}]},
            )

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["Retry-After"], "7")
        self.assertEqual(rate_limiter.keys[0].split(":")[1], "inference")

    async def test_audit_logger_records_successful_access(self) -> None:
        audit_logger = FakeAuditLogger()
        app = create_app(
            engine=FakeEngine(),
            enable_auth=True,
            auth_manager=FakeAuthManager(),
            audit_logger=audit_logger,
        )
        async with self.async_client(app) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={"X-API-Key": "operator-key"},
                json={"messages": [{"role": "user", "content": "Hi"}]},
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            any(
                event["event_type"] == "access" and event["outcome"] == "success"
                for event in audit_logger.events
            )
        )


if __name__ == "__main__":
    unittest.main()
