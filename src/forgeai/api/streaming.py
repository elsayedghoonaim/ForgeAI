"""Streaming helpers guaranteeing lease release, generator close and request abort."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from starlette.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

ErrorFrames = Callable[[Exception], Awaitable[list[str]]]


class StreamGuard:
    """Idempotent end-of-stream cleanup: abort (if unfinished), close generator, release lease."""

    def __init__(
        self,
        inner: Any,
        release: Callable[[], Awaitable[None]],
        abort: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.inner = inner
        self._release = release
        self._abort = abort
        self.completed = False
        self._task: asyncio.Future[None] | None = None

    async def _run(self) -> None:
        if not self.completed and self._abort is not None:
            with contextlib.suppress(Exception):
                await self._abort()
        aclose = getattr(self.inner, "aclose", None)
        if aclose is not None:
            with contextlib.suppress(Exception):
                await aclose()
        with contextlib.suppress(Exception):
            await self._release()

    async def finalize(self) -> None:
        """Run cleanup exactly once; it completes even if the caller is cancelled."""
        if self._task is None:
            self._task = asyncio.ensure_future(self._run())
        await asyncio.shield(self._task)


class GuardedStreamingResponse(StreamingResponse):
    """StreamingResponse that always finalizes its guard, even if the response never starts."""

    def __init__(self, content: AsyncIterator[str], guard: StreamGuard, **kwargs: Any) -> None:
        super().__init__(content, **kwargs)
        self._guard = guard

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._guard.finalize()


def guarded_stream_response(
    inner: AsyncIterator[str],
    *,
    release: Callable[[], Awaitable[None]],
    abort: Callable[[], Awaitable[None]] | None,
    on_error: ErrorFrames,
    media_type: str,
) -> StreamingResponse:
    """Wrap ``inner`` so the lease is released and the engine request aborted in all cases.

    On a mid-stream exception, ``on_error`` supplies the final frames (e.g. an error
    event) and the stream ends normally. On cancellation/disconnect the in-flight
    engine request is aborted.
    """
    guard = StreamGuard(inner, release, abort)

    async def body() -> AsyncIterator[str]:
        try:
            async for item in inner:
                yield item
            guard.completed = True
        except Exception as err:
            guard.completed = True  # the engine request already failed; nothing to abort
            for frame in await on_error(err):
                yield frame
        finally:
            await guard.finalize()

    return GuardedStreamingResponse(body(), guard, media_type=media_type)


def sse_error_frames(err: Exception) -> list[str]:
    """OpenAI-style SSE error event followed by the terminal [DONE] marker."""
    payload = {"error": {"message": str(err), "type": "server_error", "code": None}}
    return [f"data: {json.dumps(payload)}\n\n", "data: [DONE]\n\n"]
