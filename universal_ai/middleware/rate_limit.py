from __future__ import annotations

import asyncio
import logging
import time

from universal_ai.middleware.base import Middleware, Request, Response

logger = logging.getLogger("universal_ai.middleware.rate_limit")


class RateLimitMiddleware(Middleware):
    def __init__(
        self,
        requests_per_minute: int = 60,
        burst: int | None = None,
    ) -> None:
        self.rate = requests_per_minute
        self.burst = burst or min(requests_per_minute, 20)
        self._tokens = self.burst
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        new_tokens = elapsed * (self.rate / 60.0)
        self._tokens = min(self.burst, self._tokens + new_tokens)
        self._last_refill = now

    async def _acquire(self) -> None:
        async with self._lock:
            self._refill()
            while self._tokens < 1:
                wait_time = (1 - self._tokens) * (60.0 / self.rate)
                logger.debug("Rate limit reached, waiting %.2fs", wait_time)
                await asyncio.sleep(wait_time)
                self._refill()
            self._tokens -= 1

    async def process_request(self, request: Request) -> Request:
        await self._acquire()
        request.kwargs["_rate_limited"] = True
        return request

    async def process_response(self, request: Request, response: Response) -> Response:
        request.kwargs.pop("_rate_limited", None)
        return response

    async def process_error(self, request: Request, error: Exception) -> Exception:
        request.kwargs.pop("_rate_limited", None)
        return error
