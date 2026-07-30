from __future__ import annotations

import logging
import time

from universal_ai.exceptions import CircuitBreakerOpenError
from universal_ai.middleware.base import Middleware, Request, Response

logger = logging.getLogger("universal_ai.middleware.circuit_breaker")

STATE_CLOSED = "closed"
STATE_OPEN = "open"
STATE_HALF_OPEN = "half-open"


class CircuitBreakerMiddleware(Middleware):
    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60.0) -> None:
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._state = STATE_CLOSED

    @property
    def state(self) -> str:
        if self._state == STATE_OPEN:
            if time.time() - self._last_failure_time > self.reset_timeout:
                self._state = STATE_HALF_OPEN
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    @property
    def is_closed(self) -> bool:
        return self.state == STATE_CLOSED

    @property
    def is_open(self) -> bool:
        return self.state == STATE_OPEN

    async def process_request(self, request: Request) -> Request:
        if self.state == STATE_OPEN:
            elapsed = time.time() - self._last_failure_time
            remaining = self.reset_timeout - elapsed
            raise CircuitBreakerOpenError(request.provider, remaining)
        return request

    async def process_response(self, request: Request, response: Response) -> Response:
        self._failure_count = 0
        self._state = STATE_CLOSED
        return response

    async def process_error(self, request: Request, error: Exception) -> Exception:
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self.failure_threshold:
            self._state = STATE_OPEN
            logger.warning(
                "Circuit breaker OPEN for %s after %d failures",
                request.provider,
                self._failure_count,
            )
        return error

    def reset(self) -> None:
        self._failure_count = 0
        self._state = STATE_CLOSED
        self._last_failure_time = 0.0
