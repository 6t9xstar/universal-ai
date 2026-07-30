from __future__ import annotations

import asyncio
import logging
import random

from universal_ai.exceptions import RateLimitError, TimeoutError
from universal_ai.middleware.base import Middleware, Request, Response

logger = logging.getLogger("universal_ai.middleware.retry")


class RetryMiddleware(Middleware):
    RETRYABLE_ERRORS = (RateLimitError, TimeoutError, ConnectionError, OSError)

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
    ) -> None:
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter

    async def process_request(self, request: Request) -> Request:
        if "_retry_attempts" not in request.kwargs:
            request.kwargs["_retry_attempts"] = 0
        return request

    async def process_response(self, request: Request, response: Response) -> Response:
        request.kwargs.pop("_retry_attempts", None)
        request.kwargs.pop("_start_time", None)
        return response

    async def process_error(self, request: Request, error: Exception) -> Exception | None:
        attempts = request.kwargs.get("_retry_attempts", 0)

        if not isinstance(error, self.RETRYABLE_ERRORS):
            return error

        if attempts >= self.max_retries:
            logger.warning(
                "Max retries (%d) exceeded for %s/%s: %s",
                self.max_retries,
                request.provider,
                request.model,
                error,
            )
            return error

        delay = self.base_delay * (2**attempts)
        if self.jitter:
            delay = random.uniform(0, delay)
        delay = min(delay, self.max_delay)

        if isinstance(error, RateLimitError) and error.retry_after:
            delay = max(delay, error.retry_after)

        request.kwargs["_retry_attempts"] = attempts + 1

        logger.info(
            "Retrying %s/%s in %.1fs (attempt %d/%d)",
            request.provider,
            request.model,
            delay,
            attempts + 1,
            self.max_retries,
        )

        await asyncio.sleep(delay)
        return None
