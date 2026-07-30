from __future__ import annotations

import logging
import time

from universal_ai.middleware.base import Middleware, Request, Response

logger = logging.getLogger("universal_ai.middleware")


class LoggingMiddleware(Middleware):
    def __init__(self, log_level: int = logging.INFO) -> None:
        self._log_level = log_level

    async def process_request(self, request: Request) -> Request:
        logger.log(
            self._log_level,
            "Request -> %s/%s (stream=%s, tools=%d, image=%s, temp=%.2f)",
            request.provider,
            request.model,
            request.stream,
            len(request.tools),
            bool(request.image),
            request.temperature,
        )
        request.kwargs["_start_time"] = time.monotonic()
        return request

    async def process_response(self, request: Request, response: Response) -> Response:
        elapsed = 0.0
        start_time = request.kwargs.get("_start_time")
        if start_time:
            elapsed = time.monotonic() - start_time
        logger.log(
            self._log_level,
            "Response <- %s/%s (tokens=%d, cost=$%.6f, time=%.3fs, finish=%s)",
            request.provider,
            request.model,
            response.usage.total_tokens,
            response.usage.estimated_cost,
            elapsed,
            response.finish_reason,
        )
        return response

    async def process_error(self, request: Request, error: Exception) -> Exception:
        logger.error(
            "Error <- %s/%s: %s",
            request.provider,
            request.model,
            error,
        )
        return error
