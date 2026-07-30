from __future__ import annotations

import logging

from universal_ai.middleware.base import Middleware, Request, Response
from universal_ai.utils.costs import CostCalculator

logger = logging.getLogger("universal_ai.middleware.cost")


class CostTrackingMiddleware(Middleware):
    def __init__(self) -> None:
        self._total_cost: float = 0.0
        self._request_count: int = 0

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def request_count(self) -> int:
        return self._request_count

    def reset(self) -> None:
        self._total_cost = 0.0
        self._request_count = 0

    async def process_request(self, request: Request) -> Request:
        return request

    async def process_response(self, request: Request, response: Response) -> Response:
        cost = CostCalculator.calculate(
            provider=request.provider,
            model=request.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        response.metadata["cost"] = cost
        response.usage.estimated_cost = cost
        self._total_cost += cost
        self._request_count += 1
        logger.debug(
            "Cost: $%.6f for %s/%s (total: $%.6f)",
            cost,
            request.provider,
            request.model,
            self._total_cost,
        )
        return response

    async def process_error(self, request: Request, error: Exception) -> Exception:
        return error
