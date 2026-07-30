from __future__ import annotations

import asyncio

from universal_ai import AI
from universal_ai.middleware import (
    CacheMiddleware,
    CircuitBreakerMiddleware,
    CostTrackingMiddleware,
    LoggingMiddleware,
    RateLimitMiddleware,
    RetryMiddleware,
)


async def middleware_demo():
    ai = AI()
    ai.add_middleware(RetryMiddleware(max_retries=3))
    ai.add_middleware(CacheMiddleware(ttl=300))
    ai.add_middleware(RateLimitMiddleware(requests_per_minute=60))
    ai.add_middleware(CircuitBreakerMiddleware(failure_threshold=5))
    ai.add_middleware(CostTrackingMiddleware())
    ai.add_middleware(LoggingMiddleware())

    response = await ai.chat("Hello, world!")
    print(f"Response: {response.text}")
    print(f"Tokens used: {response.tokens}")


if __name__ == "__main__":
    asyncio.run(middleware_demo())
