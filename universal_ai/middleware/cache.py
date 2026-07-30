from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod

from universal_ai.middleware.base import Middleware, Request, Response

logger = logging.getLogger("universal_ai.middleware.cache")
CACHE_HIT_KEY = "_cache_hit"


class CacheBackend(ABC):
    @abstractmethod
    async def get(self, key: str) -> Response | None: ...

    @abstractmethod
    async def set(self, key: str, value: Response, ttl: int) -> None: ...

    @abstractmethod
    async def delete(self, key: str) -> None: ...

    @abstractmethod
    async def clear(self) -> None: ...


class MemoryCacheBackend(CacheBackend):
    def __init__(self) -> None:
        self._cache: dict[str, tuple[float, Response]] = {}
        import asyncio

        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Response | None:
        async with self._lock:
            if key in self._cache:
                expires, value = self._cache[key]
                if time.time() < expires:
                    return value
                del self._cache[key]
            return None

    async def set(self, key: str, value: Response, ttl: int) -> None:
        async with self._lock:
            self._cache[key] = (time.time() + ttl, value)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._cache.pop(key, None)

    async def clear(self) -> None:
        async with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)


class CacheMiddleware(Middleware):
    def __init__(self, backend: CacheBackend | None = None, ttl: int = 3600) -> None:
        self.backend = backend or MemoryCacheBackend()
        self.ttl = ttl

    def _make_key(self, request: Request) -> str:
        data = json.dumps(
            {
                "provider": request.provider,
                "model": request.model,
                "messages": request.messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "top_p": request.top_p,
                "tools": request.tools,
                "stop": request.stop,
                "frequency_penalty": request.frequency_penalty,
                "presence_penalty": request.presence_penalty,
                "json_mode": request.json_mode,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(data.encode()).hexdigest()

    async def process_request(self, request: Request) -> Request:
        key = self._make_key(request)
        cached = await self.backend.get(key)
        if cached is not None:
            logger.debug("Cache hit for %s/%s", request.provider, request.model)
            request.kwargs[CACHE_HIT_KEY] = cached
        return request

    async def process_response(self, request: Request, response: Response) -> Response:
        if CACHE_HIT_KEY in request.kwargs:
            return response
        key = self._make_key(request)
        await self.backend.set(key, response, self.ttl)
        logger.debug("Cached response for %s/%s", request.provider, request.model)
        return response

    async def process_error(self, request: Request, error: Exception) -> Exception:
        return error
