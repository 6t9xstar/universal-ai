from __future__ import annotations

import logging
import sys
from typing import Any

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import httpx

logger = logging.getLogger("universal_ai.client")


class HttpClient:
    def __init__(
        self,
        timeout: float = 30.0,
        base_url: str | None = None,
        max_connections: int = 100,
        max_keepalive: int = 20,
    ) -> None:
        self._timeout = timeout
        self._base_url = base_url
        self._max_connections = max_connections
        self._max_keepalive = max_keepalive
        self._client: httpx.AsyncClient | None = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                base_url=self._base_url,
                limits=httpx.Limits(
                    max_connections=self._max_connections,
                    max_keepalive_connections=self._max_keepalive,
                    keepalive_expiry=30,
                ),
            )
        return self._client

    async def get(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        client = self._ensure_client()
        return await client.get(url, headers=headers, params=params)

    async def post(
        self,
        url: str,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        content: bytes | None = None,
    ) -> httpx.Response:
        client = self._ensure_client()
        return await client.post(url, json=json, headers=headers, content=content)

    def stream(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
    ) -> httpx.Response:
        client = self._ensure_client()
        return client.stream(method, url, headers=headers, json=json)

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: Any,
    ) -> None:
        await self.close()

    def __repr__(self) -> str:
        status = "open" if self._client and not self._client.is_closed else "closed"
        return f"HttpClient(base_url={self._base_url!r}, status={status})"
