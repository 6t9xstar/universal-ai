"""OpenRouter provider for UniversalAI."""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

import httpx

from universal_ai.exceptions import (
    AuthenticationError,
    ModelNotFoundError,
    RateLimitError,
)
from universal_ai.models.common import (
    FinishReason,
    ProviderCapabilities,
    Usage,
)
from universal_ai.models.response import (
    ChatChunk,
    ChatResponse,
    EmbedResponse,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from universal_ai.models.request import ChatRequest, EmbedRequest

logger = logging.getLogger("universal_ai.providers.openrouter")


class OpenRouterProvider:
    name = "openrouter"
    version = "1.0.0"
    default_model = "openai/gpt-3.5-turbo"
    capabilities = ProviderCapabilities(
        chat=True,
        streaming=True,
        vision=True,
        audio=False,
        embeddings=True,
        tools=True,
        json_mode=True,
        system_message=True,
        max_tokens=128000,
        context_window=128000,
        supported_models=[],
    )

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.api_base = api_base or os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
        if not self.api_key:
            msg = "openrouter"
            raise AuthenticationError(msg, "OPENROUTER_API_KEY not set")
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.api_base,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "https://github.com/6t9xstar/universal-ai",
                    "X-Title": "UniversalAI",
                },
                timeout=30.0,
            )
        return self._client

    async def chat(self, request: ChatRequest) -> ChatResponse:
        messages = [{"role": m.role.value, "content": m.content} for m in request.messages]
        body: dict = {
            "model": request.model or self.default_model,
            "messages": messages,
            "temperature": request.temperature,
        }
        if request.max_tokens:
            body["max_tokens"] = request.max_tokens
        if request.stop:
            body["stop"] = request.stop
        if request.json_mode:
            body["response_format"] = {"type": "json_object"}
        client = self._get_client()
        resp = await client.post("/chat/completions", json=body)
        if resp.status_code == 401:
            msg = "openrouter"
            raise AuthenticationError(msg, "Invalid API key")
        if resp.status_code == 404:
            msg = "openrouter"
            raise ModelNotFoundError(msg, f"Model {body['model']} not found")
        if resp.status_code == 429:
            msg = "openrouter"
            raise RateLimitError(msg, "Rate limit exceeded")
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        finish_map = {
            "stop": FinishReason.stop,
            "length": FinishReason.length,
            "tool_calls": FinishReason.tool_calls,
        }
        finish_reason = finish_map.get(choice.get("finish_reason", "stop"), FinishReason.unknown)
        usage_data = data.get("usage", {})
        usage = Usage(
            input_tokens=usage_data.get("prompt_tokens", 0),
            output_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )
        content = choice.get("message", {}).get("content") or ""
        return ChatResponse(
            content=content,
            model=data.get("model", body["model"]),
            provider="openrouter",
            usage=usage,
            finish_reason=finish_reason,
            raw=data,
        )

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        messages = [{"role": m.role.value, "content": m.content} for m in request.messages]
        body: dict = {
            "model": request.model or self.default_model,
            "messages": messages,
            "temperature": request.temperature,
            "stream": True,
        }
        if request.max_tokens:
            body["max_tokens"] = request.max_tokens
        client = self._get_client()
        async with client.stream("POST", "/chat/completions", json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    data = json.loads(payload)
                    choice = data["choices"][0]
                    delta = choice.get("delta", {})
                    content = delta.get("content", "")
                    finish_reason = None
                    if choice.get("finish_reason"):
                        finish_reason = FinishReason(choice["finish_reason"])
                    yield ChatChunk(
                        content=content,
                        delta=content,
                        model=data.get("model", ""),
                        finish_reason=finish_reason,
                    )
                except json.JSONDecodeError:
                    continue

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        body = {
            "model": request.model,
            "input": request.input if isinstance(request.input, str) else request.input[0],
        }
        client = self._get_client()
        resp = await client.post("/embeddings", json=body)
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("data", [])
        vector = embeddings[0].get("embedding", []) if embeddings else []
        usage_data = data.get("usage", {})
        usage = Usage(
            input_tokens=usage_data.get("prompt_tokens", 0),
            output_tokens=0,
            total_tokens=usage_data.get("total_tokens", 0),
        )
        return EmbedResponse(
            vector=vector,
            model=data.get("model", request.model),
            provider="openrouter",
            usage=usage,
            raw=data,
        )

    async def health_check(self) -> dict:
        try:
            client = self._get_client()
            resp = await client.get("/models")
            return {"healthy": resp.status_code == 200}
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    async def get_latency(self) -> float:
        return 0.2

    def calculate_cost(self, estimated_tokens: int) -> float:
        return estimated_tokens * 0.000001

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def __repr__(self) -> str:
        return f"OpenRouterProvider(base_url={self.api_base!r})"
