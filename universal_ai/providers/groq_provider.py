"""Groq provider for UniversalAI."""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

import httpx

from universal_ai.exceptions import AuthenticationError, RateLimitError
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

logger = logging.getLogger("universal_ai.providers.groq")


class GroqProvider:
    name = "groq"
    version = "1.0.0"
    default_model = "mixtral-8x7b-32768"
    capabilities = ProviderCapabilities(
        chat=True,
        streaming=True,
        vision=False,
        audio=False,
        embeddings=False,
        tools=True,
        json_mode=True,
        system_message=True,
        max_tokens=32768,
        context_window=32768,
        supported_models=[
            "mixtral-8x7b-32768",
            "llama2-70b-4096",
            "llama3-8b-8192",
            "llama3-70b-8192",
        ],
    )

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.api_base = api_base or os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
        if not self.api_key:
            msg = "groq"
            raise AuthenticationError(msg, "GROQ_API_KEY not set")
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.api_base,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30.0,
            )
        return self._client

    async def chat(self, request: ChatRequest) -> ChatResponse:
        messages = [{"role": m.role.value, "content": m.content} for m in request.messages]
        body = {
            "model": request.model or self.default_model,
            "messages": messages,
            "temperature": request.temperature,
        }
        if request.max_tokens:
            body["max_tokens"] = request.max_tokens
        client = self._get_client()
        resp = await client.post("/chat/completions", json=body)
        if resp.status_code == 429:
            msg = "groq"
            raise RateLimitError(msg, "Rate limit exceeded")
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        usage_data = data.get("usage", {})
        usage = Usage(
            input_tokens=usage_data.get("prompt_tokens", 0),
            output_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )
        finish_reason = FinishReason.stop
        raw_finish = choice.get("finish_reason")
        if raw_finish:
            finish_map = {
                "stop": FinishReason.stop,
                "length": FinishReason.length,
                "tool_calls": FinishReason.tool_calls,
                "content_filter": FinishReason.content_filter,
            }
            finish_reason = finish_map.get(raw_finish, FinishReason.unknown)
        return ChatResponse(
            content=choice.get("message", {}).get("content", ""),
            model=data.get("model", body["model"]),
            provider="groq",
            usage=usage,
            finish_reason=finish_reason,
            raw=data,
        )

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        messages = [{"role": m.role.value, "content": m.content} for m in request.messages]
        body = {
            "model": request.model or self.default_model,
            "messages": messages,
            "temperature": request.temperature,
            "stream": True,
        }
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
                    if content:
                        yield ChatChunk(
                            content=content,
                            delta=content,
                            model=data.get("model", ""),
                        )
                except json.JSONDecodeError:
                    continue

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        from universal_ai.exceptions import ConfigurationError

        msg = "Groq does not support embeddings. Use OpenAI or Gemini."
        raise ConfigurationError(msg)

    async def health_check(self) -> dict:
        try:
            client = self._get_client()
            resp = await client.get("/models")
            return {"healthy": resp.status_code == 200}
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    async def get_latency(self) -> float:
        return 0.03

    def calculate_cost(self, estimated_tokens: int) -> float:
        return estimated_tokens * 0.0000003

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def __repr__(self) -> str:
        return f"GroqProvider(base_url={self.api_base!r})"
