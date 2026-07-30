"""Anthropic provider for UniversalAI."""

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
    MessageRole,
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

logger = logging.getLogger("universal_ai.providers.anthropic")


class AnthropicProvider:
    name = "anthropic"
    version = "1.0.0"
    default_model = "claude-3-sonnet-20240229"
    capabilities = ProviderCapabilities(
        chat=True,
        streaming=True,
        vision=True,
        audio=False,
        embeddings=False,
        tools=True,
        json_mode=False,
        system_message=True,
        max_tokens=200000,
        context_window=200000,
        supported_models=[
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
        ],
    )

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.api_base = api_base or os.getenv("ANTHROPIC_API_BASE", "https://api.anthropic.com")
        if not self.api_key:
            msg = "anthropic"
            raise AuthenticationError(msg, "ANTHROPIC_API_KEY not set")
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.api_base,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                },
                timeout=30.0,
            )
        return self._client

    async def chat(self, request: ChatRequest) -> ChatResponse:
        system = ""
        messages = []
        for m in request.messages:
            if m.role == MessageRole.system:
                system = m.content
            else:
                messages.append({"role": m.role.value, "content": m.content})
        body: dict = {
            "model": request.model or self.default_model,
            "messages": messages,
            "max_tokens": request.max_tokens or 4096,
            "temperature": request.temperature,
        }
        if system:
            body["system"] = system
        if request.stop:
            body["stop_sequences"] = request.stop
        client = self._get_client()
        resp = await client.post("/v1/messages", json=body)
        if resp.status_code == 401:
            msg = "anthropic"
            raise AuthenticationError(msg, "Invalid API key")
        if resp.status_code == 404:
            msg = "anthropic"
            raise ModelNotFoundError(msg, f"Model {body['model']} not found")
        if resp.status_code == 429:
            msg = "anthropic"
            raise RateLimitError(msg, "Rate limit exceeded")
        resp.raise_for_status()
        data = resp.json()
        content = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")
        stop_reason = data.get("stop_reason", "end_turn")
        finish_map = {
            "end_turn": FinishReason.stop,
            "max_tokens": FinishReason.length,
            "stop_sequence": FinishReason.stop,
        }
        finish_reason = finish_map.get(stop_reason, FinishReason.unknown)
        usage_data = data.get("usage", {})
        usage = Usage(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
        )
        usage.total_tokens = usage.input_tokens + usage.output_tokens
        return ChatResponse(
            content=content,
            model=data.get("model", body["model"]),
            provider="anthropic",
            usage=usage,
            finish_reason=finish_reason,
            raw=data,
        )

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        system = ""
        messages = []
        for m in request.messages:
            if m.role == MessageRole.system:
                system = m.content
            else:
                messages.append({"role": m.role.value, "content": m.content})
        body: dict = {
            "model": request.model or self.default_model,
            "messages": messages,
            "max_tokens": request.max_tokens or 4096,
            "temperature": request.temperature,
            "stream": True,
        }
        if system:
            body["system"] = system
        client = self._get_client()
        async with client.stream("POST", "/v1/messages", json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    data = json.loads(line[6:])
                    event_type = data.get("type", "")
                    if event_type == "content_block_delta":
                        delta = data.get("delta", {})
                        text = delta.get("text", "")
                        if text:
                            yield ChatChunk(content=text, delta=text, model="")
                    elif event_type == "message_stop":
                        usage_data = data.get("usage", {})
                        input_tokens = usage_data.get("input_tokens", 0)
                        output_tokens = usage_data.get("output_tokens", 0)
                        yield ChatChunk(
                            finish_reason=FinishReason.stop,
                            usage=Usage(
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                                total_tokens=input_tokens + output_tokens,
                            ),
                        )
                except json.JSONDecodeError:
                    continue

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        from universal_ai.exceptions import ConfigurationError

        msg = "Anthropic does not support embeddings. Use OpenAI or Gemini."
        raise ConfigurationError(msg)

    async def health_check(self) -> dict:
        try:
            client = self._get_client()
            resp = await client.post(
                "/v1/messages",
                json={
                    "model": self.default_model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                },
            )
            return {"healthy": resp.status_code in (200, 400)}
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    async def get_latency(self) -> float:
        return 0.15

    def calculate_cost(self, estimated_tokens: int) -> float:
        return estimated_tokens * 0.000003

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def __repr__(self) -> str:
        return f"AnthropicProvider(base_url={self.api_base!r})"
