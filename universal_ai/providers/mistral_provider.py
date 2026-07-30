"""Mistral provider for UniversalAI."""

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

logger = logging.getLogger("universal_ai.providers.mistral")


class MistralProvider:
    name = "mistral"
    version = "1.0.0"
    default_model = "mistral-small-latest"
    capabilities = ProviderCapabilities(
        chat=True,
        streaming=True,
        vision=False,
        audio=False,
        embeddings=True,
        tools=True,
        json_mode=False,
        system_message=True,
        max_tokens=8192,
        context_window=32768,
        supported_models=[
            "mistral-large-latest",
            "mistral-medium-latest",
            "mistral-small-latest",
            "mistral-embed",
        ],
    )

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.api_base = api_base or os.getenv("MISTRAL_API_BASE", "https://api.mistral.ai/v1")
        if not self.api_key:
            msg = "mistral"
            raise AuthenticationError(msg, "MISTRAL_API_KEY not set")
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
        messages = []
        system_content = None
        for m in request.messages:
            if m.role.value == "system":
                system_content = m.content
            else:
                messages.append({"role": m.role.value, "content": m.content})
        body: dict = {
            "model": request.model or self.default_model,
            "messages": messages,
            "temperature": request.temperature,
        }
        if system_content:
            body["system"] = system_content
        if request.max_tokens:
            body["max_tokens"] = request.max_tokens
        if request.stop:
            body["stop"] = request.stop
        client = self._get_client()
        resp = await client.post("/chat/completions", json=body)
        if resp.status_code == 401:
            msg = "mistral"
            raise AuthenticationError(msg, "Invalid API key")
        if resp.status_code == 404:
            msg = "mistral"
            raise ModelNotFoundError(msg, f"Model {body['model']} not found")
        if resp.status_code == 429:
            msg = "mistral"
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
        raw_reason = choice.get("finish_reason")
        if raw_reason:
            reason_map = {
                "stop": FinishReason.stop,
                "length": FinishReason.length,
                "model_length": FinishReason.length,
                "tool_calls": FinishReason.tool_calls,
                "content_filter": FinishReason.content_filter,
            }
            finish_reason = reason_map.get(raw_reason, FinishReason.unknown)
        return ChatResponse(
            content=choice.get("message", {}).get("content", ""),
            model=data.get("model", body["model"]),
            provider="mistral",
            usage=usage,
            finish_reason=finish_reason,
            raw=data,
        )

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        messages = []
        system_content = None
        for m in request.messages:
            if m.role.value == "system":
                system_content = m.content
            else:
                messages.append({"role": m.role.value, "content": m.content})
        body: dict = {
            "model": request.model or self.default_model,
            "messages": messages,
            "temperature": request.temperature,
            "stream": True,
        }
        if system_content:
            body["system"] = system_content
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
            provider="mistral",
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
        return 0.1

    def calculate_cost(self, estimated_tokens: int) -> float:
        return estimated_tokens * 0.000002

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def __repr__(self) -> str:
        return f"MistralProvider(base_url={self.api_base!r})"
