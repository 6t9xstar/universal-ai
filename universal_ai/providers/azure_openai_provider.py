"""Azure OpenAI provider for UniversalAI."""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

import httpx

from universal_ai.exceptions import (
    AuthenticationError,
    ConfigurationError,
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
    ToolCall,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from universal_ai.models.request import ChatRequest, EmbedRequest

logger = logging.getLogger("universal_ai.providers.azure")


class AzureOpenAIProvider:
    """Provider implementation for Azure OpenAI Service."""

    name = "azure"
    version = "1.0.0"
    default_model = "gpt-4o"
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
        supported_models=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-35-turbo"],
    )

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str = "2024-10-01-preview",
        deployment_name: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.api_base = api_base or os.getenv("AZURE_OPENAI_API_BASE", "")
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-01-preview")
        self.deployment_name = deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")
        if not self.api_key:
            msg = "azure"
            raise AuthenticationError(msg, "AZURE_OPENAI_API_KEY not set")
        if not self.api_base:
            msg = "azure"
            raise ConfigurationError(msg, "AZURE_OPENAI_API_BASE not set")
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.api_base.rstrip("/"),
                headers={"api-key": self.api_key},
                timeout=30.0,
            )
        return self._client

    def _deployment_path(self, deployment: str | None = None) -> str:
        dep = deployment or self.deployment_name or self.default_model
        return f"/openai/deployments/{dep}"

    async def chat(self, request: ChatRequest) -> ChatResponse:
        messages = []
        system_content = None
        for m in request.messages:
            if m.role == MessageRole.system:
                system_content = m.content
            else:
                messages.append({"role": m.role.value, "content": m.content})

        body: dict = {
            "messages": messages,
            "temperature": request.temperature,
        }
        if system_content:
            body["messages"].insert(0, {"role": "system", "content": system_content})
        if request.max_tokens:
            body["max_tokens"] = request.max_tokens
        if request.top_p != 1.0:
            body["top_p"] = request.top_p
        if request.json_mode:
            body["response_format"] = {"type": "json_object"}
        if request.tools:
            body["tools"] = [
                t._tool_definition if hasattr(t, "_tool_definition") else t for t in request.tools
            ]

        client = self._get_client()
        path = self._deployment_path(request.model or None)
        resp = await client.post(
            f"{path}/chat/completions",
            params={"api-version": self.api_version},
            json=body,
        )
        if resp.status_code == 401:
            msg = "azure"
            raise AuthenticationError(msg, "Invalid API key")
        if resp.status_code == 404:
            msg = "azure"
            raise ModelNotFoundError(msg, "Deployment not found")
        if resp.status_code == 429:
            msg = "azure"
            raise RateLimitError(msg, "Rate limit exceeded")
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        finish_map = {
            "stop": FinishReason.stop,
            "length": FinishReason.length,
            "tool_calls": FinishReason.tool_calls,
            "content_filter": FinishReason.content_filter,
        }
        finish_reason = finish_map.get(choice.get("finish_reason", "stop"), FinishReason.unknown)
        usage_data = data.get("usage", {})
        usage = Usage(
            input_tokens=usage_data.get("prompt_tokens", 0),
            output_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )
        content = choice.get("message", {}).get("content") or ""
        tool_calls = []
        for tc in choice.get("message", {}).get("tool_calls", []):
            tool_calls.append(
                ToolCall(
                    id=tc["id"],
                    function_name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                )
            )
        return ChatResponse(
            content=content,
            model=data.get("model", self.deployment_name or self.default_model),
            provider="azure",
            usage=usage,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            raw=data,
        )

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        messages = []
        system_content = None
        for m in request.messages:
            if m.role == MessageRole.system:
                system_content = m.content
            else:
                messages.append({"role": m.role.value, "content": m.content})
        body: dict = {
            "messages": messages,
            "temperature": request.temperature,
            "stream": True,
        }
        if system_content:
            body["messages"].insert(0, {"role": "system", "content": system_content})
        if request.max_tokens:
            body["max_tokens"] = request.max_tokens

        client = self._get_client()
        path = self._deployment_path(request.model or None)
        async with client.stream(
            "POST",
            f"{path}/chat/completions",
            params={"api-version": self.api_version},
            json=body,
        ) as resp:
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
        body = {"input": request.input if isinstance(request.input, str) else request.input[0]}
        client = self._get_client()
        path = self._deployment_path(request.model or None)
        resp = await client.post(
            f"{path}/embeddings",
            params={"api-version": self.api_version},
            json=body,
        )
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
            model=data.get("model", self.deployment_name or self.default_model),
            provider="azure",
            usage=usage,
            raw=data,
        )

    async def health_check(self) -> dict:
        try:
            client = self._get_client()
            path = self._deployment_path()
            resp = await client.get(
                f"{path}/chat/completions",
                params={"api-version": self.api_version},
            )
            return {"healthy": resp.status_code in (200, 404)}
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    async def get_latency(self) -> float:
        return 0.12

    def calculate_cost(self, estimated_tokens: int) -> float:
        return estimated_tokens * 0.000002

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def __repr__(self) -> str:
        return f"AzureOpenAIProvider(base_url={self.api_base!r})"
