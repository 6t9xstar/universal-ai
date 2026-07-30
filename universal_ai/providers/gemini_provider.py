"""Gemini provider for UniversalAI."""

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

logger = logging.getLogger("universal_ai.providers.gemini")


class GeminiProvider:
    name = "gemini"
    version = "1.0.0"
    default_model = "gemini-pro"
    capabilities = ProviderCapabilities(
        chat=True,
        streaming=True,
        vision=True,
        audio=False,
        embeddings=True,
        tools=True,
        json_mode=False,
        system_message=True,
        max_tokens=8192,
        context_window=32768,
        supported_models=["gemini-pro", "gemini-1.5-pro", "gemini-1.5-flash"],
    )

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.api_base = api_base or os.getenv(
            "GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta"
        )
        if not self.api_key:
            msg = "gemini"
            raise AuthenticationError(msg, "GEMINI_API_KEY not set")
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(base_url=self.api_base, timeout=60.0)
        return self._client

    async def chat(self, request: ChatRequest) -> ChatResponse:
        messages = []
        system_instruction = None
        for m in request.messages:
            if m.role.value == "system":
                system_instruction = m.content
            else:
                messages.append({"role": m.role.value, "parts": [{"text": m.content}]})
        body: dict = {
            "contents": messages,
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens or 2048,
            },
        }
        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        client = self._get_client()
        model = request.model or self.default_model
        resp = await client.post(
            f"/models/{model}:generateContent",
            params={"key": self.api_key},
            json=body,
        )
        if resp.status_code == 400:
            data = resp.json()
            error_msg = data.get("error", {}).get("message", "")
            if "not found" in error_msg.lower():
                msg = "gemini"
                raise ModelNotFoundError(msg, f"Model {model} not found")
        if resp.status_code == 403:
            msg = "gemini"
            raise AuthenticationError(msg, "Invalid API key")
        if resp.status_code == 429:
            msg = "gemini"
            raise RateLimitError(msg, "Rate limit exceeded")
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return ChatResponse(
                content="",
                model=model,
                provider="gemini",
                usage=Usage(),
                finish_reason=FinishReason.stop,
                raw=data,
            )
        candidate = candidates[0]
        content = ""
        parts = candidate.get("content", {}).get("parts", [])
        for part in parts:
            content += part.get("text", "")
        finish_reason = FinishReason.stop
        raw_reason = candidate.get("finishReason", "")
        reason_map = {
            "STOP": FinishReason.stop,
            "MAX_TOKENS": FinishReason.length,
            "SAFETY": FinishReason.content_filter,
            "RECITATION": FinishReason.content_filter,
        }
        finish_reason = reason_map.get(raw_reason, FinishReason.unknown)
        usage_data = data.get("usageMetadata", {})
        usage = Usage(
            input_tokens=usage_data.get("promptTokenCount", 0),
            output_tokens=usage_data.get("candidatesTokenCount", 0),
            total_tokens=usage_data.get("totalTokenCount", 0),
        )
        return ChatResponse(
            content=content,
            model=model,
            provider="gemini",
            usage=usage,
            finish_reason=finish_reason,
            raw=data,
        )

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        messages = []
        system_instruction = None
        for m in request.messages:
            if m.role.value == "system":
                system_instruction = m.content
            else:
                messages.append({"role": m.role.value, "parts": [{"text": m.content}]})
        body: dict = {
            "contents": messages,
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens or 2048,
            },
        }
        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        client = self._get_client()
        model = request.model or self.default_model
        async with client.stream(
            "POST",
            f"/models/{model}:streamGenerateContent",
            params={"key": self.api_key, "alt": "sse"},
            json=body,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if not payload.strip():
                    continue
                try:
                    data = json.loads(payload)
                    candidates = data.get("candidates", [])
                    if not candidates:
                        continue
                    candidate = candidates[0]
                    parts = candidate.get("content", {}).get("parts", [])
                    for part in parts:
                        text = part.get("text", "")
                        if text:
                            yield ChatChunk(content=text, delta=text, model=model)
                    if candidate.get("finishReason"):
                        usage_data = data.get("usageMetadata", {})
                        yield ChatChunk(
                            finish_reason=FinishReason.stop,
                            usage=Usage(
                                input_tokens=usage_data.get("promptTokenCount", 0),
                                output_tokens=usage_data.get("candidatesTokenCount", 0),
                                total_tokens=usage_data.get("totalTokenCount", 0),
                            ),
                        )
                except json.JSONDecodeError:
                    continue

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        body = {
            "model": f"models/{request.model}",
            "content": {
                "parts": [
                    {"text": request.input if isinstance(request.input, str) else request.input[0]}
                ]
            },
        }
        client = self._get_client()
        resp = await client.post(
            f"/models/{request.model}:embedContent",
            params={"key": self.api_key},
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()
        vector = data.get("embedding", {}).get("values", [])
        return EmbedResponse(
            vector=vector,
            model=request.model,
            provider="gemini",
            usage=Usage(),
            raw=data,
        )

    async def health_check(self) -> dict:
        try:
            client = self._get_client()
            resp = await client.get(
                f"/models/{self.default_model}",
                params={"key": self.api_key},
            )
            return {"healthy": resp.status_code == 200}
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    async def get_latency(self) -> float:
        return 0.12

    def calculate_cost(self, estimated_tokens: int) -> float:
        return estimated_tokens * 0.000001

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def __repr__(self) -> str:
        return f"GeminiProvider(base_url={self.api_base!r})"
