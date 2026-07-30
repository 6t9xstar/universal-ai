from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from universal_ai.models.common import Usage

logger = logging.getLogger("universal_ai.middleware")


@dataclass
class MiddlewareContext:
    request: Any
    response: Any = None
    error: Exception | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    is_retry: bool = False
    retry_attempt: int = 0
    cache_hit: bool = False
    provider: str = ""
    model: str = ""


class Request(BaseModel):
    provider: str
    model: str
    messages: list[dict]
    temperature: float = 0.7
    max_tokens: int | None = None
    top_p: float = 1.0
    stream: bool = False
    json_mode: bool = False
    image: str | bytes | None = None
    tools: list[dict] = Field(default_factory=list)
    stop: list[str] = Field(default_factory=list)
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    kwargs: dict[str, Any] = Field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Request(provider={self.provider!r}, model={self.model!r}, stream={self.stream})"


class Response(BaseModel):
    content: str
    usage: Usage = Field(default_factory=Usage)
    metadata: dict[str, Any] = Field(default_factory=dict)
    finish_reason: str = "stop"
    tool_calls: list[dict] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)
    provider: str = ""
    model: str = ""

    def __repr__(self) -> str:
        preview = self.content[:30] + "..." if len(self.content) > 30 else self.content
        return f"Response(content={preview!r}, tokens={self.usage.total_tokens})"


class Middleware(ABC):
    @abstractmethod
    async def process_request(self, request: Request) -> Request:
        return request

    @abstractmethod
    async def process_response(self, request: Request, response: Response) -> Response:
        return response

    @abstractmethod
    async def process_error(self, request: Request, error: Exception) -> Exception | None:
        return error


class MiddlewareChain:
    def __init__(self) -> None:
        self._middleware: list[Middleware] = []

    def add(self, middleware: Middleware) -> MiddlewareChain:
        self._middleware.append(middleware)
        return self

    def clear(self) -> None:
        self._middleware.clear()

    @property
    def middleware(self) -> list[Middleware]:
        return list(self._middleware)

    def build_context(self, request: Request, error: Exception | None = None) -> MiddlewareContext:
        return MiddlewareContext(
            request=request,
            error=error,
            provider=request.provider,
            model=request.model,
            cache_hit=request.kwargs.get("_cache_hit") is not None,
            is_retry=request.kwargs.get("_retry_attempts", 0) > 0,
            retry_attempt=request.kwargs.get("_retry_attempts", 0),
        )

    async def run_request(self, request: Request) -> Request:
        context = self.build_context(request)
        for m in self._middleware:
            try:
                request = await m.process_request(request)
            except Exception as e:
                logger.exception("Middleware %s failed on request: %s", type(m).__name__, e)
                raise
        return request

    async def run_response(self, request: Request, response: Response) -> Response:
        context = self.build_context(request)
        for m in self._middleware:
            try:
                response = await m.process_response(request, response)
            except Exception as e:
                logger.exception("Middleware %s failed on response: %s", type(m).__name__, e)
        return response

    async def run_error(self, request: Request, error: Exception) -> Exception | None:
        context = self.build_context(request, error=error)
        current_error = error
        for m in self._middleware:
            try:
                result = await m.process_error(request, current_error)
            except Exception as e:
                logger.exception("Middleware %s failed on error: %s", type(m).__name__, e)
                result = current_error
            if result is None:
                return None
            current_error = result
        return current_error

    def __repr__(self) -> str:
        names = [type(m).__name__ for m in self._middleware]
        return f"MiddlewareChain({', '.join(names)})"
