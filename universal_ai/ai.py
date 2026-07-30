"""Main AI class for UniversalAI - unified interface for all providers."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from typing import TYPE_CHECKING, Any

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from universal_ai.config import UniversalAIConfig
from universal_ai.conversation import Conversation
from universal_ai.events import EventEmitter
from universal_ai.exceptions import (
    ConfigurationError,
    ContextWindowExceededError,
    ProviderError,
    UniversalAIError,
)
from universal_ai.middleware.base import MiddlewareChain, Request, Response
from universal_ai.models.common import Message, MessageRole, Usage
from universal_ai.models.request import ChatRequest, EmbedRequest
from universal_ai.models.response import (
    ChatChunk,
    ChatResponse,
    EmbedResponse,
    ToolResponse,
)
from universal_ai.streaming import StreamManager
from universal_ai.tools import ToolExecutor, tool
from universal_ai.utils.detection import ProviderDetector
from universal_ai.utils.plugin_registry import PluginRegistry
from universal_ai.utils.tokens import TokenCounter

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

__all__ = ["AI", "Conversation", "tool"]

logger = logging.getLogger("universal_ai")


class AI:
    """Unified interface for AI providers.

    Usage:
        ai = AI()
        response = await ai.chat("Hello!")

        # With specific provider
        ai = AI(provider="openai", model="gpt-4o")

        # Streaming
        async for chunk in ai.stream("Write a poem"):
            print(chunk.delta, end="")
    """

    def __init__(
        self,
        config: UniversalAIConfig | None = None,
        provider: str | None = None,
        model: str = "auto",
        **kwargs: Any,
    ) -> None:
        self._config = config or UniversalAIConfig.load()
        if provider:
            self._config.provider = provider
        if model != "auto":
            self._config.model = model
        if kwargs:
            self._config = self._config.model_copy(update=kwargs)
        self._provider: Any = None
        self._provider_name: str | None = None
        self._middleware_chain = MiddlewareChain()
        self._event_emitter = EventEmitter()
        self._tool_executor = ToolExecutor()
        self._initialized = False

    async def _ensure_provider(self) -> Any:
        """Ensure we have a valid provider instance."""
        if self._provider is not None:
            return self._provider

        registry = PluginRegistry()

        if self._config.provider:
            provider_kwargs = {}
            if self._config.provider in self._config.provider_api_keys:
                provider_kwargs["api_key"] = self._config.provider_api_keys[self._config.provider]
            if self._config.api_base:
                provider_kwargs["api_base"] = self._config.api_base
            provider = registry.get_provider(self._config.provider, **provider_kwargs)
            if provider is None:
                msg = (
                    f"Provider '{self._config.provider}' not found. "
                    f"Available: {registry.list_providers()}"
                )
                raise ConfigurationError(msg)
            self._provider = provider
            self._provider_name = self._config.provider
            return self._provider

        available = await ProviderDetector.detect()
        if not available:
            msg = "No providers detected. Install Ollama or set an API key (e.g., OPENAI_API_KEY)."
            raise ConfigurationError(msg)

        for name in available:
            provider = registry.get_provider(name)
            if provider is not None:
                self._provider = provider
                self._provider_name = name
                self._config.provider = name
                logger.info("Auto-detected provider: %s", name)
                return self._provider

        msg = f"Detected providers {available} but no plugins available"
        raise ConfigurationError(msg)

    def _build_request(self, messages: list[Message], **kwargs: Any) -> ChatRequest:
        """Build a ChatRequest from messages and kwargs."""
        model = kwargs.pop("model", None)
        if model is None:
            model = self._config.model if self._config.model != "auto" else ""

        return ChatRequest(
            messages=messages,
            model=model,
            temperature=kwargs.pop("temperature", self._config.temperature),
            max_tokens=kwargs.pop("max_tokens", self._config.max_tokens),
            stream=kwargs.pop("stream", False),
            image=kwargs.pop("image", None),
            tools=kwargs.pop("tools", []),
            json_mode=kwargs.pop("json_mode", False),
            json_schema=kwargs.pop("json_schema", None),
            stop=kwargs.pop("stop", []),
            frequency_penalty=kwargs.pop("frequency_penalty", 0.0),
            presence_penalty=kwargs.pop("presence_penalty", 0.0),
            top_p=kwargs.pop("top_p", 1.0),
        )

    def _to_request(self, provider: Any, request: ChatRequest) -> Request:
        """Convert ChatRequest to middleware Request."""
        messages = [{"role": m.role.value, "content": m.content} for m in request.messages]
        return Request(
            provider=provider.name if hasattr(provider, "name") else self._provider_name or "",
            model=request.model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            stream=request.stream,
            json_mode=request.json_mode,
            image=request.image,
            tools=request.tools,
            stop=request.stop,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
        )

    async def chat(self, message: str | list[Message], **kwargs: Any) -> ChatResponse:
        """Send a chat message and get a response.

        Args:
            message: String message or list of Message objects.
            **kwargs: Override config values (temperature, max_tokens, etc.)
        """
        if isinstance(message, str):
            messages = [Message(role=MessageRole.user, content=message)]
        else:
            messages = message

        provider = await self._ensure_provider()
        request = self._build_request(messages, **kwargs)
        if not request.model:
            request.model = getattr(provider, "default_model", provider.name)

        validated_messages = self._validate_context_window(provider, request)
        request.messages = validated_messages

        middleware_request = self._to_request(provider, request)

        try:
            middleware_request = await self._middleware_chain.run_request(middleware_request)
        except Exception as e:
            raise ProviderError(provider.name, f"Middleware request failed: {e}") from e

        if hasattr(middleware_request.kwargs, "get") and "_cache_hit" in middleware_request.kwargs:
            return middleware_request.kwargs["_cache_hit"]

        try:
            if request.stream:
                response = await self._stream_to_response(provider, request)
            else:
                response = await provider.chat(request)
        except Exception as e:
            error = await self._middleware_chain.run_error(middleware_request, e)
            if error is None:
                return await self.chat(message, **kwargs)
            if isinstance(error, UniversalAIError):
                raise error
            raise ProviderError(provider.name, f"Chat failed: {error}") from error

        # Tool execution loop
        tool_calls = getattr(response, "tool_calls", []) or []
        if tool_calls and request.tools:
            response = await self._execute_tool_loop(
                provider, request, response, tool_calls, middleware_request
            )

        middleware_response = Response(
            content=response.content,
            usage=response.usage,
            metadata=dict(response.metadata) if hasattr(response, "metadata") else {},
        )

        with contextlib.suppress(Exception):
            middleware_response = await self._middleware_chain.run_response(
                middleware_request, middleware_response
            )

        return response

    def _validate_context_window(self, provider: Any, request: ChatRequest) -> list[Message]:
        """Validate that messages fit within provider context window.

        Returns messages (possibly truncated if auto_truncate is enabled).
        """
        capabilities = getattr(provider, "capabilities", None)
        if not capabilities:
            return request.messages

        context_window = getattr(capabilities, "context_window", None)
        if not context_window:
            return request.messages

        messages_dict = [{"role": m.role.value, "content": m.content} for m in request.messages]
        estimated = TokenCounter.count_messages(messages_dict)

        if estimated > context_window:
            if self._config.auto_truncate:
                logger.info(
                    "Auto-truncating messages: %d tokens -> %d context window",
                    estimated,
                    context_window,
                )
                return self._truncate_messages(request.messages, context_window)
            msg = (
                f"Estimated token count ({estimated}) exceeds context window "
                f"({context_window}) for {getattr(provider, 'name', 'provider')}. "
                f"Set auto_truncate=True to automatically truncate messages."
            )
            raise ContextWindowExceededError(
                getattr(provider, "name", "provider"), estimated, context_window
            )
        return request.messages

    def _truncate_messages(self, messages: list[Message], context_window: int) -> list[Message]:
        """Truncate messages to fit within context window, keeping system message and recent messages."""
        if not messages:
            return messages

        result: list[Message] = []
        system_messages: list[Message] = []
        other_messages: list[Message] = []

        for m in messages:
            if m.role == MessageRole.system:
                system_messages.append(m)
            else:
                other_messages.append(m)

        result.extend(system_messages)

        remaining_window = context_window - TokenCounter.count_messages(
            [{"role": m.role.value, "content": m.content} for m in system_messages]
        )

        for m in reversed(other_messages):
            msg_tokens = TokenCounter.count([m.content])
            if msg_tokens <= remaining_window:
                result.insert(len(system_messages), m)
                remaining_window -= msg_tokens
            else:
                break

        return result

    async def _execute_tool_loop(
        self,
        provider: Any,
        request: ChatRequest,
        response: ChatResponse,
        tool_calls: list[Any],
        middleware_request: Any,
        max_iterations: int = 5,
    ) -> ChatResponse:
        """Execute tool calls and continue conversation until no more tools."""
        messages = list(request.messages)

        for _iteration in range(max_iterations):
            tool_results = []
            for tc in tool_calls:
                result = await self._tool_executor.execute(tc)
                tool_results.append(result)
                messages.append(
                    Message(
                        role=MessageRole.tool,
                        content=result,
                        tool_call_id=tc.id,
                    )
                )

            follow_up_request = ChatRequest(
                messages=messages,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                tools=request.tools,
                json_mode=request.json_mode,
                stop=request.stop,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                top_p=request.top_p,
            )

            try:
                follow_up_response = await provider.chat(follow_up_request)
            except Exception as e:
                logger.warning("Tool follow-up request failed: %s", e)
                return response

            tool_calls = getattr(follow_up_response, "tool_calls", []) or []
            if not tool_calls:
                return follow_up_response

            response = follow_up_response
            request = follow_up_request

        logger.warning(
            "Max tool iterations (%d) reached for %s/%s",
            max_iterations,
            provider.name,
            request.model,
        )
        return response

    async def _stream_to_response(self, provider: Any, request: ChatRequest) -> ChatResponse:
        """Collect streaming chunks into a single response."""
        content = ""
        model = request.model
        usage = Usage()
        finish_reason = None

        async for chunk in provider.chat_stream(request):
            content += chunk.delta
            model = chunk.model or model
            if chunk.usage:
                usage = chunk.usage
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason

        return ChatResponse(
            content=content,
            model=model,
            provider=provider.name,
            usage=usage,
            finish_reason=finish_reason,
        )

    async def stream(self, message: str | list[Message], **kwargs: Any) -> AsyncIterator[ChatChunk]:
        """Stream a chat response.

        Args:
            message: String message or list of Message objects.
            **kwargs: Override config values.
        """
        if isinstance(message, str):
            messages = [Message(role=MessageRole.user, content=message)]
        else:
            messages = message

        provider = await self._ensure_provider()
        request = self._build_request(messages, stream=True, **kwargs)
        if not request.model:
            request.model = getattr(provider, "default_model", provider.name)

        stream_manager = StreamManager(provider, self._event_emitter)
        async for chunk in stream_manager.stream(request):
            yield chunk

    async def embed(self, text: str | list[str], **kwargs: Any) -> EmbedResponse:
        """Generate embeddings for text.

        Args:
            text: Text or list of texts to embed.
            **kwargs: Override config values.
        """
        provider = await self._ensure_provider()
        model = kwargs.pop("model", None)
        if model is None:
            model = self._config.model if self._config.model != "auto" else provider.name

        request = EmbedRequest(input=text, model=model)
        return await provider.embed(request)

    async def transcribe(self, audio: str | bytes, **kwargs: Any) -> str:
        """Transcribe audio to text."""
        provider = await self._ensure_provider()
        if not hasattr(provider, "transcribe"):
            msg = "Provider does not support audio transcription"
            raise NotImplementedError(msg)
        return await provider.transcribe(audio, **kwargs)

    async def speak(self, text: str, **kwargs: Any) -> bytes:
        """Generate speech from text."""
        provider = await self._ensure_provider()
        if not hasattr(provider, "speak"):
            msg = "Provider does not support text-to-speech"
            raise NotImplementedError(msg)
        return await provider.speak(text, **kwargs)

    async def image(self, prompt: str, **kwargs: Any) -> list[str]:
        """Generate an image from a text prompt. Returns list of image URLs."""
        provider = await self._ensure_provider()
        if not hasattr(provider, "generate_image"):
            msg = "Provider does not support image generation"
            raise NotImplementedError(msg)
        return await provider.generate_image(prompt, **kwargs)

    async def chat_with_tools(
        self, message: str | list[Message], tools: list, **kwargs: Any
    ) -> ToolResponse:
        """Chat with tool calling support."""
        result = await self.chat(message, tools=tools, **kwargs)
        return ToolResponse(
            content=result.content,
            model=result.model,
            provider=result.provider,
            usage=result.usage,
            tool_calls=result.tool_calls,
            finish_reason=result.finish_reason,
        )

    async def chat_json(self, message: str | list[Message], schema: type, **kwargs: Any) -> Any:
        """Chat with structured output (JSON mode)."""
        import json

        response = await self.chat(message, json_mode=True, json_schema=schema, **kwargs)
        try:
            return schema(**json.loads(response.content))
        except Exception:
            return response.content

    def add_middleware(self, middleware: Any) -> AI:
        """Add middleware to the processing chain."""
        self._middleware_chain.add(middleware)
        return self

    def register_tool(self, func: Any) -> AI:
        """Register a tool function for use in chat with tools."""
        self._tool_executor.register_tool(func)
        return self

    def on(self, event: str, handler: Any) -> AI:
        """Register an event handler."""
        self._event_emitter.on(event, handler)
        return self

    def conversation(
        self,
        system_prompt: str | None = None,
        max_turns: int | None = None,
    ) -> Conversation:
        """Create a conversation bound to this AI instance."""
        conv = Conversation(
            system_prompt=system_prompt,
            max_turns=max_turns,
            ai=self,
            tools=list(self._tool_executor._tools.values()),
        )
        return conv

    async def set_provider(self, provider: str) -> None:
        """Switch to a different provider."""
        self._config.provider = provider
        self._provider = None
        self._provider_name = None

    async def list_providers(self) -> list[dict[str, Any]]:
        """List all available providers."""
        registry = PluginRegistry()
        providers = []
        for name in registry.list_providers():
            provider = registry.get_provider(name)
            version = getattr(provider, "version", "unknown") if provider else "unknown"
            providers.append(
                {
                    "name": name,
                    "version": version,
                }
            )
        return providers

    async def provider_status(self) -> dict[str, Any]:
        """Get status of all providers."""
        registry = PluginRegistry()
        statuses = {}
        for name in registry.list_providers():
            try:
                provider = registry.get_provider(name)
                if provider and hasattr(provider, "health_check"):
                    status = await provider.health_check()
                    statuses[name] = status
                else:
                    statuses[name] = {"healthy": True, "note": "no health_check"}
            except Exception as e:
                statuses[name] = {"healthy": False, "error": str(e)}
        return statuses

    def chat_sync(self, message: str | list[Message], **kwargs: Any) -> ChatResponse:
        """Synchronous wrapper for chat()."""
        return asyncio.run(self.chat(message, **kwargs))

    def stream_sync(self, message: str | list[Message], **kwargs: Any) -> str:
        """Synchronous wrapper for stream(). Returns full text."""

        async def _collect() -> str:
            chunks = []
            async for chunk in self.stream(message, **kwargs):
                chunks.append(chunk.delta)
            return "".join(chunks)

        return asyncio.run(_collect())

    def embed_sync(self, text: str | list[str], **kwargs: Any) -> EmbedResponse:
        """Synchronous wrapper for embed()."""
        return asyncio.run(self.embed(text, **kwargs))

    def chat_with_tools_sync(
        self, message: str | list[Message], tools: list, **kwargs: Any
    ) -> ToolResponse:
        """Synchronous wrapper for chat_with_tools()."""
        return asyncio.run(self.chat_with_tools(message, tools, **kwargs))

    def chat_json_sync(self, message: str | list[Message], schema: type, **kwargs: Any) -> Any:
        """Synchronous wrapper for chat_json()."""
        return asyncio.run(self.chat_json(message, schema, **kwargs))

    def estimate_cost(self, text: str, model: str | None = None) -> float:
        """Estimate cost for a text completion."""
        from universal_ai.utils.costs import CostCalculator
        from universal_ai.utils.tokens import TokenCounter

        model_name = model or self._config.model
        if model_name == "auto":
            model_name = getattr(self._provider, "default_model", "unknown") if self._provider else "unknown"

        estimated_tokens = TokenCounter.count(text)
        return CostCalculator.calculate(
            provider=self._provider_name or "unknown",
            model=model_name,
            input_tokens=estimated_tokens,
            output_tokens=estimated_tokens,
        )

    async def close(self) -> None:
        """Clean up resources."""
        self._provider = None
        self._provider_name = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    def __repr__(self) -> str:
        provider = self._provider_name or "auto"
        model = self._config.model
        return f"AI(provider={provider!r}, model={model!r})"
