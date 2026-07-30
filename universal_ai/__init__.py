"""UniversalAI — The Requests library for AI.

A unified, provider-agnostic Python SDK for interacting with LLM providers.

Usage:
    import asyncio
    from universal_ai import AI

    async def main():
        ai = AI()
        response = await ai.chat("Hello, world!")
        print(response.content)

        # Streaming
        async for chunk in ai.stream("Tell me a story"):
            print(chunk.delta, end="")

    asyncio.run(main())

    # Specific provider
    ai = AI(provider="anthropic", model="claude-sonnet-4-20250514")
"""

from universal_ai.ai import AI
from universal_ai.config import Config
from universal_ai.conversation import Conversation
from universal_ai.exceptions import (
    AuthenticationError,
    CircuitBreakerOpenError,
    ConfigurationError,
    ContextWindowExceededError,
    ModelNotFoundError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
    StreamError,
    TimeoutError,
    UniversalAIError,
)
from universal_ai.models import (
    ChatChunk,
    ChatRequest,
    ChatResponse,
    EmbedRequest,
    EmbedResponse,
    FinishReason,
    Message,
    MessageRole,
    ProviderCapabilities,
    Tool,
    ToolCall,
    ToolResponse,
    Usage,
)
from universal_ai.tools import ToolExecutor, tool

try:
    from universal_ai.agents import Agent, Coordinator, Team
except ImportError:
    pass

try:
    from universal_ai.rag import RAG
except ImportError:
    pass

__version__ = "1.0.0"
__all__ = [
    # Core
    "AI",
    "AuthenticationError",
    "ChatChunk",
    "ChatRequest",
    "ChatResponse",
    "CircuitBreakerOpenError",
    "Config",
    "ConfigurationError",
    "ContextWindowExceededError",
    "Conversation",
    "EmbedRequest",
    "EmbedResponse",
    "FinishReason",
    # Models
    "Message",
    "MessageRole",
    "ModelNotFoundError",
    "ProviderCapabilities",
    "ProviderError",
    "ProviderUnavailableError",
    "RateLimitError",
    "StreamError",
    "TimeoutError",
    "Tool",
    "ToolCall",
    "ToolExecutor",
    "ToolResponse",
    # Exceptions
    "UniversalAIError",
    "Usage",
    # Tools
    "tool",
]
