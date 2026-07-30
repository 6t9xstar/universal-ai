"""Mock provider for integration testing."""

from __future__ import annotations

from collections.abc import AsyncIterator

from universal_ai.models.common import FinishReason, Usage
from universal_ai.models.request import ChatRequest, EmbedRequest
from universal_ai.models.response import ChatChunk, ChatResponse, EmbedResponse


class MockProvider:
    """A mock provider for testing that simulates API responses.

    Usage:
        provider = MockProvider()
        response = await provider.chat(request)
    """

    name = "mock"
    version = "0.1.0"
    default_model = "mock-model"

    def __init__(
        self,
        api_key: str | None = None,
        response_text: str = "Hello from mock provider!",
        fail: bool = False,
    ) -> None:
        self.api_key = api_key or "mock-api-key"
        self.response_text = response_text
        self.fail = fail
        self._chat_called = False
        self._stream_called = False
        self._embed_called = False
        self._last_request: ChatRequest | None = None

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Simulate a chat completion."""
        self._chat_called = True
        self._last_request = request

        if self.fail:
            from universal_ai.exceptions import ProviderError
            raise ProviderError("mock", "Simulated failure")

        return ChatResponse(
            content=self.response_text,
            model=request.model or self.default_model,
            provider="mock",
            usage=Usage(
                input_tokens=10,
                output_tokens=20,
                total_tokens=30,
            ),
            finish_reason=FinishReason.stop,
        )

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        """Simulate streaming chat completion."""
        self._stream_called = True
        self._last_request = request

        if self.fail:
            from universal_ai.exceptions import ProviderError
            raise ProviderError("mock", "Simulated stream failure")

        words = self.response_text.split()
        for i, word in enumerate(words):
            yield ChatChunk(
                content=word + " ",
                delta=word + " ",
                model=request.model or self.default_model,
                finish_reason=FinishReason.stop if i == len(words) - 1 else None,
                usage=Usage(
                    input_tokens=10,
                    output_tokens=1,
                    total_tokens=11,
                ) if i == len(words) - 1 else None,
            )

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        """Simulate embedding generation."""
        self._embed_called = True

        if self.fail:
            from universal_ai.exceptions import ProviderError
            raise ProviderError("mock", "Simulated embed failure")

        # Return a simple embedding vector
        text = request.input if isinstance(request.input, str) else request.input[0]
        vector = [float(ord(c)) / 100.0 for c in text[:10]]

        return EmbedResponse(
            vector=vector,
            model="mock-embed",
            provider="mock",
            usage=Usage(
                input_tokens=5,
                output_tokens=0,
                total_tokens=5,
            ),
        )

    async def health_check(self) -> dict:
        """Check if the mock provider is healthy."""
        return {"healthy": True, "mock": True}

    async def get_latency(self) -> float:
        """Get simulated latency."""
        return 0.01

    def calculate_cost(self, estimated_tokens: int) -> float:
        """Calculate simulated cost."""
        return estimated_tokens * 0.000001

    async def close(self) -> None:
        """Clean up resources."""

    def __repr__(self) -> str:
        return f"MockProvider(response_text={self.response_text!r})"
