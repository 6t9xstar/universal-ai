"""Integration tests for AI class with mock provider."""

from __future__ import annotations

import pytest

from tests.integration.mock_provider import MockProvider
from universal_ai import AI, Conversation, tool
from universal_ai.config import UniversalAIConfig
from universal_ai.exceptions import ConfigurationError, ProviderError
from universal_ai.models.common import Message, MessageRole
from universal_ai.models.response import ChatResponse, EmbedResponse
from universal_ai.utils.plugin_registry import PluginRegistry


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the plugin registry before each test."""
    PluginRegistry.reset()
    yield
    PluginRegistry.reset()


class TestAIInit:
    """Tests for AI class initialization."""

    def test_default_init(self):
        ai = AI()
        assert ai._config is not None
        assert ai._provider is None

    def test_init_with_provider(self):
        ai = AI(provider="openai")
        assert ai._config.provider == "openai"

    def test_init_with_model(self):
        ai = AI(model="gpt-4o")
        assert ai._config.model == "gpt-4o"

    def test_init_with_config(self):
        config = UniversalAIConfig(provider="anthropic", temperature=0.5)
        ai = AI(config=config)
        assert ai._config.provider == "anthropic"
        assert ai._config.temperature == 0.5

    def test_init_with_kwargs(self):
        ai = AI(provider="openai", temperature=0.3)
        assert ai._config.provider == "openai"
        assert ai._config.temperature == 0.3

    def test_repr(self):
        ai = AI(provider="openai", model="gpt-4o")
        # Provider name shows as "auto" until provider is actually instantiated
        assert "gpt-4o" in repr(ai)

    def test_repr_auto(self):
        ai = AI()
        assert "auto" in repr(ai)


class TestAIChat:
    """Tests for AI chat method."""

    @pytest.mark.asyncio
    async def test_chat_with_mock_provider(self):
        # Register mock provider
        registry = PluginRegistry()
        registry.register("mock", MockProvider)

        ai = AI(provider="mock")
        response = await ai.chat("Hello!")

        assert isinstance(response, ChatResponse)
        assert response.content == "Hello from mock provider!"
        assert response.provider == "mock"
        assert response.usage.total_tokens == 30

    @pytest.mark.asyncio
    async def test_chat_with_list_of_messages(self):
        registry = PluginRegistry()
        registry.register("mock", MockProvider)

        ai = AI(provider="mock")
        messages = [
            Message(role=MessageRole.user, content="Hello!"),
        ]
        response = await ai.chat(messages)

        assert isinstance(response, ChatResponse)
        assert response.content == "Hello from mock provider!"

    @pytest.mark.asyncio
    async def test_chat_no_provider_raises(self):
        # Mock the provider detection to return empty list
        from unittest.mock import AsyncMock, patch

        ai = AI()
        with patch(
            "universal_ai.utils.detection.ProviderDetector.detect",
            new_callable=AsyncMock,
            return_value=[]
        ), pytest.raises(ConfigurationError):
            await ai.chat("Hello!")


class TestAIStream:
    """Tests for AI stream method."""

    @pytest.mark.asyncio
    async def test_stream_with_mock_provider(self):
        registry = PluginRegistry()
        registry.register("mock", MockProvider)

        ai = AI(provider="mock")
        chunks = []
        async for chunk in ai.stream("Hello!"):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert all(chunk.delta for chunk in chunks)


class TestAIEmbed:
    """Tests for AI embed method."""

    @pytest.mark.asyncio
    async def test_embed_with_mock_provider(self):
        registry = PluginRegistry()
        registry.register("mock", MockProvider)

        ai = AI(provider="mock")
        response = await ai.embed("Hello world")

        assert isinstance(response, EmbedResponse)
        assert response.vector is not None
        assert response.provider == "mock"


class TestAITools:
    """Tests for AI tool calling."""

    @pytest.mark.asyncio
    async def test_register_tool(self):
        @tool
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"Weather in {city}: Sunny"

        registry = PluginRegistry()
        registry.register("mock", MockProvider)

        ai = AI(provider="mock")
        ai.register_tool(get_weather)

        assert ai._tool_executor.get_tool("get_weather") is not None


class TestAIConversation:
    """Tests for AI conversation method."""

    def test_conversation_creates_conversation(self):
        ai = AI()
        conv = ai.conversation(system_prompt="You are helpful.")

        assert isinstance(conv, Conversation)
        assert conv._system_prompt == "You are helpful."
        assert conv._ai is ai


class TestAISyncWrappers:
    """Tests for synchronous wrappers."""

    def test_chat_sync_with_mock_provider(self):
        registry = PluginRegistry()
        registry.register("mock", MockProvider)

        ai = AI(provider="mock")
        response = ai.chat_sync("Hello!")

        assert isinstance(response, ChatResponse)
        assert response.content == "Hello from mock provider!"

    def test_stream_sync_with_mock_provider(self):
        registry = PluginRegistry()
        registry.register("mock", MockProvider)

        ai = AI(provider="mock")
        result = ai.stream_sync("Hello!")

        assert isinstance(result, str)
        assert "Hello from mock provider!" in result

    def test_embed_sync_with_mock_provider(self):
        registry = PluginRegistry()
        registry.register("mock", MockProvider)

        ai = AI(provider="mock")
        response = ai.embed_sync("Hello world")

        assert isinstance(response, EmbedResponse)
        assert response.vector is not None


class TestAIContextManager:
    """Tests for AI async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        async with AI() as ai:
            assert ai is not None

    @pytest.mark.asyncio
    async def test_context_manager_close(self):
        ai = AI()
        async with ai:
            pass
        # After context manager, provider should be cleaned up
        assert ai._provider is None


class TestAIProviderManagement:
    """Tests for provider management."""

    @pytest.mark.asyncio
    async def test_set_provider(self):
        registry = PluginRegistry()
        registry.register("mock", MockProvider)

        ai = AI()
        await ai.set_provider("mock")

        assert ai._config.provider == "mock"

    @pytest.mark.asyncio
    async def test_list_providers(self):
        registry = PluginRegistry()
        registry.register("mock", MockProvider)

        ai = AI()
        providers = await ai.list_providers()

        assert len(providers) > 0
        assert any(p["name"] == "mock" for p in providers)

    @pytest.mark.asyncio
    async def test_provider_status(self):
        registry = PluginRegistry()
        registry.register("mock", MockProvider)

        ai = AI()
        statuses = await ai.provider_status()

        assert "mock" in statuses
        assert statuses["mock"]["healthy"] is True


class TestAIErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_chat_with_failing_provider(self):
        registry = PluginRegistry()
        registry.register("mock", MockProvider(fail=True))

        ai = AI(provider="mock")
        with pytest.raises(ProviderError):
            await ai.chat("Hello!")

    @pytest.mark.asyncio
    async def test_stream_with_failing_provider(self):
        registry = PluginRegistry()
        registry.register("mock", MockProvider(fail=True))

        ai = AI(provider="mock")
        with pytest.raises(ProviderError):
            async for _ in ai.stream("Hello!"):
                pass

    @pytest.mark.asyncio
    async def test_embed_with_failing_provider(self):
        registry = PluginRegistry()
        registry.register("mock", MockProvider(fail=True))

        ai = AI(provider="mock")
        with pytest.raises(ProviderError):
            await ai.embed("Hello")
