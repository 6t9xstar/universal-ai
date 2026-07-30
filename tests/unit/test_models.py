"""Tests for core models."""

import pytest
from pydantic import ValidationError

from universal_ai.models.common import (
    FinishReason,
    Message,
    MessageRole,
    ProviderCapabilities,
    Tool,
    ToolCall,
    Usage,
)
from universal_ai.models.request import ChatRequest
from universal_ai.models.response import ChatChunk, ChatResponse, EmbedResponse, ToolResponse


class TestMessage:
    def test_create_user_message(self):
        msg = Message(role=MessageRole.user, content="Hello")
        assert msg.role == MessageRole.user
        assert msg.content == "Hello"

    def test_create_system_message(self):
        msg = Message(role=MessageRole.system, content="You are helpful")
        assert msg.role == MessageRole.system

    def test_create_with_tool_calls(self):
        msg = Message(
            role=MessageRole.assistant,
            content="",
            tool_calls=[
                ToolCall(id="call_1", function_name="get_weather", arguments='{"city": "NYC"}')
            ],
        )
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].function_name == "get_weather"

    def test_repr(self):
        msg = Message(role=MessageRole.user, content="Hello world")
        assert "Message" in repr(msg)

    def test_invalid_role_raises(self):
        with pytest.raises(ValidationError):
            Message(role="invalid", content="Hello")


class TestUsage:
    def test_default_usage(self):
        usage = Usage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0

    def test_custom_usage(self):
        usage = Usage(input_tokens=10, output_tokens=20, total_tokens=30, estimated_cost=0.001)
        assert usage.input_tokens == 10
        assert usage.total_tokens == 30
        assert usage.estimated_cost == 0.001


class TestToolCall:
    def test_create_tool_call(self):
        tc = ToolCall(id="call_1", function_name="get_weather", arguments='{"city": "NYC"}')
        assert tc.id == "call_1"
        assert tc.function_name == "get_weather"


class TestTool:
    def test_create_tool(self):
        tool = Tool(name="get_weather", description="Get weather for a city")
        assert tool.name == "get_weather"
        assert tool.description == "Get weather for a city"


class TestProviderCapabilities:
    def test_default_capabilities(self):
        caps = ProviderCapabilities()
        assert caps.chat is True
        assert caps.streaming is True
        assert caps.vision is False
        assert caps.tools is False
        assert caps.max_tokens == 4096


class TestChatRequest:
    def test_create_request(self):
        msg = Message(role=MessageRole.user, content="Hi")
        req = ChatRequest(messages=[msg], model="gpt-4o")
        assert len(req.messages) == 1
        assert req.model == "gpt-4o"
        assert req.temperature == 0.7
        assert req.stream is False


class TestChatResponse:
    def test_create_response(self):
        usage = Usage(input_tokens=10, output_tokens=20, total_tokens=30)
        resp = ChatResponse(
            content="Hello there",
            model="gpt-4o",
            provider="openai",
            usage=usage,
        )
        assert resp.text == "Hello there"
        assert resp.tokens == 30

    def test_finish_reason(self):
        usage = Usage()
        resp = ChatResponse(
            content="Done",
            model="gpt-4o",
            provider="openai",
            usage=usage,
            finish_reason=FinishReason.stop,
        )
        assert resp.finish_reason == FinishReason.stop


class TestChatChunk:
    def test_create_chunk(self):
        chunk = ChatChunk(delta="Hello", model="gpt-4o")
        assert chunk.delta == "Hello"
        assert chunk.model == "gpt-4o"
        assert chunk.finish_reason is None


class TestEmbedResponse:
    def test_create_embed_response(self):
        usage = Usage(total_tokens=5)
        resp = EmbedResponse(
            vector=[0.1, 0.2, 0.3],
            model="text-embedding-3-small",
            provider="openai",
            usage=usage,
        )
        assert len(resp.vector) == 3
        assert resp.vector[0] == 0.1


class TestToolResponse:
    def test_create_tool_response(self):
        usage = Usage(total_tokens=10)
        resp = ToolResponse(
            content="72F",
            model="gpt-4o",
            provider="openai",
            usage=usage,
            tool_calls=[],
        )
        assert resp.content == "72F"
        assert resp.finish_reason == FinishReason.tool_calls
