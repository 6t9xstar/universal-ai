"""Tests for tool decorator and conversation management."""

from universal_ai.conversation import Conversation
from universal_ai.models import MessageRole
from universal_ai.tools import tool


class TestToolDecorator:
    def test_tool_decorator_adds_metadata(self):
        @tool
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"Weather in {city}"

        assert hasattr(get_weather, "_tool_definition")
        assert hasattr(get_weather, "_universal_ai_tool")
        assert get_weather._universal_ai_tool is True
        assert get_weather._tool_definition.name == "get_weather"
        assert get_weather._tool_definition.description == "Get weather for a city."

    def test_tool_decorator_multiple_params(self):
        @tool
        def calculate(a: int, b: float, operation: str = "add") -> float:
            """Calculate result."""
            return 0.0

        params = calculate._tool_definition.parameters
        assert "a" in params["properties"]
        assert "b" in params["properties"]
        assert "operation" in params["properties"]
        assert params["properties"]["a"]["type"] == "integer"
        assert params["properties"]["b"]["type"] == "number"
        assert "a" in params["required"]
        assert "b" in params["required"]
        assert "operation" not in params["required"]

    def test_tool_decorator_no_docstring(self):
        @tool
        def no_doc():
            pass

        assert no_doc._tool_definition.description == ""

    def test_tool_decorator_bool_param(self):
        @tool
        def toggle(enabled: bool) -> str:
            """Toggle something."""
            return ""

        params = toggle._tool_definition.parameters
        assert params["properties"]["enabled"]["type"] == "boolean"


class TestConversation:
    def test_create_empty(self):
        conv = Conversation()
        assert len(conv.history) == 0
        assert conv.turn_count == 0

    def test_create_with_system_prompt(self):
        conv = Conversation(system_prompt="You are helpful.")
        assert len(conv.history) == 0
        messages = conv._build_messages()
        assert len(messages) == 1
        assert messages[0].role == MessageRole.system
        assert messages[0].content == "You are helpful."

    def test_add_user_message(self):
        conv = Conversation()
        conv.add_user_message("Hello!")
        assert len(conv.history) == 1
        assert conv.history[0].role == MessageRole.user
        assert conv.history[0].content == "Hello!"

    def test_add_assistant_message(self):
        conv = Conversation()
        conv.add_assistant_message("Hi there!")
        assert len(conv.history) == 1
        assert conv.history[0].role == MessageRole.assistant
        assert conv.history[0].content == "Hi there!"

    def test_multi_turn_conversation(self):
        conv = Conversation(system_prompt="You are a tutor.")
        conv.add_user_message("What is Python?")
        conv.add_assistant_message("Python is a programming language.")
        conv.add_user_message("Is it hard?")
        conv.add_assistant_message("Not really, it's beginner-friendly.")

        assert len(conv.history) == 4
        assert conv.history[0].role == MessageRole.user
        assert conv.history[1].role == MessageRole.assistant
        assert conv.history[2].role == MessageRole.user
        assert conv.history[3].role == MessageRole.assistant

    def test_reset(self):
        conv = Conversation(system_prompt="System message")
        conv.add_user_message("Hello")
        conv.add_assistant_message("Hi")
        conv.reset()

        assert len(conv.history) == 0
        assert conv.turn_count == 0

    def test_truncate(self):
        conv = Conversation()
        for i in range(20):
            conv.add_user_message(f"Message {i}")

        conv.truncate(keep_last=5)
        assert len(conv.history) == 5
        assert conv.history[0].content == "Message 15"

    def test_messages_property(self):
        conv = Conversation()
        conv.add_user_message("Hello")
        assert isinstance(conv.history, list)
        assert len(conv.history) == 1

    def test_repr(self):
        conv = Conversation()
        conv.add_user_message("Hello")
        assert "Conversation" in repr(conv)
        assert "messages=1" in repr(conv)
