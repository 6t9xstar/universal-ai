"""Tests for plugin registry and utilities."""

from universal_ai.utils.costs import CostCalculator
from universal_ai.utils.plugin_registry import PluginRegistry
from universal_ai.utils.tokens import TokenCounter


class TestPluginRegistry:
    def setup_method(self):
        PluginRegistry.reset()

    def test_singleton(self):
        r1 = PluginRegistry()
        r2 = PluginRegistry()
        assert r1 is r2

    def test_register_provider_class(self):
        registry = PluginRegistry()

        class MockProvider:
            name = "test"

            def __init__(self, **kwargs):
                pass

        registry.register("test", MockProvider)
        assert "test" in registry.list_providers()

    def test_register_provider_instance(self):
        registry = PluginRegistry()
        instance = type("MockProvider", (), {"name": "test"})()
        registry.register("test", instance)
        assert "test" in registry.list_providers()

    def test_get_provider_instantiates(self):
        registry = PluginRegistry()

        class MockProvider:
            def __init__(self, api_key=None):
                self.api_key = api_key

        registry.register("test", MockProvider)
        provider = registry.get_provider("test", api_key="key123")
        assert provider.api_key == "key123"

    def test_get_provider_returns_instance(self):
        registry = PluginRegistry()
        instance = type("MockProvider", (), {"name": "test"})()
        registry.register("test", instance)
        provider = registry.get_provider("test")
        assert provider is instance

    def test_get_nonexistent_provider(self):
        registry = PluginRegistry()
        assert registry.get_provider("nonexistent") is None

    def test_has_provider(self):
        registry = PluginRegistry()
        assert not registry.has_provider("test")
        registry.register("test", lambda: None)
        assert registry.has_provider("test")

    def test_clear(self):
        registry = PluginRegistry()
        registry.register("test", lambda: None)
        registry.clear()
        assert len(registry.list_providers()) == 0

    def test_reset(self):
        r1 = PluginRegistry()
        r1.register("test", lambda: None)
        PluginRegistry.reset()
        r2 = PluginRegistry()
        assert "test" not in r2.list_providers()


class TestTokenCounter:
    def test_count_empty(self):
        assert TokenCounter.count("") == 0

    def test_count_short_word(self):
        assert TokenCounter.count("hi") == 1

    def test_count_long_word(self):
        assert TokenCounter.count("hello") == 2

    def test_count_multiple_words(self):
        assert TokenCounter.count("hello world") == 5

    def test_count_messages(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        count = TokenCounter.count_messages(messages)
        assert count > 0

    def test_fits_in_context(self):
        messages = [{"role": "user", "content": "Hello"}]
        assert TokenCounter.fits_in_context(messages, 4096) is True
        assert TokenCounter.fits_in_context(messages, 2) is False


class TestCostCalculator:
    def test_calculate_openai(self):
        cost = CostCalculator.calculate("openai", "gpt-4o", 1000, 500)
        assert cost > 0

    def test_calculate_ollama(self):
        cost = CostCalculator.calculate("ollama", "llama3", 1000, 500)
        assert cost >= 0.0

    def test_calculate_unknown_provider(self):
        cost = CostCalculator.calculate("unknown", "model", 1000, 500)
        assert cost > 0

    def test_get_rates(self):
        rates = CostCalculator.get_rates("openai", "gpt-4o")
        assert rates[0] == 0.00001
        assert rates[1] == 0.00003

    def test_get_rates_unknown(self):
        rates = CostCalculator.get_rates("unknown", "model")
        assert rates == CostCalculator.DEFAULT_RATES

    def test_estimate_tokens(self):
        tokens = CostCalculator.estimate_tokens("Hello world")
        assert tokens > 0
