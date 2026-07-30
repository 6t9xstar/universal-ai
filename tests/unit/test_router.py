"""Tests for provider routing."""

import pytest

from universal_ai.exceptions import ProviderUnavailableError
from universal_ai.router import (
    FallbackStrategy,
    LowestCostStrategy,
    LowestLatencyStrategy,
    RoundRobinStrategy,
    Router,
)


class MockProvider:
    def __init__(self, name: str, healthy: bool = True, latency: float = 0.1):
        self.name = name
        self._healthy = healthy
        self._latency = latency

    async def health_check(self):
        return {"healthy": self._healthy}

    def calculate_cost(self, estimated_tokens: int):
        return 0.001

    async def get_latency(self):
        return self._latency


class MockRequest:
    def __init__(self, estimated_tokens=100):
        self.estimated_tokens = estimated_tokens


class TestFallbackStrategy:
    async def test_returns_first_provider(self):
        strategy = FallbackStrategy()
        providers = [MockProvider("openai"), MockProvider("anthropic")]
        result = await strategy.select_provider(providers, MockRequest())
        assert result.name == "openai"

    async def test_empty_providers_raises(self):
        strategy = FallbackStrategy()
        with pytest.raises(ProviderUnavailableError):
            await strategy.select_provider([], MockRequest())


class TestRoundRobinStrategy:
    async def test_rotates_providers(self):
        strategy = RoundRobinStrategy()
        providers = [MockProvider("openai"), MockProvider("anthropic")]
        r1 = await strategy.select_provider(providers, MockRequest())
        r2 = await strategy.select_provider(providers, MockRequest())
        r3 = await strategy.select_provider(providers, MockRequest())
        assert r1.name == "openai"
        assert r2.name == "anthropic"
        assert r3.name == "openai"

    async def test_empty_providers_raises(self):
        strategy = RoundRobinStrategy()
        with pytest.raises(ProviderUnavailableError):
            await strategy.select_provider([], MockRequest())


class TestLowestLatencyStrategy:
    async def test_selects_lowest_latency(self):
        strategy = LowestLatencyStrategy()
        providers = [
            MockProvider("openai", latency=0.2),
            MockProvider("anthropic", latency=0.05),
        ]
        result = await strategy.select_provider(providers, MockRequest())
        assert result.name == "anthropic"

    async def test_empty_providers_raises(self):
        strategy = LowestLatencyStrategy()
        with pytest.raises(ProviderUnavailableError):
            await strategy.select_provider([], MockRequest())


class TestLowestCostStrategy:
    async def test_selects_lowest_cost(self):
        strategy = LowestCostStrategy()
        providers = [
            MockProvider("openai"),
            MockProvider("anthropic"),
        ]
        result = await strategy.select_provider(providers, MockRequest())
        assert result.name in ["openai", "anthropic"]


class TestRouter:
    async def test_fallback_strategy(self):
        providers = [MockProvider("openai"), MockProvider("anthropic")]
        router = Router(providers, strategy=FallbackStrategy())
        result = await router.route(MockRequest())
        assert result.name == "openai"

    async def test_round_robin(self):
        providers = [MockProvider("openai"), MockProvider("anthropic")]
        router = Router(providers, strategy=RoundRobinStrategy())
        r1 = await router.route(MockRequest())
        r2 = await router.route(MockRequest())
        assert r1.name != r2.name

    async def test_fallback_chain(self):
        providers = [MockProvider("openai")]
        router = Router(providers, fallback_chain=["openai"])
        result = await router.route(MockRequest())
        assert result.name == "openai"

    async def test_fallback_chain_skips_unhealthy(self):
        providers = [MockProvider("openai", healthy=False)]
        router = Router(providers, fallback_chain=["openai"])
        with pytest.raises(ProviderUnavailableError):
            await router.route(MockRequest())

    async def test_no_providers_raises(self):
        router = Router([])
        with pytest.raises(ProviderUnavailableError):
            await router.route(MockRequest())

    async def test_unhealthy_provider_falls_through(self):
        providers = [MockProvider("openai", healthy=False)]
        router = Router(providers, fallback_chain=[])
        with pytest.raises(ProviderUnavailableError):
            await router.route(MockRequest())

    async def test_repr(self):
        providers = [MockProvider("openai")]
        router = Router(providers)
        assert "Router" in repr(router)
        assert "openai" in repr(router)
