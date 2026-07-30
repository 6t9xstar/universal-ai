from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from universal_ai.exceptions import ProviderUnavailableError

logger = logging.getLogger("universal_ai.router")


class RoutingStrategy(ABC):
    @abstractmethod
    async def select_provider(self, providers: list[Any], request: Any) -> Any: ...


class FallbackStrategy(RoutingStrategy):
    async def select_provider(self, providers: list[Any], request: Any) -> Any:
        if not providers:
            msg = "No providers available"
            raise ProviderUnavailableError(msg)
        return providers[0]


class LowestLatencyStrategy(RoutingStrategy):
    async def select_provider(self, providers: list[Any], request: Any) -> Any:
        if not providers:
            msg = "No providers available"
            raise ProviderUnavailableError(msg)
        latencies = []
        for p in providers:
            try:
                latency = await p.get_latency()
            except Exception:
                latency = float("inf")
            latencies.append(latency)
        return providers[latencies.index(min(latencies))]


class LowestCostStrategy(RoutingStrategy):
    async def select_provider(self, providers: list[Any], request: Any) -> Any:
        if not providers:
            msg = "No providers available"
            raise ProviderUnavailableError(msg)
        estimated_tokens = getattr(request, "estimated_tokens", 100)
        costs = [p.calculate_cost(estimated_tokens) for p in providers]
        return providers[costs.index(min(costs))]


class RoundRobinStrategy(RoutingStrategy):
    def __init__(self) -> None:
        self._index = 0

    async def select_provider(self, providers: list[Any], request: Any) -> Any:
        if not providers:
            msg = "No providers available"
            raise ProviderUnavailableError(msg)
        provider = providers[self._index % len(providers)]
        self._index = (self._index + 1) % len(providers)
        return provider


class Router:
    def __init__(
        self,
        providers: list[Any],
        strategy: RoutingStrategy | None = None,
        fallback_chain: list[str] | None = None,
        required_capabilities: list[str] | None = None,
    ) -> None:
        self.providers = {getattr(p, "name", str(i)): p for i, p in enumerate(providers)}
        self.strategy = strategy or FallbackStrategy()
        self.fallback_chain = fallback_chain or []
        self.required_capabilities = required_capabilities or []

    async def route(self, request: Any) -> Any:
        provider_list = list(self.providers.values())
        if self.required_capabilities:
            provider_list = [p for p in provider_list if self._has_capabilities(p)]
        try:
            provider = await self.strategy.select_provider(provider_list, request)
            if provider and await self._is_healthy(provider):
                return provider
        except Exception as e:
            logger.debug("Strategy selection failed: %s", e)
        for provider_name in self.fallback_chain:
            provider = self.providers.get(provider_name)
            if provider and await self._is_healthy(provider):
                return provider
        msg = "No healthy providers available"
        raise ProviderUnavailableError(msg)

    def _has_capabilities(self, provider: Any) -> bool:
        if not self.required_capabilities:
            return True
        capabilities = getattr(provider, "capabilities", None)
        if capabilities is None:
            return True
        return all(getattr(capabilities, cap, False) for cap in self.required_capabilities)

    async def _is_healthy(self, provider: Any) -> bool:
        if not hasattr(provider, "health_check"):
            return True
        try:
            status = await provider.health_check()
            return status.get("healthy", False)
        except Exception:
            return False

    def __repr__(self) -> str:
        return (
            f"Router(providers={list(self.providers.keys())}, "
            f"strategy={type(self.strategy).__name__})"
        )
