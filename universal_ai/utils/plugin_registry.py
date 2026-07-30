from __future__ import annotations

import importlib
import logging
from typing import Any

logger = logging.getLogger("universal_ai.plugins")

_BUILTIN_PROVIDERS = [
    ("openai", "universal_ai.providers.openai_provider", "OpenAIProvider"),
    ("anthropic", "universal_ai.providers.anthropic_provider", "AnthropicProvider"),
    ("ollama", "universal_ai.providers.ollama_provider", "OllamaProvider"),
    ("groq", "universal_ai.providers.groq_provider", "GroqProvider"),
    ("gemini", "universal_ai.providers.gemini_provider", "GeminiProvider"),
    ("mistral", "universal_ai.providers.mistral_provider", "MistralProvider"),
    ("openrouter", "universal_ai.providers.openrouter_provider", "OpenRouterProvider"),
    ("huggingface", "universal_ai.providers.huggingface_provider", "HuggingFaceProvider"),
    ("azure", "universal_ai.providers.azure_openai_provider", "AzureOpenAIProvider"),
]


class PluginRegistry:
    _instance: PluginRegistry | None = None

    def __new__(cls) -> PluginRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._providers: dict[str, Any] = {}
            cls._instance._instances: dict[str, Any] = {}
            cls._instance._discover_plugins()
        return cls._instance

    def _discover_plugins(self) -> None:
        try:
            from importlib.metadata import entry_points

            eps = entry_points(group="universalai.providers")
            for ep in eps:
                try:
                    plugin_class = ep.load()
                    self._providers[ep.name] = plugin_class
                    logger.debug("Discovered provider plugin via entry point: %s", ep.name)
                except Exception as e:
                    logger.warning("Failed to load plugin %s: %s", ep.name, e)
        except Exception as e:
            logger.debug("Entry point discovery unavailable: %s", e)
        for name, module_path, class_name in _BUILTIN_PROVIDERS:
            if name in self._providers:
                continue
            try:
                module = importlib.import_module(module_path)
                provider_class = getattr(module, class_name)
                self._providers[name] = provider_class
                logger.debug("Loaded built-in provider: %s", name)
            except ImportError as e:
                logger.debug("Built-in provider %s not available: %s", name, e)
            except Exception as e:
                logger.warning("Failed to load built-in provider %s: %s", name, e)

    def register(self, name: str, provider_class_or_instance: Any) -> None:
        self._providers[name] = provider_class_or_instance
        if not isinstance(provider_class_or_instance, type):
            self._instances[name] = provider_class_or_instance

    def get_provider(self, name: str, **kwargs: Any) -> Any:
        if name in self._instances:
            return self._instances[name]
        provider_class = self._providers.get(name)
        if provider_class is None:
            return None
        if isinstance(provider_class, type):
            try:
                instance = provider_class(**kwargs)
                self._instances[name] = instance
                return instance
            except Exception as e:
                logger.exception("Failed to instantiate provider %s: %s", name, e)
                return None
        else:
            self._instances[name] = provider_class
            return provider_class

    def list_providers(self) -> list[str]:
        return list(self._providers.keys())

    def has_provider(self, name: str) -> bool:
        return name in self._providers

    def clear(self) -> None:
        self._providers.clear()
        self._instances.clear()

    @classmethod
    def reset(cls) -> None:
        cls._instance = None
