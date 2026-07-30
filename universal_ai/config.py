from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class UniversalAIConfig(BaseModel):
    provider: str | None = None
    model: str = "auto"
    fallback_providers: list[str] = Field(default_factory=list)
    api_key: str | None = None
    api_base: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    timeout: float = 30.0
    max_retries: int = 3
    stream: bool = False
    cache: bool = False
    cache_ttl: int = 3600
    cache_backend: str = "memory"
    log_level: str = "INFO"
    trace_enabled: bool = False
    metrics_enabled: bool = False
    middleware: list[str] = Field(default_factory=list)
    plugins: list[str] = Field(default_factory=list)
    auto_truncate: bool = False
    provider_api_keys: dict[str, str] = Field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"UniversalAIConfig(provider={self.provider!r}, "
            f"model={self.model!r}, temperature={self.temperature})"
        )

    @classmethod
    def from_env(cls) -> UniversalAIConfig:
        kwargs: dict[str, Any] = {}
        for env_key, config_key in [
            ("UNIVERSALAI_PROVIDER", "provider"),
            ("UNIVERSALAI_MODEL", "model"),
            ("UNIVERSALAI_API_KEY", "api_key"),
            ("UNIVERSALAI_API_BASE", "api_base"),
            ("UNIVERSALAI_LOG_LEVEL", "log_level"),
        ]:
            value = os.getenv(env_key)
            if value:
                kwargs[config_key] = value
        cache_enabled = os.getenv("UNIVERSALAI_CACHE_ENABLED")
        if cache_enabled:
            kwargs["cache"] = cache_enabled.lower() in ("true", "1", "yes")
        return cls(**kwargs) if kwargs else cls()

    @classmethod
    def from_file(cls, path: str | None = None) -> UniversalAIConfig:
        path = path or os.path.join(Path.home(), ".config", "universalai", "config.yaml")
        if not os.path.exists(path):
            path_json = path.replace(".yaml", ".json").replace(".yml", ".json")
            if os.path.exists(path_json):
                path = path_json
            else:
                return cls()
        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
            if path.endswith(".json"):
                data = json.loads(content)
            else:
                try:
                    import yaml

                    data = yaml.safe_load(content) or {}
                except ImportError:
                    data = json.loads(content)
            return cls(**data)
        except Exception:
            return cls()

    @classmethod
    def load(cls) -> UniversalAIConfig:
        defaults = cls()
        file_config = cls.from_file()
        env_config = cls.from_env()
        merged = defaults.model_copy(update=file_config.model_dump(exclude_unset=True))
        return merged.model_copy(update=env_config.model_dump(exclude_unset=True))


Config = UniversalAIConfig

__all__ = ["Config", "UniversalAIConfig"]
