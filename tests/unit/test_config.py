"""Tests for configuration system."""

from universal_ai.config import UniversalAIConfig


class TestUniversalAIConfig:
    def test_default_config(self):
        config = UniversalAIConfig()
        assert config.model == "auto"
        assert config.temperature == 0.7
        assert config.timeout == 30.0
        assert config.max_retries == 3
        assert config.cache is False
        assert config.stream is False

    def test_custom_config(self):
        config = UniversalAIConfig(
            provider="openai",
            model="gpt-4o",
            temperature=0.5,
            max_tokens=1000,
        )
        assert config.provider == "openai"
        assert config.model == "gpt-4o"
        assert config.temperature == 0.5
        assert config.max_tokens == 1000

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("UNIVERSALAI_PROVIDER", "ollama")
        monkeypatch.setenv("UNIVERSALAI_MODEL", "llama3")
        monkeypatch.setenv("UNIVERSALAI_API_KEY", "test-key")
        config = UniversalAIConfig.from_env()
        assert config.provider == "ollama"
        assert config.model == "llama3"
        assert config.api_key == "test-key"

    def test_from_env_partial(self, monkeypatch):
        monkeypatch.setenv("UNIVERSALAI_PROVIDER", "groq")
        config = UniversalAIConfig.from_env()
        assert config.provider == "groq"
        assert config.model == "auto"

    def test_env_cache_enabled(self, monkeypatch):
        monkeypatch.setenv("UNIVERSALAI_CACHE_ENABLED", "true")
        config = UniversalAIConfig.from_env()
        assert config.cache is True

    def test_env_cache_disabled(self, monkeypatch):
        monkeypatch.setenv("UNIVERSALAI_CACHE_ENABLED", "false")
        config = UniversalAIConfig.from_env()
        assert config.cache is False

    def test_merge_precedence(self, monkeypatch):
        monkeypatch.setenv("UNIVERSALAI_PROVIDER", "ollama")
        config = UniversalAIConfig.load()
        assert config.provider == "ollama"
        assert config.model == "auto"

    def test_repr(self):
        config = UniversalAIConfig(provider="openai", model="gpt-4o")
        r = repr(config)
        assert "UniversalAIConfig" in r
        assert "openai" in r

    def test_from_file_json(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text('{"provider": "anthropic", "model": "claude-3"}')
        config = UniversalAIConfig.from_file(str(config_file))
        assert config.provider == "anthropic"
        assert config.model == "claude-3"

    def test_from_file_missing(self):
        config = UniversalAIConfig.from_file("/nonexistent/path.json")
        assert config.model == "auto"
