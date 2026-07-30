"""Tests for exception hierarchy."""

from universal_ai.exceptions import (
    AuthenticationError,
    CircuitBreakerOpenError,
    ConfigurationError,
    ContextWindowExceededError,
    ModelNotFoundError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
    StreamError,
    TimeoutError,
    UniversalAIError,
)


class TestExceptionHierarchy:
    def test_base_exception(self):
        error = UniversalAIError("Base error")
        assert str(error) == "Base error"
        assert isinstance(error, Exception)

    def test_provider_error(self):
        error = ProviderError("openai", "API error", status_code=500)
        assert error.provider == "openai"
        assert error.status_code == 500
        assert error.raw == {}
        assert "openai" in str(error)

    def test_provider_error_with_raw(self):
        error = ProviderError("openai", "Error", raw={"detail": "info"})
        assert error.raw == {"detail": "info"}

    def test_authentication_error(self):
        error = AuthenticationError("openai", "Invalid key")
        assert isinstance(error, ProviderError)
        assert isinstance(error, UniversalAIError)

    def test_rate_limit_error(self):
        error = RateLimitError("openai", "Too fast", retry_after=30.0)
        assert error.retry_after == 30.0
        assert isinstance(error, ProviderError)

    def test_model_not_found_error(self):
        error = ModelNotFoundError("openai", "gpt-5 not found")
        assert isinstance(error, ProviderError)

    def test_context_window_exceeded(self):
        error = ContextWindowExceededError("openai", 5000, 4096)
        assert error.tokens == 5000
        assert error.max_tokens == 4096
        assert "5000/4096" in str(error)

    def test_timeout_error(self):
        error = TimeoutError("Request timed out")
        assert isinstance(error, UniversalAIError)
        assert not isinstance(error, ProviderError)

    def test_configuration_error(self):
        error = ConfigurationError("Invalid config")
        assert isinstance(error, UniversalAIError)

    def test_provider_unavailable_error(self):
        error = ProviderUnavailableError()
        assert str(error) == "No healthy providers available"

    def test_stream_error(self):
        error = StreamError("Stream failed")
        assert isinstance(error, UniversalAIError)

    def test_circuit_breaker_open_error(self):
        error = CircuitBreakerOpenError("openai", timeout=30.0)
        assert error.provider == "openai"
        assert error.timeout == 30.0
        assert "Circuit breaker" in str(error)

    def test_exception_inheritance_chain(self):
        assert issubclass(AuthenticationError, ProviderError)
        assert issubclass(AuthenticationError, UniversalAIError)
        assert issubclass(RateLimitError, ProviderError)
        assert issubclass(ModelNotFoundError, ProviderError)
        assert issubclass(ContextWindowExceededError, ProviderError)
        assert issubclass(TimeoutError, UniversalAIError)
        assert issubclass(ConfigurationError, UniversalAIError)
        assert issubclass(ProviderUnavailableError, UniversalAIError)
        assert issubclass(StreamError, UniversalAIError)
        assert issubclass(CircuitBreakerOpenError, UniversalAIError)
