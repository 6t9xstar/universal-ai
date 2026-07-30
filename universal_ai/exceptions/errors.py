from __future__ import annotations


class UniversalAIError(Exception):
    def __init__(self, message: str, provider: str | None = None) -> None:
        self.provider = provider
        super().__init__(message)


class ProviderError(UniversalAIError):
    def __init__(
        self,
        provider: str,
        message: str,
        status_code: int | None = None,
        raw: dict | None = None,
    ) -> None:
        self.provider = provider
        self.status_code = status_code
        self.raw = raw or {}
        super().__init__(f"[{provider}] {message}", provider)

    def __repr__(self) -> str:
        return (
            f"ProviderError(provider={self.provider!r}, "
            f"status={self.status_code}, message={str(self)!r})"
        )


class AuthenticationError(ProviderError):
    pass


class RateLimitError(ProviderError):
    def __init__(
        self,
        provider: str,
        message: str = "Rate limit exceeded",
        retry_after: float | None = None,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(provider, message)


class ModelNotFoundError(ProviderError):
    pass


class ContextWindowExceededError(ProviderError):
    def __init__(self, provider: str, tokens: int, max_tokens: int) -> None:
        self.tokens = tokens
        self.max_tokens = max_tokens
        super().__init__(provider, f"Context window exceeded: {tokens}/{max_tokens}")


class TimeoutError(UniversalAIError):
    pass


class ConfigurationError(UniversalAIError):
    pass


class ProviderUnavailableError(UniversalAIError):
    def __init__(self, message: str = "No healthy providers available") -> None:
        super().__init__(message)


class StreamError(UniversalAIError):
    pass


class CircuitBreakerOpenError(UniversalAIError):
    def __init__(self, provider: str, timeout: float = 0) -> None:
        self.provider = provider
        self.timeout = timeout
        super().__init__(
            f"Circuit breaker open for {provider}, retry in {timeout:.0f}s",
            provider=provider,
        )
