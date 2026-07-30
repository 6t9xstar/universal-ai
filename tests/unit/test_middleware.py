"""Tests for middleware system."""

import pytest

from universal_ai.exceptions import CircuitBreakerOpenError, RateLimitError
from universal_ai.middleware.base import Middleware, MiddlewareChain, Request, Response
from universal_ai.middleware.cache import CACHE_HIT_KEY, CacheMiddleware, MemoryCacheBackend
from universal_ai.middleware.circuit_breaker import CircuitBreakerMiddleware
from universal_ai.middleware.cost_tracking import CostTrackingMiddleware
from universal_ai.middleware.logging import LoggingMiddleware
from universal_ai.middleware.rate_limit import RateLimitMiddleware
from universal_ai.middleware.retry import RetryMiddleware
from universal_ai.models.common import Usage


class PassThroughMiddleware(Middleware):
    async def process_request(self, request):
        return request

    async def process_response(self, request, response):
        return response

    async def process_error(self, request, error):
        return error


class ModifyMiddleware(Middleware):
    async def process_request(self, request):
        request.kwargs["modified"] = True
        return request

    async def process_response(self, request, response):
        response.metadata["modified"] = True
        return response

    async def process_error(self, request, error):
        return error


class TestMiddlewareChain:
    async def test_empty_chain(self):
        chain = MiddlewareChain()
        req = Request(provider="openai", model="gpt-4o", messages=[])
        resp = Response(content="Test")
        result_req = await chain.run_request(req)
        result_resp = await chain.run_response(req, resp)
        assert result_req == req
        assert result_resp == resp

    async def test_add_middleware(self):
        chain = MiddlewareChain()
        chain.add(PassThroughMiddleware())
        assert len(chain.middleware) == 1

    async def test_add_chaining(self):
        chain = MiddlewareChain()
        result = chain.add(PassThroughMiddleware()).add(PassThroughMiddleware())
        assert result is chain
        assert len(chain.middleware) == 2

    async def test_clear(self):
        chain = MiddlewareChain()
        chain.add(PassThroughMiddleware())
        chain.add(PassThroughMiddleware())
        chain.clear()
        assert len(chain.middleware) == 0

    async def test_modify_request(self):
        chain = MiddlewareChain()
        chain.add(ModifyMiddleware())
        req = Request(provider="openai", model="gpt-4o", messages=[])
        result = await chain.run_request(req)
        assert result.kwargs.get("modified") is True

    async def test_modify_response(self):
        chain = MiddlewareChain()
        chain.add(ModifyMiddleware())
        req = Request(provider="openai", model="gpt-4o", messages=[])
        resp = Response(content="Test")
        result = await chain.run_response(req, resp)
        assert result.metadata.get("modified") is True

    async def test_error_chain(self):
        chain = MiddlewareChain()
        chain.add(PassThroughMiddleware())
        req = Request(provider="openai", model="gpt-4o", messages=[])
        error = Exception("test error")
        result = await chain.run_error(req, error)
        assert result is error

    async def test_error_none_return(self):
        class RetryMiddleware(Middleware):
            async def process_request(self, request):
                return request

            async def process_response(self, request, response):
                return response

            async def process_error(self, request, error):
                return None

        chain = MiddlewareChain()
        chain.add(RetryMiddleware())
        req = Request(provider="openai", model="gpt-4o", messages=[])
        result = await chain.run_error(req, Exception("test"))
        assert result is None

    async def test_repr(self):
        chain = MiddlewareChain()
        chain.add(PassThroughMiddleware())
        assert "PassThroughMiddleware" in repr(chain)


class TestLoggingMiddleware:
    async def test_logging_does_not_modify(self):
        mw = LoggingMiddleware()
        req = Request(provider="openai", model="gpt-4o", messages=[])
        result = await mw.process_request(req)
        assert result.provider == "openai"
        assert result.model == "gpt-4o"

    async def test_response_not_modified(self):
        mw = LoggingMiddleware()
        req = Request(provider="openai", model="gpt-4o", messages=[])
        resp = Response(content="Hello")
        result = await mw.process_response(req, resp)
        assert result.content == "Hello"

    async def test_error_not_modified(self):
        mw = LoggingMiddleware()
        req = Request(provider="openai", model="gpt-4o", messages=[])
        error = Exception("test")
        result = await mw.process_error(req, error)
        assert result is error


class TestCacheMiddleware:
    async def test_cache_miss(self):
        backend = MemoryCacheBackend()
        mw = CacheMiddleware(backend=backend, ttl=3600)
        req = Request(
            provider="openai", model="gpt-4o", messages=[{"role": "user", "content": "Hi"}]
        )
        result = await mw.process_request(req)
        assert CACHE_HIT_KEY not in result.kwargs

    async def test_cache_hit(self):
        backend = MemoryCacheBackend()
        mw = CacheMiddleware(backend=backend, ttl=3600)
        req = Request(
            provider="openai", model="gpt-4o", messages=[{"role": "user", "content": "Hi"}]
        )
        resp = Response(content="Cached response")
        key = mw._make_key(req)
        await backend.set(key, resp, 3600)
        result = await mw.process_request(req)
        assert CACHE_HIT_KEY in result.kwargs
        assert result.kwargs[CACHE_HIT_KEY].content == "Cached response"

    async def test_cache_store(self):
        backend = MemoryCacheBackend()
        mw = CacheMiddleware(backend=backend, ttl=3600)
        req = Request(
            provider="openai", model="gpt-4o", messages=[{"role": "user", "content": "Hi"}]
        )
        resp = Response(content="New response")
        await mw.process_response(req, resp)
        key = mw._make_key(req)
        cached = await backend.get(key)
        assert cached.content == "New response"

    async def test_cache_key_deterministic(self):
        mw = CacheMiddleware()
        req1 = Request(
            provider="openai", model="gpt-4o", messages=[{"role": "user", "content": "Hi"}]
        )
        req2 = Request(
            provider="openai", model="gpt-4o", messages=[{"role": "user", "content": "Hi"}]
        )
        assert mw._make_key(req1) == mw._make_key(req2)

    async def test_different_requests_different_keys(self):
        mw = CacheMiddleware()
        req1 = Request(
            provider="openai", model="gpt-4o", messages=[{"role": "user", "content": "Hi"}]
        )
        req2 = Request(
            provider="openai", model="gpt-4o", messages=[{"role": "user", "content": "Hello"}]
        )
        assert mw._make_key(req1) != mw._make_key(req2)

    async def test_error_not_modified(self):
        mw = CacheMiddleware()
        req = Request(provider="openai", model="gpt-4o", messages=[])
        error = Exception("test")
        result = await mw.process_error(req, error)
        assert result is error


class TestRetryMiddleware:
    async def test_no_retry_on_success(self):
        mw = RetryMiddleware(max_retries=3)
        req = Request(provider="openai", model="gpt-4o", messages=[])
        req = await mw.process_request(req)
        resp = Response(content="Success")
        result = await mw.process_response(req, resp)
        assert result.content == "Success"

    async def test_retry_on_rate_limit(self):
        mw = RetryMiddleware(max_retries=2, base_delay=0.01)
        req = Request(provider="openai", model="gpt-4o", messages=[])
        req = await mw.process_request(req)
        error = RateLimitError("openai", "Rate limited")
        result = await mw.process_error(req, error)
        assert result is None
        assert req.kwargs["_retry_attempts"] == 1

    async def test_max_retries_exceeded(self):
        mw = RetryMiddleware(max_retries=1, base_delay=0.01)
        req = Request(provider="openai", model="gpt-4o", messages=[])
        req = await mw.process_request(req)
        req.kwargs["_retry_attempts"] = 1
        error = RateLimitError("openai", "Rate limited")
        result = await mw.process_error(req, error)
        assert result is error

    async def test_no_retry_on_other_errors(self):
        mw = RetryMiddleware(max_retries=3)
        req = Request(provider="openai", model="gpt-4o", messages=[])
        req = await mw.process_request(req)
        error = Exception("Other error")
        result = await mw.process_error(req, error)
        assert result is error


class TestRateLimitMiddleware:
    async def test_basic_rate_limit(self):
        mw = RateLimitMiddleware(requests_per_minute=10)
        req = Request(provider="openai", model="gpt-4o", messages=[])
        result = await mw.process_request(req)
        assert result.kwargs.get("_rate_limited") is True

    async def test_release_on_response(self):
        mw = RateLimitMiddleware(requests_per_minute=10)
        req = Request(provider="openai", model="gpt-4o", messages=[])
        req.kwargs["_rate_limited"] = True
        resp = Response(content="Test")
        await mw.process_response(req, resp)
        assert "_rate_limited" not in req.kwargs

    async def test_release_on_error(self):
        mw = RateLimitMiddleware(requests_per_minute=10)
        req = Request(provider="openai", model="gpt-4o", messages=[])
        req.kwargs["_rate_limited"] = True
        error = Exception("test")
        result = await mw.process_error(req, error)
        assert result is error


class TestCircuitBreakerMiddleware:
    async def test_closed_allows_requests(self):
        mw = CircuitBreakerMiddleware(failure_threshold=3, reset_timeout=1.0)
        req = Request(provider="openai", model="gpt-4o", messages=[])
        result = await mw.process_request(req)
        assert result is req

    async def test_opens_after_failures(self):
        mw = CircuitBreakerMiddleware(failure_threshold=3, reset_timeout=60.0)
        req = Request(provider="openai", model="gpt-4o", messages=[])
        for _ in range(3):
            await mw.process_error(req, Exception("fail"))
        assert mw.is_open

    async def test_open_rejects_requests(self):
        mw = CircuitBreakerMiddleware(failure_threshold=1, reset_timeout=60.0)
        req = Request(provider="openai", model="gpt-4o", messages=[])
        await mw.process_error(req, Exception("fail"))
        with pytest.raises(CircuitBreakerOpenError):
            await mw.process_request(req)

    async def test_success_resets(self):
        mw = CircuitBreakerMiddleware(failure_threshold=3, reset_timeout=60.0)
        req = Request(provider="openai", model="gpt-4o", messages=[])
        resp = Response(content="Success")
        for _ in range(2):
            await mw.process_error(req, Exception("fail"))
        await mw.process_response(req, resp)
        assert mw.is_closed

    async def test_manual_reset(self):
        mw = CircuitBreakerMiddleware(failure_threshold=1, reset_timeout=60.0)
        req = Request(provider="openai", model="gpt-4o", messages=[])
        await mw.process_error(req, Exception("fail"))
        assert mw.is_open
        mw.reset()
        assert mw.is_closed


class TestCostTrackingMiddleware:
    async def test_tracks_cost(self):
        mw = CostTrackingMiddleware()
        req = Request(provider="openai", model="gpt-4o", messages=[])
        resp = Response(
            content="Hello",
            usage=Usage(input_tokens=100, output_tokens=50, total_tokens=150),
        )
        result = await mw.process_response(req, resp)
        assert "cost" in result.metadata
        assert result.metadata["cost"] > 0
        assert mw.total_cost > 0
        assert mw.request_count == 1

    async def test_cumulative_cost(self):
        mw = CostTrackingMiddleware()
        req = Request(provider="openai", model="gpt-4o", messages=[])
        resp = Response(
            content="Hello", usage=Usage(input_tokens=100, output_tokens=50, total_tokens=150)
        )
        await mw.process_response(req, resp)
        await mw.process_response(req, resp)
        assert mw.request_count == 2

    async def test_reset(self):
        mw = CostTrackingMiddleware()
        req = Request(provider="openai", model="gpt-4o", messages=[])
        resp = Response(
            content="Hello", usage=Usage(input_tokens=100, output_tokens=50, total_tokens=150)
        )
        await mw.process_response(req, resp)
        mw.reset()
        assert mw.total_cost == 0.0
        assert mw.request_count == 0
