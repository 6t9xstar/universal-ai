"""Local API server CLI command."""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

try:
    from aiohttp import web

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


async def handle_chat(request: web.Request) -> web.Response:
    from universal_ai import AI

    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)
    message = data.get("message") or data.get("prompt")
    if not message:
        return web.json_response({"error": "Message required"}, status=400)
    model = data.get("model")
    provider = data.get("provider")
    temperature = data.get("temperature")
    max_tokens = data.get("max_tokens")
    stream = data.get("stream", False)
    ai = AI()
    try:
        if provider:
            await ai.set_provider(provider)
        kwargs: dict[str, Any] = {}
        if model:
            kwargs["model"] = model
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if stream:
            response = web.StreamResponse(
                status=200,
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
            await response.prepare(request)
            async for chunk in ai.stream(message, **kwargs):
                chunk_data = {
                    "content": chunk.delta,
                    "model": chunk.model,
                    "finish_reason": chunk.finish_reason.value if chunk.finish_reason else None,
                }
                await response.write(f"data: {json.dumps(chunk_data)}\n\n".encode())
            await response.write(b"data: [DONE]\n\n")
            return response
        response = await ai.chat(message, **kwargs)
        return web.json_response(
            {
                "content": response.content,
                "model": response.model,
                "provider": response.provider,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                if response.usage
                else None,
                "finish_reason": response.finish_reason.value if response.finish_reason else None,
            }
        )
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)
    finally:
        await ai.close()


async def handle_models(request: web.Request) -> web.Response:
    from universal_ai import AI

    ai = AI()
    try:
        providers = await ai.list_providers()
        return web.json_response({"providers": providers})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)
    finally:
        await ai.close()


async def handle_health(request: web.Request) -> web.Response:
    from universal_ai import AI

    ai = AI()
    try:
        status = await ai.provider_status()
        return web.json_response({"status": "healthy", "providers": status})
    except Exception as e:
        return web.json_response({"status": "error", "error": str(e)}, status=500)
    finally:
        await ai.close()


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_post("/v1/chat", handle_chat)
    app.router.add_post("/v1/chat/completions", handle_chat)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/", handle_health)
    return app


async def cmd_serve(host: str = "127.0.0.1", port: int = 8000, provider: str | None = None) -> None:
    if not HAS_AIOHTTP:
        print("Error: aiohttp is required for the serve command", file=sys.stderr)
        print("Install with: pip install aiohttp", file=sys.stderr)
        sys.exit(1)
    print("UniversalAI API Server")
    print("=" * 40)
    print(f"Listening on: http://{host}:{port}")
    print("Endpoints:")
    print("  POST /v1/chat          - Chat completion")
    print("  POST /v1/chat/completions - Chat completion (OpenAI-compatible)")
    print("  GET  /v1/models        - List providers")
    print("  GET  /health           - Health check")
    print("\nPress Ctrl+C to stop.\n")
    app = create_app()
    try:
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await runner.cleanup()


def create_serve_parser(subparsers) -> None:
    p = subparsers.add_parser("serve", help="Start local API server")
    p.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    p.add_argument("--port", type=int, default=8000, help="Port to bind to")
    p.add_argument("-p", "--provider", help="Default provider to use")
    return p


async def handle_serve_command(args) -> None:
    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 8000)
    provider = getattr(args, "provider", None)
    await cmd_serve(host=host, port=port, provider=provider)
