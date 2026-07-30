"""Benchmark CLI commands."""

from __future__ import annotations

import time

from universal_ai.utils.plugin_registry import PluginRegistry


async def cmd_benchmark_single(
    model: str | None = None,
    provider: str | None = None,
    prompt: str = "Write a short poem about technology.",
    iterations: int = 3,
) -> None:
    from universal_ai import AI

    print("UniversalAI Benchmark")
    print("=" * 50)
    print(f"Prompt: {prompt}")
    print(f"Iterations: {iterations}")
    print()
    ai = AI()
    if provider:
        await ai.set_provider(provider)
    try:
        print("Warming up...")
        await ai.chat(prompt, model=model or "auto")
        latencies = []
        tokens_per_sec = []
        total_tokens = 0
        for i in range(iterations):
            print(f"  Iteration {i + 1}/{iterations}...", end="", flush=True)
            start = time.monotonic()
            response = await ai.chat(prompt, model=model or "auto")
            elapsed = time.monotonic() - start
            latency_ms = elapsed * 1000
            tokens = response.usage.total_tokens if response.usage else 0
            tps = tokens / elapsed if elapsed > 0 else 0
            latencies.append(latency_ms)
            tokens_per_sec.append(tps)
            total_tokens += tokens
            print(f" {latency_ms:.0f}ms, {tps:.1f} tok/s")
        avg_latency = sum(latencies) / len(latencies)
        avg_tps = sum(tokens_per_sec) / len(tokens_per_sec)
        print("\n" + "=" * 50)
        print("Results:")
        print(f"  Provider: {ai._provider_name or 'auto'}")
        print(f"  Model: {response.model if response else 'unknown'}")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Avg latency: {avg_latency:.0f}ms")
        print(f"  Avg tokens/sec: {avg_tps:.1f}")
    finally:
        await ai.close()


async def cmd_benchmark_compare(
    providers: list[str] | None = None,
    prompt: str = "Write a short poem about technology.",
    iterations: int = 2,
) -> None:
    from universal_ai import AI

    registry = PluginRegistry()
    available = registry.list_providers()
    to_test = [p for p in providers if p in available] if providers else available[:5]
    if not to_test:
        print("No providers available for benchmarking")
        return
    print("UniversalAI Provider Comparison")
    print("=" * 70)
    print(f"Prompt: {prompt}")
    print(f"Iterations: {iterations}")
    print(f"Providers: {', '.join(to_test)}")
    print()
    results = []
    for provider_name in to_test:
        print(f"\nTesting {provider_name}...")
        ai = AI()
        try:
            await ai.set_provider(provider_name)
            try:
                await ai.chat(prompt)
            except Exception as e:
                print(f"  Warmup failed: {e}")
                continue
            latencies = []
            tokens_per_sec = []
            for i in range(iterations):
                try:
                    start = time.monotonic()
                    response = await ai.chat(prompt)
                    elapsed = time.monotonic() - start
                    latency_ms = elapsed * 1000
                    tokens = response.usage.total_tokens if response.usage else 0
                    tps = tokens / elapsed if elapsed > 0 else 0
                    latencies.append(latency_ms)
                    tokens_per_sec.append(tps)
                except Exception as e:
                    print(f"  Iteration {i + 1} failed: {e}")
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                avg_tps = sum(tokens_per_sec) / len(tokens_per_sec)
                results.append(
                    {
                        "provider": provider_name,
                        "model": response.model if response else "unknown",
                        "avg_latency": avg_latency,
                        "avg_tps": avg_tps,
                        "tokens": tokens,
                    }
                )
                print(f"  Avg latency: {avg_latency:.0f}ms, Avg tok/s: {avg_tps:.1f}")
        finally:
            await ai.close()
    if results:
        print("\n" + "=" * 70)
        print("Comparison Results:")
        print("-" * 70)
        print(f"{'Provider':<15} {'Model':<20} {'Latency':<12} {'Tok/s':<10}")
        print("-" * 70)
        results.sort(key=lambda x: x["avg_tps"], reverse=True)
        for r in results:
            print(
                f"{r['provider']:<15} {r['model']:<20} {r['avg_latency']:<12.0f} {r['avg_tps']:<10.1f}"
            )
        print("-" * 70)
        winner = results[0]
        print(f"\nFastest: {winner['provider']} ({winner['avg_tps']:.1f} tok/s)")


async def handle_benchmark_command(args) -> None:
    compare = getattr(args, "compare", False)
    model = getattr(args, "model", None)
    provider = getattr(args, "provider", None)
    if compare:
        providers = [provider] if provider else None
        await cmd_benchmark_compare(providers=providers)
    else:
        await cmd_benchmark_single(model=model, provider=provider)


def create_benchmark_parser(subparsers) -> None:
    p = subparsers.add_parser("benchmark", help="Benchmark AI providers")
    p.add_argument("--model", help="Model to benchmark")
    p.add_argument("-p", "--provider", help="Provider to benchmark")
    p.add_argument("--compare", action="store_true", help="Compare multiple providers")
    p.add_argument("--iterations", type=int, default=3, help="Number of iterations")
    p.add_argument("--prompt", default="Write a short poem about technology.", help="Prompt to use")
    return p
