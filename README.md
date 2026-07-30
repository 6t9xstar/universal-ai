# UniversalAI

[![PyPI version](https://img.shields.io/pypi/v/universal-ai.svg)](https://pypi.org/project/universal-ai)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/6t9xstar/universal-ai/actions/workflows/test.yml/badge.svg)](https://github.com/6t9xstar/universal-ai/actions)
[![Codecov](https://codecov.io/gh/6t9xstar/universal-ai/branch/main/graph/badge.svg)](https://codecov.io/gh/6t9xstar/universal-ai)
[![Discord](https://img.shields.io/badge/Discord-join-blue.svg)](https://discord.gg/universal-ai)
[![Stars](https://img.shields.io/github/stars/6t9xstar/universal-ai.svg?style=social)](https://github.com/6t9xstar/universal-ai)

**The Requests library for AI** — one unified SDK for every LLM provider. Write once, run anywhere.

```
pip install universal-ai
```

```python
from universal_ai import AI

ai = AI(provider="openai", model="gpt-4o")
response = await ai.chat("What is quantum computing?")
print(response.content)
```

---

## Why UniversalAI?

Building AI applications today means juggling multiple provider SDKs, each with different APIs, error handling, and quirks. UniversalAI gives you **one clean interface** that works across all major providers:

- **Same code** works with OpenAI, Anthropic, Gemini, Ollama, Groq, Mistral, OpenRouter, HuggingFace, and Azure OpenAI
- **Switch providers** by changing one string — no code rewrite
- **Built-in resilience** with retry, caching, rate limiting, and circuit breaker middleware
- **Tool calling** works identically across all providers that support it

## Features

| Feature | Description |
|---------|-------------|
| **9 Providers** | OpenAI, Anthropic, Gemini, Ollama, Groq, Mistral, OpenRouter, HuggingFace, Azure OpenAI |
| **Async-first** | Full `async`/`await` with synchronous wrappers for scripts and notebooks |
| **Streaming** | Real-time token streaming from any provider |
| **Tool Calling** | `@tool` decorator with automatic execution loop |
| **Middleware** | Retry, cache, rate limit, circuit breaker, cost tracking, logging |
| **Routing** | Fallback, round-robin, lowest latency, lowest cost strategies |
| **Vision** | Image-aware chat with OpenAI, Anthropic, Gemini |
| **Audio** | Transcription (Whisper) and TTS with OpenAI |
| **Image Generation** | DALL-E 3 support |
| **Embeddings** | OpenAI, Gemini, Mistral, HuggingFace, Azure, OpenRouter |
| **RAG** | Built-in retrieval-augmented generation with chunking and vector store |
| **Agents** | Multi-agent orchestration with coordinator pattern |
| **Context Safety** | Automatic validation and optional truncation |
| **Cost Tracking** | Per-request and cumulative cost estimation |
| **CLI** | Full-featured `uai` command-line tool |

## Quick Start

### Installation

```bash
# Core SDK (auto-detects available providers)
pip install universal-ai

# With specific provider support
pip install universal-ai[openai]
pip install universal-ai[anthropic]
pip install universal-ai[gemini]
pip install universal-ai[ollama]

# Everything
pip install universal-ai[all]
```

### Basic Usage

```python
import asyncio
from universal_ai import AI

async def main():
    # Auto-detect provider from environment
    ai = AI()
    
    # Chat
    response = await ai.chat("Explain quantum computing in one sentence")
    print(response.content)
    
    # Streaming
    async for chunk in ai.stream("Write a haiku about programming"):
        print(chunk.delta, end="", flush=True)
    
    # Embeddings
    embed_response = await ai.embed("Hello, world!")
    print(f"Embedding dimensions: {len(embed_response.vector)}")

asyncio.run(main())
```

### With Specific Provider

```python
from universal_ai import AI

# OpenAI
ai = AI(provider="openai", model="gpt-4o")
response = await ai.chat("Hello!")

# Anthropic
ai = AI(provider="anthropic", model="claude-sonnet-4-20250514")
response = await ai.chat("Hello!")

# Local Ollama
ai = AI(provider="ollama", model="llama3")
response = await ai.chat("Hello!")
```

### Synchronous Usage

```python
from universal_ai import AI

ai = AI(provider="openai", model="gpt-4o")

# Synchronous wrappers for scripts/notebooks
response = ai.chat_sync("Hello!")
print(response.content)

# Sync streaming (returns full text)
text = ai.stream_sync("Tell me a joke")
print(text)
```

## Tool Calling

Define tools with the `@tool` decorator and let the AI use them:

```python
from universal_ai import AI, tool

@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get current weather for a city."""
    # In a real app, call a weather API
    return f"Weather in {city}: 22°{unit[0].upper()}, sunny"

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

ai = AI(provider="openai", model="gpt-4o")

# The AI will automatically call your tools
response = await ai.chat(
    "What's the weather in Paris? Also calculate 15 * 23.",
    tools=[get_weather, calculate]
)
print(response.content)
```

### Manual Tool Execution

```python
from universal_ai import AI, tool

@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

ai = AI(provider="openai", model="gpt-4o")
ai.register_tool(search)

# Tools are auto-executed in the tool loop
response = await ai.chat("Search for Python tutorials")
```

## Conversations

Multi-turn conversations with automatic history management:

```python
from universal_ai import AI

ai = AI(provider="openai", model="gpt-4o")

# Create a conversation
conv = ai.conversation(
    system_prompt="You are a helpful cooking assistant.",
    max_turns=20
)

# Send messages
response = await conv.send(message="What should I cook for dinner?")
print(response.content)

response = await conv.send(message="Can you give me a recipe?")
print(response.content)

# Access history
print(f"Turn count: {conv.turn_count}")
print(f"Messages: {len(conv.history)}")

# Reset
conv.reset()
```

## Configuration

### Environment Variables

```bash
# Provider selection
export UNIVERSALAI_PROVIDER=openai
export UNIVERSALAI_MODEL=gpt-4o

# API keys (provider-specific)
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GEMINI_API_KEY=...
export GROQ_API_KEY=gsk_...
export MISTRAL_API_KEY=...
export OPENROUTER_API_KEY=sk-or-...
export HF_API_KEY=hf_...

# Azure OpenAI
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_API_BASE=https://your-resource.openai.azure.com
export AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o

# Ollama (local)
export OLLAMA_HOST=http://localhost:11434
```

### Config File

```yaml
# ~/.config/universalai/config.yaml
provider: openai
model: gpt-4o
temperature: 0.7
max_tokens: 4096
timeout: 30
max_retries: 3
auto_truncate: true

fallback_providers:
  - anthropic
  - gemini

provider_api_keys:
  openai: sk-...
  anthropic: sk-ant-...
```

### Programmatic Configuration

```python
from universal_ai import AI, Config

config = Config(
    provider="openai",
    model="gpt-4o",
    temperature=0.7,
    max_tokens=4096,
    timeout=30,
    max_retries=3,
    auto_truncate=True,
    provider_api_keys={
        "openai": "sk-...",
        "anthropic": "sk-ant-...",
    }
)

ai = AI(config=config)
```

## Middleware

Add resilience and observability to your requests:

```python
from universal_ai import AI
from universal_ai.middleware import (
    RetryMiddleware,
    CacheMiddleware,
    RateLimitMiddleware,
    CircuitBreakerMiddleware,
    CostTrackingMiddleware,
    LoggingMiddleware,
)

ai = AI(provider="openai", model="gpt-4o")

# Add middleware in order (executed top to bottom)
ai.add_middleware(LoggingMiddleware())
ai.add_middleware(CostTrackingMiddleware())
ai.add_middleware(RetryMiddleware(max_retries=3, base_delay=1.0))
ai.add_middleware(CacheMiddleware(ttl=300))
ai.add_middleware(RateLimitMiddleware(requests_per_minute=60))
ai.add_middleware(CircuitBreakerMiddleware(failure_threshold=5))

# All requests now go through the middleware pipeline
response = await ai.chat("Hello!")
```

### Middleware Reference

| Middleware | Purpose | Key Options |
|-----------|---------|-------------|
| `RetryMiddleware` | Retry failed requests | `max_retries`, `base_delay`, `max_delay`, `jitter` |
| `CacheMiddleware` | Cache responses | `ttl`, `backend` (memory/sqlite/redis) |
| `RateLimitMiddleware` | Limit request rate | `requests_per_minute`, `burst` |
| `CircuitBreakerMiddleware` | Stop cascading failures | `failure_threshold`, `recovery_timeout` |
| `CostTrackingMiddleware` | Track API costs | — |
| `LoggingMiddleware` | Log requests/responses | `log_level` |

## Routing Strategies

Automatically select the best provider:

```python
from universal_ai import AI
from universal_ai.router import (
    Router,
    FallbackStrategy,
    RoundRobinStrategy,
    LowestLatencyStrategy,
    LowestCostStrategy,
)

# Configure fallback in config
config = Config(
    provider="openai",
    fallback_providers=["anthropic", "gemini"]
)

ai = AI(config=config)

# Or use router directly
router = Router(
    providers=["openai", "anthropic", "gemini"],
    strategy=FallbackStrategy()
)
```

### Strategy Options

| Strategy | Behavior |
|----------|----------|
| `FallbackStrategy` | Try first provider, failover to next on error |
| `RoundRobinStrategy` | Distribute requests evenly across providers |
| `LowestLatencyStrategy` | Always use the fastest responding provider |
| `LowestCostStrategy` | Always use the cheapest provider |

## RAG (Retrieval-Augmented Generation)

Build knowledge-base powered chat:

```python
from universal_ai import AI
from universal_ai.rag import RAG, TextLoader, DirectoryLoader

# Initialize RAG
rag = RAG(chunk_size=500, chunk_overlap=50, top_k=3)

# Add content
rag.add_text("Python is a high-level programming language...")
rag.add_document(Document(content="...", source="docs.txt"))
rag.add_folder("./knowledge_base")
rag.add_url("https://example.com/article.txt")
rag.add_github("owner/repo")

# Search
chunks = await rag.search("What is Python?")
for chunk in chunks:
    print(f"Score: {chunk.content[:50]}...")

# Use with AI
ai = AI(provider="openai", model="gpt-4o")
augmented_request = await rag.augment_request(chat_request)
response = await ai.chat(augmented_request)
```

## Audio & Image

### Transcription (Whisper)

```python
ai = AI(provider="openai", model="gpt-4o")

# Transcribe audio file
text = await ai.transcribe("audio.mp3")
print(text)

# Transcribe from bytes
text = await ai.transcribe(audio_bytes)
```

### Text-to-Speech

```python
# Generate speech
audio_bytes = await ai.speak("Hello, world!", voice="alloy")
with open("output.mp3", "wb") as f:
    f.write(audio_bytes)
```

### Image Generation

```python
# Generate image
urls = await ai.image("A sunset over mountains", size="1024x1024")
print(urls[0])  # URL to generated image
```

## CLI Usage

UniversalAI includes a full-featured command-line tool:

```bash
# Chat interactively
uai chat

# Chat with specific provider
uai chat -p openai -m gpt-4o

# Send a single message
uai chat "What is machine learning?"

# List available providers
uai providers

# Run diagnostics
uai doctor

# Manage configuration
uai config show
uai config set provider openai
uai config set-api-key openai

# Benchmark providers
uai benchmark --iterations 10

# Start local API server
uai serve --port 8000
```

## Provider Details

### OpenAI

```python
ai = AI(provider="openai", model="gpt-4o")

# Features: Chat, Streaming, Vision, Tools, Embeddings, Audio, Image Gen
# Requires: OPENAI_API_KEY
```

### Anthropic

```python
ai = AI(provider="anthropic", model="claude-sonnet-4-20250514")

# Features: Chat, Streaming, Vision, Tools
# Requires: ANTHROPIC_API_KEY
```

### Gemini

```python
ai = AI(provider="gemini", model="gemini-2.0-flash")

# Features: Chat, Streaming, Vision, Tools, Embeddings
# Requires: GEMINI_API_KEY
```

### Ollama (Local)

```python
ai = AI(provider="ollama", model="llama3")

# Features: Chat, Streaming, Embeddings
# Requires: Ollama running locally
# Install: https://ollama.ai
```

### Groq

```python
ai = AI(provider="groq", model="llama-3.1-70b-versatile")

# Features: Chat, Streaming, Tools
# Requires: GROQ_API_KEY
```

### Mistral

```python
ai = AI(provider="mistral", model="mistral-large-latest")

# Features: Chat, Streaming, Tools, Embeddings
# Requires: MISTRAL_API_KEY
```

### OpenRouter

```python
ai = AI(provider="openrouter", model="openai/gpt-4o")

# Features: Chat, Streaming, Vision, Tools, Embeddings
# Requires: OPENROUTER_API_KEY
```

### HuggingFace

```python
ai = AI(provider="huggingface", model="meta-llama/Llama-2-7b-chat-hf")

# Features: Chat, Streaming, Embeddings
# Requires: HF_API_KEY
```

### Azure OpenAI

```python
ai = AI(provider="azure", model="gpt-4o")

# Features: Chat, Streaming, Vision, Tools, Embeddings
# Requires: AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_BASE
```

## Error Handling

```python
from universal_ai import AI
from universal_ai.exceptions import (
    AuthenticationError,
    RateLimitError,
    ContextWindowExceededError,
    ProviderError,
    TimeoutError,
)

ai = AI(provider="openai", model="gpt-4o")

try:
    response = await ai.chat("Hello!")
except AuthenticationError as e:
    print(f"Invalid API key: {e}")
except RateLimitError as e:
    print(f"Rate limited, retry after: {e.retry_after}s")
except ContextWindowExceededError as e:
    print(f"Context too long: {e.estimated_tokens} > {e.context_window}")
except ProviderError as e:
    print(f"Provider error: {e}")
except TimeoutError:
    print("Request timed out")
```

## Context Window Safety

UniversalAI validates that messages fit within the provider's context window:

```python
from universal_ai import AI, Config

# Option 1: Raise error if too long (default)
config = Config(auto_truncate=False)
ai = AI(config=config)

# Option 2: Auto-truncate to fit
config = Config(auto_truncate=True)
ai = AI(config=config)
```

## Cost Estimation

```python
ai = AI(provider="openai", model="gpt-4o")

# Estimate cost before sending
estimated_cost = ai.estimate_cost("Hello, world!")
print(f"Estimated cost: ${estimated_cost:.6f}")

# Track actual costs with middleware
from universal_ai.middleware import CostTrackingMiddleware

cost_middleware = CostTrackingMiddleware()
ai.add_middleware(cost_middleware)

response = await ai.chat("Hello!")
print(f"Actual cost: ${response.usage.estimated_cost:.6f}")
print(f"Total cost: ${cost_middleware.total_cost:.6f}")
```

## Sync Wrappers

For scripts and notebooks where you can't use `async`:

| Async Method | Sync Wrapper |
|--------------|--------------|
| `await ai.chat(...)` | `ai.chat_sync(...)` |
| `async for chunk in ai.stream(...)` | `ai.stream_sync(...)` |
| `await ai.embed(...)` | `ai.embed_sync(...)` |
| `await ai.chat_with_tools(...)` | `ai.chat_with_tools_sync(...)` |
| `await ai.chat_json(...)` | `ai.chat_json_sync(...)` |

## Examples

See the [`examples/`](examples/) directory for complete working examples:

- `basic_chat.py` - Simple chat usage
- `streaming.py` - Real-time streaming
- `tool_calling.py` - Tool definition and execution
- `middleware_demo.py` - Middleware configuration
- `rag_demo.py` - RAG with document loading
- `multi_provider.py` - Provider switching

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Clone the repo
git clone https://github.com/6t9xstar/universal-ai.git
cd universal-ai

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .

# Run type checking
mypy .
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Community

- [Discord](https://discord.gg/universal-ai) - Join the community
- [GitHub Discussions](https://github.com/6t9xstar/universal-ai/discussions) - Ask questions
- [GitHub Issues](https://github.com/6t9xstar/universal-ai/issues) - Report bugs

## Acknowledgments

Built with love for the AI community. Inspired by [Requests](https://docs.python-requests.org/), [SQLAlchemy](https://www.sqlalchemy.org/), and [Pillow](https://python-pillow.org/).

---

<p align="center">
  <a href="https://github.com/6t9xstar/universal-ai">
    <img src="https://img.shields.io/github/stars/6t9xstar/universal-ai?style=social" alt="Star on GitHub">
  </a>
  <a href="https://discord.gg/universal-ai">
    <img src="https://img.shields.io/badge/Discord-Join-7289da?style=flat&logo=discord&logoColor=white" alt="Join Discord">
  </a>
  <a href="https://pypi.org/project/universal-ai/">
    <img src="https://img.shields.io/pypi/v/universal-ai?color=blue&label=PyPI" alt="PyPI">
  </a>
</p>
