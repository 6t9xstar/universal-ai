# Changelog

All notable changes to UniversalAI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-30

### Added
- **Core SDK**
  - `AI` class with unified interface for all providers
  - Async-first design with synchronous wrappers
  - Context manager support (`async with AI() as ai:`)
  
- **Providers** (8 total)
  - OpenAI (GPT-4, GPT-3.5, embeddings, vision, tools)
  - Anthropic (Claude, embeddings)
  - Gemini (Google AI, embeddings, vision)
  - Ollama (local models)
  - Groq (fast inference)
  - Mistral (Mistral AI, embeddings)
  - OpenRouter (multi-provider)
  - HuggingFace (Inference API)

- **Features**
  - Chat completions (`ai.chat()`)
  - Streaming (`ai.stream()`)
  - Embeddings (`ai.embed()`)
  - Tool calling with `@tool` decorator
  - Conversation management (`Conversation` class)
  - Structured output (`ai.chat_json()`)

- **Middleware**
  - Retry middleware with exponential backoff
  - Cache middleware with TTL support
  - Rate limiting middleware
  - Circuit breaker middleware
  - Cost tracking middleware
  - Logging middleware

- **Routing**
  - Fallback strategy
  - Round-robin strategy
  - Lowest latency strategy
  - Lowest cost strategy
  - Capability-aware routing

- **CLI** (`uai` command)
  - `uai chat` - Chat with AI models
  - `uai providers` - List provider status
  - `uai models` - Manage models (list/pull/remove/info)
  - `uai benchmark` - Benchmark providers
  - `uai config` - Manage configuration
  - `uai login/logout` - Provider authentication
  - `uai doctor` - System diagnostics
  - `uai serve` - Local API server

- **Configuration**
  - Environment variable support (`UNIVERSALAI_*`)
  - Config file support (YAML/JSON)
  - Layered configuration with merge precedence

- **Testing**
  - 177 unit and integration tests
  - Mock provider for testing
  - Comprehensive test coverage

### Changed
- N/A (initial release)

### Deprecated
- N/A (initial release)

### Removed
- N/A (initial release)

### Fixed
- N/A (initial release)

### Security
- Credentials stored securely with restrictive permissions
- API keys never logged or exposed in error messages
