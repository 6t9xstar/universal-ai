# UniversalAI Roadmap

This document outlines the future direction of UniversalAI. We welcome community input on priorities and features.

## Current Status (v1.0.0)

- [x] Core SDK with unified interface
- [x] 9 provider support (OpenAI, Anthropic, Gemini, Ollama, Groq, Mistral, Azure, OpenRouter, HuggingFace)
- [x] Async-first with sync wrappers
- [x] Streaming support
- [x] Tool calling with `@tool` decorator
- [x] Middleware pipeline (retry, cache, rate limit, circuit breaker, cost tracking, logging)
- [x] RAG (Retrieval-Augmented Generation)
- [x] Agent framework with coordinator pattern
- [x] CLI tool (`uai`)
- [x] 177 tests

## v1.1.0 - Enhanced Provider Support

**Target: Q3 2026**

- [ ] AWS Bedrock provider
- [ ] Google Vertex AI provider
- [ ] Together AI provider
- [ ] Fireworks AI provider
- [ ] DeepSeek provider
- [ ] LM Studio provider (local)
- [ ] vLLM provider (local)

## v1.2.0 - Advanced RAG

**Target: Q4 2026**

- [ ] PDF document loader
- [ ] HTML/web page loader
- [ ] YouTube transcript loader
- [ ] Real vector similarity search (cosine, dot product)
- [ ] Vector store backends (ChromaDB, Pinecone, Weaviate)
- [ ] Hybrid search (keyword + semantic)
- [ ] Re-ranking support
- [ ] Query decomposition

## v1.3.0 - Memory & Persistence

**Target: Q1 2027**

- [ ] SQLite memory backend
- [ ] Redis memory backend
- [ ] PostgreSQL memory backend
- [ ] Conversation persistence
- [ ] Session management
- [ ] Memory summarization
- [ ] Long-term memory with retrieval

## v1.4.0 - Advanced Agents

**Target: Q2 2027**

- [ ] Multi-agent orchestration (hierarchical)
- [ ] Agent communication protocol
- [ ] Task decomposition
- [ ] Agent memory sharing
- [ ] Tool chaining
- [ ] Parallel agent execution
- [ ] Agent evaluation metrics

## v2.0.0 - Enterprise Features

**Target: Q3 2027**

- [ ] OpenTelemetry integration
- [ ] Prometheus metrics
- [ ] Distributed tracing
- [ ] Authentication middleware
- [ ] Rate limiting per API key
- [ ] Audit logging
- [ ] Compliance features (SOC2, GDPR)

## Community-Requested Features

Vote on these features by adding a reaction to the [GitHub Issue](https://github.com/6t9xstar/universal-ai/issues):

- [ ] MCP (Model Context Protocol) support
- [ ] WebSocket streaming
- [ ] gRPC interface
- [ ] Kubernetes operator
- [ ] Terraform provider
- [ ] Docker images
- [ ] Cloud deployments (AWS, GCP, Azure)

## How to Contribute

1. Check the [CONTRIBUTING.md](CONTRIBUTING.md) guide
2. Look for issues labeled `good first issue` or `help wanted`
3. Join our [Discord](https://discord.gg/universal-ai) to discuss ideas
4. Submit a PR with your contribution

## Release Schedule

- **Major releases**: Every 6 months with breaking changes
- **Minor releases**: Every 2 months with new features
- **Patch releases**: As needed for bug fixes

## Versioning

We follow [Semantic Versioning](https://semver.org/):
- **Major**: Breaking changes
- **New features**: Backward-compatible
- **Patch**: Bug fixes only

---

*This roadmap is subject to change based on community feedback and priorities.*
