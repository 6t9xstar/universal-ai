# Contributing to UniversalAI

Thank you for your interest in contributing to UniversalAI! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Style Guide](#style-guide)
- [Adding Providers](#adding-providers)

## Code of Conduct

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing. We expect all contributors to follow it.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/universal-ai.git
   cd universal-ai
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/6t9xstar/universal-ai.git
   ```

## Development Setup

### Prerequisites

- Python 3.10 or higher
- pip or uv package manager

### Installation

1. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks** (optional):
   ```bash
   pre-commit install
   ```

### Project Structure

```
universal-ai/
├── universal_ai/           # Core SDK package
│   ├── ai.py              # Main AI class
│   ├── config.py          # Configuration management
│   ├── conversation.py    # Conversation management
│   ├── tools.py           # Tool decorator and executor
│   ├── client.py          # HTTP client
│   ├── router.py          # Provider routing
│   ├── models/            # Pydantic models
│   ├── middleware/         # Middleware components
│   ├── providers/         # Provider implementations
│   ├── streaming/         # Streaming support
│   ├── events/            # Event system
│   └── utils/             # Utilities
├── universal_ai_cli/      # CLI package
├── tests/                 # Test suite
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
└── examples/              # Usage examples
```

## Making Changes

### Branching Strategy

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** in small, focused commits
3. **Write tests** for new functionality
4. **Update documentation** if needed
5. **Push to your fork** and create a pull request

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat: Add new provider for XYZ`
- `fix: Fix cache middleware race condition`
- `docs: Update README with new examples`
- `test: Add integration tests for streaming`
- `refactor: Simplify router logic`

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run integration tests only
pytest tests/integration/

# Run with coverage
pytest --cov=universal_ai

# Run specific test file
pytest tests/unit/test_models.py

# Run specific test
pytest tests/unit/test_models.py::TestMessage::test_create_user_message
```

### Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Use descriptive test names: `test_chat_with_mock_provider`
- Follow the Arrange-Act-Assert pattern
- Mock external dependencies (API calls, file I/O)

Example:

```python
@pytest.mark.asyncio
async def test_chat_with_mock_provider():
    # Arrange
    registry = PluginRegistry()
    registry.register("mock", MockProvider)
    ai = AI(provider="mock")

    # Act
    response = await ai.chat("Hello!")

    # Assert
    assert isinstance(response, ChatResponse)
    assert response.content == "Hello from mock provider!"
```

## Pull Request Process

1. **Update documentation** for any new features
2. **Add tests** for new functionality
3. **Ensure all tests pass**: `pytest`
4. **Run linting**: `ruff check .`
5. **Run type checking**: `mypy .`
6. **Update CHANGELOG.md** with your changes
7. **Create a pull request** with a clear description

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests pass

## Checklist
- [ ] Code follows style guide
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

## Style Guide

### Python Style

- Follow PEP 8
- Use type hints for all public APIs
- Keep functions under 50 lines
- Use descriptive variable names
- Add docstrings to all public classes/functions

### Code Formatting

We use `ruff` for formatting:

```bash
# Check formatting
ruff check .

# Auto-fix formatting
ruff check --fix .
```

### Type Checking

We use `mypy` for type checking:

```bash
mypy .
```

## Adding Providers

To add a new provider:

1. **Create provider file**: `universal_ai/providers/your_provider.py`

2. **Implement the provider interface**:
   ```python
   class YourProvider:
       name = "your_provider"
       version = "0.1.0"
       default_model = "model-name"

       async def chat(self, request: ChatRequest) -> ChatResponse:
           ...

       async def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
           ...

       async def embed(self, request: EmbedRequest) -> EmbedResponse:
           ...

       async def health_check(self) -> dict:
           ...

       async def get_latency(self) -> float:
           ...

       def calculate_cost(self, estimated_tokens: int) -> float:
           ...

       async def close(self) -> None:
           ...
   ```

3. **Register in plugin_registry.py**:
   ```python
   _BUILTIN_PROVIDERS = [
       ...
       ("your_provider", "universal_ai.providers.your_provider", "YourProvider"),
   ]
   ```

4. **Add to `__init__.py`**:
   ```python
   from universal_ai.providers.your_provider import YourProvider
   ```

5. **Add optional dependency** in `pyproject.toml`:
   ```toml
   [project.optional-dependencies]
   your_provider = ["your-dependency>=1.0.0"]
   ```

6. **Write tests** in `tests/unit/test_your_provider.py`

7. **Update documentation**

## Questions?

If you have questions, feel free to:
- Open an issue
- Join our Discord community
- Reach out on Twitter

Thank you for contributing to UniversalAI!
