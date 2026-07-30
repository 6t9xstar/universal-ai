"""Tests for CLI commands."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestConfigCommands:
    """Tests for config CLI commands."""

    def test_get_config_path(self):
        from universal_ai_cli.commands.config import get_config_path

        path = get_config_path()
        assert path.name == "config.json"
        assert "universalai" in str(path)

    def test_load_config_empty(self):
        from universal_ai_cli.commands.config import load_config

        with patch("universal_ai_cli.commands.config.get_config_path") as mock_path:
            mock_path.return_value = Path(tempfile.mktemp(suffix=".json"))
            config = load_config()
            assert isinstance(config, dict)

    def test_save_and_load_config(self):
        from universal_ai_cli.commands.config import load_config, save_config

        with patch("universal_ai_cli.commands.config.get_config_path") as mock_path:
            config_file = Path(tempfile.mktemp(suffix=".json"))
            mock_path.return_value = config_file

            test_config = {"provider": "openai", "model": "gpt-4"}
            save_config(test_config)
            loaded = load_config()

            assert loaded == test_config

            # Cleanup
            config_file.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_cmd_config_show(self, capsys):
        from universal_ai_cli.commands.config import cmd_config_show

        await cmd_config_show()
        captured = capsys.readouterr()
        assert "UniversalAI Configuration" in captured.out

    @pytest.mark.asyncio
    async def test_cmd_config_set(self):
        from universal_ai_cli.commands.config import cmd_config_set, load_config

        with patch("universal_ai_cli.commands.config.get_config_path") as mock_path:
            config_file = Path(tempfile.mktemp(suffix=".json"))
            mock_path.return_value = config_file

            await cmd_config_set("test_key", "test_value")
            config = load_config()

            assert config.get("test_key") == "test_value"

            # Cleanup
            config_file.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_cmd_config_set_boolean(self):
        from universal_ai_cli.commands.config import cmd_config_set, load_config

        with patch("universal_ai_cli.commands.config.get_config_path") as mock_path:
            config_file = Path(tempfile.mktemp(suffix=".json"))
            mock_path.return_value = config_file

            await cmd_config_set("enabled", "true")
            config = load_config()

            assert config.get("enabled") is True

            # Cleanup
            config_file.unlink(missing_ok=True)


class TestLoginCommands:
    """Tests for login/logout CLI commands."""

    def test_get_credentials_path(self):
        from universal_ai_cli.commands.login import get_credentials_path

        path = get_credentials_path()
        assert path.name == "credentials.json"
        assert "universal-ai" in str(path)

    def test_provider_env_map(self):
        from universal_ai_cli.commands.login import PROVIDER_ENV_MAP

        assert "openai" in PROVIDER_ENV_MAP
        assert PROVIDER_ENV_MAP["openai"] == "OPENAI_API_KEY"

    def test_load_credentials_empty(self):
        from universal_ai_cli.commands.login import load_credentials

        with patch("universal_ai_cli.commands.login.get_credentials_path") as mock_path:
            mock_path.return_path = Path(tempfile.mktemp(suffix=".json"))
            creds = load_credentials()
            assert isinstance(creds, dict)

    @pytest.mark.asyncio
    async def test_cmd_login_status(self, capsys):
        from universal_ai_cli.commands.login import cmd_login_status

        await cmd_login_status()
        captured = capsys.readouterr()
        assert "Provider Login Status" in captured.out


class TestModelsCommands:
    """Tests for models CLI commands."""

    @pytest.mark.asyncio
    async def test_cmd_models_list(self, capsys):
        from universal_ai_cli.commands.models import cmd_models_list

        await cmd_models_list()
        captured = capsys.readouterr()
        assert "Available models" in captured.out or "No providers" in captured.out


class TestBenchmarkCommands:
    """Tests for benchmark CLI commands."""

    def test_create_benchmark_parser(self):
        import argparse

        from universal_ai_cli.commands.benchmark import create_benchmark_parser

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_benchmark_parser(subparsers)


class TestServeCommands:
    """Tests for serve CLI commands."""

    def test_serve_imports(self):
        from universal_ai_cli.commands.serve import HAS_AIOHTTP

        # Just test that the import works
        assert isinstance(HAS_AIOHTTP, bool)


class TestMainCLI:
    """Tests for main CLI module."""

    def test_create_parser(self):
        from universal_ai_cli.main import create_parser

        parser = create_parser()
        assert parser is not None

        # Test parsing chat command
        args = parser.parse_args(["chat", "hello"])
        assert args.command == "chat"
        assert args.message == "hello"

    def test_create_parser_providers(self):
        from universal_ai_cli.main import create_parser

        parser = create_parser()
        args = parser.parse_args(["providers"])
        assert args.command == "providers"

    def test_create_parser_models(self):
        from universal_ai_cli.main import create_parser

        parser = create_parser()
        args = parser.parse_args(["models", "list"])
        assert args.command == "models"
        assert args.action == "list"

    def test_create_parser_benchmark(self):
        from universal_ai_cli.main import create_parser

        parser = create_parser()
        args = parser.parse_args(["benchmark", "--compare"])
        assert args.command == "benchmark"
        assert args.compare is True

    def test_create_parser_config(self):
        from universal_ai_cli.main import create_parser

        parser = create_parser()
        args = parser.parse_args(["config", "show"])
        assert args.command == "config"
        assert args.action == "show"

    def test_create_parser_login(self):
        from universal_ai_cli.main import create_parser

        parser = create_parser()
        args = parser.parse_args(["login", "openai"])
        assert args.command == "login"
        assert args.provider == "openai"

    def test_create_parser_serve(self):
        from universal_ai_cli.main import create_parser

        parser = create_parser()
        args = parser.parse_args(["serve", "--port", "9000"])
        assert args.command == "serve"
        assert args.port == 9000
