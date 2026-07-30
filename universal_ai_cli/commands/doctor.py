"""Doctor command for UniversalAI CLI - system diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


async def cmd_doctor(args: argparse.Namespace) -> None:
    from universal_ai_cli.main import cmd_doctor as _cmd_doctor

    await _cmd_doctor(args)
