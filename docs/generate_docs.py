"""
Utility script to regenerate EvoMind documentation artifacts.

Usage:
    python docs/generate_docs.py

The script refreshes the configuration reference markdown and renders API
documentation with pdoc into ``docs/site``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from evomind.utils.config_reference import write_markdown


def build_config_reference() -> None:
    docs_dir = Path(__file__).resolve().parent
    target = docs_dir / "config_reference.md"
    write_markdown(target)
    repo_root = docs_dir.parent
    write_markdown(repo_root / "CONFIG.md")


def build_api_docs() -> None:
    docs_dir = Path(__file__).resolve().parent
    output_dir = docs_dir / "site"
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "pdoc",
        "evomind",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    build_config_reference()
    try:
        build_api_docs()
    except (ImportError, subprocess.CalledProcessError):
        print("pdoc not installed or failed to run; skipping API docs build.")


if __name__ == "__main__":
    main()
