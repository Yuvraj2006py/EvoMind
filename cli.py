"""
Command line interface for the EvoMind SDK.

Examples
--------
Run AutoML pipeline::

    python cli.py run --data data/grocery_chain_data.json --task auto --insights

List registered models::

    python cli.py list-models
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evomind import EvoMind


def _default_config_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Configuration file not found: {candidate}")


def _run_command(args: argparse.Namespace) -> None:
    config_source = _default_config_path(args.config) if args.config else None
    em = EvoMind(
        data=args.data,
        task=args.task or "auto",
        insights=not args.no_insights,
        config=config_source,
    )
    result = em.run()
    print(json.dumps({"run_id": result.run_id, "metrics": result.metrics}, indent=2))
    if args.export:
        report_path = result.export_report(args.export)
        if report_path:
            print(f"Report exported to: {report_path}")


def _list_models(_: argparse.Namespace) -> None:
    models_dir = Path("models")
    if not models_dir.exists():
        print("No models registered yet.")
        return
    for entry in sorted(models_dir.iterdir()):
        if entry.is_dir():
            print(entry.name)


def _load_model(args: argparse.Namespace) -> None:
    model_dir = Path("models") / args.model_id
    if not model_dir.exists():
        raise SystemExit(f"Model '{args.model_id}' not found in registry.")
    print(f"Model directory: {model_dir.resolve()}")
    metadata = model_dir / "metrics.json"
    if metadata.exists():
        print("Metrics:")
        print(metadata.read_text())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evomind", description="EvoMind AutoML SDK CLI")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Execute an EvoMind AutoML run.")
    run_parser.add_argument("--data", required=True, help="Path to dataset file.")
    run_parser.add_argument("--task", default="auto", help="Task identifier or 'auto'.")
    run_parser.add_argument("--config", help="Optional configuration file (YAML/JSON).")
    run_parser.add_argument("--no-insights", action="store_true", help="Disable insight generation.")
    run_parser.add_argument("--export", choices=["html", "pdf"], help="Export report in the selected format.")
    run_parser.set_defaults(func=_run_command)

    list_parser = subparsers.add_parser("list-models", help="List models registered in the local registry.")
    list_parser.set_defaults(func=_list_models)

    load_parser = subparsers.add_parser("load", help="Display information about a registered model.")
    load_parser.add_argument("model_id", help="Model identifier (directory name).")
    load_parser.set_defaults(func=_load_model)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0].startswith("--"):
        argv = ["run", *argv]
    if not argv:
        parser.print_help()
        return
    parsed = parser.parse_args(argv)
    if not hasattr(parsed, "func"):
        parser.print_help()
        return
    parsed.func(parsed)


if __name__ == "__main__":
    main()
