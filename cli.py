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
from glob import glob
from pathlib import Path

from evomind import EvoMind
from evomind.diagnostics.doctor import run_doctor
from evomind.utils.profiles import list_profiles


def _default_config_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Configuration file not found: {candidate}")


def _resolve_data_sources(sources: list[str]) -> list[str]:
    resolved: list[str] = []
    for item in sources:
        matches = glob(item)
        if matches:
            resolved.extend(matches)
        else:
            resolved.append(item)
    return resolved


def _run_command(args: argparse.Namespace) -> None:
    config_source = _default_config_path(args.config) if args.config else None
    data_sources = _resolve_data_sources(args.data)
    em = EvoMind(
        data=data_sources if len(data_sources) > 1 else data_sources[0],
        task=args.task or "auto",
        profile=args.profile,
        insights=not args.no_insights,
        config=config_source,
        run_name=args.run_name,
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


def _doctor_command(_: argparse.Namespace) -> None:
    results = run_doctor()
    for item in results:
        status = item.get("status", "unknown").upper()
        check = item.get("check", "")
        details = item.get("details")
        print(f"[{status}] {check}")
        if details:
            print(f"  {details}")


def _create_adapter_command(args: argparse.Namespace) -> None:
    adapters_dir = Path("evomind") / "adapters"
    adapters_dir.mkdir(parents=True, exist_ok=True)
    raw_name = args.name.strip().lower().replace("-", "_")
    file_path = adapters_dir / f"{raw_name}_adapter.py"
    if file_path.exists():
        raise SystemExit(f"Adapter scaffold already exists: {file_path}")
    class_name = "".join(part.capitalize() for part in raw_name.split("_")) + "Adapter"
    template = f'''"""
Scaffold for the `{raw_name}` adapter.

Use this file to implement ``load_data``, ``preprocess``, ``train`` and
``evaluate`` for your custom domain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from evomind.adapters import register_task
from evomind.adapters.base_adapter import BaseAdapter
from evomind.exceptions import EvoMindAdapterError


@register_task("{raw_name}")
class {class_name}(BaseAdapter):
    """Describe what problem this adapter solves."""

    def __init__(
        self,
        schema: Optional[Dict[str, Any]] = None,
        *,
        data: Any | None = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(schema=schema, data=data, config=config)
        self.target_column = self.target_column or "target"

    def load_data(self) -> pd.DataFrame:
        """
        Load the raw dataset.

        Replace this stub with logic that reads from files, APIs, or databases.
        Access the datasource via ``self.data_source``.
        """
        if self.data_source is None:
            raise EvoMindAdapterError("Data source missing.", context={{"adapter": "{class_name}"}})
        if isinstance(self.data_source, pd.DataFrame):
            return self.data_source.copy()
        path = Path(self.data_source)
        return pd.read_csv(path)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply light feature engineering.

        The returned dataframe must include the ``self.target_column``.
        """
        if not self.target_column:
            self.target_column = "target"
        return df

    def train(self, X, y) -> Any:
        """Train a lightweight baseline model for benchmarking."""
        raise NotImplementedError("Implement your training routine here.")

    def evaluate(self, model, X_test, y_test) -> Dict[str, float]:
        """Compute evaluation metrics for the trained model."""
        raise NotImplementedError("Implement evaluation for your model type.")
'''
    file_path.write_text(template, encoding="utf-8")
    print(f"Adapter scaffold created at {file_path}")


def _describe_config_command(args: argparse.Namespace) -> None:
    if args.key:
        print(EvoMind.explain(args.key))
        return
    EvoMind.describe_config(section=args.section, as_markdown=args.markdown, to_console=True)


def _generate_config_docs_command(args: argparse.Namespace) -> None:
    output = Path(args.output)
    path = EvoMind.generate_config_docs(output)
    print(f"Configuration reference generated at {path.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evomind", description="EvoMind AutoML SDK CLI")
    subparsers = parser.add_subparsers(dest="command")

    profile_choices = sorted(list_profiles().keys())

    run_parser = subparsers.add_parser("run", help="Execute an EvoMind AutoML run.")
    run_parser.add_argument(
        "--data",
        required=True,
        action="append",
        help="Dataset source(s). Provide multiple --data flags to load and merge several files.",
    )
    run_parser.add_argument("--task", default="auto", help="Task identifier or 'auto'.")
    run_parser.add_argument("--config", help="Optional configuration file (YAML/JSON).")
    run_parser.add_argument("--no-insights", action="store_true", help="Disable insight generation.")
    run_parser.add_argument("--export", choices=["html", "pdf"], help="Export report in the selected format.")
    run_parser.add_argument("--profile", choices=profile_choices, help="Apply a configuration profile before overrides.")
    run_parser.add_argument("--run-name", help="Optional custom name used for run directories (slugified).")
    run_parser.set_defaults(func=_run_command)

    list_parser = subparsers.add_parser("list-models", help="List models registered in the local registry.")
    list_parser.set_defaults(func=_list_models)

    load_parser = subparsers.add_parser("load", help="Display information about a registered model.")
    load_parser.add_argument("model_id", help="Model identifier (directory name).")
    load_parser.set_defaults(func=_load_model)

    doctor_parser = subparsers.add_parser("doctor", help="Run environment diagnostics.")
    doctor_parser.set_defaults(func=_doctor_command)

    adapter_parser = subparsers.add_parser("create-adapter", help="Generate a domain adapter scaffold.")
    adapter_parser.add_argument("name", help="Adapter identifier (snake_case).")
    adapter_parser.set_defaults(func=_create_adapter_command)

    describe_parser = subparsers.add_parser("describe-config", help="Display EvoMind configuration schema.")
    describe_parser.add_argument("--section", help="Optional configuration section to filter.")
    describe_parser.add_argument("--markdown", action="store_true", help="Render the output as markdown.")
    describe_parser.add_argument("--key", help="Explain a single configuration key instead of listing the table.")
    describe_parser.set_defaults(func=_describe_config_command)

    config_doc_parser = subparsers.add_parser("generate-config-docs", help="Write CONFIG.md from the schema.")
    config_doc_parser.add_argument("--output", default="CONFIG.md", help="Destination markdown file (default: CONFIG.md).")
    config_doc_parser.set_defaults(func=_generate_config_docs_command)

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

