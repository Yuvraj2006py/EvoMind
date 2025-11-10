"""
SDK entry point exposing the `EvoMind` orchestration class.

The runner coordinates dataset ingestion, automatic task detection, distributed
evolution, explainability, and reporting. It serves as the backbone for the
CLI, SDK, and API interfaces.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

import json
from omegaconf import OmegaConf
import os
import subprocess

import numpy as np
import pandas as pd
import torch

from evomind.core.metrics import fairness_metrics

from evomind.adapters import get_adapter as _get_registered_adapter, list_adapters, register_adapter as _register_external_adapter
from evomind.adapters.base_adapter import BaseTaskAdapter
from evomind.adapters.data_utils import load_dataframe
from evomind.core.data_profiler import (
    compute_mutual_information,
    detect_constant_or_duplicate_columns,
    profile_health,
    time_series_diagnostics,
    to_serialisable,
)
from evomind.core.explain import generate_explanations
from evomind.core.generic_preprocessing import generic_preprocess
from evomind.core.insight_summarizer import summarize_insights
from evomind.core.profiling import generate_profile_report
from evomind.core.schema_profiler import profile_dataset
from evomind.core.task_detector import detect_task_type
from evomind.evolution import (
    EvolutionConfig,
    EvolutionEngine,
    FitnessEvaluator,
    Genome,
    PopulationManager,
    SearchSpace,
    Trainer,
    TrainerConfig,
)
from evomind.reporting.report_builder import build_report
from evomind.utils import ConfigLoader, ExperimentLogger
from evomind.utils.config_reference import (
    CONFIG_SCHEMA,
    as_dict as _config_schema_dict,
    to_console as _config_schema_console,
    to_markdown as _config_schema_markdown,
    write_markdown as _config_write_markdown,
)
from evomind.utils.fingerprint import (
    compute_fingerprint,
    get_cached_schema,
    update_cache,
)
from evomind.utils.profiles import get_profile, list_profiles
from evomind.exceptions import EvoMindConfigError


def _slugify_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "dataset"


def _derive_data_slug(data: Union[str, Path, pd.DataFrame, Sequence[Union[str, Path]]]) -> str:
    if isinstance(data, pd.DataFrame):
        return "dataframe"
    if isinstance(data, (str, Path)):
        return _slugify_name(Path(data).stem or "dataset")
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        items = list(data)
        if not items:
            return "dataset"
        first = items[0]
        if isinstance(first, (str, Path)):
            base = Path(first).stem or "dataset"
        else:
            base = "dataset"
        if len(items) > 1:
            base = f"{base}-bundle"
        return _slugify_name(base)
    return "dataset"


def _assess_stability(metrics: Dict[str, float]) -> str:
    overfit = abs(metrics.get("overfit_indicator", 0.0))
    drift = metrics.get("sensitivity_drift", 0.0)
    cv_std = metrics.get("cv_score_std", 0.0)
    if overfit < 0.02 and drift < 0.01 and cv_std < 0.01:
        return "Stable"
    if overfit < 0.05 and drift < 0.03 and cv_std < 0.03:
        return "Moderate drift"
    return "Potential overfitting"


def _run_generic_fallback(
    df: pd.DataFrame,
    schema: Dict[str, Any],
    task_type: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_scaled, y_series = generic_preprocess(df, target=schema.get("target"))
    valid_mask = ~y_series.isna()
    if not valid_mask.all():
        X_scaled = X_scaled.loc[valid_mask]
        y_series = y_series.loc[valid_mask]
    if y_series.empty:
        raise ValueError("Target column empty after removing missing values.")
    if task_type == "classification" and y_series.nunique() <= 1:
        raise ValueError("Target column needs at least two classes for classification.")
    stratify = y_series if task_type == "classification" and y_series.nunique() > 1 else None
    from sklearn.model_selection import train_test_split

    return train_test_split(X_scaled, y_series, test_size=0.2, random_state=42, stratify=stratify)


def _prepare_dataset(
    adapter: Optional[BaseTaskAdapter],
    data_path: Optional[Path],
    df: pd.DataFrame,
    schema: Dict[str, Any],
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Dict[str, Dict], Dict[str, Any]]:
    task_type = getattr(adapter, "task_type", schema.get("task_type", "regression"))
    if adapter is None:
        X_train, X_val, y_train, y_val = _run_generic_fallback(df.copy(), schema, task_type)
        feature_names = list(X_train.columns)
        dataset_summary = {
            "schema": schema,
            "mode": "generic",
        }
        metadata = {
            "feature_names": feature_names,
            "raw_train_features": X_train,
            "raw_train_target": y_train,
        }
        processed = (
            X_train.to_numpy(dtype=np.float32),
            y_train.to_numpy(dtype=np.float32).reshape(-1, 1),
            X_val.to_numpy(dtype=np.float32),
            y_val.to_numpy(dtype=np.float32).reshape(-1, 1),
        )
        return processed, dataset_summary, metadata

    adapter.set_data_source(data_path or df.copy())
    processed_frame = adapter.prepare_training_frame()
    target_col = adapter.target_column or schema.get("target") or processed_frame.columns[-1]
    features = processed_frame.drop(columns=[target_col], errors="ignore")
    target = processed_frame[target_col]

    stratify = target if task_type == "classification" and target.nunique() > 1 else None
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=stratify
    )

    feature_names = features.columns.tolist()
    if task_type == "classification":
        classes = int(pd.Series(target).max()) + 1
        y_train_idx = pd.Series(y_train).astype(int).to_numpy()
        y_val_idx = pd.Series(y_val).astype(int).to_numpy()
        y_train_arr = np.eye(classes, dtype=np.float32)[y_train_idx]
        y_val_arr = np.eye(classes, dtype=np.float32)[y_val_idx]
    else:
        y_train_arr = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)
        y_val_arr = np.asarray(y_val, dtype=np.float32).reshape(-1, 1)

    try:
        dataset_summary = adapter.summarize_data(processed_frame)
    except Exception as exc:  # noqa: BLE001
        dataset_summary = {"schema": schema, "warning": str(exc)}

    metadata = {
        "feature_names": feature_names,
        "raw_train_features": X_train,
        "raw_train_target": y_train,
        "target_column": target_col,
    }
    processed = (
        X_train.to_numpy(dtype=np.float32),
        y_train_arr.astype(np.float32),
        X_val.to_numpy(dtype=np.float32),
        y_val_arr.astype(np.float32),
    )
    return processed, dataset_summary, metadata


@dataclass
class EvoMindResult:
    """Return payload exposed by the SDK."""

    run_id: str
    best_genome: Genome
    metrics: Dict[str, float]
    lineage: List[Dict[str, Any]]
    artifacts: Dict[str, Any]
    output_dir: Path
    engine: EvolutionEngine
    profile: Optional[str] = None

    def export_report(self, fmt: str = "html") -> Optional[Path]:
        """Return the path to a rendered report in the requested format.

        Parameters
        ----------
        fmt : str, default "html"
            Desired report format. Typical options are ``"html"`` and ``"pdf"``.

        Returns
        -------
        Path or None
            Location of the generated report. Raises :class:`ValueError` when the
            format is unavailable.
        """

        reports = self.artifacts.get("reports", {})
        if fmt in reports:
            return reports[fmt]
        raise ValueError(f"Report format '{fmt}' not available. Options: {list(reports.keys())}")

    def dashboard(self) -> Path:
        """Return the path to the Streamlit dashboard entry point."""

        return Path("evomind/dashboard/app.py")

    def launch_dashboard(self, port: int = 8501, host: str = "127.0.0.1") -> subprocess.Popen:
        """Launch the Streamlit dashboard for interactive exploration.

        Parameters
        ----------
        port : int, default 8501
            Port where Streamlit should serve the dashboard.
        host : str, default "127.0.0.1"
            Bind address for the Streamlit server.

        Returns
        -------
        subprocess.Popen
            Handle to the spawned Streamlit process. Terminate it to stop the dashboard.
        """

        env = os.environ.copy()
        env.setdefault("EVOMIND_DEFAULT_RUN", self.run_id)
        command = [
            "streamlit",
            "run",
            str(self.dashboard()),
            "--server.port",
            str(port),
            "--server.address",
            host,
        ]
        return subprocess.Popen(command, env=env)


class EvoMind:
    """Primary interface coordinating dataset understanding and evolution.

    The class exposes a high-level SDK surface that can be consumed from
    notebooks, scripts, or production services. Use :meth:`describe_config`
    for interactive documentation of all tunable parameters.
    """

    @classmethod
    def register_adapter(cls, name: str, adapter_cls: Type[BaseTaskAdapter]) -> None:
        """Expose registry helper for third-party integrations."""

        _register_external_adapter(name)(adapter_cls)

    @classmethod
    def available_adapters(cls) -> Dict[str, Type[BaseTaskAdapter]]:
        """Return the currently registered adapters."""

        return list_adapters()

    @classmethod
    def get_adapter(cls, name: str) -> Type[BaseTaskAdapter]:
        """Retrieve a registered adapter class by name."""

        return _get_registered_adapter(name)

    def _allocate_run_id(self, slug: str) -> str:
        experiments_dir = Path("experiments")
        experiments_dir.mkdir(parents=True, exist_ok=True)
        pattern = re.compile(rf"{re.escape(slug)}_(\d+)$")
        max_index = 0
        for entry in experiments_dir.iterdir():
            if entry.is_dir():
                match = pattern.match(entry.name)
                if match:
                    max_index = max(max_index, int(match.group(1)))
        return f"{slug}_{max_index + 1:03d}"

    @classmethod
    def describe_config(
        cls,
        section: Optional[str] = None,
        *,
        as_markdown: bool = False,
        to_console: bool = False,
    ) -> Union[str, Dict[str, Dict[str, Dict[str, object]]]]:
        """Return metadata describing EvoMind configuration keys.

        Parameters
        ----------
        section : str, optional
            When provided, only return information for a single section
            (for example ``"engine"``). If omitted, all sections are returned.
        as_markdown : bool, default False
            When True, the result is formatted as Markdown text suitable for
            documentation. Otherwise a nested dictionary is returned.
        to_console : bool, default False
            When True, pretty-print the configuration table to stdout. The
            return value is still provided for programmatic use.

        Returns
        -------
        dict or str
            Nested configuration metadata or a markdown string when
            ``as_markdown`` is set.
        """

        if as_markdown:
            markdown = _config_schema_markdown(section=section)
            if to_console:
                print(markdown)
            return markdown

        if to_console:
            print(_config_schema_console(section=section))
        return _config_schema_dict(section)

    @classmethod
    def explain(cls, key: str) -> str:
        """Return a human readable description for a configuration key."""

        normalized = key.strip().lower().replace("-", "_")
        for section_name, fields in CONFIG_SCHEMA.items():
            for field in fields.values():
                if field.name.lower() == normalized:
                    default_repr = "None" if field.default is None else repr(field.default)
                    description = field.description or "No description available."
                    return (
                        f"{field.name} (section={section_name}, type={field.type}, "
                        f"default={default_repr}) -> {description}"
                    )
        raise EvoMindConfigError(
            f"Unknown configuration key '{key}'.",
            context={"key": key},
        )

    @classmethod
    def generate_config_docs(cls, path: Union[str, Path] = Path("CONFIG.md")) -> Path:
        """Render the configuration reference to a markdown file."""

        target = Path(path)
        return _config_write_markdown(target)

    @classmethod
    def available_profiles(cls) -> Dict[str, Dict[str, object]]:
        """Return a mapping of available configuration profiles."""

        return list_profiles()

    def __init__(
        self,
        data: Union[str, Path, pd.DataFrame, Sequence[Union[str, Path]]],
        task: str = "auto",
        profile: Optional[str] = None,
        insights: bool = True,
        config: Optional[Union[str, Path, Dict[str, Any]]] = None,
        global_config: Optional[Union[str, Path, Dict[str, Any]]] = "configs/global.yaml",
        run_name: Optional[str] = None,
    ) -> None:
        """Create a new EvoMind orchestrator.

        Parameters
        ----------
        data : str | Path | pandas.DataFrame | Sequence[str | Path]
            Dataset source. Accepts a path to CSV/JSON files, a list of paths (they
            will be concatenated), directories containing multiple CSV/JSON files,
            or an in-memory dataframe.
        task : str, default "auto"
            Registered task adapter to use. When ``"auto"`` EvoMind detects the best
            adapter based on schema profiling.
        profile : str, optional
            Optional configuration profile (e.g. ``"fast"``, ``"balanced"``,
            ``"exhaustive"``) to merge before applying overrides.
        insights : bool, default True
            Enable explainability and profiling artefacts.
        config : str | Path | dict, optional
            Task-specific configuration overrides supplied as a YAML/JSON file or a mapping.
            global_config : str | Path | dict, optional
            Base configuration merged before ``config``. Defaults to ``configs/global.yaml``.
        run_name : str, optional
            Optional slug used to name the run directories (e.g. ``holiday_campaign``).
            When omitted, EvoMind derives the name from the dataset path.
        """

        self.data_source = data
        self.task = task
        self.profile = profile
        self.insights_enabled = insights
        self.data_slug = _derive_data_slug(data)
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._data_source_paths: List[Path] = []
        if isinstance(global_config, (str, Path)):
            gc_path = Path(global_config)
            global_loader = ConfigLoader(gc_path) if gc_path.exists() else ConfigLoader()
        elif isinstance(global_config, dict):
            global_loader = ConfigLoader(global_config)
        else:
            global_loader = ConfigLoader()

        profile_overrides: Dict[str, Any] = {}
        if profile:
            try:
                profile_overrides = get_profile(profile)
            except KeyError as exc:  # pragma: no cover - defensive
                raise ValueError(str(exc)) from exc

        overrides = None
        if isinstance(config, Dict):
            merged_overrides = OmegaConf.merge(
                OmegaConf.create(profile_overrides),
                OmegaConf.create(dict(config)),
            )
            overrides = OmegaConf.to_container(merged_overrides, resolve=True)  # type: ignore[assignment]
            config = None
        elif profile_overrides:
            overrides = profile_overrides

        self.config_loader = global_loader
        loaded = self.config_loader.load(config=config, overrides=overrides)
        self.config = loaded.to_dict()

        experiment_cfg = self.config.get("experiment", {})
        logging_cfg = self.config.get("logging", {})
        experiment_name = experiment_cfg.get("name", "EvoMind")
        logging_enabled = bool(logging_cfg.get("enable_mlflow", True))
        tracking_uri = (
            logging_cfg.get("mlflow_uri")
            or experiment_cfg.get("mlflow_uri")
            or experiment_cfg.get("mlflow_tracking_uri")
        )
        self.logger = ExperimentLogger(
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            enabled=logging_enabled,
        )
        base_slug = _slugify_name(run_name) if run_name else self.data_slug
        self.run_id = self._allocate_run_id(base_slug)
        self.output_dir = Path("experiments") / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_dataframe(self) -> Tuple[pd.DataFrame, Optional[Path]]:
        if isinstance(self.data_source, pd.DataFrame):
            self._data_source_paths = []
            return self.data_source.copy(), None
        if isinstance(self.data_source, (str, Path)):
            data_path = Path(self.data_source)
            if not data_path.exists():
                raise FileNotFoundError(f"Dataset not found: {data_path}")
            self._data_source_paths = [data_path]
            return load_dataframe(data_path), data_path
        if isinstance(self.data_source, Sequence):
            sequence_items = list(self.data_source)
            if len(sequence_items) == 1 and isinstance(sequence_items[0], (str, Path, pd.DataFrame)):
                single_source = sequence_items[0]
                self.data_source = single_source  # type: ignore[assignment]
                return self._load_dataframe()
            frames: List[pd.DataFrame] = []
            origin_paths: List[Path] = []
            for entry in sequence_items:
                if isinstance(entry, pd.DataFrame):
                    origin_paths.append(Path("in-memory"))
                    frames.append(entry.copy())
                    continue
                if isinstance(entry, (str, Path)):
                    path = Path(entry)
                else:
                    continue
                if not path.exists():
                    raise FileNotFoundError(f"Dataset not found: {path}")
                if path.is_dir():
                    files = sorted(list(path.glob("*.csv")) + list(path.glob("*.json")))
                    if not files:
                        raise FileNotFoundError(f"No CSV/JSON files found in directory: {path}")
                    for file in files:
                        frames.append(load_dataframe(file))
                        origin_paths.append(file)
                else:
                    frames.append(load_dataframe(path))
                    origin_paths.append(path)
            if not frames:
                raise ValueError("No datasets were loaded from the provided sources.")
            combined = pd.concat(frames, ignore_index=True)
            self._data_source_paths = origin_paths
            return combined, None
        raise TypeError("Unsupported data source type. Provide a path, DataFrame, or list of sources.")

    def _detect_task(self, df: pd.DataFrame, schema: Dict[str, Any]) -> str:
        if self.task and self.task != "auto":
            return self.task
        detected = detect_task_type(df, schema)
        return detected or "regression"

    def _build_adapter(
        self,
        task_name: str,
        schema: Dict[str, Any],
        data_source: Optional[Union[Path, pd.DataFrame]],
    ) -> Optional[BaseTaskAdapter]:
        resolved_name = task_name
        try:
            adapter_cls = _get_registered_adapter(task_name)
        except KeyError:
            message = "⚙️ No domain adapter detected. Running in Generic Mode."
            logging.info(message)
            print(message)
            logging.debug("Available adapters: %s", ", ".join(sorted(list_adapters().keys())))
            fallback = schema.get("task_type") or "regression"
            resolved_name = fallback
            try:
                adapter_cls = _get_registered_adapter(fallback)
            except KeyError:
                adapter_cls = _get_registered_adapter("regression")
        adapter = adapter_cls(
            schema=schema,
            data=data_source,
            config=self.config.get("adapters", {}).get(resolved_name, {}),
        )
        if hasattr(adapter, "schema"):
            adapter.schema = schema
        if hasattr(adapter, "set_data_source"):
            adapter.set_data_source(data_source)
        return adapter

    def _create_population(self, population_size: int) -> PopulationManager:
        search_space = SearchSpace()
        population = PopulationManager(search_space=search_space, population_size=population_size)
        population.seed()
        return population

    def _write_self_diagnosis(
        self,
        health_score: float,
        anomalies: Dict[str, Any],
        fairness: Dict[str, float],
    ) -> Path:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"self_diagnosis_{self.run_id}.log"
        lines = [
            f"Run ID: {self.run_id}",
            f"Health Score: {health_score:.2f}",
        ]
        if anomalies.get("constant"):
            lines.append(f"Constant columns: {', '.join(anomalies['constant'])}")
        if anomalies.get("duplicate"):
            lines.append(f"Duplicate columns: {', '.join(anomalies['duplicate'])}")
        high_corr = anomalies.get("high_correlation") or []
        if high_corr:
            formatted = [f"{entry['features'][0]}~{entry['features'][1]} ({entry['correlation']:.2f})" for entry in high_corr]
            lines.append("High multicollinearity: " + ", ".join(formatted))
        if fairness:
            lines.append(f"Fairness - demographic parity gap: {fairness.get('demographic_parity', 0.0):.3f}")
            lines.append(f"Fairness - equal opportunity gap: {fairness.get('equal_opportunity', 0.0):.3f}")
        log_path.write_text("\n".join(lines), encoding="utf-8")
        return log_path

    def _write_model_card(self, metrics: Dict[str, float], artifacts: Dict[str, Any]) -> Path:
        """Persist a lightweight model card summarising key artefacts."""

        card_path = self.output_dir / "model_card.html"
        reports = artifacts.get("reports", {})
        report_link = reports.get("html") if isinstance(reports, dict) else None
        fairness = artifacts.get("fairness", {}) or {}
        insight_summary = artifacts.get("insight_summary", "")

        metric_rows = "".join(
            f"<tr><td>{name}</td><td>{value:.4f}</td></tr>"
            for name, value in sorted(metrics.items())
            if isinstance(value, (int, float))
        )
        fairness_rows = "".join(
            f"<tr><td>{name}</td><td>{value:.4f}</td></tr>"
            for name, value in sorted(fairness.items())
            if isinstance(value, (int, float))
        ) or "<tr><td colspan='2'>Not available</td></tr>"

        card_html = f"""
        <html>
        <head>
            <meta charset='utf-8'>
            <title>EvoMind Model Card</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 2rem; background: #111; color: #f0f0f0; }}
                h1, h2 {{ color: #4dd0e1; }}
                table {{ width: 100%; border-collapse: collapse; margin-bottom: 1.5rem; }}
                td, th {{ border: 1px solid #333; padding: 0.5rem; text-align: left; }}
                a {{ color: #80cbc4; }}
                .section {{ margin-bottom: 2rem; }}
            </style>
        </head>
        <body>
            <h1>EvoMind Model Card</h1>
            <p><strong>Run ID:</strong> {self.run_id}</p>
            <div class='section'>
                <h2>Summary</h2>
                <p>{insight_summary or 'No insight summary recorded.'}</p>
            </div>
            <div class='section'>
                <h2>Performance Metrics</h2>
                <table>
                    <tbody>
                        {metric_rows or '<tr><td colspan="2">No metrics logged.</td></tr>'}
                    </tbody>
                </table>
            </div>
            <div class='section'>
                <h2>Fairness Diagnostics</h2>
                <table>
                    <tbody>
                        {fairness_rows}
                    </tbody>
                </table>
            </div>
            <div class='section'>
                <h2>Related Artefacts</h2>
                <ul>
                    <li>Lineage Manifest: {self.output_dir / 'manifest.json'}</li>
                    <li>Metrics: {self.output_dir / 'metrics.json'}</li>
                    <li>Report (HTML): {report_link or 'Not generated'}</li>
                </ul>
            </div>
        </body>
        </html>
        """

        card_path.write_text(card_html, encoding="utf-8")
        return card_path

    def _persist_run_outputs(
        self,
        history_records: List[Dict[str, Any]],
        artifacts: Dict[str, Any],
        metrics: Dict[str, float],
    ) -> Dict[str, Path]:
        paths: Dict[str, Path] = {}
        if history_records:
            history_df = pd.DataFrame(history_records)
            history_path = self.output_dir / "history.csv"
            history_df.to_csv(history_path, index=False)
            paths["history"] = history_path

        metrics_path = self.output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        paths["metrics"] = metrics_path

        lineage_path = self.output_dir / "lineage.json"
        lineage_path.write_text(json.dumps(artifacts.get("lineage", []), indent=2), encoding="utf-8")
        paths["lineage"] = lineage_path

        health_path = self.output_dir / "health.json"
        health_path.write_text(json.dumps(artifacts.get("health", {}), indent=2), encoding="utf-8")
        paths["health"] = health_path

        summary_path = self.output_dir / "dataset_summary.json"
        summary_path.write_text(json.dumps(artifacts.get("dataset_summary", {}), indent=2), encoding="utf-8")
        paths["dataset_summary"] = summary_path

        fairness_path = None
        if artifacts.get("fairness"):
            fairness_path = self.output_dir / "fairness.json"
            fairness_path.write_text(json.dumps(artifacts["fairness"], indent=2), encoding="utf-8")
            paths["fairness"] = fairness_path

        report_refs = {fmt: str(path) for fmt, path in artifacts.get("reports", {}).items() if path}
        explanations: Dict[str, Any] = {}
        for key, value in artifacts.get("explanations", {}).items():
            if isinstance(value, Path):
                explanations[key] = str(value)
            elif isinstance(value, dict):
                explanations[key] = {inner_key: str(inner_val) for inner_key, inner_val in value.items()}
            else:
                explanations[key] = value

        manifest = {
            "run_id": self.run_id,
            "run_dir": str(self.output_dir.resolve()),
            "model_metrics": metrics,
            "model_stability": _assess_stability(metrics),
            "insight_summary": artifacts.get("insight_summary"),
            "history": {"latest": str(paths["history"])} if "history" in paths else {},
            "lineage": {"latest": str(lineage_path)} if lineage_path.exists() else {},
            "reports": report_refs,
            "dataset_profile": {
                "profile_html": str(artifacts.get("profile_path")),
                "health_json": str(health_path),
                "summary_json": str(summary_path),
            },
            "explanations": explanations,
            "fairness": artifacts.get("fairness"),
            "fairness_json": str(fairness_path) if fairness_path else None,
            "diagnosis_log": str(artifacts.get("diagnosis_log")) if artifacts.get("diagnosis_log") else None,
            "recorded_at": self.run_timestamp,
            "executor": artifacts.get("executor_stats"),
            "config": artifacts.get("resolved_config"),
            "data_sources": artifacts.get("data_sources"),
        }
        manifest["profile"] = self.profile
        manifest["fingerprint"] = artifacts.get("fingerprint")

        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        paths["manifest"] = manifest_path

        latest_manifest_path = Path("experiments") / "manifest.json"
        latest_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        mutual_info_data = artifacts.get("mutual_info")
        if mutual_info_data:
            mutual_df = pd.DataFrame(mutual_info_data)
            mutual_info_path = self.output_dir / "mutual_info.csv"
            mutual_df.to_csv(mutual_info_path, index=False)
            paths["mutual_info"] = mutual_info_path
            manifest["mutual_info"] = str(mutual_info_path)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            latest_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        else:
            manifest["mutual_info"] = None

        card_path = self._write_model_card(metrics, artifacts)
        paths["model_card"] = card_path
        artifacts["model_card"] = str(card_path)
        manifest["model_card"] = str(card_path)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        latest_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return paths
    def _register_model(self, genome: Genome, metrics: Dict[str, float]) -> Path:
        registry_dir = Path("models") / self.run_id
        registry_dir.mkdir(parents=True, exist_ok=True)
        model_path = registry_dir / "model.pt"
        if genome.trained_model is not None:
            torch.save(genome.trained_model.state_dict(), model_path)
        (registry_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (registry_dir / "config.json").write_text(json.dumps(self.config, indent=2, default=str), encoding="utf-8")
        return registry_dir
    def _collect_artifacts(
        self,
        adapter: BaseTaskAdapter,
        df: pd.DataFrame,
        dataset_summary: Dict[str, Any],
        metadata: Dict[str, Any],
        best_genome: Genome,
        metrics: Dict[str, float],
        schema: Dict[str, Any],
        lineage: List[Dict[str, Any]],
        processed_dataset: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        sensitive_feature: Optional[str],
        fingerprint: Optional[str],
    ) -> Dict[str, Any]:
        artifacts: Dict[str, Any] = {}
        profile_path = generate_profile_report(df, self.output_dir / "artifacts" / "data_profile.html")
        column_health, health_score = profile_health(df)
        anomalies = detect_constant_or_duplicate_columns(df)
        mutual_info = compute_mutual_information(df, schema.get("target", ""), schema.get("task_type", "regression"))
        ts_diag = {}
        if schema.get("datetime"):
            ts_diag = time_series_diagnostics(df, schema["datetime"][0], schema.get("target", ""))

        explanations: Dict[str, Any] = {}
        if self.insights_enabled and best_genome.trained_model is not None:
            feature_names = metadata.get("feature_names", [])
            X_train = metadata["raw_train_features"]
            explanation_dir = self.output_dir / "artifacts" / "explainability"
            explanations = generate_explanations(
                best_genome.trained_model,
                np.asarray(X_train)[:200],
                feature_names,
                explanation_dir,
                task_type=getattr(adapter, "task_type", schema.get("task_type", "regression")),
            )

        fairness_scores: Dict[str, float] = {}
        if (
            sensitive_feature
            and sensitive_feature in getattr(metadata.get("raw_train_features"), "columns", [])
            and getattr(adapter, "task_type", "") == "classification"
            and best_genome.trained_model is not None
        ):
            try:
                X_train_proc, y_train_arr, _, _ = processed_dataset
                sensitive_values = np.asarray(metadata["raw_train_features"][sensitive_feature])
                model = best_genome.trained_model.eval()
                with torch.no_grad():
                    logits = model(torch.as_tensor(X_train_proc, dtype=torch.float32))
                if logits.ndim > 1 and logits.shape[1] > 1:
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    preds = np.argmax(probs, axis=1)
                else:
                    probs = torch.sigmoid(logits).cpu().numpy().reshape(-1, 1)
                    preds = (probs[:, 0] > 0.5).astype(int)
                y_true = y_train_arr
                if y_true.ndim > 1 and y_true.shape[1] > 1:
                    y_true = np.argmax(y_true, axis=1)
                else:
                    y_true = y_true.reshape(-1)
                fairness_scores = fairness_metrics(y_true=y_true, y_pred=preds, sensitive_attr=sensitive_values)
            except Exception as exc:  # pragma: no cover - fairness optional
                logging.debug("Fairness metric computation skipped: %s", exc)

        diagnosis_path = self._write_self_diagnosis(health_score, anomalies, fairness_scores)

        insight_summary = summarize_insights(
            metrics=metrics,
            feature_importance=explanations.get("feature_importance"),
            health_score=health_score,
            anomalies=anomalies,
            fairness=fairness_scores if fairness_scores else None,
        )

        report_context = {
            "run_id": self.run_id,
            "insight_summary": insight_summary,
            "data_profile": {
                "health_score": health_score,
                "rows": int(dataset_summary.get("shape", {}).get("rows", df.shape[0])),
                "columns": int(dataset_summary.get("shape", {}).get("columns", df.shape[1])),
                "missing_columns": len(anomalies.get("constant", [])),
                "outlier_columns": sum(1 for entry in column_health if entry.outlier_pct > 0.05),
                "details": to_serialisable(column_health),
            },
            "model_metrics": metrics,
            "stability_label": _assess_stability(metrics),
            "top_features": mutual_info[:5],
            "feature_images": [str(path) for path in explanations.values() if isinstance(path, Path) and path.is_file()],
            "correlations": dataset_summary.get("correlation", {}),
            "correlation_image": None,
            "fairness": fairness_scores,
            "mutual_info": mutual_info,
        }
        reports = build_report(self.output_dir / "reports", report_context)

        artifacts.update(
            {
                "profile_path": profile_path,
                "health": {
                    "details": to_serialisable(column_health),
                    "score": health_score,
                    "rows": int(df.shape[0]),
                    "columns": int(df.shape[1]),
                    "missing_columns": len(anomalies.get("constant", [])),
                    "outlier_columns": sum(1 for entry in column_health if entry.outlier_pct > 0.05),
                    "anomalies": anomalies,
                    "time_series": ts_diag,
                },
                "dataset_summary": dataset_summary,
                "explanations": explanations,
                "reports": reports,
                "lineage": lineage,
                "fairness": fairness_scores,
                "mutual_info": mutual_info,
                "insight_summary": insight_summary,
                "diagnosis_log": str(diagnosis_path),
                "report_context": report_context,
                "profile": self.profile,
                "fingerprint": fingerprint,
            }
        )
        return artifacts

    def run(self) -> EvoMindResult:
        """Execute the full AutoML workflow and return artefacts.

        Returns
        -------
        EvoMindResult
            Rich result object containing metrics, lineage, persisted artefacts,
            and helper utilities for exporting reports or launching the dashboard.
        """

        df, data_path = self._load_dataframe()
        fingerprint = compute_fingerprint(df if data_path is None else None, data_path)
        cached_schema = get_cached_schema(fingerprint) if fingerprint else None
        if cached_schema:
            schema = copy.deepcopy(cached_schema)
        else:
            schema = profile_dataset(df)
        schema["task_type"] = schema.get("task_type") or detect_task_type(df, schema)
        if fingerprint:
            update_cache(fingerprint, copy.deepcopy(schema), schema.get("task_type"))

        adapter = self._build_adapter(
            self._detect_task(df, schema),
            schema,
            data_path if data_path is not None else df.copy(),
        )
        dataset, dataset_summary, metadata = _prepare_dataset(adapter, data_path, df, schema)
        metadata["source_files"] = [str(path) for path in self._data_source_paths]

        engine_cfg = dict(self.config.get("engine", {}))
        if not engine_cfg:
            legacy_scheduler = self.config.get("scheduler", {})
            legacy_evolution = self.config.get("evolution", {})
            engine_cfg = {
                "generations": legacy_scheduler.get("generations", 5),
                "population": self.config.get("population_size", 20),
                "elite_fraction": legacy_evolution.get("elite_fraction", 0.2),
                "mutation_rate": legacy_evolution.get("mutation_rate", 0.3),
                "crossover_rate": legacy_evolution.get("crossover_rate", 0.6),
            }

        population_size = int(engine_cfg.get("population", 20))
        generations = int(engine_cfg.get("generations", 5))

        population = self._create_population(population_size)
        trainer_defaults = self.config.get("trainer", {})
        trainer_cfg = TrainerConfig(
            epochs=int(engine_cfg.get("epochs", trainer_defaults.get("epochs", 5))),
            batch_size=int(engine_cfg.get("batch_size", trainer_defaults.get("batch_size", 32))),
            learning_rate=float(engine_cfg.get("learning_rate", trainer_defaults.get("learning_rate", 1e-3))),
        )
        trainer = Trainer(trainer_cfg, self.logger)
        engine = EvolutionEngine(
            population=population,
            trainer=trainer,
            fitness_evaluator=FitnessEvaluator(),
            search_space=population.search_space,
            logger=self.logger,
            config=EvolutionConfig(
                elite_fraction=float(engine_cfg.get("elite_fraction", 0.2)),
                mutation_rate=float(engine_cfg.get("mutation_rate", 0.3)),
                crossover_rate=float(engine_cfg.get("crossover_rate", 0.6)),
                parallel_backend="ray" if engine_cfg.get("parallel", True) else "threads",
                bayes_rounds=int(engine_cfg.get("bayes_rounds", 0)),
                ensemble_top_k=int(engine_cfg.get("ensemble_top_k", 3)),
            ),
        )

        lineage: List[Dict[str, Any]] = []
        history_records: List[Dict[str, Any]] = []
        executor_history: List[Dict[str, Any]] = []
        best_genome = None
        metrics: Dict[str, float] = {}

        with self.logger.start_run(run_name=self.run_id):
            for generation in range(generations):
                summary, best, generation_lineage, exec_stats = engine.run_generation(adapter, dataset, generation)
                lineage.extend(generation_lineage)
                history_records.append({"generation": generation, **summary})
                if exec_stats:
                    executor_history.append(exec_stats)
                best_genome = best
                metrics = best.metrics
                engine.evolve()
                self.logger.log_metrics({"generation_best_fitness": summary["best_fitness"]}, step=generation)

        if best_genome is None:
            raise RuntimeError("Evolution did not produce any genomes. Check dataset quality.")

        # Retrain best genome to capture final model artefact and refreshed metrics.
        retrained_metrics, retrained_model = trainer.train(best_genome.clone(), adapter, dataset)
        best_genome.metrics = retrained_metrics
        best_genome.trained_model = retrained_model
        metrics = retrained_metrics

        artifacts = self._collect_artifacts(
            adapter=adapter,
            df=df,
            dataset_summary=dataset_summary,
            metadata=metadata,
            best_genome=best_genome,
            metrics=metrics,
            schema=schema,
            lineage=lineage,
            processed_dataset=dataset,
            sensitive_feature=self.config.get("data", {}).get("sensitive_feature"),
            fingerprint=fingerprint,
        )
        artifacts["executor_stats"] = {
            "backend": getattr(engine.executor, "backend", "threads"),
            "max_workers": engine.executor.max_workers,
            "history": executor_history,
        }
        artifacts["data_sources"] = [str(path) for path in self._data_source_paths]
        artifacts["resolved_config"] = json.loads(json.dumps(self.config, default=str))

        artifact_paths = self._persist_run_outputs(history_records, artifacts, metrics)
        artifacts["paths"] = {key: str(path) for key, path in artifact_paths.items()}
        registry_dir = self._register_model(best_genome, metrics)
        artifacts["model_registry_dir"] = str(registry_dir)

        return EvoMindResult(
            run_id=self.run_id,
            best_genome=best_genome,
            metrics=metrics,
            lineage=lineage,
            artifacts=artifacts,
            output_dir=self.output_dir,
            engine=engine,
            profile=self.profile,
        )

