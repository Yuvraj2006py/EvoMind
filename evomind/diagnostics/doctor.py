"""
Environment diagnostics for the EvoMind SDK.

The diagnostics are intentionally lightweight so they can run quickly from the
CLI (`evomind doctor`) and during CI checks. Each diagnostic returns a
dictionary with a human-readable description, status, and optional details.
"""

from __future__ import annotations

import importlib
import platform
import shutil
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional


MIN_PYTHON = (3, 9)
REQUIRED_BINARIES = {
    "git": "Git is required for version control and changelog generation.",
}
CRITICAL_DEPENDENCIES = [
    "numpy",
    "pandas",
    "torch",
    "sklearn",
    "omegaconf",
]
OPTIONAL_DEPENDENCIES = [
    "ray",
    "weasyprint",
    "streamlit",
    "optuna",
    "mlflow",
    "plotly",
]


@dataclass
class CheckResult:
    """Structured diagnostic result."""

    check: str
    status: str
    details: Optional[str] = None

    def as_dict(self) -> Dict[str, Optional[str]]:
        payload = asdict(self)
        if payload["details"] is None:
            payload.pop("details")
        return payload


def _status(ok: bool) -> str:
    return "pass" if ok else "fail"


def _warn(message: str) -> CheckResult:
    return CheckResult(check=message, status="warn")


def _check_python_version() -> CheckResult:
    current = sys.version_info
    ok = current >= MIN_PYTHON
    details = f"Detected Python {current.major}.{current.minor}.{current.micro}"
    if not ok:
        details += f" (requires >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]})"
    return CheckResult(check="Python runtime", status=_status(ok), details=details)


def _check_binary(name: str, hint: str) -> CheckResult:
    exists = shutil.which(name) is not None
    status = _status(exists)
    details = hint if not exists else shutil.which(name)
    return CheckResult(check=f"Binary '{name}' availability", status=status, details=details)


def _check_dependency(module_name: str, /, *, optional: bool = False) -> CheckResult:
    label = module_name.replace("_", " ")
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - dependent on external env
        status = "warn" if optional else "fail"
        details = f"{exc.__class__.__name__}: {exc}"
        return CheckResult(
            check=f"Python package '{label}' import",
            status=status,
            details=details,
        )
    version = getattr(module, "__version__", "unknown")
    status = "pass"
    details = f"{version}"
    if optional and module_name == "torch":
        # Optional flag only applies to other libs – torch is critical.
        optional = False

    if module_name == "torch":
        try:
            import torch  # noqa: WPS433

            gpu = "enabled" if torch.cuda.is_available() else "CPU-only"
            details = f"{version} ({gpu})"
        except Exception:  # pragma: no cover - defensive
            pass
    if module_name == "ray":
        try:
            import ray  # noqa: WPS433

            details = f"{version} ({'running' if ray.is_initialized() else 'not initialised'})"
        except Exception:  # pragma: no cover - defensive
            details = version

    return CheckResult(
        check=f"Python package '{label}' import",
        status=status,
        details=details,
    )


def _check_paths() -> Iterable[CheckResult]:
    base = Path.cwd()
    expected = {
        "Global config": base / "configs" / "global.yaml",
        "Default config schema": base / "configs" / "config_default.yaml",
        "Experiments directory": base / "experiments",
        "Model registry": base / "models",
    }
    for label, path in expected.items():
        if path.exists():
            status = "pass"
            details = str(path.resolve())
        else:
            status = "warn"
            details = "Will be created automatically." if path.suffix else f"Missing at {path.resolve()}"
        yield CheckResult(check=label, status=status, details=details)


def _check_platform() -> CheckResult:
    details = f"{platform.system()} {platform.release()} ({platform.machine()})"
    return CheckResult(check="Platform", status="pass", details=details)


def run_doctor() -> List[Dict[str, Optional[str]]]:
    """
    Execute environment diagnostics and return structured results.

    Returns
    -------
    list of dict
        Each dictionary contains `check`, `status`, and optional `details`.
        Status is one of ``pass``, ``warn``, or ``fail``.
    """

    results: List[CheckResult] = [
        _check_platform(),
        _check_python_version(),
    ]

    for name, hint in REQUIRED_BINARIES.items():
        results.append(_check_binary(name, hint))

    for module in CRITICAL_DEPENDENCIES:
        results.append(_check_dependency(module))

    for module in OPTIONAL_DEPENDENCIES:
        results.append(_check_dependency(module, optional=True))

    results.extend(_check_paths())

    # Convert to dictionaries for CLI friendliness.
    return [result.as_dict() for result in results]
