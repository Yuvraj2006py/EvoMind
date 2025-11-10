"""
Centralised exception hierarchy for EvoMind.

The SDK surfaces rich, user friendly errors by raising typed exceptions instead
of generic ``ValueError`` or ``RuntimeError`` instances.  This allows the CLI,
SDK, and dashboard layers to present actionable remediation hints while keeping
logs structured for observability pipelines.
"""

from __future__ import annotations

from typing import Any


class EvoMindError(Exception):
    """Base class for all EvoMind specific exceptions."""

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.context = context or {}


class EvoMindAdapterError(EvoMindError):
    """Raised when an adapter violates the required interface or schema."""


class EvoMindConfigError(EvoMindError):
    """Raised for configuration or profile related issues."""


class EvoMindRuntimeError(EvoMindError):
    """Raised for runtime orchestration issues."""


__all__ = [
    "EvoMindError",
    "EvoMindAdapterError",
    "EvoMindConfigError",
    "EvoMindRuntimeError",
]
