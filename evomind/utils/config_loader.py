"""
Unified configuration loader for EvoMind.

This module normalises configuration handling across the CLI, SDK, and API
layers. Configurations can be provided as dictionaries, JSON/YAML files, or
OmegaConf strings and are merged on top of global defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Union

import yaml
from omegaconf import DictConfig, OmegaConf


ConfigLike = Union[str, Path, Mapping[str, Any], DictConfig]


@dataclass
class LoadedConfig:
    """Container that exposes both OmegaConf and plain-dict views."""

    data: DictConfig

    def to_dict(self) -> Dict[str, Any]:
        return OmegaConf.to_container(self.data, resolve=True)  # type: ignore[return-value]

    def __getitem__(self, item: str) -> Any:
        return self.data[item]


class ConfigLoader:
    """
    Load and merge EvoMind configuration sources.

    Parameters
    ----------
    global_config : Optional[ConfigLike]
        Optional path or mapping containing default configuration values.
    """

    def __init__(self, global_config: Optional[ConfigLike] = None) -> None:
        self._global_conf = self._coerce(global_config) if global_config is not None else OmegaConf.create({})

    def _coerce(self, source: ConfigLike) -> DictConfig:
        """Convert arbitrary config-like inputs into an OmegaConf instance."""
        if isinstance(source, DictConfig):
            return source
        if isinstance(source, Mapping):
            return OmegaConf.create(dict(source))
        if isinstance(source, Path):
            return self._load_path(source)
        if isinstance(source, str):
            potential_path = Path(source)
            if potential_path.exists():
                return self._load_path(potential_path)
            try:
                parsed = yaml.safe_load(source)
            except yaml.YAMLError as exc:  # pragma: no cover - defensive branch.
                raise ValueError(f"Failed to parse configuration string: {exc}") from exc
            if not isinstance(parsed, MutableMapping):
                raise ValueError("Configuration string must evaluate to a mapping.")
            return OmegaConf.create(dict(parsed))
        raise TypeError(f"Unsupported configuration source: {type(source)!r}")

    def _load_path(self, path: Path) -> DictConfig:
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        suffix = path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            return OmegaConf.load(path)
        if suffix == ".json":
            return OmegaConf.create(yaml.safe_load(path.read_text(encoding="utf-8")))
        raise ValueError(f"Unsupported configuration file format: '{suffix}'. Expected YAML or JSON.")

    def load(
        self,
        config: Optional[ConfigLike] = None,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> LoadedConfig:
        """Merge global defaults with optional additional configuration and overrides."""

        merged = self._global_conf.copy()

        if config is not None:
            merged = OmegaConf.merge(merged, self._coerce(config))

        if overrides:
            merged = OmegaConf.merge(merged, dict(overrides))

        return LoadedConfig(merged)
