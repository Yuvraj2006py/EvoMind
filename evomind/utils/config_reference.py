"""
Utilities for loading EvoMind configuration metadata and generating references.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import yaml

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "configs" / "config_default.yaml"


@dataclass(frozen=True)
class ConfigField:
    """Structured representation of a configuration field."""

    section: str
    name: str
    type: str
    default: object
    description: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "section": self.section,
            "key": self.name,
            "type": self.type,
            "default": self.default,
            "description": self.description,
        }


def _load_schema() -> Dict[str, Dict[str, ConfigField]]:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Configuration schema file not found: {SCHEMA_PATH}")
    raw: Mapping[str, Mapping[str, Mapping[str, object]]] = yaml.safe_load(
        SCHEMA_PATH.read_text(encoding="utf-8")
    )
    schema: Dict[str, Dict[str, ConfigField]] = {}
    for section, entries in raw.items():
        section_map: Dict[str, ConfigField] = {}
        for key, meta in entries.items():
            section_map[key] = ConfigField(
                section=section,
                name=key,
                type=str(meta.get("type", "Any")),
                default=meta.get("default"),
                description=str(meta.get("description", "")).strip(),
            )
        schema[section] = section_map
    return schema


CONFIG_SCHEMA: Dict[str, Dict[str, ConfigField]] = _load_schema()


def iter_fields(section: Optional[str] = None) -> Iterable[ConfigField]:
    """Yield configuration fields optionally filtered by section."""

    if section:
        if section not in CONFIG_SCHEMA:
            raise KeyError(f"Unknown config section '{section}'. Options: {list(CONFIG_SCHEMA)}")
        yield from CONFIG_SCHEMA[section].values()
        return

    for section_fields in CONFIG_SCHEMA.values():
        yield from section_fields.values()


def as_dict(section: Optional[str] = None) -> Dict[str, Dict[str, Dict[str, object]]]:
    """Return configuration metadata as a nested dictionary."""

    if section:
        fields = {field.name: field.as_dict() for field in iter_fields(section)}
        return {section: fields}
    return {
        sec: {field.name: field.as_dict() for field in section_fields.values()}
        for sec, section_fields in CONFIG_SCHEMA.items()
    }


def to_markdown(section: Optional[str] = None) -> str:
    """Render the configuration reference as a markdown table."""

    if section:
        if section not in CONFIG_SCHEMA:
            raise KeyError(f"Unknown config section '{section}'. Options: {list(CONFIG_SCHEMA)}")
        sections = {section: CONFIG_SCHEMA[section]}
    else:
        sections = CONFIG_SCHEMA
    title = (
        f"# EvoMind Configuration Reference - {section.title()}\n"
        if section
        else "# EvoMind Configuration Reference\n"
    )
    lines = [title, ""]
    for sec_name, fields in sections.items():
        lines.append(f"## {sec_name.title()}")
        lines.append("")
        lines.append("| Key | Type | Default | Description |")
        lines.append("| --- | --- | --- | --- |")
        for field in fields.values():
            default_repr = "`None`" if field.default is None else f"`{field.default}`"
            description = field.description.replace("|", "\\|")
            lines.append(f"| `{field.name}` | `{field.type}` | {default_repr} | {description} |")
        lines.append("")
    return "\n".join(lines)


def write_markdown(path: Path, section: Optional[str] = None) -> Path:
    """Persist the markdown representation to the specified path."""

    content = to_markdown(section=section)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def to_console(section: Optional[str] = None) -> str:
    """Format the configuration reference as a console-friendly table."""

    if section:
        if section not in CONFIG_SCHEMA:
            raise KeyError(f"Unknown config section '{section}'. Options: {list(CONFIG_SCHEMA)}")
        sections = {section: CONFIG_SCHEMA[section]}
    else:
        sections = CONFIG_SCHEMA

    lines = []
    for sec_name, fields in sections.items():
        lines.append(f"[{sec_name.upper()}]")
        for field in fields.values():
            default_repr = "None" if field.default is None else repr(field.default)
            description = field.description or "No description available."
            lines.append(
                f"  - {field.name} (type={field.type}, default={default_repr}): {description}"
            )
        lines.append("")
    return "\n".join(lines).strip()


__all__ = [
    "ConfigField",
    "CONFIG_SCHEMA",
    "iter_fields",
    "as_dict",
    "to_markdown",
    "write_markdown",
    "to_console",
]
