"""
Dataset fingerprint cache utilities.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

CACHE_PATH = Path("logs") / "fingerprint_cache.json"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def compute_fingerprint(df: Optional[pd.DataFrame] = None, source_path: Optional[Path] = None) -> Optional[str]:
    """Compute a fingerprint for the dataset based on file metadata or dataframe content."""

    if source_path is not None:
        try:
            stat = source_path.stat()
        except FileNotFoundError:
            return None
        payload = f"{source_path.resolve()}::{stat.st_size}::{stat.st_mtime_ns}"
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    if df is None:
        return None

    hashed = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.md5(hashed).hexdigest()


def load_cache() -> Dict[str, Dict[str, object]]:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_cache(cache: Dict[str, Dict[str, object]]) -> None:
    _ensure_parent(CACHE_PATH)
    CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def get_cached_schema(fingerprint: str) -> Optional[Dict[str, object]]:
    cache = load_cache()
    entry = cache.get(fingerprint)
    if entry:
        return entry.get("schema")
    return None


def update_cache(fingerprint: str, schema: Dict[str, object], task_type: Optional[str] = None) -> None:
    cache = load_cache()
    cache[fingerprint] = {
        "schema": schema,
    }
    if task_type:
        cache[fingerprint]["task_type"] = task_type
    save_cache(cache)
