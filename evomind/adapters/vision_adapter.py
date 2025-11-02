"""
Computer vision adapter for EvoMind built on flattened image features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

from evomind.adapters import register_task
from evomind.adapters.classification_adapter import ClassificationAdapter


@register_task("vision")
class VisionAdapter(ClassificationAdapter):
    """Handle image classification datasets stored as folders of images."""

    def __init__(self, image_size: int = 32, schema: Optional[Dict[str, object]] = None) -> None:
        super().__init__(schema=schema, default_target="label")
        self.image_size = image_size

    def _load_dataframe(self, path: Path) -> pd.DataFrame:  # type: ignore[override]
        if path.is_dir():
            records: List[np.ndarray] = []
            labels: List[str] = []
            for class_dir in sorted(path.iterdir()):
                if not class_dir.is_dir():
                    continue
                for image_path in class_dir.glob("*.*"):
                    try:
                        image = Image.open(image_path).convert("L").resize((self.image_size, self.image_size))
                        records.append(np.asarray(image, dtype=np.float32).flatten())
                        labels.append(class_dir.name)
                    except Exception:
                        continue
            if not records:
                raise ValueError("No images found for vision dataset.")
            columns = [f"pixel_{idx}" for idx in range(records[0].size)]
            df = pd.DataFrame(records, columns=columns)
            df["label"] = labels
            return df
        return super()._load_dataframe(path)
