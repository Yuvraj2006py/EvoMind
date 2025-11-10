from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from evomind.adapters.classification_adapter import ClassificationAdapter
from evomind.core.task_detector import detect_task_type
from evomind.evolution.genome import Genome
from evomind.evolution.search_space import LayerSpec


def _write_csv(path: Path, df: pd.DataFrame) -> Path:
    path.write_text(df.to_csv(index=False))
    return path


def test_classification_adapter_end_to_end(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "num_feature": [1, 2, 3, 4, 5, 6],
            "cat_feature": ["A", "B", "A", "B", "A", "B"],
            "target": ["yes", "no", "yes", "no", "yes", "no"],
        }
    )
    data_path = _write_csv(tmp_path / "cls.csv", df)

    adapter = ClassificationAdapter(schema={"target": "target"}, data=data_path)
    processed = adapter.preprocess(adapter.load_data())
    X = processed.drop(columns=[adapter.target_column]).to_numpy(dtype=np.float32)
    y = processed[adapter.target_column].to_numpy(dtype=int)

    assert X.shape[1] == len(adapter.feature_names_)

    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, random_state=42, test_size=0.5)
    genome = Genome(layers=[LayerSpec("dense", {"units": 8, "activation": "relu"})])
    output_dim = int(np.max(y) + 1)
    model = adapter.build_model(genome, input_dim=X.shape[1], output_dim=output_dim)
    model.eval()

    torch_val = torch.from_numpy(X_val).float()
    y_val_tensor = torch.from_numpy(np.eye(output_dim)[y_val]).float()

    metrics = adapter.evaluate(model, torch_val, y_val_tensor)
    assert {"val_loss", "val_accuracy", "precision", "recall", "f1_score"}.issubset(metrics.keys())
    assert isinstance(adapter.fitness(metrics), float)

    schema = {"target": "target", "numeric": ["num_feature"], "text": [], "datetime": []}
    detected = detect_task_type(df, schema)
    assert detected == "classification"
