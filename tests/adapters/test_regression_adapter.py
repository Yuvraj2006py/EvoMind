from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from evomind.adapters.regression_adapter import RegressionAdapter
from evomind.core.task_detector import detect_task_type
from evomind.evolution.genome import Genome
from evomind.evolution.search_space import LayerSpec


def _write_csv(path: Path, df: pd.DataFrame) -> Path:
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def test_regression_adapter_end_to_end(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "feature_a": [float(i) for i in range(12)],
            "feature_b": [i * 2 for i in range(12)],
            "target": [i * 10.0 for i in range(12)],
        }
    )
    data_path = _write_csv(tmp_path / "reg.csv", df)

    adapter = RegressionAdapter(schema={"target": "target"}, data=data_path)
    processed = adapter.preprocess(adapter.load_data())

    feature_matrix = processed.drop(columns=[adapter.target_column]).to_numpy(dtype=np.float32)
    target_vector = processed[adapter.target_column].to_numpy(dtype=np.float32)
    assert feature_matrix.shape[1] == len(adapter.feature_names_)

    X_train, X_val, y_train, y_val = train_test_split(feature_matrix, target_vector, test_size=0.2, random_state=42)

    genome = Genome(layers=[LayerSpec("dense", {"units": 8, "activation": "relu"})])
    model = adapter.build_model(genome, input_dim=feature_matrix.shape[1], output_dim=1)
    model.eval()

    torch_val = torch.from_numpy(X_val).float()
    y_val_tensor = torch.from_numpy(y_val.reshape(-1, 1)).float()

    metrics = adapter.evaluate(model, torch_val, y_val_tensor)
    assert {"val_loss", "val_mae", "rmse", "r2_score"}.issubset(metrics.keys())
    assert isinstance(adapter.fitness(metrics), float)

    schema = {"target": "target", "numeric": ["feature_a", "feature_b"], "text": [], "datetime": []}
    detected = detect_task_type(df, schema)
    assert detected == "regression"


def test_regression_adapter_coerces_numeric_target(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5, 6],
            "Price": ["₹1,000", "₹2,500", None, "3,750", "₹4,200", "In Progress"],
        }
    )
    data_path = _write_csv(tmp_path / "prices.csv", df)

    adapter = RegressionAdapter(schema={"target": "Price"}, data=data_path)
    processed = adapter.preprocess(adapter.load_data())
    series = processed[adapter.target_column]
    assert series.dtype.kind in {"f", "i"}
    assert series.notna().all()
