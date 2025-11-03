from pathlib import Path

import pandas as pd
import torch

from evomind.adapters.regression_adapter import RegressionAdapter
from evomind.core.task_detector import detect_task_type
from evomind.evolution.genome import Genome
from evomind.evolution.search_space import LayerSpec


def _write_csv(path: Path, df: pd.DataFrame) -> Path:
    path.write_text(df.to_csv(index=False))
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

    adapter = RegressionAdapter()
    X_train, y_train, X_val, y_val = adapter.load_data(data_path)
    X_train_processed, X_val_processed = adapter.preprocess(X_train, X_val)

    assert X_train_processed.shape[1] == len(adapter.feature_names_)

    genome = Genome(layers=[LayerSpec("dense", {"units": 8, "activation": "relu"})])
    model = adapter.build_model(genome, input_dim=X_train_processed.shape[1], output_dim=1)
    model.eval()

    torch_val = torch.from_numpy(X_val_processed).float()
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

    metrics = adapter.evaluate_model(model, torch_val, y_val_tensor)
    assert {"val_loss", "val_mae", "rmse", "r2_score"}.issubset(metrics.keys())
    assert isinstance(adapter.fitness(metrics), float)

    schema = {"target": "target", "numeric": ["feature_a", "feature_b"], "text": [], "datetime": []}
    detected = detect_task_type(df, schema)
    assert detected == "regression"
