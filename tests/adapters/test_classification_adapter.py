from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

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

    adapter = ClassificationAdapter()
    X_train, y_train, X_val, y_val = adapter.load_data(data_path)
    X_train_processed, X_val_processed = adapter.preprocess(X_train, X_val)

    assert X_train_processed.shape[1] == len(adapter.feature_names_)
    assert adapter.label_encoder is not None

    genome = Genome(layers=[LayerSpec("dense", {"units": 8, "activation": "relu"})])
    output_dim = len(adapter.label_encoder.classes_)
    model = adapter.build_model(genome, input_dim=X_train_processed.shape[1], output_dim=output_dim)
    model.eval()

    torch_val = torch.from_numpy(X_val_processed).float()
    y_val_tensor = F.one_hot(
        torch.tensor(y_val.values, dtype=torch.int64),
        num_classes=output_dim,
    ).float()

    metrics = adapter.evaluate_model(model, torch_val, y_val_tensor)
    assert {"val_loss", "val_accuracy", "precision", "recall", "f1_score"}.issubset(metrics.keys())
    assert isinstance(adapter.fitness(metrics), float)

    schema = {"target": "target", "numeric": ["num_feature"], "text": [], "datetime": []}
    detected = detect_task_type(df, schema)
    assert detected == "classification"
