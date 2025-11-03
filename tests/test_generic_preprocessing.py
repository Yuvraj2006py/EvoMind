import pandas as pd

from evomind.pipelines.runner import _run_generic_fallback


def test_generic_fallback_infers_numeric_target_and_drops_missing():
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5],
            "Price": ["1,000", "2500", None, "3499", "4,200"],
        }
    )
    schema = {"target": "Price"}

    X_train, X_val, y_train, y_val = _run_generic_fallback(df, schema, "regression")

    for target_split in (y_train, y_val):
        assert target_split.notna().all()
        assert target_split.dtype.kind in {"f", "i"}
