# Task Adapters

EvoMind automatically detects the task type and routes datasets to the appropriate adapter. The adapters encapsulate schema profiling, preprocessing, model search space, and metric selection.

## Built-in Adapters

| Adapter | Task Type | Highlights |
| ------- | --------- | ---------- |
| `TabularRegressionAdapter` | Regression | Feature scaling, missing value imputation, ensemble-ready search space. |
| `TabularClassificationAdapter` | Classification | Label encoding, class balancing heuristics, probability calibration. |
| `TimeSeriesAdapter` | Forecasting | Seasonal decomposition, lag feature synthesis, trend awareness. |
| `RetailDemandAdapter` | Retail | Promotion/holiday feature unions, hierarchical aggregation support. |
| `GenericAdapter` | Fallback | Used when schema inspection cannot determine a specialised adapter. |

## Writing a Custom Adapter

1. Implement a subclass of `BaseTaskAdapter`.
2. Define the `task_type`, `load_data`, and `preprocess` methods.
3. Register the adapter in `evomind/adapters/__init__.py` by updating the `TASK_REGISTRY`.
4. Add unit tests under `tests/adapters/` to validate the adapter with synthetic data.

Use configuration overrides or profiles to tweak adapter behaviour without editing code.
