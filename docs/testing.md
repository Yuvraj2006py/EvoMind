# Testing & Quality

EvoMind ships with an expanded test suite to safeguard adapters, configuration handling, and end-to-end pipelines.

## Test Layout

- `tests/test_config_loader.py` – configuration merging and validation.
- `tests/adapters/` – unit tests for regression, classification, and other adapters.
- `tests/integration/test_full_run.py` – smoke test that executes the full AutoML workflow on synthetic data and asserts artefact creation.

## Running Tests

```bash
pytest -q --disable-warnings
coverage run -m pytest
coverage report -m
```

CI runs these commands via GitHub Actions (`.github/workflows/test.yml`).

## Writing New Tests

1. Create synthetic datasets inline to avoid dependence on external files.
2. Use fixtures to share adapters or configuration snippets.
3. Assert on metrics, artefacts, and error handling to capture regressions.
4. Aim for deterministic runs by seeding RNGs and limiting evolutionary generations.
