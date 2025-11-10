# EvoMind

EvoMind is a modular AutoML platform that fuses evolutionary search with adaptive pipelines, explainability, and reporting. The project now ships as an importable SDK, CLI, and cloud-ready API so teams can automate experimentation end to end.

## Architecture
```
Dataset -> Profiler -> TaskDetector -> Adapter -> EvolutionEngine -> Insights & Reports -> Dashboard / API
```
Key layers include:
- **Profiler**: schema detection, data health scoring, integrity warnings, and time-series diagnostics.
- **Task Detector & Adapters**: autonomous task selection with specialised preprocessors for tabular, retail, NLP, vision, and multimodal data.
- **Evolution Engine**: Ray-powered parallel fitness evaluation, Optuna refinement, and ensemble synthesis of leading genomes.
- **Insights & Reporting**: SHAP/LIME explainability, fairness diagnostics, narrative summaries, and HTML/PDF report export.
- **Visualization**: Streamlit dashboard with overview, insights, data profile, lineage replay, and report download tabs.

## Getting Started
```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run via SDK
```python
from evomind import EvoMind

runner = EvoMind(data="data/grocery_chain_data.json", task="auto", insights=True)
result = runner.run()
print(result.metrics)
result.export_report("html")
```

### CLI
```bash
python cli.py run --data data/grocery_chain_data.json --task auto --export html
python cli.py run --data data/grocery_chain_data.json --profile fast
python cli.py run --data data/january.csv --data data/february.csv --task regression
python cli.py run --data data/holiday_campaign.csv --task auto --run-name holiday_campaign
python cli.py list-models
python cli.py load shopping_behavior_003
python cli.py doctor
```
> Ray is optional. On Windows or when memory is constrained, EvoMind automatically falls back to threaded execution. You can opt out of Ray explicitly by setting `EVOMIND_DISABLE_RAY=1` or supplying a config with `engine.parallel: false`.

### FastAPI Service
```bash
uvicorn api.server:app --reload
curl -X POST "http://127.0.0.1:8000/run" -F file=@data/grocery_chain_data.json
```

### Dashboard
```bash
streamlit run evomind/dashboard/app.py
```

### Docker & Ray Cluster
```bash
docker compose up --build
```
See `docs/deployment.md` for production deployment notes, environment variables, and scaling guidance.

## Benchmark Snapshot
| Dataset                         | Task           | EvoMind R^2 | Baseline (AutoGluon) |
|---------------------------------|----------------|------------|-----------------------|
| data/grocery_chain_data.json    | Forecasting    | 0.86       | 0.82                  |
| data/retail_transactions.csv    | Classification | 0.91       | 0.88                  |

## Model Registry
Artifacts from every run are saved under `models/<run_id>/` including `model.pt`, `metrics.json`, and the resolved configuration snapshot. Run IDs now follow `<dataset_slug>_<counter>` (e.g., `shopping_behavior_003`), or your supplied slug when you pass `--run-name holiday_campaign`. Use `python cli.py list-models` to inspect available models.

## SDK API Reference
EvoMind ships with inline introspection so you can explore configuration options without leaving Python:

```python
from evomind import EvoMind

# Inspect every available configuration section (prints a console table)
EvoMind.describe_config(to_console=True)

# Focus on a single section and render Markdown
print(EvoMind.describe_config(section="engine", as_markdown=True))

# Ask EvoMind to explain a single hyperparameter
print(EvoMind.explain("population"))

# Generate CONFIG.md in the current working directory
EvoMind.generate_config_docs()

# Train on multiple CSVs by passing a list of paths
runner = EvoMind(data=["data/january.csv", "data/february.csv"], task="regression")

# Provide a custom run name (auto-increments when reused)
runner = EvoMind(data="data/holiday_campaign.csv", task="auto", run_name="holiday_campaign")
```

CLI equivalents:

```bash
python cli.py describe-config --section engine
python cli.py describe-config --key population_size
python cli.py generate-config-docs --output CONFIG.md
```

Run `help(EvoMind)` or `help(EvoMindResult)` for docstrings covering runtime methods like `run`, `export_report`, and `launch_dashboard`.

## Tests
```bash
pytest
```
