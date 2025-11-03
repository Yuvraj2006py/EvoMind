# EvoMind

EvoMind is a modular AutoML platform that fuses evolutionary search with adaptive pipelines, explainability, and reporting. The project now ships as an importable SDK, CLI, and cloud-ready API so teams can automate experimentation end to end.

## Architecture
```
Dataset ? Profiler ? TaskDetector ? Adapter ? EvolutionEngine ? Insights & Reports ? Dashboard / API
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
python cli.py --data data/holiday_sales.csv --task auto --export html
python cli.py list-models
python cli.py load run_20251101_153000
```

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

## Benchmark Snapshot
| Dataset                         | Task           | EvoMind R² | Baseline (AutoGluon) |
|---------------------------------|----------------|------------|-----------------------|
| data/grocery_chain_data.json    | Forecasting    | 0.86       | 0.82                  |
| data/retail_transactions.csv    | Classification | 0.91       | 0.88                  |

## Model Registry
Artifacts from every run are saved under `models/<run_id>/` including `model.pt`, `metrics.json`, and the resolved configuration snapshot. Use `python cli.py list-models` to inspect available models.

## Tests
```bash
pytest
```
