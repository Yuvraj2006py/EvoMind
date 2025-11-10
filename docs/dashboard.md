# Dashboard

The Streamlit dashboard provides an interactive exploration surface for EvoMind runs.

## Tabs & Features

- **Overview:** Key metrics, fitness curve, stability label, and quick artefact links.
- **Insights:** SHAP and LIME visualisations, permutation importance, and what-if sliders for sensitivity analysis.
- **Data Profile:** Data health score, missingness heatmaps, correlation matrix, and an interactive NetworkX correlation graph.
- **Lineage:** Generation slider to replay evolutionary progress, diversity index, and parent-child lineage map with modal model cards.
- **Reports:** Download rendered HTML/PDF reports, model cards, and MLflow run links.

## Launch Locally

```bash
streamlit run evomind/dashboard/app.py
```

Pass a specific run directory (or your custom run name) to inspect historical executions:

```bash
streamlit run evomind/dashboard/app.py -- --run_id shopping_behavior_003
```

## Customisation

Use the configuration system to adjust visual defaults (theme, plot density) or override feature lists. The dashboard consumes the experiment artefacts saved under `experiments/<run_id>/` (e.g., `experiments/shopping_behavior_003/` or `experiments/holiday_campaign_002/` when you reuse `--run-name holiday_campaign`).
