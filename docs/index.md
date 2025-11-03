# EvoMind Developer Guide

Welcome to the EvoMind documentation portal. EvoMind is a modular AutoML SDK that blends evolutionary search, adaptive preprocessing, explainability, and reporting. The project ships as an importable library, CLI toolkit, Streamlit dashboard, and FastAPI microservice.

## Quick Links
- [Configuration Reference](config_reference.md)
- [Adapters](adapters.md)
- [Dashboard](dashboard.md)
- [API](api.md)
- [Testing & Quality](testing.md)
- [Deployment](deployment.md)

## Generate Documentation

Regenerate the documentation bundle after modifying the SDK:

```bash
python docs/generate_docs.py
```

The command refreshes the Markdown reference files and builds API docs into `docs/site/` using `pdoc`.
