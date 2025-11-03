# API

EvoMind exposes a FastAPI service for remote execution and integration with external systems.

## Quickstart

```bash
uvicorn evomind.api.server:app --host 0.0.0.0 --port 8000
```

Submit a dataset:

```bash
curl -X POST "http://localhost:8000/run" \
     -F file=@data/grocery_chain_data.json
```

The response contains metrics and a pointer to the generated report directory.

## Endpoints

| Method | Path | Description |
| ------ | ---- | ----------- |
| `POST` | `/run` | Upload a dataset and trigger an EvoMind pipeline run. |
| `GET`  | `/health` | Optional health-check endpoint for liveness probes. |

Extend the API by importing the `EvoMind` class and exposing additional metadata (profiling summaries, fairness diagnostics, etc.).

## Deployment

Combine the API with the dashboard and Ray workers using the provided `docker-compose.yml`. See `docs/deployment.md` for production hardening tips (TLS, authentication, scaling).
