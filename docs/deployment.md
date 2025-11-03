# Deployment Guide

This document explains how to run the full EvoMind stack (API service, dashboard, and Ray workers) with Docker Compose. It also covers environment configuration and production hardening tips.

## Prerequisites

- Docker Engine 24 or newer
- Docker Compose v2
- (Optional) GPU drivers if you intend to use GPU-enabled Ray images

## Quick Start

```bash
docker compose up --build
```

This command launches the multi-service stack defined in `docker-compose.yml`:

- **evomind-api** - FastAPI service that also boots a Ray head node.
- **evomind-dashboard** - Streamlit dashboard for visual analytics.
- **ray-worker** - Worker node that connects to the Ray head for distributed evaluations.

The services mount the project directory so code changes are reflected immediately. For production, build immutable images and remove the bind mount.

## Service Overview

| Service             | Purpose                                         | Default Ports |
|--------------------|-------------------------------------------------|---------------|
| `evomind-api`      | FastAPI endpoint and Ray head                   | 8000, 8265    |
| `evomind-dashboard`| Streamlit analytics dashboard                   | 8501          |
| `ray-worker`       | Ray worker node connecting to the API head node | n/a           |

### Environment Variables

| Variable                 | Default | Description                                                  |
|--------------------------|---------|--------------------------------------------------------------|
| `EVOMIND_API_PORT`       | 8000    | Published port for the FastAPI service.                      |
| `EVOMIND_DASHBOARD_PORT` | 8501    | Published port for the Streamlit dashboard.                  |
| `EVOMIND_API_URL`        | derived | Dashboard uses this to reach the API (set automatically in Compose). |
| `RAY_HEAD_ADDR`          | derived | Ray worker connection string (`ray://evomind-api:10001`).    |

Override ports by exporting variables before running `docker compose up`:

```bash
export EVOMIND_API_PORT=9000
export EVOMIND_DASHBOARD_PORT=8600
docker compose up --build
```

## Production Considerations

- **Immutable images:** Use `Dockerfile.prod` to build a runtime image and push it to your registry. Avoid mounting the project directory in production.
- **Secrets management:** Configure sensitive values (for example API tokens or database URLs) via Docker secrets or an external vault. Never bake credentials into images.
- **TLS termination:** Place a reverse proxy (NGINX or Traefik) in front of the API and dashboard for HTTPS termination and rate limiting.
- **Health checks:** Add Compose `healthcheck` entries or integrate readiness checks in your orchestrator (Kubernetes, ECS). Expose a `/health` endpoint where needed.
- **Scaling Ray:** Increase `ray-worker` replicas or deploy dedicated Ray nodes as workloads grow. Ensure the `ray://` connection string remains reachable from all workers.
- **Persistent storage:** Mount `experiments/`, `models/`, and `logs/` to durable volumes if you need to retain artifacts outside the container lifecycle.

## Running the API Locally (Optional)

For quick verification without Docker, install dependencies and run:

```bash
uvicorn evomind.api.server:app --reload
```

## Updating Images

1. Rebuild the production image:
   ```bash
   docker build -f Dockerfile.prod -t evomind:latest .
   ```
2. Push to your registry (optional):
   ```bash
   docker tag evomind:latest ghcr.io/<user>/evomind:latest
   docker push ghcr.io/<user>/evomind:latest
   ```
3. Update your Compose file or orchestration manifests to pull the new image.

---
For additional configuration and API usage details, refer to `docs/config_reference.md`, `docs/api.md`, and the README.
