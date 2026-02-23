# Autorma — Automated Refund Item Classification

An end-to-end MLOps system for classifying returned e-commerce items using computer vision. Built with production-grade practices: model versioning, containerised services, batch inference, and real-time monitoring.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Asset Management](#asset-management)
3. [Quick Start](#quick-start)
4. [Running Batch Jobs](#running-batch-jobs)
5. [Monitoring](#monitoring)
6. [Troubleshooting](#troubleshooting)

---

## System Overview

| Service | Purpose | Port |
|---|---|---|
| MLflow | Model registry & experiment tracking | 5000 |
| Model Service | FastAPI inference API | 8000 |
| Orchestrator | Batch inference runner | — |
| Prometheus | Metrics collection | 9090 |
| Pushgateway | Batch metrics ingestion | 9091 |
| Grafana | Dashboards | 3000 |
| Streamlit UI | Manual batch interface | 8501 |

**Model:** EfficientNet-B0 fine-tuned on 5 return categories — Shirts, Watches, Casual Shoes, Tops, Handbags. Test accuracy: **96.53%**

---

## Asset Management

The following files are not in Git and must be downloaded before running the system:

| Asset | Size | Location | Download | Required |
|---|---|---|---|---|
| Training Dataset | ~1GB | `data/processed/` | [Google Drive](https://drive.google.com/drive/folders/1g1V4I3WL8FfXLZfkXqrTYcCR8etojiBY?usp=drive_link) | Yes |
| Trained Model v1 | ~50MB | `models/v1/` | [Google Drive](https://drive.google.com/drive/folders/1IQ4wyuTYO0TuQvKg0bIZ3n1kTZQpuvtp?usp=drive_link) | Yes |

See [docs/ASSETS.md](docs/ASSETS.md) for detailed download and verification instructions.

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.12+ with [uv](https://docs.astral.sh/uv/)
- 8GB RAM minimum
- Downloaded assets (see above)

### 1. Clone and install

```bash
git clone https://github.com/DanielPopoola/autorma.git
cd autorma
uv sync
```

### 2. Create required directories

```bash
mkdir -p data/inference/{input,output,checkpoints} mlflow_data/artifacts logs
```

### 3. Start core services with Docker

MLflow and the Model Service run in Docker. Start them together:

```bash
docker compose up --build -d
```

Wait for both to be healthy (takes ~60s on first build — torch is large):

```bash
docker compose ps  # Both should show "healthy"
```

Services are accessible at:
- MLflow UI: http://localhost:5000
- Model Service: http://localhost:8000/docs

### 4. Register the model

This only needs to be done once (or after clearing the MLflow database). Run while Docker services are running:

```bash
MLFLOW_TRACKING_URI=http://localhost:5000 python scripts/register_model.py
MLFLOW_TRACKING_URI=http://localhost:5000 python scripts/set_production.py
```

> ⚠️ **Always register while the Dockerised MLflow server is running.** Registering against a locally-run MLflow instance records host-absolute artifact paths that containers cannot resolve. See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for the full explanation of why this matters.

After registering, restart the model service to load the model:

```bash
docker compose restart model-service
```

Verify it loaded: `curl http://localhost:8000/health`

### 5. Start the Streamlit UI (optional)

```bash
streamlit run streamlit-ui/app.py
```

Access at: http://localhost:8501

### 6. Start the monitoring stack (optional)

```bash
cd monitoring && docker compose up -d
```

Access Grafana at http://localhost:3000 (admin/admin).

---

## Running Batch Jobs

The orchestrator runs as a one-shot container — triggered manually or by cron.

### Manual run

```bash
# Populate the input directory with test images
find data/processed/test -name "*.jpg" | shuf -n 50 | xargs -I {} cp {} data/inference/input/

# Run the orchestrator
docker compose --profile manual up orchestrator
```

Results are written to `data/inference/output/` as JSON.

### Scheduled via cron

```bash
crontab -e

# Add: run nightly at 2 AM
0 2 * * * cd /home/youruser/autorma && docker compose --profile manual run --rm orchestrator >> logs/cron.log 2>&1
```

### Idempotency

The orchestrator checkpoints after each batch to `data/inference/checkpoints/checkpoint.json`. Re-running will skip already-processed images. To force a full rerun:

```bash
rm data/inference/checkpoints/checkpoint.json
```

---

## Model Management

**Register a new model version:**

```bash
MLFLOW_TRACKING_URI=http://localhost:5000 python scripts/register_model.py
MLFLOW_TRACKING_URI=http://localhost:5000 python scripts/set_production.py
```

**Roll back to a previous version:**

```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
client = mlflow.MlflowClient()
client.set_registered_model_alias("refund-classifier", "production", "1")  # version number
```

Then restart the model service: `docker compose restart model-service`

---

## Monitoring

With the monitoring stack running:

- **Prometheus:** http://localhost:9090
- **Pushgateway:** http://localhost:9091
- **Grafana:** http://localhost:3000 — request rate, latency, class distribution, batch success rate

Key metrics exposed by the model service at `/metrics`:

- `api_requests_total{endpoint, status}`
- `api_request_duration_seconds`
- `prediction_confidence`
- `predictions_by_class_total{class_name}`
- `images_processed_total`

---

## Troubleshooting

**Model service fails with "No such file or directory" on an artifact path**

The model was registered against a non-Docker MLflow instance. The artifact path was recorded as a host-absolute path containers can't reach. Fix:

1. Delete the registered model and its experiment in the MLflow UI (http://localhost:5000)
2. Ensure Docker is running: `docker compose up -d`
3. Re-register: `MLFLOW_TRACKING_URI=http://localhost:5000 python scripts/register_model.py`
4. `docker compose restart model-service`

**Model service exits immediately**

`docker logs model-service` — MLflow likely wasn't healthy when the service started. Run `docker compose restart model-service`.

**Orchestrator can't find images**

`data/inference/input/` on your host is mounted into the container. Confirm images are there: `ls data/inference/input/`.

**Grafana shows no data**

Run a batch job to generate traffic first, then expand the time range to "Last 6 hours".

---

## 👤 Author

Built as a final year project demonstrating end-to-end ML systems engineering.

**Stack:** Python · PyTorch · FastAPI · MLflow · Docker · Prometheus · Grafana · Streamlit

## 📄 License

MIT — see LICENSE file.