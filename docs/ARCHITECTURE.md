# System Architecture

This document covers the architecture of the Refund Item Classification System — component responsibilities, data flows, and design decisions.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Component Details](#component-details)
3. [Data Flow](#data-flow)
4. [Design Decisions](#design-decisions)
5. [Scalability Considerations](#scalability-considerations)

---

## Architecture Overview

The system follows a microservices-inspired architecture with clear separation of concerns. Core ML services (MLflow, Model Service) run in Docker containers on a shared network. Supporting services (monitoring, UI) are optional but production-ready.

```
┌──────────────────────────────────────────────────────────────────┐
│                      PRESENTATION LAYER                          │
│  ┌─────────────────────┐       ┌──────────────────────────┐     │
│  │   Streamlit UI      │       │   Grafana Dashboard      │     │
│  │   Port: 8501        │       │   Port: 3000             │     │
│  │   - Upload images   │       │   - View metrics         │     │
│  │   - Trigger batches │       │   - Monitor health       │     │
│  │   - View results    │       │   - Analyse trends       │     │
│  └──────────┬──────────┘       └────────────▲─────────────┘     │
└─────────────┼──────────────────────────────────┼─────────────────┘
              │ HTTP                             │ PromQL
              ▼                                 │
┌──────────────────────────────────────────────────────────────────┐
│                   APPLICATION LAYER (Docker)                     │
│   Network: ml-network                                            │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           Batch Orchestrator                            │    │
│  │           orchestrator/batch_inference.py               │    │
│  │           (profile: manual - runs on demand)            │    │
│  │                                                         │    │
│  │  - Scans data/inference/input/ for new images          │    │
│  │  - Batches images, calls Model Service                 │    │
│  │  - Saves results to data/inference/output/             │    │
│  │  - Checkpoints progress for crash recovery             │    │
│  │  - Pushes batch metrics to Pushgateway                 │    │
│  └────────────────────┬────────────────────────────────────┘    │
│                       │ HTTP POST /predict                       │
│                       ▼                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           Model Service (FastAPI)                       │    │
│  │           model_service/app.py                          │    │
│  │           Port: 8000                                    │    │
│  │                                                         │    │
│  │  - Loads production model from MLflow on startup       │    │
│  │  - POST /predict  -- batch image classification        │    │
│  │  - GET  /health   -- liveness check                    │    │
│  │  - GET  /metrics  -- Prometheus scrape endpoint        │    │
│  └────────────────────┬────────────────────────────────────┘    │
│                       │ mlflow-artifacts:// over HTTP           │
│                       ▼                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           MLflow Server                                 │    │
│  │           Port: 5000                                    │    │
│  │                                                         │    │
│  │  - Backend store: SQLite (mlflow_data/mlflow.db)       │    │
│  │  - Artifact proxy: --serve-artifacts                   │    │
│  │  - Artifact destination: /mlflow_data/artifacts        │    │
│  │  - Model Registry: refund-classifier                   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────────────────┐  ┌──────────────────────────────┐    │
│  │  Filesystem Storage  │  │  Prometheus + Pushgateway    │    │
│  │                      │  │  Ports: 9090 / 9091          │    │
│  │  data/inference/     │  │                              │    │
│  │  ├── input/          │  │  Scrapes: model-service:8000 │    │
│  │  ├── output/         │  │  Receives: orchestrator push │    │
│  │  └── checkpoints/    │  │                              │    │
│  │                      │  │  Feeds into Grafana          │    │
│  │  mlflow_data/        │  │                              │    │
│  │  ├── mlflow.db       │  │                              │    │
│  │  └── artifacts/      │  │                              │    │
│  └──────────────────────┘  └──────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                      SCHEDULING LAYER                            │
│   cron (2 AM daily) -> docker compose --profile manual run      │
│                         orchestrator                             │
└──────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Model Service (FastAPI)

**Location:** `model_service/app.py`

**Purpose:** Stateless HTTP inference API. Loads the production model from MLflow at startup and serves it for the lifetime of the container.

**API Contract:**

```
POST /predict
Request:  { "image_paths": ["/data/inference/input/img1.jpg"] }
Response: { "predictions": [{ "predicted_class": "shirts",
                               "confidence": 0.98,
                               "all_probabilities": {...} }],
            "model_version": "production" }

GET /health  ->  { "status": "healthy", "model_loaded": true, "model_version": "production" }
GET /metrics ->  Prometheus text format
```

**Startup sequence:**
1. Set `MLFLOW_TRACKING_URI` to `http://mlflow:5000`
2. Load `models:/refund-classifier@production` via MLflow proxy
3. If production alias missing, fall back to `@latest`
4. Serve requests

**Metrics exposed:**

| Metric | Type | Description |
|---|---|---|
| `api_requests_total{endpoint,status}` | Counter | All requests |
| `api_request_duration_seconds` | Histogram | Latency per endpoint |
| `prediction_confidence` | Histogram | Confidence scores |
| `predictions_by_class_total{class_name}` | Counter | Per-class counts |
| `model_loaded` | Gauge | 1 if model loaded |
| `images_processed_total` | Counter | Cumulative images |

---

### 2. MLflow Server

**Location:** `mlflow.Dockerfile`

**Purpose:** Model registry, experiment tracking, and artifact proxy.

**Critical configuration:**

```bash
mlflow server \
  --backend-store-uri sqlite:////mlflow_data/mlflow.db \
  --artifacts-destination file:///mlflow_data/artifacts \
  --serve-artifacts \
  --host 0.0.0.0 \
  --port 5000
```

The `--serve-artifacts` flag is non-negotiable for Docker deployments. Without it, MLflow returns raw `file://` URIs pointing to wherever the artifact was originally logged. Containers on the `ml-network` resolve artifacts via `mlflow-artifacts://` URIs over HTTP through the MLflow server — they never need direct filesystem access to `mlflow_data/artifacts/`. The volume is mounted only for persistence.

**Model aliasing workflow:**

```
Register version -> set alias "production" -> model-service loads @production on startup
                                           -> rollback: reassign alias to older version -> restart service
```

---

### 3. Batch Orchestrator

**Location:** `orchestrator/batch_inference.py`

**Purpose:** Scheduled batch job. Scans the input directory, classifies images in batches via the Model Service, writes results to JSON, checkpoints progress.

**Algorithm:**

```
1. Load checkpoint (set of already-processed image paths)
2. Scan data/inference/input/ for .jpg/.png files
3. Filter out already-processed images
4. If none: exit cleanly
5. Split remainder into batches of N
6. For each batch:
   a. POST image paths to Model Service /predict
   b. Append results to output JSON
   c. Write updated checkpoint (atomic)
7. Push batch metrics to Pushgateway
```

Runs as a Docker container with `--profile manual` — only starts when explicitly invoked and exits after one pass. No persistent process.

**Volumes mounted:**
- `./data/inference` -> `/data/inference` (input images and output results)
- `./logs` -> `/app/logs`

---

### 4. Monitoring Stack

**Location:** `monitoring/docker-compose.yml`

**Components:** Prometheus (scrapes model-service), Pushgateway (receives orchestrator metrics), Grafana (dashboards).

The monitoring stack runs separately from the core ML stack and is optional for local development.

**Prometheus scrape config:**

```yaml
scrape_configs:
  - job_name: 'model_service'
    static_configs:
      - targets: ['model-service:8000']
    scrape_interval: 15s

  - job_name: 'pushgateway'
    static_configs:
      - targets: ['pushgateway:9091']
    scrape_interval: 15s
```

> If running Prometheus outside the `ml-network`, use `host.docker.internal:8000` or the WSL IP instead of `model-service:8000`.

---

### 5. Streamlit UI

**Location:** `streamlit-ui/app.py`

**Purpose:** User-friendly interface for manual batch processing and results review. Runs on the host, not in Docker — it's a dev/demo tool only.

---

## Data Flow

### Batch inference (primary flow)

```
[cron / manual trigger]
        |
        v
  Orchestrator container starts
        |
        |-- reads checkpoint
        |-- scans data/inference/input/
        |-- filters processed images
        |
        v
  POST /predict to model-service:8000
  { "image_paths": ["/data/inference/input/img1.jpg", ...] }
        |
        v
  Model Service
  |-- loads image from shared volume
  |-- runs EfficientNet-B0 inference
  `-- returns predictions + confidence scores
        |
        v
  Orchestrator writes results JSON
  -> data/inference/output/predictions_<timestamp>.json
  -> updates checkpoint
  -> pushes metrics to Pushgateway
        |
        v
  Prometheus scrapes Pushgateway -> Grafana displays batch metrics
```

### Model load flow (on model-service startup)

```
model-service starts
  |
  |-- sets MLFLOW_TRACKING_URI=http://mlflow:5000
  |-- requests models:/refund-classifier@production
  |
  v
MLflow server
  |-- resolves alias -> version number
  |-- looks up artifact URI -> mlflow-artifacts:/1/models/.../
  `-- streams artifact files back over HTTP
  |
  v
model-service loads model into memory -> ready to serve /predict
```

This proxy flow is why `--serve-artifacts` is required. Without it, MLflow returns a `file://` URI and the client tries to read from disk directly — which fails inside any container that doesn't mount `mlflow_data/artifacts` at the exact recorded path.

---

## Design Decisions

### Why Docker for core services?

The original setup ran MLflow and the Model Service as local processes. This caused a critical artifact path bug: when `register_model.py` ran on the host, MLflow recorded artifact locations as absolute host paths (`/home/user/autorma/mlflow_data/...`). Any container mounting the same directory sees it at a different path, so model loading fails.

Running MLflow in Docker with `--serve-artifacts` solves this permanently: artifact URIs are recorded as `mlflow-artifacts://...` and resolved by the MLflow server over HTTP, making them location-independent.

**The key rule:** artifact URIs are immutable once written. They are recorded at log time, not serve time. Server config changes only affect new experiments. This is why adding `--serve-artifacts` to an existing MLflow instance doesn't fix old registered models — you must delete the old experiment and re-register.

### Why `--serve-artifacts` and not `--default-artifact-root`?

`--default-artifact-root` sets where new artifacts are stored but still returns `file://` URIs to clients. The client then tries to access the artifact directly from disk. `--serve-artifacts` with `--artifacts-destination` makes the server own the entire download path — clients always go through HTTP, never touching the filesystem directly.

### Why batch processing over real-time?

Returns arrive throughout the day; classification results are only needed by the next morning. Batch processing means simpler infrastructure (no streaming, no always-on GPU), cheaper operation, and natural idempotency via checkpointing.

### Why separate Model Service from Orchestrator?

| Concern | Monolith | Separate service |
|---|---|---|
| Testing | Full pipeline required | Model testable in isolation |
| Deployment | Change one, restart all | Update independently |
| Scaling | Scale everything together | Scale model service only |
| Technology | Locked in | Orchestrator could be any language |

### Why MLflow?

MLflow's model registry with alias-based promotion (`@production`, `@latest`) gives a clean rollback story: reassign an alias, restart the service. No code changes required. `--serve-artifacts` makes this work correctly across container boundaries.

---

## Scalability Considerations

### Current capacity

Single CPU-only Model Service instance. Estimated throughput: ~10 images/batch, ~1s/batch -> ~36,000 images/hour. For a 6-hour nightly window that is ~216,000 images — far beyond current requirements.

### When and how to scale

**Vertical:** More RAM allows larger batch sizes. A GPU gives 10-100x faster inference with no code changes.

**Horizontal:** Run multiple model-service replicas behind a load balancer. The orchestrator round-robins across service URLs. MLflow server and filesystem storage remain shared state.

**Cloud migration:** S3 replaces the local artifact volume. Point `--artifacts-destination` at `s3://bucket/mlflow-artifacts`. SageMaker or Cloud Run replaces the model-service container. Everything else stays the same.

Don't scale prematurely. The current architecture handles 10x current load without changes.

---

## Future Improvements

**Near-term:**
- Data drift detection — monitor input image statistics, alert on distribution shift
- Retry logic in the orchestrator for transient Model Service failures
- Model quantization (INT8) for faster CPU inference

**Long-term:**
- Cloud deployment (S3 + ECS or Cloud Run)
- Active learning loop — low-confidence predictions flagged for human review, used to retrain
- A/B testing via traffic splitting at the Model Service

---

## Conclusion

This architecture prioritises simplicity and correctness over premature optimisation. Each component has one clear responsibility. The Docker network (`ml-network`) provides service discovery without manual IP management. The `--serve-artifacts` flag on the MLflow server is the keystone that makes artifact resolution work correctly across all container boundaries.