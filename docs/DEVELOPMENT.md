# Development Guide

This document covers the development process, implementation details, and lessons learned while building the Refund Item Classification System.

---

## Table of Contents

1. [Asset Management & Reproducibility](#asset-management--reproducibility)
2. [Development Timeline](#development-timeline)
3. [Implementation Notes](#implementation-notes)
4. [Technology Choices](#technology-choices)
5. [Common Commands](#common-commands)
6. [Debugging Guide](#debugging-guide)
7. [Lessons Learned](#lessons-learned)
8. [Future Development](#future-development)

---

## Asset Management & Reproducibility

### Files Not in Git

```gitignore
data/                 # Training/test datasets (~750MB)
models/               # Model checkpoints (~50MB each)
mlflow_data/          # MLflow artifacts and DB (grows over time)
logs/                 # Runtime logs (regenerated)
*.pth                 # PyTorch model weights
*.pkl                 # Pickle files
__pycache__/          # Python cache
.venv/                # Virtual environment
```

### Why These Are Excluded

| Directory/File | Reason | Hosted Where |
|----------------|--------|--------------|
| `data/` | Large binary files | Google Drive |
| `models/*.pth` | GitHub 100MB file limit | Google Drive |
| `mlflow_data/` | Database grows over time, user-specific | Local only |
| `logs/` | Runtime-generated | Not needed |
| `.venv/` | Environment-specific | Not needed |

### Reproducibility Strategy

**Data:** Source is Kaggle Fashion Product Images, sampled with seed=42 via `notebooks/01_data_preparation.ipynb`.

**Model:** All hyperparameters logged in `training_metadata.json`. Training reproducible via `notebooks/02_model_training.ipynb`.

**MLflow:** First run auto-creates database. Artifacts are stored in `mlflow_data/` and tracked per-run.

---

### New Developer Onboarding

**From a fresh clone:**

```bash
# 1. Clone
git clone https://github.com/DanielPopoola/autorma.git
cd autorma

# 2. Download and extract assets
# Dataset: https://drive.google.com/drive/folders/1g1V4I3WL8FfXLZfkXqrTYcCR8etojiBY
# Model:   https://drive.google.com/drive/folders/1IQ4wyuTYO0TuQvKg0bIZ3n1kTZQpuvtp
unzip dataset.zip -d data/
unzip model_v1.zip -d models/

# 3. Create directories
mkdir -p mlflow_data/{artifacts,mlruns}
mkdir -p data/inference/{input,output,checkpoints}
mkdir -p logs

# 4. Install Python dependencies
uv sync

# 5. Start MLflow (Docker — recommended)
docker compose up mlflow -d

# 6. Register model (must happen while MLflow is running)
MLFLOW_TRACKING_URI=http://localhost:5000 python scripts/register_model.py
MLFLOW_TRACKING_URI=http://localhost:5000 python scripts/set_production.py

# 7. Start remaining services
docker compose up model-service -d
cd monitoring && docker-compose up -d && cd ..
streamlit run streamlit-ui/app.py
```

**Total setup time:** ~30 minutes including downloads.

> **Why must model registration happen after MLflow is running?**
> MLflow records artifact paths at log time, not load time. If you register while MLflow is running in Docker with `--serve-artifacts`, it records the URI as `mlflow-artifacts:/...` — a path that routes through the HTTP proxy. If you register against a local MLflow instance first and then try to load in Docker, it records an absolute host filesystem path that the container can't access.

---

### Retraining from Scratch

```bash
# 1. Dataset prep (Kaggle or locally)
# Open notebooks/01_data_preparation.ipynb, run all cells (seed=42)

# 2. Training (Google Colab recommended for GPU)
# Open notebooks/02_model_training.ipynb
# Download: best_model.pth, training_metadata.json to models/v2/

# 3. Register new version
MLFLOW_TRACKING_URI=http://localhost:5000 python scripts/register_model.py

# 4. Promote to production
MLFLOW_TRACKING_URI=http://localhost:5000 python scripts/set_production.py

# 5. Restart Model Service to pick up new version
docker compose restart model-service
```

---

## Development Timeline

### Stage 1: Dataset Preparation (Week 1)

Explored Fashion Product Images on Kaggle. Identified 5 categories: Shirts, Watches, Casual Shoes, Tops, Handbags. Sampled 500 images per class, split 70/15/15 into train/val/test. Downloaded as zip (~750MB).

**Challenge:** Dataset used `articleType` column, not `masterCategory`.
**Solution:** Explored schema first with pandas before writing the sampling script.

---

### Stage 2: MLflow Setup (Week 1)

Started MLflow with SQLite backend and file-based artifact store.

**Critical lesson learned:** MLflow must be started with `--serve-artifacts` and `--artifacts-destination` (not `--default-artifact-root`) when running in Docker. Using the wrong flags causes artifact URIs to be recorded as absolute host filesystem paths — paths the container can't reach. See [Lessons Learned](#lessons-learned) for the full explanation.

---

### Stage 3: Model Training (Week 2)

Used EfficientNet-B0 with transfer learning on Google Colab (T4 GPU). Trained for 15 epochs (~12 minutes). Achieved 96.53% test accuracy. Downloaded checkpoint and metadata, then registered in MLflow.

---

### Stage 4: Model Service (Week 3)

Built FastAPI service that loads the model from MLflow at startup using the `@production` alias.

```python
# Correct — uses alias syntax (current MLflow API)
model_uri = "models:/refund-classifier@production"

# Wrong — uses stage syntax (deprecated in MLflow 2.x)
model_uri = "models:/refund-classifier/Production"
```

Added Prometheus metrics: request counts, latency histograms, prediction confidence, class distribution.

---

### Stage 5: Batch Orchestrator (Week 3)

Built `orchestrator/batch_inference.py` implementing checkpoint-based batch processing.

**Key algorithm:**
```
1. Load checkpoint
2. Scan input directory
3. Filter already-processed images
4. Split into batches of 10
5. For each batch: call /predict, save results, update checkpoint
6. Push metrics to Pushgateway
7. Save final JSON
```

**Testing idempotency:**
```bash
python orchestrator/batch_inference.py
python orchestrator/batch_inference.py  # Should print "No new images to process"
```

---

### Stage 6: Monitoring Stack (Week 4)

Set up Prometheus + Pushgateway + Grafana via Docker Compose. Configured scrape targets. Built Grafana dashboard with panels for: total images processed, request rate, confidence distribution, predictions by class, batch success rate, batch duration.

**WSL networking note:** Prometheus running in Docker cannot reach `localhost` — it sees the container network, not the host. Use the WSL IP address in `prometheus.yml`:

```bash
# Get WSL IP
ip addr show eth0 | grep -oP '(?<=inet\s)\d+(\.\d+){3}'
```

---

### Stage 7: Streamlit UI (Week 4-5)

Built `streamlit-ui/app.py` with three tabs: Upload & Classify, Results History, and About.

Key integration pattern — calling the orchestrator from Streamlit:
```python
subprocess.run(
    ["python", "orchestrator/batch_inference.py"],
    capture_output=True,
    cwd="/absolute/path/to/project"  # Must be absolute
)
```

---

### Stage 8: Docker Setup (Week 5)

Containerised MLflow and Model Service. Key decisions:

- MLflow uses `--serve-artifacts` so the model service downloads artifacts over HTTP rather than reading the filesystem directly. This makes the setup portable — no shared filesystem tricks needed.
- Model registration must happen against the running Docker MLflow instance (not a local one) so artifact URIs are recorded correctly.
- Orchestrator runs with `--profile manual` so it only starts when explicitly invoked.

---

## Technology Choices

### Why uv?

- 10-100x faster than pip
- Deterministic via `uv.lock`
- Single source of truth in `pyproject.toml`

### Why Python?

Best ML ecosystem. Only consider Go/Rust for the orchestration layer at massive scale.

### Why EfficientNet-B0?

Good accuracy/size tradeoff (20MB checkpoint). Reasonable CPU inference speed (~100-200ms/image). Well-supported by `timm`.

Alternatives considered: ResNet50 (larger, slower), MobileNetV2 (faster but lower accuracy), ViT (overkill, needs more data).

### Why FastAPI?

Automatic OpenAPI docs, Pydantic validation, async support, easy Prometheus integration. Flask lacks type safety and auto-docs; Django is too heavy.

### Why MLflow?

Open source, self-hosted, built-in model registry and experiment tracking. Weights & Biases and Neptune are cloud-based and cost money. DVC is good for data versioning but not model serving.

### Why Prometheus + Grafana?

Industry standard, free, time-series optimised, excellent alerting. CloudWatch is AWS-only; Datadog is expensive; ELK is log-focused.

### Why Streamlit?

Pure Python, fast to build, built-in components for file upload and charts. Perfect for ML demos and internal tooling.

---

## Common Commands

### Starting Services

**Docker (recommended):**
```bash
docker compose up mlflow -d
docker compose up model-service -d
docker compose --profile manual run orchestrator
```

**MLflow (local):**
```bash
ABS_PATH=$(pwd)
mlflow server \
  --backend-store-uri sqlite:///$ABS_PATH/mlflow_data/mlflow.db \
  --artifacts-destination file:///$ABS_PATH/mlflow_data/artifacts \
  --serve-artifacts \
  --host 0.0.0.0 \
  --port 5000
```

**Model Service (local):**
```bash
cd model_service
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**Monitoring:**
```bash
cd monitoring
docker-compose up -d
docker-compose logs -f
docker-compose down
```

**Streamlit:**
```bash
streamlit run streamlit-ui/app.py
```

### Running Batch Jobs

```bash
# Copy test images to input
find data/processed/test -name "*.jpg" | shuf -n 50 | xargs -I {} cp {} data/inference/input/

# Run orchestrator locally
python orchestrator/batch_inference.py

# Run orchestrator via Docker
docker compose --profile manual run orchestrator

# View results
cat data/inference/output/predictions_*.json | jq
```

### Model Management

```bash
# Register new version
MLFLOW_TRACKING_URI=http://localhost:5000 python scripts/register_model.py

# Promote to production
MLFLOW_TRACKING_URI=http://localhost:5000 python scripts/set_production.py

# Rollback (Python)
python -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5000')
client = mlflow.MlflowClient()
client.set_registered_model_alias('refund-classifier', 'production', '1')
"
```

---

## Debugging Guide

### Model Service Won't Start

1. Check MLflow is healthy: `curl http://localhost:5000`
2. Check model is registered: visit http://localhost:5000 → Models
3. Check production alias is set: Models → refund-classifier → Aliases
4. Check `MLFLOW_TRACKING_URI` env var is set correctly in the container
5. Check Docker logs: `docker logs model-service`

**Common error:** `OSError: No such file or directory: '/home/user/...mlflow_data/artifacts/...'`

This means the model was registered against a local MLflow instance (not Docker). The artifact URI recorded in the database is an absolute host path the container can't reach. Fix: delete the model from the MLflow UI, then re-register with `MLFLOW_TRACKING_URI=http://localhost:5000`.

### Orchestrator Fails

1. Check Model Service is healthy: `curl http://localhost:8000/health`
2. Check input directory has images: `ls data/inference/input/`
3. Check checkpoint isn't corrupted: `cat data/inference/checkpoints/checkpoint.json`
4. Run with verbose output: `python orchestrator/batch_inference.py 2>&1 | head -50`

### Grafana Shows No Data

1. Generate traffic first: run a batch job or hit `/predict`
2. Expand time range to "Last 6 hours"
3. Verify Prometheus data source is configured
4. Check Prometheus targets: http://localhost:9090/targets (all should be UP)
5. If using WSL: ensure `prometheus.yml` uses WSL IP, not `localhost`

---

## Lessons Learned

1. **MLflow artifact URIs are immutable.** They're recorded at log time. If you log against a local server, the URI is a host filesystem path. If you log against a Docker server with `--serve-artifacts`, the URI is an `mlflow-artifacts:/` proxy path. You can't fix a wrong URI after the fact — you have to re-register the model.

2. **`--serve-artifacts` and `--default-artifact-root` are different flags for different purposes.** `--default-artifact-root` sets where artifacts are stored and records that path in the database. `--artifacts-destination` + `--serve-artifacts` stores artifacts there but records an `mlflow-artifacts:/` URI that routes through the HTTP server. The second pattern is required for Docker.

3. **Absolute paths everywhere.** Relative paths cause "works on my machine" issues. The orchestrator especially must use absolute paths since it can be called from Streamlit (different cwd) or cron.

4. **Checkpointing is essential for batch jobs.** Early version re-processed all images on crash. Saving checkpoint after each mini-batch means at most 10 images of lost progress.

5. **WSL has its own network stack.** `localhost` inside a Docker container on WSL is not the same as `localhost` on the host. Use the WSL IP in Prometheus scrape targets.

6. **MLflow stages are deprecated.** Old tutorials use `models:/name/Production` (slash + stage name). Current MLflow uses `models:/name@alias`. If you follow old docs, the API calls silently fail or behave unexpectedly.

7. **Instrument metrics from the start.** Adding Prometheus metrics as an afterthought means retrofitting counters into code paths you've already forgotten. Easier to add them alongside each feature.

8. **UI makes demos significantly better.** Showing `curl` commands in a defense presentation is unconvincing. A Streamlit UI built in a day is worth it.

9. **Test each component immediately after building it.** Integration bugs found late are much harder to debug than bugs found close to when the code was written.

10. **Production ML is 10% model training, 90% everything else.** The model took 12 minutes to train. The serving, orchestration, monitoring, and reliability work took weeks.

---

## Future Development

### Immediate (1-3 Months)

- **Data drift detection:** Monitor input image statistics, alert on distribution shift
- **Retry logic:** Exponential backoff for transient Model Service failures
- **Model quantization:** INT8 for 2-4x CPU speedup with minimal accuracy loss
- **PostgreSQL MLflow backend:** Replace SQLite for concurrent access support

### Medium-Term (3-6 Months)

- **Dynamic batching:** Adjust batch size based on queue depth
- **A/B testing:** Route a percentage of traffic to a shadow model version
- **Active learning loop:** Flag low-confidence predictions for human review, feed labels back into training

### Long-Term (Cloud Migration)

| Current | Cloud equivalent |
|---------|-----------------|
| Local filesystem artifacts | S3 / GCS |
| SQLite MLflow backend | RDS / Cloud SQL |
| Cron job | Cloud Scheduler / Airflow |
| Docker Compose | Kubernetes / Cloud Run |
| Pushgateway | Managed Prometheus |

The clean component boundaries mean migration can happen one piece at a time — no big-bang rewrite needed.

---

**Next Developer:** Read ARCHITECTURE.md first for the big picture, then this file for implementation context. Follow the Docker Quick Start in README.md. You'll have the system running in under 30 minutes.