# Refund Item Classification System

An end-to-end machine learning system for automated classification of returned items in an e-commerce warehouse. Built with production-grade MLOps practices including model versioning, batch inference pipelines, monitoring, and a user-friendly interface.

## 🎯 Project Overview

This system demonstrates a complete ML deployment workflow that goes beyond model training to include:
- Automated batch inference pipeline with checkpoint recovery
- Model versioning and registry with MLflow
- Production monitoring with Prometheus and Grafana
- RESTful model serving with FastAPI
- Interactive UI for manual batch processing
- Scheduled automation via cron jobs

**Key Metrics:**
- Model Accuracy: 96.53% on test set
- Processing Speed: ~5 seconds per batch (10 images)
- Categories: 5 (Casual Shoes, Handbags, Shirts, Tops, Watches)
- Total Dataset: 2,500 images

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interfaces                         │
│  ┌──────────────────┐         ┌─────────────────────┐      │
│  │  Streamlit UI    │         │  Grafana Dashboard  │      │
│  │  (Manual Upload) │         │  (Monitoring)       │      │
│  └────────┬─────────┘         └──────────▲──────────┘      │
└───────────┼────────────────────────────────┼─────────────────┘
            │                                │
            ▼                                │
┌─────────────────────────────────────────────────────────────┐
│                  Orchestration Layer                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Batch Orchestrator (Python Script)                  │   │
│  │  - Scans input directory for new images              │   │
│  │  - Manages checkpoints for recovery                  │   │
│  │  - Calls Model Service API                           │   │
│  │  - Saves results and updates metrics                 │   │
│  └────────┬───────────────────────────────────┬─────────┘   │
└───────────┼───────────────────────────────────┼─────────────┘
            │                                   │
            │ HTTP POST /predict                │ Metrics
            ▼                                   ▼
┌──────────────────────────┐      ┌──────────────────────────┐
│   Model Service (API)    │      │   Prometheus + Pushgateway│
│  ┌────────────────────┐  │      │  - Scrapes /metrics      │
│  │  FastAPI Server    │  │      │  - Stores time series    │
│  │  ┌──────────────┐  │  │      │  - Feeds Grafana         │
│  │  │ EfficientNet │  │  │      └──────────────────────────┘
│  │  │ B0 Model     │  │  │
│  │  └──────────────┘  │  │
│  └────────────────────┘  │
│         ▲                │
└─────────┼────────────────┘
          │ Load model
          │
┌─────────┴────────────────┐
│   MLflow Registry        │
│  - Model versioning      │
│  - Experiment tracking   │
│  - Production/Staging    │
└──────────────────────────┘

Scheduled Automation:
  Cron (2 AM daily) → Batch Orchestrator → Process overnight returns
```

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed system design.

---

## 📁 Project Structure

```
autorma/
├── data/
│   ├── processed/              # Training/val/test datasets
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── inference/              # Batch processing data
│       ├── input/              # New images to classify
│       ├── output/             # Prediction results
│       └── checkpoints/        # Recovery checkpoints
│
├── model-service/              # FastAPI prediction service
│   ├── app.py
│
├── orchestrator/               # Batch inference pipeline
│   ├── batch_inference.py
│   ├── metrics_pusher.py
│
├── streamlit-ui/               # Web interface
│   ├── app.py
│
├── monitoring/                 # Prometheus + Grafana
│   ├── docker-compose.yml
│   └── prometheus.yml
│
├── mlflow_data/                # MLflow artifacts and metadata
│   ├── artifacts/
│   └── mlflow.db
│
├── models/                     # Trained model checkpoints
│   └── v1/
│       ├── best_model.pth
│       └── training_metadata.json
│
├── scripts/                    # Utility scripts
│   ├── register_model.py
│   └── set_production.py
│
├── notebooks/               # Training notebooks
|  ├── 01_data_prepartion.ipynb
│  └── 02_train_model.ipynb
│
├── logs/                       # Application logs
│
├── README.md
├── docs/
└── requirements.txt
```

---

## 📦 Asset Management

**Important:** Due to file sizes, the following assets are **not included** in the Git repository and must be downloaded separately.

| Asset | Size | Location | Download Link | Required? |
|-------|------|----------|---------------|-----------|
| Training Dataset | ~1GB | `data/processed/` | [Google Drive - Dataset](https://drive.google.com/drive/folders/1g1V4I3WL8FfXLZfkXqrTYcCR8etojiBY?usp=drive_link) | ✅ Yes |
| Trained Model v1 | ~50MB | `models/v1/` | [Google Drive - Model](https://drive.google.com/drive/folders/1IQ4wyuTYO0TuQvKg0bIZ3n1kTZQpuvtp?usp=drive_link) | ✅ Yes |
| MLflow Database | Variable | `mlflow_data/` | Auto-created on first run | ⚠️ Auto |
| Sample Results | <1MB | `data/inference/output/` | Optional (generated during use) | ❌ No |

📘 **For detailed download instructions and troubleshooting, see [ASSETS.md](docs/ASSETS.md)**

### First-Time Setup Checklist

Before running the system, ensure you have:

- [ ] Downloaded and extracted **dataset.zip** to `data/processed/`
- [ ] Downloaded and extracted **model_v1.zip** to `models/v1/`
- [ ] Verified directory structure (see below)

**Verify Setup:**
```bash
# Check dataset structure
tree data/processed -L 2

# Expected output:
# data/processed/
# ├── train/
# │   ├── casual shoes/
# │   ├── handbags/
# │   ├── shirts/
# │   ├── tops/
# │   └── watches/
# ├── val/
# │   └── (same structure)
# ├── test/
# │   └── (same structure)
# └── dataset_info.json

# Check model exists
ls -lh models/v1/

# Expected output:
# best_model.pth
# training_metadata.json
```

**Alternative: Use Setup Script**

```bash
# Run automated setup (downloads from configured links)
./scripts/setup_assets.sh

# Follow prompts to download missing files
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- 8GB RAM minimum
- Docker & Docker Compose (for monitoring)
- WSL2 (if on Windows)
- uv (https://docs.astral.sh/uv/)
- **Downloaded assets** (see Asset Management section above)

### 1. Clone and Setup

```bash
git clone https://github.com/DanielPopoola/autorma.git
cd autorma

# Verify you've downloaded required assets
ls data/processed/train  # Should show category folders
ls models/v1/            # Should show .pth and .json files


# Install dependencies with uv (recommended)
uv sync

# Alternative: Generate requirements.txt for pip users
uv pip compile pyproject.toml -o requirements.txt
pip install -r requirements.txt
```

### 2. Register Model in MLflow

Before starting services, register the downloaded model:

```bash
# Start MLflow server first (see step 3)
# Then in another terminal:
python scripts/register_model.py
python scripts/set_production.py
```

### 3. Start MLflow Server

```bash
ABS_PATH=$(pwd)
mlflow server \
  --backend-store-uri sqlite:///$ABS_PATH/mlflow_data/mlflow.db \
  --default-artifact-root file://$ABS_PATH/mlflow_data/artifacts \
  --host 0.0.0.0 \
  --port 5000
```

Access at: http://localhost:5000

### 4. Start Model Service

```bash
# In a new terminal
cd model-service
uvicorn app:app --host 0.0.0.0 --port 8000
```

Test health: `curl http://localhost:8000/health`

### 5. Start Monitoring Stack

```bash
cd monitoring
docker-compose up -d
```

Access:
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090

### 6. Run Streamlit UI

```bash
streamlit run streamlit-ui/app.py
```

Access at: http://localhost:8501

---

## 📊 Usage

### Manual Batch Processing (via UI)

1. Open Streamlit UI at http://localhost:8501
2. Navigate to "Upload & Classify" tab
3. Upload images (JPG/PNG)
4. Click "Run Classification"
5. View results in real-time
6. Download results as CSV

### Automated Batch Processing (via CLI)

```bash
# Place images in input directory
cp /path/to/images/* data/inference/input/

# Run batch inference
python orchestrator/batch_inference.py

# View results
cat data/inference/output/predictions_*.json
```

### Scheduled Automation (Cron)

The system runs automatically every night at 2 AM:

```bash
# Edit crontab
crontab -e

# Add this line
0 2 * * * cd /path/to/autorma && source .venv/bin/activate && python orchestrator/batch_inference.py >> logs/cron.log 2>&1
```

---

## 🔄 Model Update Workflow

### Training a New Model

1. Train model on Colab (see notebooks/)
2. Download checkpoint to `models/v{N}/`
3. Register in MLflow:

```bash
python scripts/register_model.py
```

### Promoting to Production

```bash
python scripts/set_production.py
```

This updates the production alias. Model Service will load the new version on next restart.

### Rollback

```python
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = mlflow.MlflowClient()

# Rollback to version 1
client.set_registered_model_alias("refund-classifier", "production", "1")
```

Restart Model Service to apply.

---

## 📈 Monitoring

### Key Metrics Tracked

**Model Service:**
- Request rate and latency (p50, p95, p99)
- Prediction confidence distribution
- Images processed per class
- API success/failure rate

**Batch Orchestrator:**
- Images processed per run
- Batch processing duration
- Success rate
- Failed images count

### Accessing Dashboards

**Grafana:** http://localhost:3000
- Dashboard: "Refund Classifier Monitoring"
- Real-time metrics visualization
- Historical trends

**Prometheus:** http://localhost:9090
- Raw metrics queries
- Target health status

---

## 🧪 Testing

### Test Model Service

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "image_paths": ["/absolute/path/to/test/image.jpg"]
  }'
```

### Test Batch Processing

```bash
# Copy test images
find data/processed/test -name "*.jpg" | shuf -n 20 | xargs -I {} cp {} data/inference/input/

# Run orchestrator
python orchestrator/batch_inference.py

# Verify results
ls -lh data/inference/output/
```

### Verify Monitoring

```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Check metrics endpoint
curl http://localhost:8000/metrics | grep images_processed
```

---

## 🛠️ Troubleshooting

### Model Service won't start

**Issue:** `model_loaded` shows 0 in metrics

**Solution:**
1. Check MLflow is running: `curl http://localhost:5000/health`
2. Verify model is registered: Check MLflow UI
3. Ensure production alias is set: `python scripts/set_production.py`

### Prometheus can't scrape Model Service

**Issue:** Targets show DOWN in Prometheus

**Solution (WSL):**
1. Get WSL IP: `ip addr show eth0 | grep -oP '(?<=inet\s)\d+(\.\d+){3}'`
2. Update `monitoring/prometheus.yml` with your IP
3. Restart: `docker-compose restart prometheus`

### Batch processing fails

**Issue:** Images not being processed

**Solution:**
1. Check Model Service is running: `curl http://localhost:8000/health`
2. Verify image paths are absolute
3. Check logs: `tail -f logs/orchestrator.log`
4. Look for checkpoint issues: `cat data/inference/checkpoints/checkpoint.json`

---

## 📚 Documentation

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - Detailed system design and component breakdown
- [DEVELOPMENT.md](docs/DEVELOPMENT.md) - Development guide and implementation notes

---

## 🎓 Key Learnings & Design Decisions

### Why Batch Processing?

- **Cost Efficiency:** Process overnight during low-traffic hours
- **Resource Optimization:** Batch GPU inference is more efficient than single predictions
- **Business Alignment:** Returns are processed daily, not real-time
- **Simplicity:** Avoids complexity of real-time streaming systems

### Why Separate Model Service?

- **Testability:** Can test inference independently
- **Deployability:** Update orchestration logic without reloading model
- **Scalability:** Can scale Model Service separately if needed
- **Technology Flexibility:** Could rewrite orchestrator in Go without touching ML code

### Why MLflow?

- **Reproducibility:** Track experiments, hyperparameters, metrics
- **Version Control:** Manage model versions with aliases
- **Easy Rollback:** Quickly revert to previous model if needed
- **Team Collaboration:** Multiple data scientists can share experiments

### Why Prometheus + Grafana?

- **Industry Standard:** Production monitoring pattern
- **Time-Series Data:** Perfect for tracking metrics over time
- **Alerting Ready:** Can add alerts on metric thresholds
- **Visualization:** Grafana provides professional dashboards

---

## 🔄 Reproducibility

### Recreating the System from Scratch

**1. Dataset Preparation:**
- Source: [Fashion Product Images Dataset on Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)
- Process documented in: `notebooks/01_data_preparation.ipynb`
- Categories: Casual Shoes, Handbags, Shirts, Tops, Watches
- 500 images per class, 70/15/15 train/val/test split
- **N.B :** - Run `notebooks/01_data_preparation.ipynb` in Kaggle notebooks not on local PC.

**2. Model Training:**
- Training notebook: Available in Google Colab or local Jupyter
- Architecture: EfficientNet-B0 (pretrained on ImageNet)
- Hyperparameters logged in: `models/v1/training_metadata.json`
- Training environment: Google Colab with T4 GPU
- Training time: ~12 minutes for 15 epochs

**3. Model Registration:**
```bash
# After training and downloading checkpoint
python scripts/register_model.py
python scripts/set_production.py
```

**4. System Deployment:**
- Follow Quick Start guide above
- All configuration is in code (no manual setup required)

### For Evaluators / New Developers

**To verify this project:**
1. Download assets from links in Asset Management section
2. Follow Quick Start instructions
3. System should be running within 30 minutes

**To retrain from scratch:**
1. Download raw dataset from Kaggle
2. Run data preparation notebook
3. Run training notebook on Colab
4. Register new model version
5. Promote to production

All steps are documented and reproducible.

---

## 👤 Author

Built as a final year project demonstrating end-to-end ML systems engineering.

**Technologies:** Python, PyTorch, FastAPI, MLflow, Prometheus, Grafana, Streamlit, Docker

---

## 📄 License

MIT License - See LICENSE file for details
