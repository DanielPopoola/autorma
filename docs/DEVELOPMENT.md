# Development Guide

This document covers the development process, implementation details, and lessons learned while building the Refund Item Classification System.

---

## Table of Contents

1. [Development Timeline](#development-timeline)
2. [Implementation Notes](#implementation-notes)
3. [Technology Choices](#technology-choices)
4. [Common Commands](#common-commands)
5. [Debugging Guide](#debugging-guide)
6. [Lessons Learned](#lessons-learned)
7. [Future Development](#future-development)

---

## Development Timeline

### Stage 1: Dataset Preparation (Week 1)

**Goal:** Organized, balanced dataset ready for training

**Process:**
1. Explored Fashion Product Images dataset on Kaggle
2. Identified 5 relevant categories: Shirts, Watches, Casual Shoes, Tops, Handbags
3. Used Kaggle notebook to sample 500 images per class
4. Split into train/val/test (70/15/15)
5. Downloaded as zip (~150MB)
6. Extracted to local `data/processed/` directory

**Key Files Created:**
- `data/processed/train/` - 1,750 images
- `data/processed/val/` - 375 images
- `data/processed/test/` - 375 images
- `data/processed/dataset_info.json` - Metadata

**Challenges:**
- Dataset had different category names than expected (used `articleType` instead of `masterCategory`)
- Kaggle dataset was large; needed to sample subset to save bandwidth

**Solution:**
- Explored dataset structure first with pandas
- Created sampling script in Kaggle notebook
- Only downloaded processed subset

---

### Stage 2: MLflow Setup (Week 1)

**Goal:** Local MLflow tracking server operational

**Process:**
1. Installed MLflow: `pip install mlflow`
2. Created directory structure for artifacts
3. Started server with SQLite backend
4. Accessed UI to verify setup

**Key Command:**
```bash
ABS_PATH=$(pwd)
mlflow server \
  --backend-store-uri sqlite:///$ABS_PATH/mlflow_data/mlflow.db \
  --default-artifact-root file://$ABS_PATH/mlflow_data/artifacts \
  --host 0.0.0.0 \
  --port 5000
```

**Challenges:**
- Initial command used relative paths, broke when changing directories
- Needed to use absolute paths for artifact root

**Solution:**
- Captured `$(pwd)` in variable
- Used absolute path interpolation in command

**Created Alias:**
```bash
# In ~/.bashrc
alias mlflow-start='cd ~/autorma && ABS_PATH=$(pwd) && mlflow server --backend-store-uri sqlite:///$ABS_PATH/mlflow_data/mlflow.db --default-artifact-root file://$ABS_PATH/mlflow_data/artifacts --host 0.0.0.0 --port 5000'
```

---

### Stage 3: Model Training (Week 2)

**Goal:** Trained model with >90% accuracy, logged in MLflow

**Process:**
1. Uploaded dataset to Google Drive
2. Created Colab notebook with training code
3. Used EfficientNet-B0 with transfer learning
4. Trained for 15 epochs (~12 minutes on Colab GPU)
5. Achieved 96.53% test accuracy
6. Downloaded model checkpoint and metadata
7. Registered in local MLflow

**Model Details:**
- **Architecture:** EfficientNet-B0 (pretrained on ImageNet)
- **Fine-tuning:** All layers trained (not just classifier head)
- **Optimizer:** Adam (lr=0.0001)
- **Data Augmentation:** Random flips, rotations, color jitter
- **Batch Size:** 32
- **Final Model Size:** ~20MB

**Key Code Pattern:**
```python
# Transfer learning setup
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=5)

# Train all parameters
for param in model.parameters():
    param.requires_grad = True

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2)
```

**Registration Script:**
```bash
python scripts/register_model.py
```

**Challenges:**
- Colab can't directly connect to local MLflow server
- Need to download model and register locally

**Solution:**
- Train on Colab, save checkpoint to Google Drive
- Download to local machine
- Run registration script locally

**Setting Production Alias:**
```bash
python scripts/set_production.py
```

This replaced the old "stages" approach (Production/Staging) with the new "aliases" approach.

---

### Stage 4: Model Service API (Week 2)

**Goal:** FastAPI server serving predictions from MLflow model

**Process:**
1. Created `model-service/app.py` with FastAPI
2. Added startup event to load model from MLflow
3. Implemented `/predict` endpoint
4. Added Prometheus metrics instrumentation
5. Tested with curl and Swagger UI

**Key Implementation Details:**

**Model Loading:**
```python
@app.on_event("startup")
def load_model():
    global model, model_version
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    model = mlflow.pyfunc.load_model("models:/refund-classifier/production")
    model_version = "production"
```

**Prediction Endpoint:**
```python
@app.post("/predict")
def predict(request: PredictRequest):
    predictions = model.predict({"image_paths": request.image_paths})
    return PredictResponse(predictions=predictions, model_version=model_version)
```

**Running:**
```bash
cd model-service
uvicorn app:app --host 0.0.0.0 --port 8000
```

**Testing:**
```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"image_paths": ["/absolute/path/to/image.jpg"]}'
```

**Challenges:**
- Model loaded in sync startup blocked event loop
- Image paths must be absolute, not relative

**Solution:**
- FastAPI's `@app.on_event("startup")` runs before accepting requests
- Documented absolute path requirement in API contract

---

### Stage 5: Batch Orchestrator (Week 3)

**Goal:** Automated batch processing with checkpoint recovery

**Process:**
1. Created `orchestrator/batch_inference.py`
2. Implemented checkpoint loading/saving
3. Added batch splitting logic
4. Integrated Model Service API calls
5. Added error handling for corrupted images
6. Implemented metrics pushing to Prometheus

**Key Algorithm:**
```python
1. Load checkpoint
2. Scan input directory
3. Filter already-processed images
4. Split into batches of 10
5. For each batch:
   - Call Model Service
   - Save results
   - Update checkpoint
6. Push metrics
7. Save final results JSON
```

**Checkpoint Design:**
```json
{
  "processed_images": ["img1.jpg", "img2.jpg"],
  "last_run": "2024-02-12T02:15:33Z"
}
```

**Running:**
```bash
python orchestrator/batch_inference.py
```

**Challenges:**
- Need to handle both successful and failed images
- Checkpoint must be atomic (not corrupted on crash)
- Running from different directories broke paths

**Solution:**
- Failed images logged separately, don't block batch
- Write checkpoint after each batch (not just at end)
- Used absolute paths everywhere, documented in README

**Testing Idempotency:**
```bash
# Run twice - second run should find no new images
python orchestrator/batch_inference.py
python orchestrator/batch_inference.py  # "No new images to process"
```

---

### Stage 6: Monitoring Stack (Week 4)

**Goal:** Prometheus + Grafana monitoring operational

**Process:**
1. Added Prometheus client to Model Service
2. Created metrics for requests, latency, predictions
3. Set up Prometheus + Pushgateway + Grafana via Docker
4. Configured scrape targets
5. Built Grafana dashboard with 6 panels

**Docker Compose Setup:**
```bash
cd monitoring
docker-compose up -d
```

**Prometheus Configuration:**
```yaml
scrape_configs:
  - job_name: 'model-service'
    static_configs:
      - targets: ['<WSL_IP>:8000']  # Replace with actual IP
  
  - job_name: 'pushgateway'
    static_configs:
      - targets: ['pushgateway:9091']
```

**Getting WSL IP:**
```bash
ip addr show eth0 | grep -oP '(?<=inet\s)\d+(\.\d+){3}'
```

**Grafana Dashboard Panels:**
1. Total Images Processed (Stat)
2. API Request Rate (Time series)
3. Prediction Confidence Distribution (Bar chart)
4. Predictions by Class (Pie chart)
5. Batch Success Rate (Gauge)
6. Batch Duration (Stat)

**Challenges:**
- WSL networking: Prometheus in Docker can't reach localhost
- Grafana UI changed - "stages" deprecated, now uses "aliases"
- Panel configuration not intuitive

**Solution:**
- Used WSL IP address in Prometheus targets
- Updated to use model aliases instead of stages
- Manual panel creation via UI (JSON import issues)

**Accessing Services:**
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090
- Pushgateway: http://localhost:9091

---

### Stage 7: Streamlit UI (Week 4-5)

**Goal:** User-friendly interface for manual batch processing

**Process:**
1. Created `streamlit-ui/app.py`
2. Added file uploader component
3. Integrated with orchestrator via subprocess
4. Added results visualization
5. Created history browser

**Running:**
```bash
streamlit run streamlit-ui/app.py
```

**Key Features:**
- Multi-file upload with preview
- "Run Classification" button triggers orchestrator
- Real-time progress indicator
- Results table with confidence scores
- CSV download
- Historical results browser
- Class distribution charts

**Challenges:**
- Streamlit runs in different directory than orchestrator
- Need to call orchestrator with correct paths
- Results need to be read after subprocess completes

**Solution:**
```python
# Call orchestrator from project root
subprocess.run(
    ["python", "orchestrator/batch_inference.py"],
    capture_output=True,
    cwd="/absolute/path/to/project"
)
```

**UI Tabs:**
1. **Upload & Classify:** Main workflow
2. **Results History:** Browse past runs
3. **About:** System documentation

---

## Technology Choices

### Why Python?

**Pros:**
- Best ML ecosystem (PyTorch, transformers, scikit-learn)
- Fast prototyping
- Rich data manipulation (pandas, numpy)
- Easy integration (MLflow, Streamlit, FastAPI)

**Cons:**
- Slower than Go/Rust
- GIL limits concurrency (but not critical for batch jobs)

**Verdict:** Right choice for ML system. Only consider Go/Rust for orchestration layer at massive scale.

### Why EfficientNet-B0?

**Alternatives Considered:**
- ResNet50: Larger (98MB), slower on CPU
- MobileNetV2: Faster, but lower accuracy
- ViT (Vision Transformer): Overkill, requires more data

**Why EfficientNet-B0:**
- Good accuracy/size tradeoff (20MB)
- Reasonable CPU inference speed (~100-200ms/image)
- Pretrained on ImageNet (good transfer learning)
- Well-supported by timm library

### Why FastAPI?

**Alternatives Considered:**
- Flask: Simpler, but no async, no auto-docs
- Django: Too heavy for simple API
- gRPC: Overkill, harder to debug
- Raw Uvicorn/Starlette: No validation layer

**Why FastAPI:**
- Automatic OpenAPI docs (Swagger UI)
- Type validation with Pydantic
- Async support (future-proof)
- Easy Prometheus integration
- Modern, actively maintained

### Why MLflow?

**Alternatives Considered:**
- Weights & Biases: Cloud-based, costs money
- Neptune: Similar to W&B
- Custom solution: More work, less features
- DVC: Good for data versioning, not model registry

**Why MLflow:**
- Open source, self-hosted
- Built-in model registry
- Experiment tracking
- Artifact storage
- Industry standard

### Why Prometheus + Grafana?

**Alternatives Considered:**
- CloudWatch: AWS-only
- Datadog: Expensive
- ELK Stack: Heavier, log-focused
- Custom dashboard: More work

**Why Prometheus + Grafana:**
- Industry standard
- Free and open source
- Time-series optimized
- Great visualization
- Alerting support

### Why Streamlit?

**Alternatives Considered:**
- React + FastAPI backend: More work
- Gradio: Good for demos, less flexible
- Plotly Dash: Similar, more verbose
- Jupyter notebook: Not user-friendly

**Why Streamlit:**
- Pure Python (no frontend skills needed)
- Fast iteration
- Built-in components
- Perfect for ML demos
- Easy deployment

---

## Common Commands

### Starting Services

**MLflow Server:**
```bash
ABS_PATH=$(pwd)
mlflow server \
  --backend-store-uri sqlite:///$ABS_PATH/mlflow_data/mlflow.db \
  --default-artifact-root file://$ABS_PATH/mlflow_data/artifacts \
  --host 0.0.0.0 \
  --port 5000

# Or use alias (if configured)
mlflow-start
```

**Model Service:**
```bash
cd model-service
uvicorn app:app --host 0.0.0.0 --port 8000

# With auto-reload for development
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**Monitoring Stack:**
```bash
cd monitoring
docker-compose up -d      # Start in background
docker-compose logs -f    # View logs
docker-compose down       # Stop all services
```

**Streamlit UI:**
```bash
streamlit run streamlit-ui/app.py
```

### Running Batch Jobs

**Manual Batch Processing:**
```bash
# Copy test images to input
find data/processed/test -name "*.jpg" | shuf -n 50 | xargs -I {} cp {} data/inference/input/

# Run orchestrator
python orchestrator/batch_inference.py

# View results
cat data/inference/output/predictions_*.json | jq
```

**Scheduled via Cron:**
```bash
# Edit crontab
crontab -e

# Add entry for 2 AM daily
0 2 * * * cd /home/user/autorma && source .venv/bin/activate && python orchestrator/batch_inference.py >> logs/cron.log 2>&1

# Test cron (every 5 minutes)
*/5 * * * * cd /home/user/autorma && source .venv/bin/activate && python orchestrator/batch_inference.py >> logs/cron.log 2>&1
```

### Model Management

**Register New Model:**
```bash
# After training and downloading checkpoint
python scripts/register_model.py
```

**Set Production Alias:**
```bash
python scripts/set_production.py
```

**Rollback to Previous Version:**
```python
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = mlflow.MlflowClient()

# Set version 1 as production
client.set_registered_model_alias("refund-classifier", "production", "1")
```

### Testing & Debugging

**Test Model Service:**
```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "image_paths": ["/home/user/autorma/data/processed/test/shirts/12345.jpg"]
  }'

# Batch prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "image_paths": [
      "/home/user/autorma/data/processed/test/shirts/1.jpg",
      "/home/user/autorma/data/processed/test/watches/2.jpg"
    ]
  }' | jq
```

**Check Metrics:**
```bash
# Model Service metrics
curl http://localhost:8000/metrics

# Filter specific metric
curl http://localhost:8000/metrics | grep images_processed

# Prometheus query
curl 'http://localhost:9090/api/v1/query?query=images_processed_total'
```

**View Logs:**
```bash
# Orchestrator logs
tail -f logs/orchestrator.log

# Cron logs
tail -f logs/cron.log

# Docker logs
docker logs prometheus
docker logs grafana
docker logs pushgateway
```

### Cleanup

**Clear Processed Images:**
```bash
rm -rf data/inference/input/*
rm -rf data/inference/output/*
rm -rf data/inference/checkpoints/*
```

**Reset Checkpoint:**
```bash
rm data/inference/checkpoints/checkpoint.json
```

**Stop All Services:**
```bash
# Kill MLflow
pkill -f "mlflow server"

# Kill Model Service
pkill -f "uvicorn app:app"

# Kill Streamlit
pkill -f "streamlit run"

# Stop monitoring
cd monitoring && docker-compose down
```

---

## Debugging Guide

### Model Service Won't Start

**Symptom:** `model_loaded` metric shows 0

**Debug Steps:**
1. Check MLflow is running: `curl http://localhost:5000/health`
2. Check model is registered: Visit MLflow UI
3. Check production alias: MLflow UI → Models → refund-classifier → Aliases
4. Check logs: Look at uvicorn output for errors

**Common Issues:**
- MLflow not running → Start it
- No model registered → Run `python scripts/register_model.py`
- Production alias not set → Run `python scripts/set_production.py`
- Wrong tracking URI → Check `mlflow.set_tracking_uri()` in app.py

### Prometheus Can't Scrape Targets

**Symptom:** Targets show DOWN in Prometheus UI

**Debug Steps:**
1. Check Model Service is running: `curl http://localhost:8000/health`
2. Check WSL IP hasn't changed: `ip addr show eth0`
3. Check prometheus.yml has correct IP
4. Check metrics endpoint works: `curl http://localhost:8000/metrics`
5. Check Docker networking: `docker network ls`

**Common Issues (WSL):**
- `host.docker.internal` doesn't work → Use actual WSL IP
- IP changed after reboot → Update prometheus.yml
- Firewall blocking → Check Windows Defender

### Batch Processing Fails

**Symptom:** Orchestrator exits with error

**Debug Steps:**
1. Check Model Service health: `curl http://localhost:8000/health`
2. Check image paths are absolute
3. Check images exist: `ls data/inference/input/`
4. Check logs: `cat logs/orchestrator.log`
5. Try single image prediction via curl

**Common Issues:**
- Relative paths used → Use absolute paths
- Model Service down → Start it
- Corrupted images → Check failed_images in results JSON
- Permissions issue → Check file ownership

### Streamlit UI Doesn't Show Results

**Symptom:** Classification completes but no results shown

**Debug Steps:**
1. Check orchestrator actually ran: `ls data/inference/output/`
2. Check results JSON is valid: `cat data/inference/output/*.json | jq`
3. Check Streamlit logs in terminal
4. Verify subprocess exit code in Streamlit

**Common Issues:**
- Orchestrator called with wrong working directory
- Results saved to wrong location
- JSON parsing error → Check file format

### Grafana Shows No Data

**Symptom:** Dashboard panels are empty

**Debug Steps:**
1. Check Prometheus is scraping: http://localhost:9090/targets
2. Check metrics exist: Query in Prometheus UI
3. Check time range in Grafana (top-right)
4. Check data source is configured correctly
5. Check queries in panel editor

**Common Issues:**
- No traffic generated → Run batch job first
- Time range too narrow → Expand to "Last 6 hours"
- Wrong data source → Select Prometheus
- Query syntax error → Test in Prometheus first

---

## Lessons Learned

### 1. Start Simple, Add Complexity Later

**Initial Plan:** Complex multi-worker queue system

**Reality:** Single orchestrator script was sufficient

**Lesson:** Build for current requirements. The simple solution handles 10x current load. Optimize when needed, not before.

### 2. Absolute Paths Matter

**Problem:** Orchestrator worked in some directories, failed in others

**Solution:** Use absolute paths everywhere, document in README

**Lesson:** Relative paths cause 80% of "works on my machine" issues. Absolute paths or proper config management.

### 3. Checkpointing is Essential

**Problem:** Early version re-processed all images on crash

**Solution:** Save checkpoint after each batch

**Lesson:** For batch jobs, checkpoint state frequently. Recovery is critical.

### 4. WSL Networking is Tricky

**Problem:** Prometheus in Docker couldn't reach localhost services

**Solution:** Use WSL IP address instead of localhost

**Lesson:** WSL has its own network stack. localhost != host.docker.internal.

### 5. Documentation Saves Time

**Problem:** Spent time re-figuring out commands weeks later

**Solution:** Created DEVELOPMENT.md with all commands

**Lesson:** Document as you build. Future-you will thank present-you.

### 6. Monitoring is Not Optional

**Problem:** Hard to debug performance issues without data

**Solution:** Added Prometheus metrics from day 1

**Lesson:** Instrumentation should be part of initial implementation, not an afterthought.

### 7. UI Makes Demos 10x Better

**Problem:** Showing curl commands in defense would be boring

**Solution:** Built Streamlit UI in 1 day

**Lesson:** For demos/presentations, invest in UI. It's worth it.

### 8. MLflow Stages → Aliases

**Problem:** Followed old tutorials, stages were deprecated

**Solution:** Switched to aliases pattern

**Lesson:** Always check official docs, not just tutorials. ML tooling evolves fast.

### 9. Test Early, Test Often

**Problem:** Found bugs in orchestrator during final testing

**Solution:** Should have tested each component immediately after building

**Lesson:** Integration testing earlier would have caught issues sooner.

### 10. Keep It Production-Like

**Problem:** Some projects build toy systems that don't resemble production

**Solution:** Used real production patterns (monitoring, logging, versioning)

**Lesson:** Learn production practices even in academic projects. It's the differentiator.

---

## Future Development

### Immediate Next Steps

1. **Containerize Model Service**
   ```dockerfile
   FROM python:3.10-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY app.py .
   CMD ["uvicorn", "app:app", "--host", "0.0.0.0"]
   ```

2. **Add Unit Tests**
   ```python
   # tests/test_orchestrator.py
   def test_checkpoint_loading():
       checkpoint = load_checkpoint()
       assert "processed_images" in checkpoint
   
   def test_batch_splitting():
       images = [f"img_{i}.jpg" for i in range(25)]
       batches = split_into_batches(images, batch_size=10)
       assert len(batches) == 3
   ```

3. **Add CI/CD**
   ```yaml
   # .github/workflows/test.yml
   name: Test
   on: [push]
   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v2
         - run: pip install -r requirements.txt
         - run: pytest tests/
   ```

### Short-Term Enhancements

1. **Data Drift Detection**
   - Monitor input image statistics
   - Alert if distribution shifts
   - Automatic retraining trigger

2. **Model A/B Testing**
   - Run v1 and v2 in parallel
   - Route 50/50 traffic
   - Compare metrics

3. **Better Error Handling**
   - Retry logic for API failures
   - Circuit breaker pattern
   - Dead letter queue

4. **Performance Optimization**
   - Model quantization (INT8)
   - ONNX Runtime for faster CPU inference
   - Dynamic batching

### Long-Term Vision

1. **Cloud Deployment**
   - AWS Lambda + SageMaker
   - Auto-scaling based on load
   - S3 for image storage

2. **Active Learning Loop**
   - Human reviews low-confidence predictions
   - Labels fed back into training
   - Continuous improvement

3. **Multi-Model Ensemble**
   - EfficientNet + ResNet + ViT
   - Ensemble predictions
   - Confidence calibration

4. **Real-Time Streaming**
   - Kafka for image events
   - Stream processing
   - Hybrid batch + real-time

---

## Development Best Practices

### Code Organization

```
autorma/
├── model-service/       # Each component is self-contained
│   ├── app.py
│   └── requirements.txt
├── orchestrator/
│   ├── batch_inference.py
│   ├── metrics_pusher.py
│   └── requirements.txt
└── streamlit-ui/
    ├── app.py
    └── requirements.txt
```

**Principle:** Each component has its own dependencies. No shared code that creates tight coupling.

### Configuration Management

**Current:** Hardcoded values in files

**Better:**
```python
# config.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    model_service_url: str = "http://localhost:8000"
    mlflow_tracking_uri: str = "http://localhost:5000"
    batch_size: int = 10
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### Logging Standards

**Current:** Mix of print() and logging

**Better:**
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Processing batch 1")
```

### Error Handling

**Current:** Try/except with basic logging

**Better:**
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def call_model_service(images):
    response = requests.post(...)
    response.raise_for_status()
    return response.json()
```

---

## Contribution Guidelines

### Code Style

- Use Black for formatting: `black .`
- Use isort for imports: `isort .`
- Use mypy for type checking: `mypy .`
- Follow PEP 8

### Git Workflow

```bash
# Feature branch
git checkout -b feature/add-data-drift-detection

# Commit with descriptive message
git commit -m "feat: Add data drift detection using KS test"

# Push and create PR
git push origin feature/add-data-drift-detection
```

### Commit Message Format

```
type(scope): description

[optional body]

[optional footer]
```

**Types:** feat, fix, docs, style, refactor, test, chore

**Examples:**
- `feat(orchestrator): Add retry logic for failed batches`
- `fix(model-service): Correct confidence score calculation`
- `docs(readme): Update setup instructions for WSL`

---

## Conclusion

This project demonstrates end-to-end ML systems engineering. The key takeaway: **production ML is 10% model training, 90% everything else** (serving, monitoring, orchestration, error handling).

The architecture prioritizes simplicity and maintainability. Each component does one thing well. The system is designed to scale when needed, but not prematurely optimized.

**Next Developer:** Read ARCHITECTURE.md first, then this file. Run through the Quick Start in README.md. You'll have the system running in 30 minutes.
