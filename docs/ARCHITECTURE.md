# System Architecture

This document provides an in-depth look at the architecture of the Refund Item Classification System, explaining design decisions, component responsibilities, and data flows.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Component Details](#component-details)
3. [Data Flow](#data-flow)
4. [Design Decisions](#design-decisions)
5. [Scalability Considerations](#scalability-considerations)

---

## Architecture Overview

The system follows a **microservices-inspired architecture** with clear separation of concerns:

```
┌──────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                         │
│  ┌──────────────────────┐       ┌──────────────────────────┐    │
│  │   Streamlit UI       │       │   Grafana Dashboard      │    │
│  │   Port: 8501         │       │   Port: 3000             │    │
│  │   - Upload images    │       │   - View metrics         │    │
│  │   - Trigger batches  │       │   - Monitor health       │    │
│  │   - View results     │       │   - Analyze trends       │    │
│  └──────────┬───────────┘       └────────────▲─────────────┘    │
└─────────────┼──────────────────────────────────┼──────────────────┘
              │                                  │
              │ Calls orchestrator               │ Queries metrics
              ▼                                  │
┌──────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                            │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              Batch Orchestrator (Python)                   │  │
│  │              orchestrator/batch_inference.py               │  │
│  │                                                            │  │
│  │  Responsibilities:                                         │  │
│  │  • Scan input directory for unprocessed images            │  │
│  │  • Load/save checkpoints for crash recovery               │  │
│  │  • Split images into mini-batches (10 per batch)          │  │
│  │  • Call Model Service API for predictions                 │  │
│  │  • Handle failed images (log for human review)            │  │
│  │  • Save results to output directory                       │  │
│  │  • Push metrics to Prometheus Pushgateway                 │  │
│  │                                                            │  │
│  └──────────┬───────────────────────────────────┬─────────────┘  │
│             │                                   │                │
│             │ HTTP POST /predict                │ Push metrics   │
│             ▼                                   ▼                │
│  ┌─────────────────────────┐      ┌─────────────────────────┐   │
│  │   Model Service API     │      │    Pushgateway          │   │
│  │   Port: 8000            │      │    Port: 9091           │   │
│  │   FastAPI + PyTorch     │      │    Metrics sink         │   │
│  └────────┬────────────────┘      └──────────▲──────────────┘   │
└───────────┼────────────────────────────────────┼──────────────────┘
            │                                    │
            │ Load model                         │ Scrape
            ▼                                    │
┌──────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                │
│                                                                   │
│  ┌──────────────────────┐        ┌──────────────────────────┐   │
│  │   MLflow Registry    │        │   Prometheus TSDB        │   │
│  │   Port: 5000         │        │   Port: 9090             │   │
│  │                      │        │                          │   │
│  │  • Model versions    │        │  • Time-series metrics   │   │
│  │  • Experiments       │        │  • Scrape configs        │   │
│  │  • Artifacts         │        │  • Retention policies    │   │
│  │  • Metadata          │        │                          │   │
│  └──────────────────────┘        └──────────────────────────┘   │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Filesystem Storage                           │   │
│  │                                                           │   │
│  │  data/inference/                                          │   │
│  │    ├── input/       ← New images arrive here             │   │
│  │    ├── output/      ← Prediction results saved here      │   │
│  │    └── checkpoints/ ← Recovery state stored here         │   │
│  └───────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                      SCHEDULING LAYER                             │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │   Cron (2 AM daily)                                        │  │
│  │   └─> python orchestrator/batch_inference.py              │  │
│  └────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Model Service (FastAPI)

**Location:** `model-service/app.py`

**Purpose:** Stateless HTTP API for model inference

**Key Features:**
- Loads production model from MLflow Registry on startup
- Exposes `/predict` endpoint for batch predictions
- Exposes `/metrics` endpoint for Prometheus scraping
- Exposes `/health` endpoint for status checks

**API Contract:**

```python
# Request
POST /predict
{
  "image_paths": ["/absolute/path/to/img1.jpg", "/absolute/path/to/img2.jpg"]
}

# Response
{
  "predictions": [
    {
      "predicted_class": "shirts",
      "confidence": 0.98,
      "all_probabilities": {
        "shirts": 0.98,
        "tops": 0.01,
        "casual shoes": 0.005,
        "handbags": 0.003,
        "watches": 0.002
      }
    }
  ],
  "model_version": "production"
}
```

**Metrics Exposed:**
- `api_requests_total{endpoint, status}` - Counter of requests
- `api_request_duration_seconds{endpoint}` - Histogram of latencies
- `prediction_confidence` - Histogram of confidence scores
- `predictions_by_class_total{class_name}` - Counter per class
- `model_loaded` - Gauge (0 or 1)
- `images_processed_total` - Counter of total images

**Lifecycle:**
1. Startup: Load model from MLflow
2. Runtime: Serve predictions, expose metrics
3. Shutdown: Clean up resources (handled by FastAPI)

**Why FastAPI?**
- Async support (though not critical for batch use case)
- Automatic OpenAPI docs
- Type validation with Pydantic
- Easy Prometheus integration
- Production-ready ASGI server (Uvicorn)

---

### 2. Batch Orchestrator

**Location:** `orchestrator/batch_inference.py`

**Purpose:** Coordinate batch processing workflow

**Algorithm:**

```python
1. Load checkpoint (what's been processed)
2. Scan input directory for new images
3. Filter out already-processed images
4. Split into mini-batches of 10
5. For each batch:
   a. Call Model Service /predict endpoint
   b. Save predictions
   c. Update checkpoint
   d. Handle errors (log failed images)
6. Push metrics to Prometheus Pushgateway
7. Save final results JSON
```

**Checkpoint Structure:**

```json
{
  "processed_images": ["img1.jpg", "img2.jpg", ...],
  "last_run": "2024-02-12T02:15:33Z"
}
```

**Error Handling:**
- Corrupted images → Logged to `failed_images` array
- Model Service down → Entire batch fails, checkpoint not updated
- Partial batch failure → Successful images saved, failed ones logged

**Why Checkpointing?**
- **Crash Recovery:** If orchestrator crashes mid-run, can resume
- **Idempotency:** Running twice doesn't duplicate work
- **Debugging:** Can see exactly what's been processed

**Why Mini-Batches of 10?**
- GPU efficiency (batch inference faster than single)
- Checkpoint granularity (don't lose too much progress on crash)
- Manageable error scope (corruption affects ≤10 images)

---

### 3. MLflow Registry

**Location:** `mlflow_data/`

**Purpose:** Model versioning and experiment tracking

**Key Concepts:**

**Model Versions:**
```
refund-classifier
├── Version 1 (alias: production)
├── Version 2 (alias: staging)
└── Version 3 (no alias)
```

**Aliases vs Stages:**
- Old MLflow: Used "stages" (Production, Staging, Archived)
- New MLflow: Uses "aliases" (more flexible, custom names)
- This project uses aliases: `production`, `staging`

**Model Loading:**
```python
# Model Service loads via alias
model_uri = "models:/refund-classifier/production"
model = mlflow.pyfunc.load_model(model_uri)
```

**Promotion Workflow:**
1. Train new model (v3)
2. Register in MLflow
3. Set `staging` alias for testing
4. If good, set `production` alias
5. Model Service restart picks up new version

**Rollback:**
```python
# Point production alias back to v1
client.set_registered_model_alias("refund-classifier", "production", "1")
```

---

### 4. Monitoring Stack

**Components:**
- **Prometheus:** Time-series database, scrapes metrics
- **Pushgateway:** Accepts pushed metrics from batch jobs
- **Grafana:** Visualization and dashboards

**Why Prometheus + Grafana?**
- **Industry Standard:** Most companies use this stack
- **Pull Model:** Prometheus scrapes targets (Model Service)
- **Push Model:** Batch jobs push to Pushgateway
- **Flexible Queries:** PromQL for complex aggregations
- **Alerting Ready:** Can add alerts on thresholds

**Scrape Configs:**

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'model-service'
    static_configs:
      - targets: ['<WSL_IP>:8000']  # Scrapes /metrics endpoint
    scrape_interval: 15s
  
  - job_name: 'pushgateway'
    static_configs:
      - targets: ['pushgateway:9091']  # Scrapes pushed metrics
    scrape_interval: 15s
```

**Key Metrics Queries:**

```promql
# Request rate per endpoint
rate(api_requests_total[5m])

# 95th percentile latency
histogram_quantile(0.95, rate(api_request_duration_seconds_bucket[5m]))

# Batch success rate
batch_success_rate

# Predictions distribution
predictions_by_class_total
```

---

### 5. Streamlit UI

**Location:** `streamlit-ui/app.py`

**Purpose:** User-friendly interface for manual batch processing

**Pages:**

**Tab 1: Upload & Classify**
- Multi-file uploader
- Image preview grid
- "Run Classification" button (calls orchestrator as subprocess)
- Results table with confidence scores
- CSV download

**Tab 2: Results History**
- Dropdown to select past batch runs
- Metrics summary (total, success, failed, duration)
- Class distribution bar chart
- Detailed results table

**Tab 3: About**
- System architecture explanation
- Model performance stats
- Workflow description

**Why Streamlit?**
- Pure Python (no frontend code)
- Perfect for ML demos
- Built-in components (file upload, charts, tables)
- Fast iteration

**Integration:**
- Saves uploaded files to `data/inference/input/`
- Calls orchestrator via `subprocess.run()`
- Reads results from `data/inference/output/`
- Shows system status via Model Service `/health` endpoint

---

## Data Flow

### Nightly Batch Processing Flow

```
2:00 AM - Cron triggers
     │
     ├─> Orchestrator starts
     │
     ├─> Load checkpoint.json
     │   └─> Already processed: 500 images
     │
     ├─> Scan data/inference/input/
     │   └─> Found: 50 new images
     │
     ├─> Filter: 50 - 0 = 50 to process
     │
     ├─> Split into batches: [10, 10, 10, 10, 10]
     │
     ├─> Batch 1 (images 1-10)
     │   ├─> POST /predict to Model Service
     │   ├─> Model Service:
     │   │   ├─> Load images
     │   │   ├─> Preprocess (resize, normalize)
     │   │   ├─> Forward pass through EfficientNet-B0
     │   │   ├─> Softmax → probabilities
     │   │   └─> Return predictions
     │   ├─> Orchestrator receives predictions
     │   ├─> Save to results array
     │   └─> Update checkpoint (now 510 processed)
     │
     ├─> Batch 2 (images 11-20)
     │   └─> ... repeat ...
     │
     ├─> ... Batches 3-5 ...
     │
     ├─> All batches complete
     │
     ├─> Push metrics to Pushgateway
     │   ├─> batch_duration_seconds: 5.2
     │   ├─> batch_images_processed: 50
     │   ├─> batch_success_rate: 1.0
     │   └─> predictions_class_shirts: 12
     │
     ├─> Save results to output/predictions_20240212_020533.json
     │
     └─> Orchestrator exits (exit code 0)

2:05 AM - Batch complete
```

### Manual Processing Flow (via UI)

```
User opens Streamlit UI
     │
     ├─> Upload 20 images via file uploader
     │
     ├─> Click "Run Classification"
     │
     ├─> Streamlit:
     │   ├─> Save uploaded files to data/inference/input/
     │   └─> subprocess.run("python orchestrator/batch_inference.py")
     │
     ├─> Orchestrator runs (same flow as nightly)
     │
     ├─> Results saved to output/predictions_*.json
     │
     ├─> Streamlit:
     │   ├─> Read latest results file
     │   ├─> Display results table
     │   └─> Offer CSV download
     │
     └─> User downloads results
```

---

## Design Decisions

### Why Batch Processing Instead of Real-Time?

**Business Context:**
- Returns arrive throughout the day
- Classification results needed by next morning
- No real-time decision required

**Benefits:**
- **Cost:** Process during off-peak hours
- **Efficiency:** Batch GPU inference is faster
- **Simplicity:** No streaming infrastructure needed
- **Resource Optimization:** Can use cheaper CPU for orchestrator, GPU only for model

**Trade-offs:**
- Latency: Results not immediate (acceptable for this use case)
- Complexity: Need checkpoint mechanism

### Why Separate Model Service?

**Alternative:** Monolithic script that loads model and processes images

**Why Separate:**

| Aspect | Monolithic | Separate Service |
|--------|------------|------------------|
| **Testing** | Must run full pipeline | Can test model in isolation |
| **Deployment** | Change orchestrator → reload model | Update independently |
| **Technology** | Locked into Python | Could rewrite orchestrator in Go |
| **Scaling** | Scale everything together | Scale model service separately |
| **Development** | Tight coupling | Clear API contract |

**When Monolithic Might Be Better:**
- Very small scale (< 100 images/day)
- No plans to evolve system
- Single developer, no team

### Why MLflow?

**Alternatives:**
- Manual versioning (save models with timestamps)
- Git LFS for model files
- Custom model registry

**Why MLflow:**
- **Experiment Tracking:** Log hyperparameters, metrics
- **Reproducibility:** Can recreate any experiment
- **Versioning:** Built-in model registry
- **Promotion Workflow:** Staging → Production
- **Easy Rollback:** Change alias, restart service
- **Team Collaboration:** Shared experiment database

**MLflow Overhead:**
- Running server (always-on process)
- Learning curve for team
- SQLite bottleneck (at scale, use PostgreSQL)

### Why Filesystem Storage?

**Alternatives:**
- S3/Object storage
- Database (PostgreSQL)
- Message queue (Kafka)

**Why Filesystem:**
- **Simplicity:** No external dependencies
- **Debugging:** Easy to inspect files
- **Cost:** No cloud costs
- **Performance:** Local disk is fast

**When to Switch:**
- **S3:** When deploying to cloud
- **Database:** When need transactional guarantees
- **Message Queue:** When moving to streaming

---

## Scalability Considerations

### Current System Capacity

**Constraints:**
- Single Model Service instance (CPU-only)
- Sequential batch processing
- Local filesystem storage

**Estimated Throughput:**
- ~10 images per batch
- ~1 second per batch (CPU inference)
- ~3600 batches/hour
- **~36,000 images/hour**

For nightly processing (6-hour window):
- **~216,000 images/night**

### Scaling Strategies

#### 1. Vertical Scaling (Upgrade Hardware)

**Current:** 8GB RAM, Core i5 CPU

**Upgrade:**
- 16GB RAM → Larger batch sizes (20-30 images)
- GPU (even low-end) → 10-100x faster inference
- SSD → Faster image loading

**Expected Improvement:** 5-10x throughput

#### 2. Horizontal Scaling (Add More Instances)

**Strategy:**
- Run multiple Model Service instances
- Load balancer in front (or round-robin in orchestrator)
- Partition input directory (orchestrator-1 processes `/input/part1/`, orchestrator-2 processes `/input/part2/`)

**Changes Needed:**
```python
# orchestrator/batch_inference.py
MODEL_SERVICES = [
    "http://model-service-1:8000",
    "http://model-service-2:8000",
    "http://model-service-3:8000"
]

# Round-robin across services
service_url = MODEL_SERVICES[batch_num % len(MODEL_SERVICES)]
```

**Expected Improvement:** Linear with number of instances (3 instances = 3x throughput)

#### 3. Distributed Processing (Message Queue)

**Current:** Orchestrator directly calls Model Service

**With Queue:**
```
Orchestrator → RabbitMQ/Redis Queue → Worker Pool → Model Service
```

**Benefits:**
- Decouple submission from processing
- Workers pull tasks at their own pace
- Better fault tolerance (queue persists tasks)
- Dynamic scaling (add/remove workers)

**Tools:**
- Celery + Redis
- RabbitMQ
- AWS SQS

#### 4. Cloud Deployment

**Current:** Runs on single machine (WSL)

**Cloud Architecture:**
```
S3 (images) → Lambda (orchestrator) → SageMaker (model) → DynamoDB (results)
```

**Benefits:**
- Auto-scaling
- Pay per use
- Managed services (no ops)

**Costs:**
- SageMaker inference: ~$0.05/hour (CPU)
- Lambda: ~$0.20/million requests
- S3: ~$0.023/GB/month

### When to Scale?

**Current System Handles:**
- 50-500 images/night comfortably
- <5 second batch processing time

**Scale When:**
- Processing time > batch window (can't finish overnight)
- Manual intervention needed for failures
- Metrics show degraded performance
- Business growth demands it

**Golden Rule:** Don't scale prematurely. Current architecture can handle 10x growth before changes needed.

---

## Security Considerations

### Current Security Posture

**Implemented:**
- Model Service only accepts local filesystem paths (no arbitrary URLs)
- Checkpoints prevent duplicate processing
- Failed images logged (no silent failures)

**Not Implemented (Future):**
- Authentication on Model Service API
- Encryption of model files
- Input validation (malicious images)
- Rate limiting
- HTTPS/TLS

### Production Security Checklist

- [ ] Add API key authentication to Model Service
- [ ] Validate image file signatures (prevent malicious uploads)
- [ ] Run Model Service as non-root user
- [ ] Enable HTTPS with valid certificates
- [ ] Implement rate limiting (prevent DoS)
- [ ] Encrypt MLflow database
- [ ] Add audit logging (who accessed what)
- [ ] Scan uploaded images for malware
- [ ] Implement CORS policies
- [ ] Use secrets management (not hardcoded credentials)

---

## Disaster Recovery

### Failure Scenarios & Mitigation

| Failure | Impact | Recovery |
|---------|--------|----------|
| **Model Service crashes** | Batch run fails | Checkpoint allows resume; restart service |
| **Orchestrator crashes mid-batch** | Partial processing | Next run loads checkpoint, continues |
| **Corrupted image in batch** | Batch fails | Error logged, remaining images processed |
| **MLflow server down** | Model Service can't start | Keep last-known-good model cached |
| **Disk full** | Can't save results | Monitor disk space, auto-cleanup old results |
| **Prometheus down** | No metrics | System continues, just no monitoring |

### Backup Strategy

**Critical Data:**
- MLflow database (`mlflow_data/mlflow.db`)
- Model checkpoints (`models/v*/`)
- Training metadata
- Configuration files

**Backup Plan:**
- Daily backup of MLflow DB to S3/external storage
- Model files versioned in Git LFS (or artifact store)
- Configuration in version control
- Results archived after 30 days

---

## Monitoring & Alerting

### Key Metrics to Alert On

| Metric | Threshold | Action |
|--------|-----------|--------|
| `batch_success_rate` | < 0.95 | Investigate failed images |
| `batch_duration_seconds` | > 600 | Check Model Service health |
| `model_loaded` | == 0 | Restart Model Service |
| `api_request_duration_seconds` (p95) | > 5s | Check resource usage |
| Disk space | < 10% free | Clean old results |

### Grafana Alerting (Future)

```yaml
# Example alert rule
- alert: BatchSuccessRateLow
  expr: batch_success_rate < 0.95
  for: 5m
  annotations:
    summary: "Batch processing success rate dropped below 95%"
    description: "Last batch had {{ $value }}% success rate"
```

---

## Future Enhancements

### Short-Term (Next 3-6 Months)

1. **Dockerize All Services**
   - Model Service → Docker image
   - Orchestrator → Scheduled container
   - Single `docker-compose up` to start everything

2. **Add Data Drift Detection**
   - Monitor input image statistics
   - Alert if distribution shifts significantly
   - Automatic model retraining trigger

3. **Improve Error Handling**
   - Retry logic for transient failures
   - Circuit breaker for Model Service
   - Dead letter queue for permanently failed images

4. **Performance Optimization**
   - Model quantization (INT8) for faster CPU inference
   - Dynamic batching based on input size
   - Caching for repeated images

### Long-Term (6-12 Months)

1. **Cloud Migration**
   - AWS: S3 + Lambda + SageMaker
   - GCP: Cloud Storage + Cloud Run + Vertex AI
   - Azure: Blob Storage + Functions + ML Studio

2. **Active Learning**
   - Flag low-confidence predictions for human review
   - Use human labels to retrain model
   - Continuous improvement loop

3. **Multi-Model Support**
   - Run multiple specialized models
   - Ensemble predictions
   - A/B testing of model versions

4. **Real-Time Processing**
   - Add streaming endpoint for urgent returns
   - Keep batch processing for bulk
   - Hybrid architecture

---

## Conclusion

This architecture prioritizes **simplicity**, **maintainability**, and **production-readiness** over premature optimization. Each component has a clear responsibility, components communicate via well-defined interfaces, and the system is designed to scale when needed.

The key insight: **Build for current requirements, design for future growth.** The clean boundaries (orchestrator ↔ model service, MLflow registry, monitoring) make it easy to swap components or scale pieces independently when business needs change.