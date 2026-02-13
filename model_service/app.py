from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import mlflow.pyfunc
from pathlib import Path
import logging
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from fastapi.responses import Response
import time

from config import get_settings


logger = logging.getLogger(__name__)

model = None
model_version = None

settings = get_settings()

# Prometheus metrics
request_count = Counter(
    "api_requests_total", "Total API requests", ["endpoint", "status"]
)
request_duration = Histogram(
    "api_request_duration_seconds", "Request duration", ["endpoint"]
)
prediction_confidence = Histogram(
    "prediction_confidence",
    "Prediction confidence scores",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
)
predictions_by_class = Counter(
    "predictions_by_class_total", "Predictions by class", ["class_name"]
)
model_loaded = Gauge("model_loaded", "Whether model is loaded")
images_processed = Counter("images_processed_total", "Total images processed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_version

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    try:
        model_uri = "models:/refund-classifier@production"
        logger.info(f"Loading model from: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        model_version = "production"
        logger.info(f"✓ Model loaded successfully: {model_version}")
    except Exception as e:
        logger.warning(f"Production model not found, loading latest: {e}")
        model_uri = "models:/refund-classifier@latest"
        model = mlflow.pyfunc.load_model(model_uri)
        model_version = "latest"
        logger.info(f"✓ Model loaded successfully: {model_version}")

    yield

    logger.info("Shutting down...")


app = FastAPI(title="Refund Classifier API", lifespan=lifespan)


class PredictRequest(BaseModel):
    image_paths: list[str]


class PredictResponse(BaseModel):
    predictions: list[dict]
    model_version: str


@app.get("/health")
def health():
    request_count.labels(endpoint="/health", status="success").inc()
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": model_version,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    start_time = time.time()

    if model is None:
        request_count.labels(endpoint="/predict", status="error").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    for path in request.image_paths:
        if not Path(path).exists():
            request_count.labels(endpoint="/predict", status="error").inc()
            raise HTTPException(status_code=400, detail=f"Image not found: {path}")

    try:
        predictions = model.predict({"image_paths": request.image_paths})

        # Track metrics
        for pred in predictions:
            prediction_confidence.observe(pred["confidence"])
            predictions_by_class.labels(class_name=pred["predicted_class"]).inc()
            images_processed.inc()

        duration = time.time() - start_time
        request_duration.labels(endpoint="/predict").observe(duration)
        request_count.labels(endpoint="/predict", status="success").inc()

        return PredictResponse(predictions=predictions, model_version=model_version)
    except Exception as e:
        request_count.labels(endpoint="/predict", status="error").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
