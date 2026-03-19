import io
import time
import logging
from contextlib import asynccontextmanager

import torch
import timm
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── model state ────────────────────────────────────────────────────────────────
model = None
idx_to_class = None

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── prometheus metrics ─────────────────────────────────────────────────────────
request_count    = Counter("api_requests_total", "Total API requests", ["endpoint", "status"])
request_duration = Histogram("api_request_duration_seconds", "Request duration", ["endpoint"])
pred_confidence  = Histogram("prediction_confidence", "Confidence scores", buckets=[.5,.6,.7,.8,.9,.95,1.0])
pred_by_class    = Counter("predictions_by_class_total", "Predictions by class", ["class_name"])
model_loaded_g   = Gauge("model_loaded", "Whether model is loaded")
images_processed = Counter("images_processed_total", "Total images processed")


def _predict(img: Image.Image) -> dict:
    tensor = TRANSFORM(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    idx = torch.argmax(probs).item()
    return {
        "predicted_class": idx_to_class[idx],
        "confidence": float(probs[idx]),
        "all_probabilities": {idx_to_class[i]: float(probs[i]) for i in range(len(probs))},
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, idx_to_class
    checkpoint = torch.load("best_model.pth", map_location="cpu", weights_only=False)
    idx_to_class = {v: k for k, v in checkpoint["class_to_idx"].items()}
    m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(idx_to_class))
    m.load_state_dict(checkpoint["model_state_dict"])
    m.eval()
    model = m
    model_loaded_g.set(1)
    logger.info(f"✓ Model loaded | classes: {list(idx_to_class.values())}")
    yield


app = FastAPI(title="Refund Classifier API", lifespan=lifespan)


@app.get("/health")
def health():
    request_count.labels(endpoint="/health", status="success").inc()
    return {"status": "healthy", "model_loaded": model is not None, "model_version": "v1"}


@app.post("/predict")
async def predict(files: list[UploadFile] = File(...)):
    if model is None:
        raise HTTPException(503, "Model not loaded")

    start = time.time()
    predictions = []
    try:
        for file in files:
            img = Image.open(io.BytesIO(await file.read())).convert("RGB")
            pred = _predict(img)
            pred["image_name"] = file.filename
            predictions.append(pred)
            pred_confidence.observe(pred["confidence"])
            pred_by_class.labels(class_name=pred["predicted_class"]).inc()
            images_processed.inc()

        request_duration.labels(endpoint="/predict").observe(time.time() - start)
        request_count.labels(endpoint="/predict", status="success").inc()
        return {"predictions": predictions, "model_version": "v1"}

    except Exception as e:
        request_count.labels(endpoint="/predict", status="error").inc()
        raise HTTPException(500, str(e))


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)