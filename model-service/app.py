from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import mlflow.pyfunc
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
model_version = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_version
    
    mlflow_tracking_uri = "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
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
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": model_version
    }

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    for path in request.image_paths:
        if not Path(path).exists():
            raise HTTPException(status_code=400, detail=f"Image not found: {path}")
    
    try:
        predictions = model.predict({"image_paths": request.image_paths})
        return PredictResponse(
            predictions=predictions,
            model_version=model_version
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))