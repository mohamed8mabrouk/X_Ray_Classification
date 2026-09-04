import io
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from starlette.concurrency import run_in_threadpool
from PIL import Image

from .model import ImageClassifier
from .schema import PredictionResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ml_models = {}
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}
MODEL_PATH = Path(__file__).resolve().parents[2] / "model" / "bone_fracture_model.keras"

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading TensorFlow model")
    ml_models["classifier"] = ImageClassifier(
        str(MODEL_PATH)
    )
    logger.info("TensorFlow model is ready")
    yield
    ml_models.clear()

app = FastAPI(
    title="TensorFlow Image Prediction API",
    lifespan=lifespan,
)

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded="classifier" in ml_models,
    )

@app.get("/health/ready")
def readiness():
    if "classifier" not in ml_models:
        raise HTTPException(503, "AI Model Is Currently Unavailable")
    return {"status": "ready", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, "Unsupported File Type")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Failed To Read Image")

    start = time.perf_counter()
    classifier = ml_models["classifier"]
    result = await run_in_threadpool(classifier.predict, image)
    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info("Prediction: %s in %.1fms", result["label"], elapsed_ms)

    label = result["label"]

    if label == 1:
     label = "Fractured"
    else:
     label = "Not Fractured"

    return PredictionResponse(
    label=label,
    processing_time_ms=round(elapsed_ms, 2)
)