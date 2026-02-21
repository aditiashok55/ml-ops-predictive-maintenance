import mlflow.xgboost
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging
import time
import os

# ── Logging setup ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── App init ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Predictive Maintenance API",
    description="Predicts Remaining Useful Life (RUL) of jet engines from sensor readings",
    version="1.0.0"
)

# ── Model loading ─────────────────────────────────────────────────────
MODEL_URI = os.getenv("MODEL_URI", "models:/rul-predictor/1")
model = None

@app.on_event("startup")
async def load_model():
    global model
    logger.info(f"Loading model from {MODEL_URI}...")
    try:
        model = mlflow.xgboost.load_model(MODEL_URI)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


# ── Request / Response schemas ────────────────────────────────────────
class SensorReading(BaseModel):
    engine_id: int = Field(..., description="Engine ID")
    cycle: int = Field(..., description="Current cycle number")
    op_setting_1: float = Field(..., ge=0.0, le=1.0)
    op_setting_2: float = Field(..., ge=0.0, le=1.0)
    op_setting_3: float = Field(..., ge=0.0, le=1.0)
    sensor_2: float = Field(..., ge=0.0, le=1.0)
    sensor_3: float = Field(..., ge=0.0, le=1.0)
    sensor_4: float = Field(..., ge=0.0, le=1.0)
    sensor_7: float = Field(..., ge=0.0, le=1.0)
    sensor_8: float = Field(..., ge=0.0, le=1.0)
    sensor_9: float = Field(..., ge=0.0, le=1.0)
    sensor_11: float = Field(..., ge=0.0, le=1.0)
    sensor_12: float = Field(..., ge=0.0, le=1.0)
    sensor_13: float = Field(..., ge=0.0, le=1.0)
    sensor_14: float = Field(..., ge=0.0, le=1.0)
    sensor_15: float = Field(..., ge=0.0, le=1.0)
    sensor_17: float = Field(..., ge=0.0, le=1.0)
    sensor_20: float = Field(..., ge=0.0, le=1.0)
    sensor_21: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    engine_id: int
    cycle: int
    predicted_rul: float
    warning_level: str
    latency_ms: float


# ── Helper: build features ────────────────────────────────────────────
def build_single_features(data: dict) -> pd.DataFrame:
    """
    Build feature vector for a single reading.
    For rolling features we use the raw value (no history available at inference).
    """
    sensor_cols = [k for k in data.keys() if k.startswith("sensor_")]
    op_cols = [k for k in data.keys() if k.startswith("op_setting_")]

    row = {col: data[col] for col in sensor_cols + op_cols}

    # Rolling features default to raw value at inference time
    for col in sensor_cols:
        row[f"{col}_mean5"] = data[col]
        row[f"{col}_std5"] = 0.0

    return pd.DataFrame([row])


# ── Routes ────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()

    try:
        data = reading.model_dump()
        features = build_single_features(data)
        predicted_rul = float(model.predict(features)[0])
        predicted_rul = max(0.0, round(predicted_rul, 2))
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = round((time.time() - start) * 1000, 2)

    # Warning level logic
    if predicted_rul <= 30:
        warning_level = "CRITICAL"
    elif predicted_rul <= 70:
        warning_level = "WARNING"
    else:
        warning_level = "NORMAL"

    logger.info(
        f"engine={reading.engine_id} cycle={reading.cycle} "
        f"RUL={predicted_rul} level={warning_level} latency={latency_ms}ms"
    )

    return PredictionResponse(
        engine_id=reading.engine_id,
        cycle=reading.cycle,
        predicted_rul=predicted_rul,
        warning_level=warning_level,
        latency_ms=latency_ms
    )


@app.get("/")
def root():
    return {
        "service": "Predictive Maintenance API",
        "version": "1.0.0",
        "docs": "/docs"
    }