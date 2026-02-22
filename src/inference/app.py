import mlflow.xgboost
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from prometheus_fastapi_instrumentator import Instrumentator
import logging
import time
import os

# ── Logging setup ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_URI = os.getenv("MODEL_URI", "models:/rul-predictor/2")
model = None


# ── Lifespan (replaces deprecated on_event) ───────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info(f"Loading model from {MODEL_URI}...")
    try:
        model = mlflow.xgboost.load_model(MODEL_URI)
        logger.info("Model loaded successfully from registry")
    except Exception as e:
        logger.warning(f"Registry load failed, trying local fallback: {e}")
        try:
            client = mlflow.tracking.MlflowClient()
            latest = client.get_latest_versions("rul-predictor")[0]
            local_path = "/app/mlruns/" + latest.source.split("mlruns/")[-1]
            logger.info(f"Loading from local path: {local_path}")
            model = mlflow.xgboost.load_model(local_path)
            logger.info("Model loaded successfully from local path")
        except Exception as e2:
            logger.error(f"Failed to load model: {e2}")
            raise
    yield
    # Shutdown logic can go here if needed


# ── App init ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Predictive Maintenance API",
    description="Predicts Remaining Useful Life (RUL) of jet engines from sensor readings",
    version="1.0.0",
    lifespan=lifespan
)

Instrumentator().instrument(app).expose(app)


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
    sensor_cols = [k for k in data.keys() if k.startswith("sensor_")]
    op_cols = [k for k in data.keys() if k.startswith("op_setting_")]
    row = {col: data[col] for col in sensor_cols + op_cols}
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