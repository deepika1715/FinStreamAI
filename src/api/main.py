# Copyright (c) 2025 Deepika Sharma. Licensed under Apache 2.0.

import pickle
import numpy as np
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from src.api.schemas import TransactionRequest, PredictionResponse, HealthResponse
from src.monitoring.prediction_logger import init_db, log_prediction
from src.monitoring.model_watcher import ModelWatcher

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Shared app state -- passed to ModelWatcher for zero-downtime swaps ────────
app_state = {
    'model': None,
    'feature_names': [],
    'model_loaded': False
}

# ── Load model on startup ─────────────────────────────────────────────────────
MODEL_PATH    = "models/fraud_model.pkl"
FEATURES_PATH = "models/feature_names.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        app_state['model'] = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f:
        app_state['feature_names'] = pickle.load(f)
    app_state['model_loaded'] = True
    logger.info(
        f"Model loaded successfully — "
        f"{len(app_state['feature_names'])} features"
    )
except Exception as e:
    logger.error(f"Failed to load model: {e}")

# ── Convenience references ────────────────────────────────────────────────────
def get_model():
    return app_state['model']

def get_feature_names():
    return app_state['feature_names']

def is_model_loaded():
    return app_state['model_loaded']

# ── Initialise prediction database ───────────────────────────────────────────
init_db()
logger.info("Prediction database initialised")

# ── Start model watcher ───────────────────────────────────────────────────────
watcher = ModelWatcher(app_state, check_interval=30)
watcher.start()

# ── Prometheus metrics ────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "fraud_api_requests_total",
    "Total prediction requests",
    ["result"]
)
REQUEST_LATENCY = Histogram(
    "fraud_api_latency_seconds",
    "Prediction request latency"
)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="FinStreamAI Fraud Detection API",
    description="Real-time fraud detection using XGBoost — with zero-downtime model swaps",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health_check():
    """Returns API health status and model availability."""
    return HealthResponse(
        status="healthy" if is_model_loaded() else "degraded",
        model_loaded=is_model_loaded(),
        kafka_available=True
    )

@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest):
    """Score a single transaction for fraud probability."""
    if not is_model_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    with REQUEST_LATENCY.time():
        features = np.array([
            [getattr(transaction, f) for f in get_feature_names()]
        ])
        fraud_prob = float(get_model().predict_proba(features)[0][1])
        is_fraud   = fraud_prob >= 0.5

        if fraud_prob < 0.3:
            risk_level = "LOW"
        elif fraud_prob < 0.7:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        REQUEST_COUNT.labels(
            result="fraud" if is_fraud else "legitimate"
        ).inc()

        logger.info(
            f"Transaction scored — amount: £{transaction.Amount:.2f}, "
            f"fraud_prob: {fraud_prob:.4f}, risk: {risk_level}"
        )

    log_prediction(
        amount=transaction.Amount,
        fraud_probability=round(fraud_prob, 4),
        risk_level=risk_level,
        is_fraud=is_fraud
    )

    return PredictionResponse(
        is_fraud=is_fraud,
        fraud_probability=round(fraud_prob, 4),
        risk_level=risk_level,
        amount=transaction.Amount,
        api_model_version="xgboost-v1"
    )

@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)