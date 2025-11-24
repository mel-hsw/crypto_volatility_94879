"""
FastAPI application for real-time volatility prediction.
Provides /health, /predict, /version, and /metrics endpoints.
"""

import os
import time
import logging
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from starlette.responses import Response
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Import inference logic
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.infer import VolatilityPredictor, prepare_features_for_inference

# Setup structured logging with JSON format
try:
    from pythonjsonlogger import jsonlogger

    # Configure JSON logger
    logHandler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    logHandler.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(logHandler)
    logger.propagate = False
except ImportError:
    # Fallback to standard logging if json logger not available
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

# Prometheus metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status"],
)

prediction_latency_seconds = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
)

predictions_total = Counter(
    "predictions_total", "Total number of predictions", ["model_version", "prediction"]
)

model_loaded = Gauge(
    "model_loaded", "Whether the model is loaded (1) or not (0)", ["model_version"]
)

# Initialize FastAPI app
app = FastAPI(
    title="Crypto Volatility Detection API",
    description="Real-time volatility spike prediction API",
    version="1.0.0",
)

# Initialize rate limiter (generous limits for demo - 1000/min per IP)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Global model predictor
predictor: Optional[VolatilityPredictor] = None
MODEL_VARIANT = os.getenv("MODEL_VARIANT", "ml")  # 'ml' or 'baseline'
MODEL_VERSION = os.getenv("MODEL_VERSION", "random_forest")
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(
        Path(__file__).parent.parent
        / "models"
        / "artifacts"
        / MODEL_VERSION
        / "model.pkl"
    ),
)
BASELINE_MODEL_PATH = os.getenv(
    "BASELINE_MODEL_PATH",
    str(
        Path(__file__).parent.parent / "models" / "artifacts" / "baseline" / "model.pkl"
    ),
)


# Request/Response models
class FeatureRequest(BaseModel):
    """
    Model expects 10 features. Missing features will be filled with 0.0.
    Required features:
    - log_return_300s, spread_mean_300s, trade_intensity_300s, order_book_imbalance_300s
    - spread_mean_60s, order_book_imbalance_60s, price_velocity_300s
    - realized_volatility_300s, order_book_imbalance_30s, realized_volatility_60s
    """

    features: Dict[str, float] = Field(
        ...,
        description="Dictionary of feature values. Missing features will be filled with 0.0.",
        example={
            "log_return_300s": 0.001,
            "spread_mean_300s": 0.5,
            "trade_intensity_300s": 100,
            "order_book_imbalance_300s": 0.6,
            "spread_mean_60s": 0.3,
            "order_book_imbalance_60s": 0.55,
            "price_velocity_300s": 0.0001,
            "realized_volatility_300s": 0.002,
            "order_book_imbalance_30s": 0.52,
            "realized_volatility_60s": 0.0015,
        },
    )


class PredictionResponse(BaseModel):
    """Prediction response."""

    prediction: int = Field(..., description="Binary prediction (0=normal, 1=spike)")
    probability: float = Field(..., description="Prediction probability [0, 1]")
    alert: bool = Field(..., description="Whether to alert (prediction == 1)")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    model_version: str = Field(..., description="Model version used")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    kafka_connected: bool = Field(..., description="Whether Kafka is connected")


class VersionResponse(BaseModel):
    """Version information."""

    version: str = Field(..., description="API version")
    model: str = Field(..., description="Model name")
    model_path: str = Field(..., description="Path to model file")
    api_build_date: str = Field(..., description="API build date")


def load_model():
    """Load the prediction model based on MODEL_VARIANT."""
    global predictor

    # Determine which model to load based on MODEL_VARIANT
    if MODEL_VARIANT == "baseline":
        model_path = BASELINE_MODEL_PATH
        model_name = "baseline"
    else:
        model_path = MODEL_PATH
        model_name = MODEL_VERSION

    if predictor is not None:
        logger.info(f"Model already loaded: {model_name}")
        return

    try:
        logger.info(
            "loading_model",
            extra={
                "model_variant": MODEL_VARIANT,
                "model_path": model_path,
                "model_name": model_name,
            },
        )
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        predictor = VolatilityPredictor(model_path)
        model_loaded.labels(model_version=model_name).set(1)
        logger.info(
            "model_loaded",
            extra={
                "model_name": model_name,
                "model_variant": MODEL_VARIANT,
                "model_path": model_path,
            },
        )
    except Exception as e:
        logger.error(
            "model_load_failed",
            extra={
                "model_path": model_path,
                "error": str(e),
                "error_type": type(e).__name__,
            },
            exc_info=True,
        )
        model_loaded.labels(model_version=model_name).set(0)
        raise


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """Middleware to add correlation ID for request tracing."""
    # Get correlation ID from header or generate new one
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    request.state.correlation_id = correlation_id

    response = await call_next(request)

    # Add correlation ID to response headers
    response.headers["X-Correlation-ID"] = correlation_id

    return response


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware to track HTTP metrics and structured logging."""
    start_time = time.time()
    correlation_id = getattr(request.state, "correlation_id", "unknown")

    # Log request start
    logger.info(
        "request_started",
        extra={
            "correlation_id": correlation_id,
            "method": request.method,
            "endpoint": request.url.path,
            "client_ip": get_remote_address(request),
        },
    )

    try:
        response = await call_next(request)
    except Exception as e:
        # Log error
        logger.error(
            "request_failed",
            extra={
                "correlation_id": correlation_id,
                "method": request.method,
                "endpoint": request.url.path,
                "error": str(e),
                "error_type": type(e).__name__,
            },
            exc_info=True,
        )
        raise

    # Record metrics
    duration = time.time() - start_time
    status_code = response.status_code
    method = request.method
    endpoint = request.url.path

    http_requests_total.labels(
        method=method, endpoint=endpoint, status=status_code
    ).inc()

    # Log request completion
    logger.info(
        "request_completed",
        extra={
            "correlation_id": correlation_id,
            "method": method,
            "endpoint": endpoint,
            "status_code": status_code,
            "duration_ms": duration * 1000,
        },
    )

    return response


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.
    Returns service status and component health.
    """
    try:
        # Check model
        model_status = predictor is not None

        # Check Kafka (simplified - just check if we can import)
        kafka_status = True
        try:
            import kafka  # noqa: F401

            # Could add actual connection check here
        except ImportError:
            kafka_status = False

        status = "healthy" if (model_status and kafka_status) else "degraded"

        return HealthResponse(
            status=status,
            timestamp=datetime.utcnow().isoformat() + "Z",
            model_loaded=model_status,
            kafka_connected=kafka_status,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
@limiter.limit(
    "1000/minute"
)  # Generous limit for demo (1000 requests per minute per IP)
async def predict(request: Request, feature_request: FeatureRequest):
    """
    Make a volatility prediction.

    Accepts feature dictionary and returns prediction, probability, and alert status.
    """
    correlation_id = getattr(request.state, "correlation_id", "unknown")

    if predictor is None:
        logger.error("model_not_loaded", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        logger.info(
            "prediction_requested",
            extra={
                "correlation_id": correlation_id,
                "features_count": len(feature_request.features),
                "model_version": MODEL_VERSION,
            },
        )

        # Convert features to DataFrame (single row)
        # Add required columns that prepare_features expects
        features_dict = feature_request.features.copy()

        # Ensure timestamp exists (required by prepare_features)
        if "timestamp" not in features_dict:
            features_dict["timestamp"] = datetime.utcnow().isoformat()

        # Ensure product_id exists
        if "product_id" not in features_dict:
            features_dict["product_id"] = "BTC-USD"

        # Model expects these 10 features (from train.py prepare_features)
        # Fill missing features with 0.0 (safe default for most features)
        expected_features = [
            "log_return_300s",
            "spread_mean_300s",
            "trade_intensity_300s",
            "order_book_imbalance_300s",
            "spread_mean_60s",
            "order_book_imbalance_60s",
            "price_velocity_300s",
            "realized_volatility_300s",
            "order_book_imbalance_30s",
            "realized_volatility_60s",
        ]

        # Ensure return_range_60s is computed if not provided directly
        # Try to compute from return_max_60s and return_min_60s if available
        if "return_range_60s" not in features_dict:
            if "return_max_60s" in features_dict and "return_min_60s" in features_dict:
                features_dict["return_range_60s"] = (
                    features_dict["return_max_60s"] - features_dict["return_min_60s"]
                )

        # Fill missing features with defaults (including return_range_60s if still missing)
        for feat in expected_features:
            if feat not in features_dict:
                features_dict[feat] = 0.0

        features_df = pd.DataFrame([features_dict])

        # Prepare features using same logic as training
        # This ensures feature consistency
        prepared_features = prepare_features_for_inference(features_df)

        # Make prediction
        start_time = time.time()
        result = predictor.predict(prepared_features)
        inference_time = time.time() - start_time

        # Determine model name for metrics
        model_name = "baseline" if MODEL_VARIANT == "baseline" else MODEL_VERSION

        # Record metrics
        prediction_latency_seconds.observe(inference_time)
        predictions_total.labels(
            model_version=model_name, prediction=str(result["prediction"])
        ).inc()

        # Log successful prediction
        logger.info(
            "prediction_completed",
            extra={
                "correlation_id": correlation_id,
                "prediction": result["prediction"],
                "probability": result["probability"],
                "alert": result["alert"],
                "inference_time_ms": result["inference_time_ms"],
                "model_version": model_name,
            },
        )

        return PredictionResponse(
            prediction=result["prediction"],
            probability=result["probability"],
            alert=result["alert"],
            inference_time_ms=result["inference_time_ms"],
            model_version=f"{model_name} ({MODEL_VARIANT})",
        )
    except KeyError as e:
        logger.error(
            "missing_feature",
            extra={"correlation_id": correlation_id, "missing_feature": str(e)},
        )
        raise HTTPException(
            status_code=400, detail=f"Missing required feature: {str(e)}"
        )
    except Exception as e:
        logger.error(
            "prediction_failed",
            extra={
                "correlation_id": correlation_id,
                "error": str(e),
                "error_type": type(e).__name__,
            },
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/version", response_model=VersionResponse)
async def version():
    """
    Get API and model version information.
    """
    return VersionResponse(
        version="1.0.0",
        model=MODEL_VERSION,
        model_path=MODEL_PATH,
        api_build_date=datetime.utcnow().strftime("%Y-%m-%d"),
    )


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.
    Returns metrics in Prometheus format.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Crypto Volatility Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Make predictions (POST)",
            "/version": "API version info",
            "/metrics": "Prometheus metrics",
            "/docs": "API documentation (Swagger)",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
