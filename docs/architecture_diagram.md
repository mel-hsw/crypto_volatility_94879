# System Architecture Diagram

## Week 4 - Replay Mode Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA SOURCE                              │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         Coinbase WebSocket API                           │   │
│  │    (wss://advanced-trade-ws.coinbase.com)               │   │
│  └───────────────────────┬────────────────────────────────┘   │
│                            │                                      │
│                            │ WebSocket Stream                     │
│                            │                                      │
└────────────────────────────┼──────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      INGESTION LAYER                             │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         Kafka Producer                                    │   │
│  │         (scripts/ws_ingest.py)                            │   │
│  │                                                           │   │
│  │  • Subscribes to ticker channel                         │   │
│  │  • Publishes to Kafka topic: ticks.raw                  │   │
│  │  • Optional: Save to disk (NDJSON)                      │   │
│  └───────────────────────┬────────────────────────────────┘   │
│                            │                                      │
└────────────────────────────┼──────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MESSAGE BROKER (Kafka + Zookeeper)           │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         Kafka Broker (Zookeeper Mode)                   │   │
│  │         Port: 9092                                        │   │
│  │                                                           │   │
│  │  Topics:                                                 │   │
│  │  • ticks.raw      (raw ticker data)                     │   │
│  │  • ticks.features (computed features)                   │   │
│  └───────────────────────┬────────────────────────────────┘   │
│                            │                                      │
└────────────────────────────┼──────────────────────────────────────┘
                             │
                             │ ticks.raw
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING                           │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         Feature Pipeline                                 │   │
│  │         (features/featurizer.py)                        │   │
│  │                                                           │   │
│  │  • Consumes from ticks.raw                              │   │
│  │  • Computes windowed features (30s, 60s, 300s)          │   │
│  │  • Publishes to ticks.features                          │   │
│  │  • Saves to parquet (data/processed/features.parquet)   │   │
│  └───────────────────────┬────────────────────────────────┘   │
│                            │                                      │
└────────────────────────────┼──────────────────────────────────────┘
                             │
                             │ ticks.features
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREDICTION API                               │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         FastAPI Service                                  │   │
│  │         (api/app.py)                                     │   │
│  │         Port: 8000                                       │   │
│  │                                                           │   │
│  │  Endpoints:                                              │   │
│  │  • GET  /health    - Health check                        │   │
│  │  • POST /predict   - Make predictions                    │   │
│  │  • GET  /version   - API version info                    │   │
│  │  • GET  /metrics   - Prometheus metrics                 │   │
│  │                                                           │   │
│  │  Model: XGBoost (Stratified)                             │   │
│  │  • Loads from models/artifacts/xgboost_stratified/      │   │
│  │  • Inference time: < 1ms per sample                      │   │
│  └───────────────────────┬────────────────────────────────┘   │
│                            │                                      │
└────────────────────────────┼──────────────────────────────────────┘
                             │
                             │ Predictions & Metrics
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MONITORING & TRACKING                         │
│                                                                   │
│  ┌──────────────────────┐      ┌────────────────────────────┐   │
│  │      MLflow          │      │      Prometheus           │   │
│  │      Port: 5001      │      │      (via /metrics)        │   │
│  │                      │      │                            │   │
│  │  • Model tracking    │      │  • HTTP request metrics    │   │
│  │  • Experiment logs   │      │  • Prediction latency      │   │
│  │  • Artifact storage  │      │  • Prediction counts       │   │
│  └──────────────────────┘      └────────────────────────────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Ingestion Layer
- **Component**: `scripts/ws_ingest.py`
- **Function**: Connects to Coinbase WebSocket, streams ticker data
- **Output**: Kafka topic `ticks.raw`
- **Optional**: Saves raw data to `data/raw/*.ndjson`

### 2. Message Broker
- **Component**: Kafka (Zookeeper mode)
- **Port**: 9092
- **Topics**:
  - `ticks.raw`: Raw ticker data from Coinbase
  - `ticks.features`: Computed features ready for prediction
- **Mode**: Zookeeper mode (traditional, stable)

### 3. Feature Engineering
- **Component**: `features/featurizer.py`
- **Function**: 
  - Consumes from `ticks.raw`
  - Computes windowed features (30s, 60s, 300s windows)
  - Publishes to `ticks.features`
  - Saves to parquet for batch processing
- **Features**: 10+ features including returns, volatility, spreads, trade intensity

### 4. Prediction API
- **Component**: `api/app.py` (FastAPI)
- **Port**: 8000
- **Endpoints**:
  - `/health`: Service health check
  - `/predict`: POST endpoint for predictions
  - `/version`: API and model version info
  - `/metrics`: Prometheus metrics
- **Model**: XGBoost (Stratified) - PR-AUC 0.7815

### 5. Monitoring & Tracking
- **MLflow**: Model versioning, experiment tracking
- **Prometheus**: Metrics collection via `/metrics` endpoint
- **Future**: Grafana dashboards (Week 6)

## Data Flow (Replay Mode)

1. **Replay Data** → Load NDJSON file → Publish to Kafka `ticks.raw`
2. **Feature Pipeline** → Consume from Kafka → Compute features → Publish to `ticks.features`
3. **API** → Consume features (or receive via POST) → Predict → Return results
4. **Monitoring** → Log predictions to MLflow → Expose metrics to Prometheus

## Deployment

All services run via Docker Compose:
```bash
cd docker
docker compose up -d
```

Services:
- `zookeeper`: Zookeeper coordination service
- `kafka`: Kafka broker
- `mlflow-server`: MLflow tracking server
- `volatility-api`: FastAPI prediction service
- `prometheus`: Prometheus metrics collection
- `grafana`: Grafana visualization dashboard

