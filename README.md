# Crypto Volatility Detection

**Author:** Melissa Wong  
**Course:** Operationalize AI  
**Date:** November 2025

Real-time cryptocurrency volatility detection system using Coinbase WebSocket API, Kafka, and MLflow.

---

## 📋 Project Overview

This project builds a complete ML pipeline to detect short-term volatility spikes in cryptocurrency markets using streaming data from Coinbase. The system predicts whether significant price volatility will occur in the next 60 seconds.

**Key Technologies:** Python, Kafka, MLflow, Docker, Evidently, scikit-learn

---

## 🏗️ Project Structure

```
.
├── docker/
│   ├── docker-compose.yaml       # Infrastructure (Kafka, Zookeeper, MLflow)
│   └── Dockerfile.ingestor       # Containerized data ingestion
├── scripts/
│   ├── ws_ingest.py              # WebSocket data ingestion
│   ├── kafka_consume_check.py    # Stream validation
│   ├── replay.py                 # Reproducibility verification
│   ├── generate_evidently_report.py  # Data drift monitoring
│   └── generate_eval_report.py   # Model evaluation PDF
├── features/
│   └── featurizer.py             # Real-time feature computation
├── models/
│   ├── baseline.py               # Baseline z-score detector
│   ├── train.py                  # Training pipeline
│   ├── infer.py                  # Inference & benchmarking
│   └── artifacts/                # Saved models and plots
├── notebooks/
│   └── eda.ipynb                 # Exploratory data analysis
├── data/
│   ├── raw/                      # Raw ticker data (NDJSON)
│   ├── processed/                # Engineered features (Parquet)
│   └── reports/                  # Evidently reports
├── docs/
│   ├── scoping_brief.pdf         # Problem definition
│   ├── feature_spec.md           # Feature documentation
│   ├── model_card_v1.md          # Model documentation
│   └── genai_appendix.md         # GenAI usage disclosure
├── reports/
│   └── model_eval.pdf            # Model evaluation report
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Docker Desktop installed and running
- Python 3.9+ with pip (pyenv recommended for version management)
- Git

### 1. Clone & Setup

```bash
git clone <repository-url>
cd operationaliseai

# Install Python 3.9 if using pyenv (optional)
pyenv install 3.9.25
pyenv local 3.9.25

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Infrastructure

```bash
cd docker
docker compose up -d

# Verify services running
docker compose ps
# Expected: kafka, zookeeper, mlflow all "Up"
```

**Services:**
- Kafka: localhost:9092
- Zookeeper: localhost:2182
- MLflow: http://localhost:5001

### 3. Create Kafka Topics

```bash
# For Docker container names like docker-kafka-1
docker exec -it docker-kafka-1 kafka-topics --create \
  --topic ticks.raw \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1

docker exec -it docker-kafka-1 kafka-topics --create \
  --topic ticks.features \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1

# Verify
docker exec -it docker-kafka-1 kafka-topics --list --bootstrap-server localhost:9092
```

### 4. Configure Hostname Resolution (for local execution)

```bash
echo "127.0.0.1 kafka" | sudo tee -a /etc/hosts
```

---

## 📊 Complete Pipeline Workflow

### Milestone 1: Data Ingestion (15-60 minutes)

**Collect real-time ticker data from Coinbase:**

```bash
# Collect data (60 minutes recommended for good analysis)
python scripts/ws_ingest.py --pair BTC-USD --minutes 60 --save-disk

# In another terminal, validate stream
python scripts/kafka_consume_check.py --topic ticks.raw --min 100
```

**Verify data collected:**
```bash
ls -lh data/raw/
wc -l data/raw/ticks_BTCUSD_*.ndjson
```

### Milestone 2: Feature Engineering (15-30 minutes)

**Generate features from raw data:**

```bash
# Process all raw data into features
python scripts/replay.py \
  --raw "data/raw/ticks_BTCUSD_*.ndjson" \
  --out data/processed/features.parquet

# Verify features created
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/features.parquet')
print(f'Samples: {len(df)}')
print(f'Columns: {list(df.columns)}')
print(f'Spike rate: {df[\"volatility_spike\"].mean():.2%}')
"
```

**Run EDA to select threshold:**

```bash
jupyter notebook notebooks/eda.ipynb
# Follow notebook to analyze features and select volatility threshold
```

**Generate drift report:**

```bash
python scripts/generate_evidently_report.py
open data/reports/evidently_report.html
```

### Milestone 3: Model Training & Evaluation (30 minutes)

**Train all models:**

```bash
python models/train.py \
  --features data/processed/features.parquet \
  --models baseline logistic xgboost \
  --mlflow-uri http://localhost:5001
```

**View results in MLflow:**
```bash
open http://localhost:5001
```

**Benchmark inference performance:**

```bash
python models/infer.py \
  --model models/artifacts/logistic_regression/model.pkl \
  --features data/processed/features.parquet \
  --mode benchmark \
  --n-samples 1000
```

**Generate evaluation report:**

```bash
python scripts/generate_eval_report.py \
  --features data/processed/features.parquet \
  --artifacts models/artifacts \
  --output reports/model_eval.pdf

open reports/model_eval.pdf
```

**Complete documentation:**
```bash
# Fill in actual metrics from MLflow and reports
nano docs/model_card_v1.md
nano docs/genai_appendix.md
```

---

## 📄 Data Format

### Raw Ticker Data (NDJSON)

```json
{
  "timestamp": "2025-11-08T20:15:42.123456",
  "product_id": "BTC-USD",
  "price": "76543.21",
  "volume_24h": "12345.67890123",
  "low_24h": "75000.00",
  "high_24h": "77000.00",
  "best_bid": "76543.20",
  "best_ask": "76543.22",
  "raw": { ... }
}
```

### Feature Engineering

**Computed Features:**
- `price_return_1min` - 1-minute price return
- `price_return_5min` - 5-minute price return  
- `price_volatility_5min` - Rolling standard deviation
- `bid_ask_spread` - Absolute bid-ask spread
- `bid_ask_spread_bps` - Spread in basis points
- `volume_24h_pct_change` - 24-hour volume change

**Target Variable:**
- `volatility_spike` - Binary (1 = spike, 0 = normal)
- Threshold: 90th percentile of rolling volatility

---

## 🎯 Project Milestones

### ✅ Milestone 1: Streaming Setup (Complete)
- Kafka & MLflow infrastructure via Docker
- Real-time WebSocket data ingestion  
- Reconnection & heartbeat monitoring
- Stream validation & containerization
- Scoping brief with success metrics

### ✅ Milestone 2: Feature Engineering (Complete)
- Real-time feature computation pipeline
- Replay script for reproducibility
- EDA with threshold selection
- Evidently data quality reports
- Feature specification documentation

### ✅ Milestone 3: Modeling & Evaluation (Complete)
- Baseline z-score model
- ML models (Logistic Regression, XGBoost)
- MLflow experiment tracking
- Inference benchmarking (< 2x real-time)
- Model evaluation PDF report
- Model Card v1.0
- GenAI usage appendix

---

## 📈 Performance Results

### Model Performance (Test Set)

| Model | PR-AUC | F1-Score | Precision | Recall | Inference Time |
|-------|--------|----------|-----------|--------|----------------|
| Baseline (Z-Score) | 0.3149 | 0.0000 | 0.0000 | 0.0000 | < 1ms |
| Logistic Regression | 0.2449 | 0.2759 | 0.1773 | 0.6218 | < 1ms |
| XGBoost | 0.2323 | 0.2586 | 0.3059 | 0.2239 | < 1ms |

*Note: All models trained with reduced feature set (10 features) to minimize multicollinearity. Features include log return volatility, return statistics, spread volatility, and trade intensity. Removed perfectly correlated features (return_std_* and log_return_mean_*) which improved Logistic Regression PR-AUC by +6.6% (0.2298 → 0.2449). Baseline achieves PR-AUC 0.3149 but has 0% recall (threshold too conservative). Logistic Regression detects 62.18% of spikes with 17.73% precision. XGBoost achieves PR-AUC 0.2323 with high precision (30.59%) but lower recall (22.39%).*

### Requirements Met

- ✅ **Inference < 2x real-time:** All models < 120s requirement (typically < 1ms per sample)
- ✅ **Reproducibility:** Replay matches live features
- ✅ **Data Quality:** Monitored with Evidently reports
- ⚠️ **PR-AUC:** Model achieves 0.0699 PR-AUC with 89.77% recall (prioritizes spike detection over precision)

---

## 🧪 Testing & Validation

### Quick System Check

```bash
# 1. Infrastructure
cd docker && docker compose ps

# 2. Data exists
ls data/raw/*.ndjson
ls data/processed/features.parquet

# 3. Models trained
ls models/artifacts/*/model.pkl

# 4. MLflow has runs
open http://localhost:5001

# 5. Reports generated
ls reports/*.pdf
ls data/reports/*.html
```

### Troubleshooting

**Kafka connection issues:**
```bash
# Add hostname resolution
echo "127.0.0.1 kafka" | sudo tee -a /etc/hosts

# Check Kafka running
docker ps | grep kafka
```

**MLflow unhealthy but working:**
```bash
curl http://localhost:5001/health
# Should return "OK"
```

**Not enough data:**
```bash
# Check sample count
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/features.parquet')
print(f'Samples: {len(df)} (need 500+)')
print(f'Spike rate: {df[\"volatility_spike\"].mean():.2%} (target: 5-15%)')
"
```

---

## 📚 Documentation

- **Scoping Brief** (`docs/scoping_brief.pdf`): Problem definition, success metrics, risk assumptions
- **Feature Spec** (`docs/feature_spec.md`): Feature engineering details and rationale
- **Model Card** (`docs/model_card_v1.md`): Model documentation following Mitchell et al. (2019)
- **GenAI Appendix** (`docs/genai_appendix.md`): Transparent AI usage disclosure
- **Evaluation Report** (`reports/model_eval.pdf`): Comprehensive performance analysis

---

## 🔐 Security & Ethics

- **No secrets committed:** Credentials in `.env` (gitignored)
- **Public data only:** Free Coinbase WebSocket API
- **No automated trading:** Analysis and alerting only; human oversight required
- **Transparent AI usage:** Full disclosure in GenAI appendix

---

## 🛠️ Technology Stack

**Core:**
- Python 3.9+
- Apache Kafka (KRaft mode)
- MLflow 2.10.2
- Docker Compose

**ML & Analysis:**
- scikit-learn 1.3.2
- XGBoost 2.0.3
- pandas, NumPy
- Evidently 0.4.11
- Jupyter

**Data Formats:**
- NDJSON (raw ticks)
- Parquet (features)
- PDF (reports)

---

## 📞 Support

**Common Issues:**
1. Kafka not accessible → Check Docker containers, verify hostname resolution
2. No data collected → Verify WebSocket connection, check Coinbase API status
3. Models not training → Ensure features.parquet exists with 500+ samples
4. Poor model performance → Collect more data, adjust threshold, add features

**Resources:**
- [Coinbase API Docs](https://docs.cloud.coinbase.com/advanced-trade-api/docs/ws-overview)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Evidently Documentation](https://docs.evidentlyai.com/)
- [Model Cards Paper](https://arxiv.org/abs/1810.03993)

---

## 🎓 Learning Outcomes

This project demonstrates:

1. **Streaming Data Engineering:** Kafka producers/consumers, real-time processing
2. **Feature Engineering:** Time-series features, rolling windows, labeling strategies
3. **ML Operations:** Experiment tracking, model versioning, reproducibility
4. **Model Evaluation:** PR-AUC for imbalanced data, confusion matrices, performance benchmarking
5. **Production Readiness:** Inference latency validation, data drift monitoring
6. **Documentation:** Model cards, scoping briefs, transparent AI usage

---

## 🔄 Future Enhancements

- **Deployment:** REST API for real-time predictions
- **Monitoring:** Grafana dashboard for live alerts
- **Automation:** Scheduled retraining pipeline
- **Scaling:** Multi-pair support (ETH-USD, etc.)
- **Advanced Models:** LSTM for sequence modeling

