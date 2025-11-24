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
├── api/ # FastAPI application
│ ├── init.py
│ └── app.py # Main API endpoints (/predict, /health, /version, /metrics)
├── docker/ # Docker configuration
│ ├── compose.yaml # Main Docker Compose (Kafka, Zookeeper, MLflow, API, Prometheus, Grafana)
│ ├── compose-kraft.yaml # Kafka KRaft mode (alternative)
│ ├── Dockerfile.api # API service containerization
│ ├── Dockerfile.ingestor # Data ingestion containerization
│ ├── grafana/
│ │ ├── dashboards/
│ │ │ ├── crypto-volatility.json # Pre-configured Grafana dashboard
│ │ │ └── dashboard.yml # Dashboard provisioning config
│ │ └── datasources/
│ │ └── prometheus.yml # Prometheus datasource config
│ ├── prometheus/
│ │ ├── prometheus.yml # Prometheus configuration
│ │ └── alerts.yml # Prometheus alerting rules
│ └── mlflow_data/ # MLflow data persistence (volumes)
├── scripts/ # Utility scripts
│ ├── ws_ingest.py # WebSocket data ingestion → Kafka
│ ├── replay.py # Reproducibility verification (offline feature generation)
│ ├── replay_to_kafka.py # Replay NDJSON to Kafka (E2E testing)
│ ├── consolidate_data.py # Consolidate feature files & create stratified splits
│ ├── feature_analysis.py # Feature engineering & selection analysis
│ ├── generate_evidently_report.py # Data drift monitoring reports
│ ├── generate_eval_report.py # Model evaluation PDF generation
│ ├── load_test.py # API load testing script
│ ├── add_labels.py # Add volatility spike labels to features
│ ├── kafka_consume_check.py # Stream validation
│ ├── run_e2e_simple.sh # Simplified end-to-end test (ingestion + API)
│ ├── run_e2e_test.sh # Full end-to-end pipeline test
│ └── retrain_all_models.sh # Retrain all models script
├── features/ # Feature engineering
│ └── featurizer.py # Real-time feature computation (FeatureComputer, FeaturePipeline)
├── models/ # Model training & inference
│ ├── baseline.py # Baseline z-score detector
│ ├── train.py # Training pipeline (Random Forest, XGBoost, Logistic Regression, Baseline)
│ ├── infer.py # Inference & benchmarking (VolatilityPredictor)
│ └── artifacts/ # Saved models and metadata
│ ├── random_forest/
│ │ ├── model.pkl # Trained Random Forest model
│ │ ├── threshold_metadata.json # Optimal threshold (0.7057)
│ │ └── .png # PR/ROC curves, feature importance
│ └── baseline/
│ ├── model.pkl # Baseline model
│ └── .png # Evaluation plots
├── tests/ # Test suite
│ ├── init.py
│ └── test_api_integration.py # API integration tests (pytest)
├── notebooks/ # Jupyter notebooks
│ └── eda.ipynb # Exploratory data analysis
├── data/ # Data storage
│ ├── raw/ # Raw ticker data (NDJSON)
│ │ ├── ticks_BTCUSD_.ndjson # Historical tick data
│ │ └── ticks_10min_sample.ndjson # Sample data for testing
│ └── processed/ # Engineered features (Parquet)
│ ├── features_consolidated.parquet # Consolidated dataset (26,881 samples)
│ ├── features_consolidated_train.parquet # Training set (stratified split)
│ ├── features_consolidated_val.parquet # Validation set (stratified split)
│ ├── features_consolidated_test.parquet # Test set (stratified split)
│ ├── features_replay.parquet # Replay features
│ └── features_.parquet # Other feature files
├── docs/ # Documentation
│ ├── architecture_diagram.md # System architecture diagram
│ ├── feature_spec.md # Feature specification (v1.2)
│ ├── model_card_v1.md # Model card documentation
│ ├── selection_rationale.md # Model selection rationale
│ ├── performance_summary.md # Performance metrics & SLO compliance
│ ├── runbook.md # Operations runbook (startup, troubleshooting, recovery)
│ ├── slo.md # Service Level Objectives
│ ├── drift_summary.md # Data drift monitoring strategy
│ ├── genai_appendix.md # GenAI usage disclosure
│ ├── scoping_brief.pdf # Problem definition
│ ├── grafana_dashboard_screenshot.png # Dashboard screenshot
│ └── load_test_results.json # Load test results
├── reports/ # Generated reports
│ ├── evidently/ # Evidently drift reports
│ │ ├── data_drift_report.html
│ │ ├── train_test_drift_report.html
│ │ └── .json # Report metadata
│ ├── feature_analysis/ # Feature analysis outputs
│ │ ├── feature_importance.png
│ │ ├── feature_importance.csv
│ │ ├── correlation_heatmap.png
│ │ └── recommended_features.txt
│ └── model_eval.pdf # Model evaluation report
├── .github/ # GitHub configuration (if exists)
│ └── workflows/
│ └── ci.yml # CI/CD pipeline (lint, test, replay)
├── requirements.txt # Python dependencies (local development)
├── requirements-api.txt # Python dependencies (Docker API service only)
├── PRE_SUBMISSION_CHECKLIST.md # Pre-submission checklist
└── README.md # This file
```

---

## 🚀 Quick Start (≤10 Lines)

```bash
git clone <repository-url> && cd operationaliseai
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cd docker && docker compose up -d
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"features":{"log_return_300s":0.001,"spread_mean_300s":0.5,"trade_intensity_300s":100,"order_book_imbalance_300s":0.6,"spread_mean_60s":0.3,"order_book_imbalance_60s":0.55,"price_velocity_300s":0.0001,"realized_volatility_300s":0.002,"order_book_imbalance_30s":0.52,"realized_volatility_60s":0.0015}}'
```

**That's it!** The API is running at `http://localhost:8000`. See detailed setup below.

---

## 📋 Detailed Setup

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

# Initialize pyenv in your shell (if not already in ~/.zshrc)
eval "$(pyenv init -)"

# If using conda/anaconda, deactivate it first
conda deactivate

# Create virtual environment (uses pyenv's Python 3.9.25)
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Verify Python version
python --version  # Should show Python 3.9.25

# Install dependencies
pip install -r requirements.txt
```

**Note on Requirements Files:**
- **`requirements.txt`** - For local development (scripts, notebooks, training). Includes Evidently (requires `pydantic<2`).
- **`requirements-api.txt`** - For Docker API container only. Uses `pydantic>=2.0` (required by FastAPI). Docker handles this automatically - you don't need to install it manually.

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
- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (credentials: see `.env` file or use defaults from `docker/compose.yaml`)

### 3. Create Kafka Topics

```bash
# For Docker container names like kafka
docker exec -it kafka kafka-topics --create \
  --topic ticks.raw \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1

docker exec -it kafka kafka-topics --create \
  --topic ticks.features \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1

# Verify
docker exec -it kafka kafka-topics --list --bootstrap-server localhost:9092
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

# Option1: Process from Kafka stream (for live streaming data)
python features/featurizer.py \
  --output_file data/processed/features.parquet
# Labels added automatically when pipeline finishes

# Alternative: Uses FeatureComputer (same as featurizer) + adds volatility_spike labels
python scripts/replay.py \
  --raw "data/raw/ticks_BTCUSD_*.ndjson" \
  --out data/processed/features.parquet \
  --add-labels \
  --label-threshold-percentile 90

# Verify features created
python -c "
import pandas as pd
# Check labeled file (used for training)
df = pd.read_parquet('data/processed/features_replay.parquet')
print(f'Samples: {len(df)}')
print(f'Columns: {len(df.columns)} columns')
if 'volatility_spike' in df.columns:
    print(f'Spike rate: {df[\"volatility_spike\"].mean():.2%}')
else:
    print('Note: volatility_spike column not found. Add labels using: python scripts/add_labels.py --features data/processed/features_replay.parquet')
"
```

**Run EDA to select threshold:**

```bash
jupyter notebook notebooks/eda.ipynb
# Follow notebook to analyze features and select volatility threshold
```

**Generate drift report:**

```bash
# Compare early vs late data (data drift)
python scripts/generate_evidently_report.py \
  --features data/processed/features_replay.parquet \
  --output reports/evidently/data_drift_report.html

# Compare train vs test split (train/test drift)
python scripts/generate_evidently_report.py \
  --features data/processed/features_replay.parquet \
  --report_type train_test \
  --output reports/evidently/train_test_drift_report.html

# Open reports
open reports/evidently/data_drift_report.html
```

**Consolidate all available data for better performance:**

```bash
# Consolidate all feature files and create balanced splits
python scripts/consolidate_data.py --splits

# This creates:
# - data/processed/features_consolidated.parquet (all data)
# - data/processed/features_consolidated_train.parquet
# - data/processed/features_consolidated_val.parquet
# - data/processed/features_consolidated_test.parquet

# Train with consolidated data and stratified splits (recommended)
python models/train.py \
  --features data/processed/features_consolidated.parquet \
  --split-method stratified \
  --models random_forest
```

### Milestone 3: Model Training & Evaluation (30 minutes)

**Train all models:**

```bash
# Train models with consolidated data and stratified splits (recommended)
python models/train.py \
  --features data/processed/features_consolidated.parquet \
  --split-method stratified \
  --models baseline logistic xgboost random_forest \
  --mlflow-uri http://localhost:5001

# Or use time-based split (original method)
python models/train.py \
  --features data/processed/features_replay.parquet \
  --split-method time_based \
  --models random_forest \
  --mlflow-uri http://localhost:5001
```

**Threshold Optimization:**
- The Random Forest model automatically optimizes the **probability threshold** during training
- Uses optimal F1 threshold (0.7057) which maximizes F1-score on validation set
- This threshold achieves excellent performance: validation PR-AUC 0.9806, test PR-AUC 0.9859
- Alternative threshold for 10% spike rate (0.8050) is computed for reference but not used
- Threshold is saved to `models/artifacts/random_forest/threshold_metadata.json`
- Inference code automatically loads and uses the optimized threshold
- This ensures the model predicts spikes appropriately: high precision (95.7%) and recall (93.7%)

**View results in MLflow:**
```bash
open http://localhost:5001
```

**Benchmark inference performance:**

```bash
python models/infer.py \
  --model models/artifacts/random_forest/model.pkl \
  --features data/processed/features_replay.parquet \
  --mode benchmark \
  --n-samples 1000
```

**Generate evaluation report:**

```bash
python scripts/generate_eval_report.py \
  --features data/processed/features_consolidated.parquet \
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

**Current Feature Set (v1.2):**

Features are computed over rolling windows: **30s, 60s, and 300s**

**1. Momentum & Volatility (Price Trends):**
- `log_return_{window}s` - Log returns over fixed lookback periods
- `realized_volatility_{window}s` - Rolling std dev of 1-second returns (target proxy, volatility clusters)
- `price_velocity_{window}s` - Rolling mean of absolute 1-second price changes

**2. Liquidity & Microstructure (Market Nerves):**
- `spread_mean_{window}s` - Rolling mean of bid-ask spread (smoothed)
- `order_book_imbalance_{window}s` - Rolling mean of buy/sell volume ratio at top of book

**3. Activity (Market Energy):**
- `trade_intensity_{window}s` - Rolling sum of tick count (trade intensity)
- `volume_velocity_{window}s` - Rolling sum of trade sizes (may be 0.0 if unavailable)

**Top 10 Features Used in Random Forest Model:**
1. `order_book_imbalance_300s` (18.8% importance)
2. `trade_intensity_300s` (17.1% importance)
3. `spread_mean_300s` (14.9% importance)
4. `log_return_300s`, `spread_mean_60s`, `order_book_imbalance_60s`
5. `price_velocity_300s`, `realized_volatility_300s`
6. `order_book_imbalance_30s`, `realized_volatility_60s`

**Target Variable:**
- `volatility_spike` - Binary (1 = spike, 0 = normal)
- Threshold: 90th percentile of future volatility (chunk-aware, 60-second lookahead)

---

## 📈 Performance Results

### Model Performance (Test Set)

**Current Model Performance (November 24, 2025) - Stratified Split with Consolidated Data:**
| Model | PR-AUC (Test) | PR-AUC (Val) | Improvement vs Baseline |
|-------|---------------|--------------|-------------------------|
| **Random Forest** | **0.9859** | 0.9806 | **+849%** (9.5x better) |
| Baseline (Z-Score) | 0.1039 | 0.0861 | Baseline |
| XGBoost | [To be retrained] | [To be retrained] | - |
| Logistic Regression | [To be retrained] | [To be retrained] | - |

**Note:** Baseline model has very low recall (4.42% test, 2.33% validation), missing 95.58% of volatility spikes. Random Forest significantly outperforms baseline with 93.7% recall.

**Key Findings:**
- **Current Production Model: Random Forest** achieves PR-AUC 0.9859, F1 0.9471, Recall 93.7%, Precision 95.7%, outperforming baseline by 849% (9.5x better)
- **Consolidated Dataset:** Trained on 26,881 samples from consolidated data (5 feature files, ~350 hours)
- **Stratified Splits:** Balanced spike rates (~10.67%) across all splits eliminate validation/test imbalance
- **Model Selection:** Random Forest selected based on best test performance and interpretable feature importance
- **Top Features:** `order_book_imbalance_300s` (18.8%), `trade_intensity_300s` (17.1%), `spread_mean_300s` (14.9%)
- **Threshold Optimization:** Probability threshold optimized to 0.7057 (maximizes F1-score), achieving excellent performance on both validation and test
- **Feature Set:** 10 top features from new v1.2 feature set (Momentum & Volatility, Liquidity & Microstructure, Activity)
- **Baseline Performance:** PR-AUC 0.1039 (test), 0.0861 (validation) with very low recall (4.42% test, 2.33% validation)

### Requirements Met

- ✅ **Inference < 2x real-time:** All models < 120s requirement (typically < 1ms per sample)
- ✅ **Reproducibility:** Replay matches live features
- ✅ **Data Quality:** Monitored with Evidently reports
- ✅ **PR-AUC:** Current model (Random Forest) achieves 0.9859 PR-AUC (test set) with top 10 features

---

## 🧪 Testing & Validation

### Quick System Check

```bash
# 1. Infrastructure
cd docker && docker compose ps

# 2. Data exists
ls data/raw/*.ndjson
ls data/processed/features_consolidated.parquet
ls data/processed/features_replay.parquet

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
df = pd.read_parquet('data/processed/features_consolidated.parquet')
print(f'Samples: {len(df)} (need 500+)')
print(f'Spike rate: {df[\"volatility_spike\"].mean():.2%} (target: 5-15%)')
"
```

---

## 🚨 Data Drift Response Guide

When data drift is detected, follow these steps to maintain model performance:

### Step 1: Assess Drift Severity

```bash
# Generate drift report
python scripts/generate_evidently_report.py \
  --features data/processed/features_replay.parquet \
  --output reports/evidently/data_drift_report.html

# Open and review the report
open reports/evidently/data_drift_report.html
```

**Key Metrics to Check:**
- **Dataset Drift Metric**: Overall drift score (0-1, higher = more drift)
- **Column Drift Metrics**: Per-feature drift detection
- **Missing Values**: Sudden increase may indicate data pipeline issues
- **Distribution Shifts**: Visual comparison of feature distributions

**Severity Levels:**
- **Low (< 0.3)**: Minor distribution shifts, likely normal market variation
- **Medium (0.3-0.6)**: Significant shifts, investigate root cause
- **High (> 0.6)**: Major drift, immediate action required

### Step 2: Investigate Root Cause

**Common Causes:**
1. **Market Regime Change**: Crypto markets are volatile; new regimes are normal
   - Check: Compare drift timing with major market events
   - Action: May require retraining with recent data

2. **Data Pipeline Issues**: Changes in data collection or processing
   - Check: Review `features/featurizer.py` logs for errors
   - Check: Verify Kafka/WebSocket connection stability
   - Action: Fix pipeline, reprocess data

3. **Feature Engineering Changes**: Code changes affecting feature computation
   - Check: Git diff of `features/featurizer.py`
   - Action: Ensure consistency or document intentional changes

4. **Data Quality Degradation**: Missing data, outliers, or gaps
   - Check: Review `DatasetMissingValuesMetric` in Evidently report
   - Check: Run EDA notebook to visualize data quality
   - Action: Fix data source, filter bad data

### Step 3: Evaluate Model Impact

```bash
# Test current model on drifted data
python models/infer.py \
  --model models/artifacts/random_forest/model.pkl \
  --features data/processed/features_replay.parquet \
  --mode benchmark \
  --n-samples 1000

# Compare performance metrics
# If PR-AUC drops significantly (< 0.20), retraining is critical
```

**Decision Matrix:**

| Drift Severity | Model Performance | Action |
|----------------|-------------------|--------|
| Low | PR-AUC > 0.20 | Monitor, no action |
| Low | PR-AUC < 0.20 | Investigate, consider retraining |
| Medium | PR-AUC > 0.20 | Retrain with recent data |
| Medium | PR-AUC < 0.20 | **Retrain immediately** |
| High | Any | **Retrain immediately**, investigate root cause |

### Step 4: Retrain Model (if needed)

```bash
# 1. Collect recent data (if needed)
python scripts/ws_ingest.py --pair BTC-USD --minutes 60 --save-disk

# 2. Regenerate features with latest data
python scripts/replay.py \
  --raw "data/raw/ticks_BTCUSD_*.ndjson" \
  --out data/processed/features_replay.parquet \
  --add-labels

# 3. Retrain models
python models/train.py \
  --features data/processed/features_replay.parquet \
  --models baseline logistic xgboost random_forest \
  --mlflow-uri http://localhost:5001

# 4. Compare new vs old model in MLflow
open http://localhost:5001
```

**Retraining Best Practices:**
- Use time-based split (most recent 15% as test set)
- Compare new model performance with previous version in MLflow
- If new model performs worse, investigate further
- Document retraining reason and results

### Step 5: Deploy Updated Model

```bash
# Copy best-performing model to production location
cp models/artifacts/random_forest/model.pkl models/artifacts/production/model.pkl

# Update inference script to use new model
# Test inference latency
python models/infer.py \
  --model models/artifacts/production/model.pkl \
  --features data/processed/features_replay.parquet \
  --mode benchmark \
  --n-samples 1000
```

### Step 6: Document & Monitor

**Documentation Checklist:**
- [ ] Record drift detection date and severity
- [ ] Document root cause analysis findings
- [ ] Note retraining date and new model performance
- [ ] Update model card with new training data range
- [ ] Log action taken in MLflow experiment notes

**Ongoing Monitoring:**
```bash
# Set up weekly drift checks (add to cron or scheduled job)
# Weekly: Generate drift report
python scripts/generate_evidently_report.py \
  --features data/processed/features_replay.parquet \
  --output reports/evidently/weekly_drift_$(date +%Y%m%d).html

# Weekly: Evaluate model on recent data
python models/infer.py \
  --model models/artifacts/production/model.pkl \
  --features data/processed/features_replay.parquet \
  --mode benchmark \
  --n-samples 1000
```

### Automated Drift Detection (Future Enhancement)

Consider implementing:
- Scheduled drift reports (cron job)
- Alert system when drift exceeds threshold
- Automated retraining pipeline when drift detected
- Model performance monitoring dashboard

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
- scikit-learn >=1.4.0
- XGBoost 2.0.3
- pandas, NumPy
- Evidently >=0.4.40,<0.5.0
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
3. Models not training → Ensure features_consolidated.parquet or features_replay.parquet exists with 500+ samples
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

