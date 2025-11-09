# Crypto Volatility Detection - Milestone 1

**Author:** Melissa Wong  
**Course:** Operationalize AI  
**Date:** November 8, 2025

Real-time cryptocurrency volatility detection system using Coinbase WebSocket API, Kafka, and MLflow.

---

## 📋 Milestone 1 Objectives

✅ Launch Kafka and MLflow infrastructure using Docker Compose  
✅ Ingest real-time Coinbase WebSocket ticker data  
✅ Implement reconnect/resubscribe and heartbeat monitoring  
✅ Stream data to Kafka topic `ticks.raw`  
✅ Validate data flow with consumer script  
✅ Define problem scope and success criteria  
✅ Containerize ingestion service  

---

## 🏗️ Project Structure

```
.
├── docker/
│   ├── docker-compose.yaml       # Infrastructure setup (Kafka, Zookeeper, MLflow)
│   └── Dockerfile.ingestor       # Containerized data ingestion service
├── scripts/
│   ├── ws_ingest.py              # WebSocket data ingestion with reconnect logic
│   └── kafka_consume_check.py    # Kafka stream validation tool
├── data/
│   └── raw/                      # Local mirror of raw ticker data (NDJSON)
├── docs/
│   └── scoping_brief.pdf         # Problem definition and success metrics
├── config.yaml                   # Configuration (optional)
├── .env                          # Environment variables (not committed)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Docker Desktop installed and running
- Python 3.9+ with pip
- Git

### 1. Clone Repository

```bash
git clone <repository-url>
cd operationaliseai
```

### 2. Start Infrastructure

```bash
cd docker
docker compose up -d
```

Verify all services are running:
```bash
docker compose ps
```

Expected output:
- ✅ `kafka` - Running on port 9092
- ✅ `zookeeper` - Running on port 2182
- ✅ `mlflow` - Running on port 5001

Access MLflow UI: http://localhost:5001

### 3. Create Kafka Topics

```bash
# Create raw ticks topic
docker exec -it kafka kafka-topics --create \
  --topic ticks.raw \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1

# Create features topic (for Milestone 2)
docker exec -it kafka kafka-topics --create \
  --topic ticks.features \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1

# Verify topics exist
docker exec -it kafka kafka-topics --list --bootstrap-server localhost:9092
```

### 4. Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create `.env` file in project root (if needed):
```bash
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
```

**Note:** For running locally, add hostname resolution:
```bash
echo "127.0.0.1 kafka" | sudo tee -a /etc/hosts
```

---

## 📊 Running Data Ingestion

### Option A: Local Execution (Development)

```bash
# From project root with virtual environment activated
python scripts/ws_ingest.py --pair BTC-USD --minutes 15 --save-disk
```

**Arguments:**
- `--pair`: Trading pair (e.g., BTC-USD, ETH-USD)
- `--minutes`: Duration to run (default: 15)
- `--save-disk`: Mirror data to `data/raw/` directory

### Option B: Docker Container (Production-like)

```bash
# Build container
docker build -f docker/Dockerfile.ingestor -t crypto-ingestor .

# Run container
docker run --rm \
  --network docker_crypto-network \
  -v $(pwd)/data:/app/data \
  -e KAFKA_BOOTSTRAP_SERVERS=kafka:9092 \
  crypto-ingestor \
  python scripts/ws_ingest.py --pair BTC-USD --minutes 15 --save-disk
```

---

## ✅ Validating Data Flow

### Check Kafka Messages

In a separate terminal:

```bash
# Validate at least 100 messages received
python scripts/kafka_consume_check.py --topic ticks.raw --min 100
```

### Inspect Raw Data Files

```bash
# View most recent data file
ls -lth data/raw/ | head -5

# Preview contents (first 5 lines)
head -5 data/raw/ticks_BTCUSD_*.ndjson
```

### Monitor with Kafka Console Consumer

```bash
docker exec -it kafka kafka-console-consumer \
  --topic ticks.raw \
  --bootstrap-server localhost:9092 \
  --from-beginning \
  --max-messages 10
```

---

## 📄 Data Format

### Raw Ticker Message Schema

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

**Fields:**
- `timestamp`: ISO 8601 capture time (UTC)
- `product_id`: Trading pair identifier
- `price`: Last trade price
- `volume_24h`: 24-hour trading volume
- `low_24h` / `high_24h`: 24-hour price range
- `best_bid` / `best_ask`: Top of order book
- `raw`: Complete Coinbase WebSocket message

---

## 🧪 Testing & Verification

### Test Checklist

- [ ] All Docker services show "Up" status
- [ ] Kafka topics created successfully
- [ ] Local ingestion runs for 15 minutes without errors
- [ ] At least 100 messages received in `ticks.raw`
- [ ] Container builds without errors
- [ ] Container runs and streams data successfully
- [ ] MLflow UI accessible at http://localhost:5001

### Run All Tests

```bash
# 1. Check infrastructure
cd docker && docker compose ps

# 2. Run ingestion (15 minutes)
python scripts/ws_ingest.py --pair BTC-USD --minutes 15 --save-disk

# 3. Validate messages (in separate terminal)
python scripts/kafka_consume_check.py --topic ticks.raw --min 100

# 4. Test container
docker build -f docker/Dockerfile.ingestor -t crypto-ingestor .
docker run --rm --network docker_crypto-network \
  -e KAFKA_BOOTSTRAP_SERVERS=kafka:9092 \
  crypto-ingestor \
  python scripts/ws_ingest.py --pair BTC-USD --minutes 2
```

---

## 🔧 Troubleshooting

### Kafka Connection Issues

**Problem:** `DNS lookup failed for kafka:9092`

**Solution:**
```bash
# Add hostname resolution (for local execution)
echo "127.0.0.1 kafka" | sudo tee -a /etc/hosts
```

### MLflow Shows "Unhealthy"

**Check if actually working:**
```bash
curl http://localhost:5001/health
# Should return: OK
```

If returns `OK`, MLflow is functional despite health check status.

### No Messages in Kafka

**Debugging steps:**
```bash
# 1. Check WebSocket connection in logs
# Look for: "WebSocket connected to wss://..."

# 2. Verify topics exist
docker exec -it kafka kafka-topics --list --bootstrap-server localhost:9092

# 3. Check Kafka producer logs for errors
```

### Port Conflicts

**Ports in use:**
- 5001: MLflow UI
- 9092: Kafka broker
- 2182: Zookeeper

**Change ports in `docker/docker-compose.yaml` if needed**

---

## 📚 Key Components

### WebSocket Ingestion (`ws_ingest.py`)

**Features:**
- ✅ Auto-reconnect on connection loss
- ✅ Heartbeat monitoring (30-second timeout)
- ✅ Graceful shutdown (Ctrl+C)
- ✅ Dual output: Kafka stream + local NDJSON files
- ✅ Structured logging

**Error Handling:**
- Exponential backoff for reconnection
- Message validation before Kafka publish
- Connection state tracking

### Kafka Consumer Validator (`kafka_consume_check.py`)

**Purpose:** Verify streaming pipeline health

**Usage:**
```bash
python scripts/kafka_consume_check.py --topic ticks.raw --min 100
```

**Output:**
- Message count validation
- Sample message preview
- Success/failure status

---

## 📖 Documentation

### Scoping Brief

See `docs/scoping_brief.pdf` for:
- Use case and business context
- 60-second volatility prediction goal
- Success metrics (PR-AUC ≥ 0.70)
- Risk assumptions and constraints
- Labeling strategy

---

## 🔐 Security Notes

- **No secrets committed:** API keys and credentials in `.env` (gitignored)
- **Public data only:** Using free Coinbase WebSocket API
- **No trading:** Analysis and detection only

---

---

## 🎯 Milestone 2: Feature Engineering & Analysis

**Status:** ✅ Complete

### Objectives Achieved

✅ Built streaming feature engineering pipeline  
✅ Implemented replay script for reproducibility  
✅ Conducted exploratory data analysis (EDA)  
✅ Selected volatility spike threshold (τ)  
✅ Generated Evidently data quality report  
✅ Documented feature specifications  

### New Components

```
features/
├── featurizer.py                 # Kafka consumer for real-time feature computation
└── __init__.py

scripts/
├── replay.py                     # Reproducibility verification script
├── generate_evidently_report.py  # Data drift monitoring
└── check_milestone2.py           # Milestone 2 verification checklist

notebooks/
└── eda.ipynb                     # Exploratory analysis & threshold selection

docs/
└── feature_spec.md               # Feature engineering documentation

data/
├── processed/
│   ├── features.parquet          # Live-computed features
│   └── features_replay.parquet   # Replay-computed features
└── reports/
    └── evidently_report.html     # Data quality & drift report
```

### Running Milestone 2 Pipeline

#### 1. Install Additional Dependencies

```bash
pip install -r requirements.txt
```

New dependencies include:
- `pandas==2.1.4` - DataFrame operations
- `pyarrow==14.0.1` - Parquet file support
- `evidently==0.4.11` - Drift monitoring
- `jupyter==1.0.0` - Notebook environment

#### 2. Run Feature Engineering Pipeline

**Real-time streaming mode:**
```bash
python features/featurizer.py --topic_in ticks.raw --topic_out ticks.features
```

This consumes from `ticks.raw`, computes rolling window features, and publishes to `ticks.features`.

**Features computed:**
- Price returns (1min, 5min rolling windows)
- Volatility (rolling standard deviation)
- Bid-ask spread dynamics
- Volume-weighted metrics
- Trade intensity indicators

Let it run for 10-15 minutes to accumulate sufficient data.

#### 3. Verify Reproducibility with Replay

```bash
python scripts/replay.py \
  --raw "data/raw/*.ndjson" \
  --out data/processed/features_replay.parquet
```

This re-processes raw data through the same feature pipeline and compares outputs to verify deterministic behavior.

**Expected output:**
```
✓ Features match between live and replay
✓ Row counts identical
✓ Reproducibility verified
```

#### 4. Run Exploratory Data Analysis

```bash
jupyter notebook notebooks/eda.ipynb
```

**Analysis includes:**
- Feature distribution visualization
- Correlation analysis
- Volatility pattern identification
- Threshold selection (90th percentile)
- Label generation strategy

**Key finding:** Volatility threshold (τ) set at [YOUR VALUE]% based on 90th percentile of rolling standard deviation.

#### 5. Generate Evidently Report

```bash
python scripts/generate_evidently_report.py
```

Generates `data/reports/evidently_report.html` with:
- Data quality metrics
- Feature drift detection
- Distribution comparisons (early vs late data windows)
- Missing value analysis
- Statistical test results

**View report:**
```bash
open data/reports/evidently_report.html
```

#### 6. Verify Milestone Completion

```bash
python scripts/check_milestone2.py
```

Checklist includes:
- ✓ Feature pipeline files present
- ✓ Processed features exist
- ✓ Replay features match live features
- ✓ EDA notebook completed
- ✓ Feature specification documented
- ✓ Evidently report generated

---

## 📊 Feature Engineering Details

### Feature Categories

**1. Price-Based Features**
- `price_return_1min` - 1-minute price return
- `price_return_5min` - 5-minute price return
- `price_volatility_5min` - Rolling standard deviation

**2. Spread Features**
- `bid_ask_spread` - Absolute spread
- `bid_ask_spread_bps` - Spread in basis points

**3. Volume Features**
- `volume_24h_pct_change` - 24-hour volume change
- `trade_intensity` - Trades per minute (estimated)

**4. Target Variable**
- `volatility_spike` - Binary label (1 = spike detected, 0 = normal)

### Labeling Strategy

**Definition of Volatility Spike:**
```python
# Look-ahead window: 60 seconds
future_volatility = rolling_std(returns, window=60s)
threshold = percentile_90(historical_volatility)
label = 1 if future_volatility >= threshold else 0
```

### Data Quality Findings

From Evidently report:
- **Missing data rate:** [X]%
- **Feature drift detected:** [Yes/No]
- **Data distribution shifts:** [Description]
- **Recommended actions:** [Retraining schedule, monitoring thresholds]

---

## 🔄 Reproducibility Verification

### Replay Testing

The replay script ensures deterministic feature computation:

```bash
# Step 1: Collect live data
python features/featurizer.py --duration 15

# Step 2: Replay same data
python scripts/replay.py --raw "data/raw/*.ndjson"

# Step 3: Compare outputs
python scripts/check_milestone2.py
```

**Success criteria:**
- Feature values match to floating-point precision
- Row counts identical between live and replay
- Timestamps align correctly

---

## 📈 Next Steps (Milestone 3)

- [ ] Train baseline model (rule-based)
- [ ] Train ML model (Logistic Regression / XGBoost)
- [ ] Log experiments to MLflow
- [ ] Implement model serving pipeline
- [ ] Set up automated retraining
- [ ] Deploy monitoring dashboard

---

## 🛠️ Technology Stack

- **Language:** Python 3.9
- **Streaming:** Apache Kafka (KRaft mode)
- **Experiment Tracking:** MLflow 2.10.2
- **Container Orchestration:** Docker Compose
- **Data Processing:** Pandas, NumPy
- **Data Formats:** NDJSON, Parquet
- **Quality Monitoring:** Evidently 0.4.11
- **Analysis:** Jupyter Notebooks

---

## 🐛 Troubleshooting Milestone 2

### Featurizer Not Processing Messages

**Check:**
```bash
# Verify Kafka topics exist
docker exec -it kafka kafka-topics --list --bootstrap-server localhost:9092

# Check raw data is flowing
python scripts/kafka_consume_check.py --topic ticks.raw --min 10
```

### Replay Features Don't Match

**Common causes:**
- Timestamp sorting issues
- Floating-point precision differences
- Missing data handling inconsistencies

**Fix:** Ensure consistent data sorting and NaN handling in both pipelines.

### Evidently Report Empty

**Check:**
```bash
# Verify features.parquet exists and has data
ls -lh data/processed/features.parquet
python -c "import pandas as pd; print(pd.read_parquet('data/processed/features.parquet').shape)"
```

Need at least 100+ rows for meaningful drift analysis.

---

# Milestone 3: Modeling, Tracking & Evaluation

**Goal:** Train baseline and ML models, track experiments with MLflow, and generate comprehensive evaluation reports.

---

## 📋 Quick Start

### Prerequisites
- Milestones 1 & 2 completed
- Features generated in `data/processed/features.parquet`
- MLflow running at http://localhost:5001

### Installation

```bash
# Install additional dependencies for Milestone 3
pip install -r requirements.txt

# Verify MLflow is accessible
curl http://localhost:5001/health
```

---

## 🎯 Milestone 3 Objectives

✅ Train baseline model (z-score rule-based)  
✅ Train ML model (Logistic Regression + optional XGBoost)  
✅ Use time-based train/val/test splits  
✅ Log experiments to MLflow  
✅ Compute PR-AUC and F1@threshold metrics  
✅ Generate model evaluation report (PDF)  
✅ Create Model Card v1  
✅ Generate Evidently drift report  
✅ Verify inference speed (< 2x real-time)  

---

## 📂 New File Structure

```
models/
├── baseline.py              # Baseline z-score detector
├── train.py                 # Training pipeline for all models
├── infer.py                 # Inference and benchmarking
└── artifacts/               # Saved models and plots
    ├── baseline/
    │   ├── model.pkl
    │   ├── pr_curve.png
    │   └── roc_curve.png
    ├── logistic_regression/
    │   ├── model.pkl
    │   ├── pr_curve.png
    │   ├── roc_curve.png
    │   └── feature_importance.png
    └── xgboost/              # Optional
        └── ...

scripts/
└── generate_eval_report.py  # PDF evaluation report generator

reports/
├── model_eval.pdf           # Comprehensive evaluation report
└── evidently_drift.html     # Data drift report (refreshed)

docs/
├── model_card_v1.md         # Model documentation
└── genai_appendix.md        # GenAI usage disclosure
```

---

## 🚀 Step-by-Step Workflow

### Step 1: Train All Models

Train baseline and ML models with MLflow tracking:

```bash
python models/train.py \
  --features data/processed/features.parquet \
  --models baseline logistic xgboost \
  --mlflow-uri http://localhost:5001
```

**What it does:**
- Loads features and creates time-based train/val/test splits
- Trains 3 models: baseline (z-score), logistic regression, XGBoost
- Logs all experiments to MLflow with parameters and metrics
- Saves models to `models/artifacts/`
- Generates PR/ROC curves for each model

**Expected output:**
```
Loading data...
Data splits:
  Train: 1500 samples (10.2% spikes)
  Val:   300 samples (9.8% spikes)
  Test:  300 samples (10.5% spikes)

=== Training Baseline Model ===
Validation PR-AUC: 0.7234
Test PR-AUC: 0.7156

=== Training Logistic Regression ===
Validation PR-AUC: 0.7892
Test PR-AUC: 0.7745

=== Training XGBoost ===
Validation PR-AUC: 0.8123
Test PR-AUC: 0.7998

=== Model Comparison (Test Set) ===
Model                PR-AUC     F1         Precision  Recall    
------------------------------------------------------------
baseline             0.7156     0.6543     0.7012     0.6134    
logistic             0.7745     0.7234     0.7456     0.7023    
xgboost              0.7998     0.7543     0.7823     0.7289    

All models logged to MLflow: http://localhost:5001
```

### Step 2: View MLflow Experiments

```bash
# MLflow should already be running from Milestone 1
# If not, start it:
cd docker && docker compose up -d mlflow

# Access MLflow UI
open http://localhost:5001
```

**In MLflow UI, verify:**
- Experiment: "crypto-volatility-detection"
- At least 2 runs (baseline + logistic, or all 3)
- Metrics: `test_pr_auc`, `test_f1_score`, `test_precision`, `test_recall`
- Artifacts: PR curves, ROC curves, saved models

### Step 3: Run Inference Benchmark

Verify inference meets <  2x real-time requirement (< 120 seconds for 60-second windows):

```bash
# Benchmark best model (e.g., logistic regression)
python models/infer.py \
  --model models/artifacts/logistic_regression/model.pkl \
  --features data/processed/features.parquet \
  --mode benchmark \
  --n-samples 1000
```

**Expected output:**
```
=== Inference Benchmark ===
Model: models/artifacts/logistic_regression/model.pkl
Requirement: < 120 seconds for 60-second prediction window
Testing with 1000 samples...

Single prediction test:
  Time: 0.23 ms
  Prediction: 0 (prob: 0.124)

Batch prediction test (1000 samples):
  Total time: 0.187 seconds
  Avg per sample: 0.19 ms
  Throughput: 5347.6 predictions/second

=== Real-Time Performance Check ===
Prediction window: 60 seconds
Maximum allowed inference time: 120 seconds
Actual batch time: 0.187 seconds

✓ PASSED: Inference is 641.7x faster than requirement

Alert rate: 10.2% (102 / 1000)
```

### Step 4: Test Live Inference Simulation

```bash
python models/infer.py \
  --model models/artifacts/logistic_regression/model.pkl \
  --features data/processed/features.parquet \
  --mode live
```

**Shows streaming predictions:**
```
=== Live Inference Simulation ===

Streaming predictions (showing first 10):
Timestamp                 Prediction   Probability  Alert    Time (ms) 
--------------------------------------------------------------------------------
2025-11-09 01:24:55       0            0.123          0.18      
2025-11-09 01:24:56       0            0.089          0.15      
2025-11-09 01:25:12       1            0.847        🚨  0.21      
...
```

### Step 5: Generate Evaluation Report

Create comprehensive PDF report:

```bash
python scripts/generate_eval_report.py \
  --features data/processed/features.parquet \
  --artifacts models/artifacts \
  --output reports/model_eval.pdf
```

**Report includes:**
- Model comparison table
- PR/ROC curves comparison
- Confusion matrices
- Feature importance plots

```bash
# View report
open reports/model_eval.pdf
```

### Step 6: Refresh Evidently Drift Report

Generate updated drift report comparing train vs test distributions:

```bash
python scripts/generate_evidently_report.py \
  --features data/processed/features.parquet \
  --output reports/evidently_drift.html
```

```bash
# View report
open reports/evidently_drift.html
```

### Step 7: Complete Model Card

Fill in the template with your actual results:

```bash
# Edit docs/model_card_v1.md
# Replace placeholders like [INSERT VALUE] with actual numbers from MLflow and reports
```

**Key sections to complete:**
- Training date and metrics
- Test set performance (PR-AUC, F1, precision, recall)
- Data statistics (samples, spike rate, time range)
- Threshold value (τ)
- Inference latency measurements

### Step 8: Document GenAI Usage

Update `docs/genai_appendix.md` with honest assessment of AI assistance used.

---

## 📊 Key Deliverables Checklist

### Code Files
- [x] `models/baseline.py` - Baseline z-score detector
- [x] `models/train.py` - Training pipeline
- [x] `models/infer.py` - Inference module
- [x] `scripts/generate_eval_report.py` - Report generator

### Trained Models
- [x] `models/artifacts/baseline/model.pkl`
- [x] `models/artifacts/logistic_regression/model.pkl`
- [x] `models/artifacts/xgboost/model.pkl` (optional)

### Reports & Documentation
- [x] `reports/model_eval.pdf` - Performance evaluation
- [x] `reports/evidently_drift.html` - Data drift analysis
- [x] `docs/model_card_v1.md` - Model documentation
- [x] `docs/genai_appendix.md` - AI usage disclosure

### MLflow Tracking
- [x] At least 2 logged runs (baseline + ML)
- [x] Metrics: PR-AUC, F1, Precision, Recall
- [x] Artifacts: Plots and models

---

## 🧪 Testing & Validation

### Verify MLflow Logging

```bash
# Check runs exist
mlflow experiments list
mlflow runs list --experiment-id 0

# Or use Python
python -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5001')
client = mlflow.tracking.MlflowClient()
runs = client.search_runs(experiment_ids=['0'])
print(f'Found {len(runs)} runs')
for run in runs:
    print(f\"  {run.info.run_name}: PR-AUC = {run.data.metrics.get('test_pr_auc', 'N/A')}\")
"
```

### Verify Inference Performance

**Requirements:**
- Single prediction: < 10ms (recommended)
- Batch 1000 samples: < 1 second (recommended)
- Overall: < 2x real-time (< 120 seconds for 60-second windows) **REQUIRED**

```bash
# Quick test
python models/infer.py \
  --model models/artifacts/logistic_regression/model.pkl \
  --features data/processed/features.parquet \
  --mode benchmark \
  --n-samples 100
```

### Verify Report Quality

**Model evaluation PDF should include:**
- ✓ Title page with date
- ✓ Metrics comparison table with PR-AUC highlighted
- ✓ PR curves for all models
- ✓ ROC curves for all models
- ✓ Confusion matrices
- ✓ Feature importance (for applicable models)

---

## 📈 Expected Results

### Performance Targets

| Metric | Baseline Target | ML Target | Stretch Goal |
|--------|----------------|-----------|--------------|
| PR-AUC | ≥ 0.60 | ≥ 0.70 | ≥ 0.80 |
| F1-Score | ≥ 0.50 | ≥ 0.65 | ≥ 0.75 |
| Inference Time | < 120s | < 10s | < 1s |

### Typical Results

Based on 2,000-5,000 feature samples with ~10% spike rate:

**Baseline (Z-Score):**
- PR-AUC: 0.65-0.72
- F1-Score: 0.55-0.65
- Simple, fast, interpretable

**Logistic Regression:**
- PR-AUC: 0.72-0.80
- F1-Score: 0.65-0.75
- Good balance of performance and speed

**XGBoost:**
- PR-AUC: 0.75-0.82
- F1-Score: 0.70-0.78
- Best performance, slightly slower

---

## 🐛 Troubleshooting

### MLflow Connection Issues

```bash
# Check if MLflow is running
docker ps | grep mlflow

# Restart if needed
cd docker && docker compose restart mlflow

# Test connection
curl http://localhost:5001/health
```

### Model Training Fails

**Issue:** "Not enough data" or "No positive samples"

**Solution:**
```bash
# Check feature data
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/features.parquet')
print(f'Total samples: {len(df)}')
print(f'Spike rate: {df[\"volatility_spike\"].mean():.2%}')
print(f'Missing values: {df.isnull().sum().sum()}')
"

# Need at least 500+ samples with 5-15% spike rate
```

### Inference Benchmark Fails

**Issue:** "KeyError: 'price_volatility_5min'"

**Solution:** Baseline model only needs volatility feature, handled automatically in code. Check that features.parquet has all required columns.

### PDF Report Generation Fails

**Issue:** Missing matplotlib backend

**Solution:**
```bash
pip install matplotlib --upgrade
# or
pip install reportlab
```

---

## 🎓 Learning Outcomes

After completing Milestone 3, you should understand:

1. **Model Training Workflows**
   - Time-based train/val/test splits
   - Handling class imbalance
   - Hyperparameter selection

2. **Experiment Tracking**
   - Logging parameters, metrics, artifacts to MLflow
   - Comparing multiple model runs
   - Model versioning

3. **Model Evaluation**
   - PR-AUC vs ROC-AUC for imbalanced data
   - Precision-recall tradeoffs
   - Confusion matrix interpretation

4. **Production Considerations**
   - Inference latency requirements
   - Real-time performance validation
   - Model documentation (Model Cards)

5. **Monitoring & Drift**
   - Data distribution shifts
   - Model performance degradation
   - Retraining triggers

---

## 📚 Next Steps (Post-Milestone 3)

- **Deployment:** Containerize inference service
- **API:** Build REST API for real-time predictions
- **Monitoring Dashboard:** Visualize live predictions and alerts
- **Automated Retraining:** Schedule weekly model updates
- **A/B Testing:** Compare model versions in production
- **Alert System:** Send notifications when spikes detected

---

## 📞 Support

**Common Issues:**
1. MLflow not accessible → Check Docker containers
2. Models not training → Verify feature data exists
3. Inference too slow → Use simpler model or optimize features
4. Poor PR-AUC → Adjust threshold, add features, or collect more data

**Resources:**
- MLflow Docs: https://mlflow.org/docs/latest/index.html
- Evidently Docs: https://docs.evidentlyai.com/
- Model Cards Paper: https://arxiv.org/abs/1810.03993


## 📞 Support

For issues or questions:
1. Check troubleshooting sections above
2. Review Docker logs: `docker compose logs <service-name>`
3. Verify environment configuration
4. Run verification script: `python scripts/check_milestone2.py`
5. Consult course materials

---

## 📝 License

Educational project for Operationalize AI course.

---

## 📅 Project Timeline

**Milestone 1:** ✅ Complete (November 8, 2025) - Streaming infrastructure  
**Milestone 2:** ✅ Complete (November [DATE], 2025) - Feature engineering & analysis  
**Milestone 3:** 🔄 In Progress - Model training & deployment