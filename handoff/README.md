# Team Handoff Package

**Project:** Crypto Volatility Detection  
**Author:** Melissa Wong  
**Date:** November 13, 2025  
**Version:** 1.1

**Decision:** **Selected-base** (XGBoost Stratified Split Model)

---

## Quick Summary

This handoff package contains everything needed to integrate the crypto volatility detection model into your team project.

**Selected Model:** XGBoost (Stratified Split) - PR-AUC 0.7815, Recall 97.31%, Precision 52.87%

**Package Includes:**
- ✅ Docker infrastructure (compose.yaml, Dockerfile.ingestor, .env.example)
- ✅ Complete documentation (feature_spec.md, model_card_v1.md)
- ✅ Trained models (XGBoost stratified + alternatives)
- ✅ 10-minute raw data slice + features
- ✅ Evaluation reports (model_eval.pdf, Evidently report)
- ✅ Sample predictions file
- ✅ Exact integration steps below

---

## Package Contents

### Docker & Infrastructure
- `docker/compose.yaml` - Docker Compose configuration (Kafka, Zookeeper, MLflow)
- `docker/Dockerfile.ingestor` - Containerized data ingestion service

### Documentation
- `docs/feature_spec.md` - Feature specification and labeling strategy (v1.1)
- `docs/model_card_v1.md` - Model documentation with performance metrics (v1.1)
- `docs/genai_appendix.md` - GenAI usage disclosure

### Models & Artifacts
- `models/artifacts/` - Trained models (baseline, logistic regression, XGBoost)
  - **Time-based split models:** `baseline/`, `logistic_regression/`, `xgboost/`
  - **Stratified split models:** `baseline_stratified/`, `logistic_regression_stratified/`, `xgboost_stratified/`
  - Model files (`.pkl`)
  - Evaluation plots (PR curves, ROC curves, feature importance)

### Scripts (Reference - see main repo)
- `scripts/generate_eval_report.py` - Generate comprehensive PDF evaluation report
- `scripts/generate_evidently_report.py` - Generate data drift reports
- `scripts/replay.py` - Reproducibility verification
- `scripts/add_labels.py` - Add volatility spike labels to features

### Data
- `data/raw/` - 10-minute raw data slice (NDJSON format)
- `data/processed/features_sample.parquet` - Corresponding features for raw data slice

### Reports
- `reports/model_eval.pdf` - Model evaluation report with metrics and comparisons
- `reports/evidently/train_test_drift_report.html` - Data drift analysis (train vs test)

### Predictions
- `predictions.parquet` - Sample predictions on test set

### Dependencies
- `requirements.txt` - Python package dependencies

---

## Model Selection: **Selected-base**

**Decision:** Use XGBoost model trained with stratified split as the base for team integration.

**Type:** Selected-base (not composite)

**Rationale:**
- **Best Performance:** XGBoost (stratified) achieves PR-AUC 0.7815 with 97.31% recall and 52.87% precision
- **High Sensitivity:** Detects 97.31% of volatility spikes (excellent for alerting systems)
- **Balanced Performance:** Good precision (52.87%) reduces false alarm rate
- **Stratified Splitting:** Balanced spike rates across train/val/test improve model generalization
- **Feature Engineering:** Well-documented, chunk-aware label creation ensures correct forward-looking volatility
- **Production-Ready:** All infrastructure (Docker, Kafka, MLflow) is containerized and ready

**Model Performance (XGBoost Stratified - Test Set):**
- **PR-AUC:** 0.7815 (excellent for imbalanced data)
- **Recall:** 97.31% (detects nearly all spikes)
- **Precision:** 52.87% (good balance - when it predicts spike, usually correct)
- **F1-Score:** 0.6851
- **ROC-AUC:** See MLflow for details

**Alternative Models Available:**
- **XGBoost (Time-Based):** PR-AUC 0.7359, Precision 87.41%, Recall 25.88% (high precision, lower recall)
- **Logistic Regression (Stratified):** PR-AUC 0.2491, Recall 80.75% (more interpretable, lower performance)
- **Baseline (Stratified):** PR-AUC 0.2295 (rule-based, good for comparison)

---

## Exact Integration Steps

### Step 1: Setup Infrastructure
```bash
cd handoff/docker
cp .env.example .env  # Edit .env if needed (usually no changes required)
docker compose up -d

# Verify services are running
docker compose ps
# Expected: kafka, zookeeper, mlflow all "Up"
```

**Services:**
- Kafka: localhost:9092
- Zookeeper: localhost:2182
- MLflow: http://localhost:5001

### Step 2: Install Dependencies
```bash
# From handoff directory root
pip install -r requirements.txt
```

### Step 3: Load Model and Run Inference
```bash
# From handoff directory root
# Recommended: XGBoost stratified (best performance)
python -c "
import pickle
import pandas as pd
import numpy as np

# Load model
with open('models/artifacts/xgboost_stratified/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load features
features = pd.read_parquet('data/processed/features_sample.parquet')

# Prepare features (same as training)
feature_cols = ['log_return_std_30s', 'log_return_std_60s', 'log_return_std_300s',
                'return_mean_60s', 'return_mean_300s', 'return_min_30s',
                'spread_std_300s', 'spread_mean_60s', 'tick_count_60s']

# Add derived feature
if 'return_max_60s' in features.columns and 'return_min_60s' in features.columns:
    features['return_range_60s'] = features['return_max_60s'] - features['return_min_60s']
    feature_cols.append('return_range_60s')

X = features[[col for col in feature_cols if col in features.columns]].fillna(0)

# Make predictions
predictions = model.predict(X)
probabilities = model.predict_proba(X)[:, 1]

print(f'Predictions: {predictions.sum()} spikes out of {len(predictions)} samples')
print(f'Spike rate: {predictions.mean():.2%}')
print(f'Average probability: {probabilities.mean():.4f}')
"

# Or use the inference script (from main repo)
# python models/infer.py --model models/artifacts/xgboost_stratified/model.pkl --features data/processed/features_sample.parquet
```

### Step 4: Verify Integration
```bash
# Check model file exists
ls -lh models/artifacts/xgboost_stratified/model.pkl

# Check predictions file
python -c "import pandas as pd; df = pd.read_parquet('predictions.parquet'); print(f'Predictions: {len(df)} samples'); print(df.head())"

# View reports
open reports/model_eval.pdf
open reports/evidently/train_test_drift_report.html
```

### Step 5: View Results
- **MLflow UI:** http://localhost:5001 (experiment: `crypto-volatility-detection-stratified`)
- **Evidently Report:** `reports/evidently/train_test_drift_report.html`
- **Model Evaluation:** `reports/model_eval.pdf`

---

## Key Files for Team Integration

1. **Feature Pipeline:** `features/featurizer.py` (main repo)
   - Chunk-aware label creation with `--add-labels` flag
   - Forward-looking volatility calculation (60-second horizon)
   
2. **Training Scripts:** 
   - `models/train.py` - Time-based split (default)
   - `models/train_stratified.py` - Stratified split (recommended, see main repo)

3. **Inference Script:** `models/infer.py` (main repo)

4. **Best Model:** `models/artifacts/xgboost_stratified/model.pkl`
   - Alternative: `models/artifacts/logistic_regression_stratified/model.pkl` (more interpretable)

5. **Feature Spec:** `docs/feature_spec.md` (v1.1)

6. **Evaluation Scripts:** (main repo)
   - `scripts/generate_eval_report.py` - Generate PDF evaluation report
   - `scripts/generate_evidently_report.py` - Generate data drift reports

---

## Important Notes

### Feature Set
- **Model uses 10 features** (reduced from full set to minimize multicollinearity)
- Features: log return volatility (30s/60s/300s), return statistics, spread metrics, trade intensity
- Derived feature: `return_range_60s` (return_max - return_min)

### Label Creation
- **Threshold:** 90th percentile (configurable)
- **Chunk-aware:** Volatility calculated only within continuous data segments (gaps >300s define boundaries)
- **Forward-looking:** 60-second horizon, correctly implemented to avoid look-ahead bias
- **Spike rate:** ~10% in labeled data (when balanced via stratified split)

### Data Splitting
- **Time-based split:** Default, maintains temporal order but causes spike rate imbalance
- **Stratified split:** Recommended, balances spike rates (10% across all splits) and improves performance
- **Impact:** Stratified splitting improved XGBoost PR-AUC from 0.7359 to 0.7815

### Model Selection
- **Best Performance:** XGBoost (stratified) - PR-AUC 0.7815, Recall 97.31%
- **High Precision:** XGBoost (time-based) - Precision 87.41%, Recall 25.88%
- **Interpretable:** Logistic Regression (stratified) - PR-AUC 0.2491, Recall 80.75%

### Monitoring
- All metrics logged to MLflow at http://localhost:5001
- Data drift monitoring via Evidently reports
- Model evaluation reports available in `reports/model_eval.pdf`

---

## Contact

For questions about this handoff package, refer to:
- Model Card: `docs/model_card_v1.md`
- Feature Spec: `docs/feature_spec.md`
- GenAI Usage: `docs/genai_appendix.md`

