# Handoff Summary

**Decision Type:** Selected-base

**Selected Model:** XGBoost (Stratified Split)
- Model Path: `models/artifacts/xgboost_stratified/model.pkl`
- Performance: PR-AUC 0.7815, Recall 97.31%, Precision 52.87%

## Quick Start (3 Steps)

1. **Setup Infrastructure:**
   ```bash
   cd docker
   cp .env.example .env
   docker compose up -d
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Inference:**
   ```bash
   python -c "
   import pickle
   import pandas as pd
   model = pickle.load(open('models/artifacts/xgboost_stratified/model.pkl', 'rb'))
   features = pd.read_parquet('data/processed/features_sample.parquet')
   # ... (see README.md for full code)
   "
   ```

## All Required Files Present ✅

- ✅ docker/compose.yaml
- ✅ docker/Dockerfile.ingestor  
- ✅ docker/.env.example
- ✅ docs/feature_spec.md
- ✅ docs/model_card_v1.md
- ✅ models/artifacts/xgboost_stratified/model.pkl
- ✅ requirements.txt
- ✅ data/raw/ (10-minute slice)
- ✅ data/processed/features_sample.parquet
- ✅ reports/model_eval.pdf
- ✅ reports/evidently/train_test_drift_report.html
- ✅ predictions.parquet

See README.md for complete integration steps.
