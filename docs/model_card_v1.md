# Model Card: Crypto Volatility Detection v1.1

**Date:** November 13, 2025  
**Author:** Melissa Wong  
**Project:** Real-Time Cryptocurrency Volatility Detection

---

## Model Details

### Model Description
This model predicts short-term volatility spikes in cryptocurrency markets (specifically BTC-USD) using real-time tick data from Coinbase. The model predicts whether significant price volatility will occur in the next 60 seconds.

**Model Type:** XGBoost (Best Performing), Logistic Regression, Baseline  
**Framework:** XGBoost, scikit-learn  
**Version:** 1.1  
**Training Date:** November 13, 2025

### Model Architecture

**Best Model: XGBoost (Stratified Split)**
- **Input Features:** 10 engineered features from real-time tick data
  - **Log Return Volatility:** `log_return_std_30s`, `log_return_std_60s`, `log_return_std_300s`
  - **Return Statistics:** `return_mean_60s`, `return_mean_300s`, `return_min_30s`
  - **Spread Volatility:** `spread_std_300s`, `spread_mean_60s`
  - **Trade Intensity:** `tick_count_60s`
  - **Derived:** `return_range_60s` (return_max - return_min)

- **Output:** Binary classification (0 = normal volatility, 1 = spike)

- **Training Details:**
  - Algorithm: XGBoost Gradient Boosting
  - Class balancing: Applied via `scale_pos_weight`
  - Hyperparameters:
    - max_depth: 5
    - learning_rate: 0.1
    - n_estimators: 100
    - objective: binary:logistic
    - eval_metric: aucpr
    - random_state: 42
  - **Data Split:** Stratified (balanced spike rates across train/val/test)

**Alternative Models:**
- **Logistic Regression:** L2 regularization, class_weight='balanced', max_iter=1000
- **Baseline:** Composite z-score across 8 features (DEFAULT_FEATURES)

---

## Intended Use

### Primary Use Case
Real-time detection of cryptocurrency volatility spikes to enable:
- **Risk Management:** Early warning system for traders
- **Trading Strategy Triggers:** Signal generation for algorithmic trading
- **Market Monitoring:** Surveillance and anomaly detection

### Target Users
- Cryptocurrency traders (retail and institutional)
- Risk management teams
- Market makers and liquidity providers
- Researchers analyzing market dynamics

### Out-of-Scope Use Cases
- **Not for automated trading:** This model is for detection/alerting only; human oversight required
- **Not for other assets:** Model trained specifically on BTC-USD; not validated for other cryptocurrencies or financial instruments
- **Not for long-term prediction:** Designed for 60-second horizon only
- **Not production-ready:** This is v1.0 for educational purposes; requires further validation for production deployment

---

## Training Data

### Data Source
- **API:** Coinbase Advanced Trade WebSocket (public ticker channel)
- **Trading Pair:** BTC-USD
- **Collection Period:** November 8-9, 2025 (15:12:31 - 01:25:17 UTC)
- **Total Samples:** 52,524 feature samples (after windowing and feature computation)

### Data Splits

**Time-Based Split (Default):**
- **Training:** 70% (~6,740 samples, 6.60% spike rate)
- **Validation:** 15% (~1,444 samples, 2.42% spike rate)
- **Test:** 15% (~1,445 samples, 33.43% spike rate)

**Stratified Split (Balanced):**
- **Training:** 70% (~6,740 samples, 10.0% spike rate)
- **Validation:** 15% (~1,444 samples, 10.0% spike rate)
- **Test:** 15% (~1,445 samples, 10.0% spike rate)

*Note: Stratified splitting balances spike rates across splits, improving model performance by reducing temporal clustering effects.*

### Labeling Strategy
**Definition of Volatility Spike:**
- Look-ahead window: 60 seconds
- Metric: Rolling standard deviation of price returns
- Threshold (τ): 0.000026 (90th percentile of historical distribution)
- Label = 1 if future volatility ≥ τ, else 0

**Class Balance:**
- Negative samples (normal): 90.0%
- Positive samples (spike): 10.0%

### Data Quality
- **Missing values:** 0.01% (filled with 0)
- **Outliers:** Handled through feature normalization
- **Data drift:** Monitored using Evidently reports

---

## Evaluation

### Metrics

**Primary Metric: PR-AUC (Precision-Recall Area Under Curve)**

**Best Model: XGBoost (Stratified Split)**
- **Validation:** 0.6126
- **Test:** 0.7815

**Secondary Metrics (Test Set - XGBoost Stratified):**
| Metric | Value | Interpretation |
|--------|-------|----------------|
| Precision | 0.5287 | Of predicted spikes, 52.87% are true spikes |
| Recall | 0.9731 | Of true spikes, 97.31% are detected |
| F1-Score | 0.6851 | Harmonic mean of precision and recall |
| ROC-AUC | [See MLflow] | Overall discrimination ability |
| Accuracy | [See MLflow] | Overall correct predictions |

**Confusion Matrix (Test Set):**
```
                Predicted Negative    Predicted Positive
Actual Negative        [TN]                [FP]
Actual Positive        [FN]                [TP]
```
*Note: Full confusion matrix available in MLflow runs*

### Model Comparison

**Time-Based Split (Default):**
| Model | PR-AUC (Test) | F1-Score | Precision | Recall |
|-------|--------|----------|-----------|--------|
| Baseline (Z-Score) | 0.2881 | 0.0000 | 0.0000 | 0.0000 |
| Logistic Regression | 0.2549 | 0.4241 | 0.3100 | 0.6708 |
| XGBoost | 0.7359 | 0.3994 | 0.8741 | 0.2588 |

**Stratified Split (Balanced Spike Rates):**
| Model | PR-AUC (Test) | F1-Score | Precision | Recall |
|-------|--------|----------|-----------|--------|
| Baseline (Z-Score) | 0.2295 | 0.1694 | 0.1356 | 0.2257 |
| Logistic Regression | 0.2491 | 0.5013 | 0.3635 | 0.8075 |
| XGBoost | **0.7815** | **0.6851** | **0.5287** | **0.9731** |

**Key Findings:**
- **XGBoost (Stratified):** Best overall performance with PR-AUC 0.7815, achieving 97.31% recall and 52.87% precision. Excellent for spike detection use cases requiring high sensitivity.
- **XGBoost (Time-Based):** High precision (87.41%) but lower recall (25.88%), suitable for precision-focused applications. PR-AUC 0.7359.
- **Logistic Regression:** Moderate performance with balanced precision/recall. Stratified split improves recall to 80.75% (vs 67.08% time-based).
- **Baseline:** Composite z-score approach (8 features) shows improved performance with stratified split (F1: 0.1694 vs 0.0000 time-based), but still underperforms ML models.

**Impact of Stratified Splitting:**
Stratified splitting (balancing spike rates across train/val/test) significantly improves model performance, especially for XGBoost:
- XGBoost PR-AUC: 0.7359 → 0.7815 (+6.2%)
- XGBoost Recall: 25.88% → 97.31% (+275% relative improvement)
- Logistic Regression Recall: 67.08% → 80.75% (+20% relative improvement)

This suggests temporal clustering of spikes in the original time-based split was hurting model performance.

**Feature Set:** All models trained with reduced feature set (10 features) to minimize multicollinearity. Features include log return volatility (log_return_std_30s/60s/300s), return statistics (return_mean_60s/300s, return_min_30s), spread volatility (spread_std_300s, spread_mean_60s), trade intensity (tick_count_60s), and derived feature (return_range_60s). Baseline model uses composite z-score across 8 features matching DEFAULT_FEATURES.

### Performance Requirements
- **Latency:** Inference must complete in < 120 seconds (2x real-time for 60-second windows)
- **Actual Performance:** See `models/infer.py` benchmark results
- **Status:** ✓ Meets requirement (inference completes in milliseconds, well under 2x real-time)

---

## Ethical Considerations

### Potential Harms
1. **Financial Risk:** False negatives (missed spikes) could lead to unexpected losses if users rely solely on model predictions
2. **False Alarms:** False positives may cause unnecessary concern or suboptimal trading decisions
3. **Market Manipulation:** Model could potentially be gamed if architecture is public
4. **Over-Reliance:** Users may over-trust automated predictions without understanding limitations

### Mitigation Strategies
1. Model outputs are alerts/signals only; not automated trading actions
2. Human-in-the-loop required for all trading decisions
3. Continuous monitoring for data drift and model degradation
4. Regular retraining on recent data
5. Clear documentation of limitations and uncertainty

### Fairness & Bias
- **Market Access:** Public Coinbase data accessible to all users; no privileged information
- **Transparency:** Model architecture and features are documented
- **No PII:** No personally identifiable information used

---

## Limitations

### Technical Limitations
1. **Market Regime Changes:** Model trained on specific market conditions; performance may degrade during unprecedented events (black swans, exchange outages)
2. **Data Latency:** Assumes Coinbase API latency < 1 second; actual latency may impact performance
3. **Feature Drift:** Market microstructure can change over time, requiring model retraining
4. **Sample Imbalance:** Volatility spikes are rare events (~10% of data); model may be conservative

### Known Failure Modes
1. **Low Liquidity Periods:** Weekends and holidays have lower trading activity; model less reliable
2. **Flash Crashes:** Extreme, rapid price movements may not be captured in 60-second windows
3. **Exchange Issues:** Coinbase downtime or data quality issues directly impact predictions
4. **Cascading Volatility:** Multiple spikes in quick succession may not be handled well

### Recommended Use
- Use as one signal among many in trading strategy
- Combine with other technical indicators and fundamental analysis
- Implement position sizing and stop-losses
- Monitor model performance continuously
- Retrain weekly or when drift detected

---

## Maintenance & Monitoring

### Retraining Schedule
- **Frequency:** Weekly (recommended)
- **Trigger Conditions:**
  - PR-AUC drops below [INSERT THRESHOLD]
  - Evidently drift report shows significant distribution shift
  - Major market events (e.g., regulatory changes, exchange incidents)

### Monitoring Plan
1. **Real-Time Metrics:**
   - Inference latency
   - Prediction distribution (spike rate)
   - Alert frequency

2. **Batch Metrics (Daily):**
   - PR-AUC on recent data
   - Precision/Recall trends
   - Feature distributions

3. **Weekly Analysis:**
   - Evidently drift reports
   - False positive/negative analysis
   - Model performance by time of day and day of week

### Incident Response
- **High False Positive Rate:** Increase decision threshold or retrain
- **High False Negative Rate:** Decrease threshold, add features, or retrain
- **Latency Issues:** Optimize inference code or simplify model
- **Data Quality Issues:** Implement data validation and fallback strategies

---

## Model Lineage

### Training Environment
- **MLflow Tracking URI:** http://localhost:5001
- **Experiment Name:** crypto-volatility-detection
- **Run IDs:** 
  - Baseline: See MLflow UI for latest run
  - Logistic Regression: See MLflow UI for latest run
- **Git Commit:** [Run `git rev-parse HEAD` to get current commit]

### Artifacts
- Model file: `models/artifacts/[model_name]/model.pkl`
- Training script: `models/train.py`
- Features: `data/processed/features.parquet`
- Evaluation plots: `models/artifacts/[model_name]/pr_curve.png`, `roc_curve.png`

### Dependencies
```
Python: 3.9+
scikit-learn: 1.3.0
pandas: 2.1.4
numpy: 1.26.2
mlflow: 2.9.2
xgboost: 2.0.0 (if applicable)
```

---

## References

1. Coinbase Advanced Trade API Documentation
2. Evidently AI - Data and ML Model Monitoring
3. [Your scoping brief reference]
4. [Any relevant papers on volatility prediction]

---

## Changelog

### v1.1 (November 13, 2025)
- **Major Performance Improvement:** Fixed future volatility calculation (chunk-aware, forward-looking)
- **Best Model:** XGBoost with stratified split achieves PR-AUC 0.7815, Recall 97.31%, Precision 52.87%
- **Stratified Splitting:** Implemented balanced train/val/test splits, improving XGBoost PR-AUC from 0.7359 to 0.7815
- **Baseline Update:** Composite z-score across 8 features (matching DEFAULT_FEATURES)
- **Evaluation:** Both time-based and stratified splits available; stratified recommended for better performance

### v1.0 (November 9, 2025)
- Initial model release with reduced feature set (10 features) to minimize multicollinearity
- Baseline: Z-score rule-based detector (PR-AUC: 0.3149)
- Logistic Regression: PR-AUC 0.2449, Precision 17.73%, Recall 62.18% (+6.6% improvement after feature reduction)
- XGBoost: PR-AUC 0.2323, Precision 30.59%, Recall 22.39%
- Features: 10 engineered features including log return volatility (log_return_std_30s/60s/300s), return statistics, spread volatility, and trade intensity. Removed perfectly correlated features (return_std_* and log_return_mean_*).
- Evaluation: Time-based train/val/test split (70/15/15)

---

## Contact

**Maintainer:** Melissa Wong  
**Course:** Operationalize AI  
**Institution:** [Your Institution]

For questions or issues, contact: [your-email@example.com]

---

**Model Card Template:** Adapted from Mitchell et al. (2019) - "Model Cards for Model Reporting"
