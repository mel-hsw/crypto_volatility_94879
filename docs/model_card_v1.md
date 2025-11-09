# Model Card: Crypto Volatility Detection v1.0

**Date:** [INSERT DATE]  
**Author:** Melissa Wong  
**Project:** Real-Time Cryptocurrency Volatility Detection

---

## Model Details

### Model Description
This model predicts short-term volatility spikes in cryptocurrency markets (specifically BTC-USD) using real-time tick data from Coinbase. The model predicts whether significant price volatility will occur in the next 60 seconds.

**Model Type:** [Logistic Regression / XGBoost]  
**Framework:** scikit-learn / XGBoost  
**Version:** 1.0  
**Training Date:** [INSERT DATE]

### Model Architecture
- **Input Features:** 6 engineered features from real-time tick data
  - `price_return_1min`: 1-minute price return
  - `price_return_5min`: 5-minute price return
  - `price_volatility_5min`: Rolling 5-minute volatility (std dev of returns)
  - `bid_ask_spread`: Absolute bid-ask spread
  - `bid_ask_spread_bps`: Spread in basis points
  - `volume_24h_pct_change`: 24-hour volume percentage change

- **Output:** Binary classification (0 = normal volatility, 1 = spike)

- **Training Details:**
  - Algorithm: [Logistic Regression with L2 regularization / XGBoost]
  - Class balancing: Applied (class_weight='balanced')
  - Hyperparameters: [INSERT KEY PARAMS]

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
- **Collection Period:** [INSERT DATES]
- **Total Samples:** [INSERT NUMBER] ticks
- **Feature Samples:** [INSERT NUMBER] (after windowing and feature computation)

### Data Splits (Time-Based)
- **Training:** 70% ([INSERT NUMBER] samples, [INSERT %] spike rate)
- **Validation:** 15% ([INSERT NUMBER] samples, [INSERT %] spike rate)
- **Test:** 15% ([INSERT NUMBER] samples, [INSERT %] spike rate)

### Labeling Strategy
**Definition of Volatility Spike:**
- Look-ahead window: 60 seconds
- Metric: Rolling standard deviation of price returns
- Threshold (τ): [INSERT VALUE] (90th percentile of historical distribution)
- Label = 1 if future volatility ≥ τ, else 0

**Class Balance:**
- Negative samples (normal): [INSERT %]
- Positive samples (spike): [INSERT %]

### Data Quality
- **Missing values:** [INSERT %] (filled with 0)
- **Outliers:** Handled through feature normalization
- **Data drift:** Monitored using Evidently reports

---

## Evaluation

### Metrics

**Primary Metric: PR-AUC (Precision-Recall Area Under Curve)**
- **Validation:** [INSERT VALUE]
- **Test:** [INSERT VALUE]

**Secondary Metrics (Test Set):**
| Metric | Value | Interpretation |
|--------|-------|----------------|
| Precision | [INSERT] | Of predicted spikes, what % are true spikes |
| Recall | [INSERT] | Of true spikes, what % are detected |
| F1-Score | [INSERT] | Harmonic mean of precision and recall |
| ROC-AUC | [INSERT] | Overall discrimination ability |
| Accuracy | [INSERT] | Overall correct predictions |

**Confusion Matrix (Test Set):**
```
                Predicted Negative    Predicted Positive
Actual Negative        [TN]                [FP]
Actual Positive        [FN]                [TP]
```

### Baseline Comparison

| Model | PR-AUC | F1-Score | Precision | Recall |
|-------|--------|----------|-----------|--------|
| Baseline (Z-Score) | [INSERT] | [INSERT] | [INSERT] | [INSERT] |
| [Your ML Model] | [INSERT] | [INSERT] | [INSERT] | [INSERT] |
| **Improvement** | [INSERT]% | [INSERT]% | [INSERT]% | [INSERT]% |

**Key Finding:** [INSERT - e.g., "The ML model achieves 15% higher PR-AUC than the baseline, demonstrating improved spike detection capability."]

### Performance Requirements
- **Latency:** Inference must complete in < 120 seconds (2x real-time for 60-second windows)
- **Actual Performance:** [INSERT] ms average inference time
- **Status:** ✓ Meets requirement ([INSERT]x faster than deadline)

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
- **Run ID:** [INSERT MLFLOW RUN ID]
- **Git Commit:** [INSERT GIT HASH]

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

### v1.0 (YYYY-MM-DD)
- Initial model release
- Baseline: Z-score rule-based detector
- ML Model: [Logistic Regression / XGBoost]
- Features: 6 engineered features from tick data
- Evaluation: Time-based train/val/test split

---

## Contact

**Maintainer:** Melissa Wong  
**Course:** Operationalize AI  
**Institution:** [Your Institution]

For questions or issues, contact: [your-email@example.com]

---

**Model Card Template:** Adapted from Mitchell et al. (2019) - "Model Cards for Model Reporting"
