# Feature Specification

## Project: Real-Time Crypto Volatility Detection

**Date:** November 9, 2025  
**Author:** Melissa Wong

---

## 1. Problem Definition

**Use Case:** Detect short-term volatility spikes in cryptocurrency markets to enable proactive risk management and trading decisions.

**Prediction Goal:** Predict whether a volatility spike will occur in the next 60 seconds based on real-time market data.

---

## 2. Target Variable

### Target Horizon
**60 seconds** - We predict volatility in the next minute.

### Volatility Proxy
**Rolling standard deviation of midprice returns** over the future 60-second window.

Mathematically:
```
σ_future = std(r_t+1, r_t+2, ..., r_t+n)
where:
  r_i = (price_i - price_{i-1}) / price_{i-1}
  n = number of ticks in 60 seconds
```

### Label Definition
Binary classification:
```
label = 1  if σ_future >= τ  (volatility spike)
label = 0  if σ_future < τ   (normal conditions)
```

### Chosen Threshold (τ)
**Value:** `0.000026` (from EDA analysis, 90th percentile)

**Justification:**
- Selected at the **90th percentile** of observed future volatility
- Based on percentile analysis in EDA (see `notebooks/eda.ipynb`)
- This threshold captures the top 10% of volatile periods
- Results in exactly **10.0%** positive class (spikes) - 5,251 out of 52,524 samples

**Trade-offs:**
- Higher threshold → fewer false positives, but might miss moderate spikes
- Lower threshold → more sensitivity, but higher false alarm rate
- Current threshold balances detection rate with actionable signal quality

---

## 3. Features

### 3.1 Raw Features

| Feature | Description | Type | Source |
|---------|-------------|------|--------|
| `timestamp` | Event timestamp | datetime | Coinbase WebSocket |
| `product_id` | Trading pair (e.g., BTC-USD) | string | Coinbase WebSocket |
| `price` | Midprice: (best_bid + best_ask) / 2 | float | Computed |
| `best_bid` | Best bid price | float | Coinbase WebSocket |
| `best_ask` | Best ask price | float | Coinbase WebSocket |
| `spread` | Bid-ask spread (absolute) | float | Computed |
| `spread_bps` | Bid-ask spread in basis points | float | Computed |

### 3.2 Windowed Features

Features are computed over rolling windows: **60s (1min) and 300s (5min)**

#### Features Used in Model (10 features)

The current model uses a reduced feature set to minimize multicollinearity:

| Feature Name | Description | Window | Type |
|--------------|-------------|--------|------|
| `log_return_std_30s` | 30-second log return volatility | 30s | float |
| `log_return_std_60s` | 60-second log return volatility | 60s | float |
| `log_return_std_300s` | 300-second log return volatility | 300s | float |
| `return_mean_60s` | Mean return over 60-second window | 60s | float |
| `return_mean_300s` | Mean return over 300-second (5min) window | 300s | float |
| `return_min_30s` | Minimum return in 30-second window | 30s | float |
| `spread_std_300s` | 300-second spread volatility | 300s | float |
| `spread_mean_60s` | 60-second spread mean | 60s | float |
| `tick_count_60s` | Trading intensity (tick count) | 60s | int |
| `return_range_60s` | Return range (max - min) | 60s | float |


### 3.3 Feature Engineering Rationale

**Why these features?**

1. **Multiple time windows** capture both short-term noise and longer-term trends
2. **Return statistics** directly measure price movement patterns
3. **Spread metrics** indicate market liquidity and stress
4. **Tick intensity** proxies for trading activity and information flow

---

## 4. Data Processing Pipeline

### 4.1 Real-Time Pipeline
```
Coinbase WebSocket → Kafka (ticks.raw) → Featurizer → Kafka (ticks.features) → Parquet
```

### 4.2 Replay Pipeline (for reproducibility)
```
NDJSON files → replay.py → FeatureComputer → Parquet
```

**Validation:** Replay and live features must match exactly (verified via `scripts/replay.py`)

---

## 5. Data Quality Considerations

### 5.1 Missing Data Handling
- **Midprice missing:** Skip tick (requires both bid and ask)
- **Insufficient window data:** Features set to `None` or 0 for counts
- **Timestamp issues:** Use current system time as fallback

### 5.2 Edge Cases
- **Market reconnections:** Feature windows reset when buffer empty
- **Extreme outliers:** Not filtered in feature computation (model's job)
- **Time gaps:** No interpolation; gaps natural in windowed features

### 5.3 Known Limitations
- Features lag reality by ~100-500ms (typical Kafka + compute latency)
- Window sizes fixed (not adaptive to market regime)
- No handling of trading halts or circuit breakers
- Volume-based features not available (ticker channel doesn't provide volume data)

---

## 6. Feature Statistics

**From EDA (`notebooks/eda.ipynb`) and processed data:**

| Metric | Value |
|--------|-------|
| Total samples | 52,524 |
| Time range | 2025-11-08 15:12:31 to 2025-11-09 01:25:17 (~10.2 hours) |
| Positive class % | 10.00% |
| Missing data % | 0.01% |
| Avg ticks/second | 1.43 |

**Feature Statistics (mean, std):**
- `return_mean_60s`: mean=-0.000000, std=0.000002
- `return_mean_300s`: mean=-0.000000, std=0.000001
- `return_std_300s`: mean=0.000040, std=0.000013
- `spread`: mean=0.271610, std=0.948744
- `spread_bps`: mean=0.026680, std=0.093189

---

## 7. Next Steps (Milestone 3)

1. **Train models** using these features
2. **Evaluate** using PR-AUC (primary metric)
3. **Monitor drift** between train and test distributions
4. **Iterate** on features based on model performance

---

## Appendix: Feature Correlation

**Correlation with target variable (`volatility_spike`):**

Top 3 features correlated with future volatility:
1. `return_std_300s`: r = 0.1917 (strongest predictor)
2. `return_mean_60s`: r = 0.0416
3. `return_mean_300s`: r = 0.0357

**Interpretation:**
- `return_std_300s` (5-minute volatility) shows the strongest positive correlation with future volatility spikes, confirming that recent volatility is a key indicator
- Short-term return means show weaker but positive correlations
- Spread features (`spread`, `spread_bps`) show minimal correlation with future volatility in this dataset