# Feature Specification

## Project: Real-Time Crypto Volatility Detection

**Date:** November 13, 2025  
**Author:** Melissa Wong  
**Version:** 1.1

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

**Implementation:** Chunk-aware forward-looking calculation that respects data collection gaps.

Mathematically:
```
σ_future = std(r_t+1, r_t+2, ..., r_t+n)
where:
  r_i = (price_i - price_{i-1}) / price_{i-1}
  n = number of ticks in [t, t+60 seconds]
  
Constraint: Only ticks within the same data collection chunk are considered
(no calculation across gaps > gap_threshold_seconds)
```

**Key Changes (v1.1):**
- **Chunk-aware calculation:** Volatility computed only within continuous data segments
- **Forward-looking:** For each timestamp, finds all ticks in the next 60 seconds and computes std of returns
- **Gap handling:** Data collection gaps (>300s default) define chunk boundaries
- **Iterative method:** Correctly handles variable tick density and ensures no look-ahead bias

### Label Definition
Binary classification:
```
label = 1  if σ_future >= τ  (volatility spike)
label = 0  if σ_future < τ   (normal conditions)
```

### Chosen Threshold (τ)
**Value:** 90th percentile (configurable, default: 90)

**Justification:**
- Selected at the **90th percentile** of observed future volatility within each data chunk
- Based on percentile analysis in EDA (see `notebooks/eda.ipynb`)
- This threshold captures the top 10% of volatile periods
- **Chunk-aware:** Threshold calculated separately for each data collection chunk to account for temporal variations
- Results in approximately **10.0%** positive class (spikes) when data is balanced

**Trade-offs:**
- Higher threshold → fewer false positives, but might miss moderate spikes
- Lower threshold → more sensitivity, but higher false alarm rate
- Current threshold balances detection rate with actionable signal quality

**Configuration:**
- Default: `label_threshold_percentile=90` (configurable in `featurizer.py`)
- Gap threshold: `label_gap_threshold_seconds=300` (5 minutes) defines chunk boundaries

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

Features are computed over rolling windows: **30s, 60s (1min), and 300s (5min)**

#### Feature Categories

**Price-based Features:**
- **Simple Returns**: `r_t = (p_t - p_{t-1}) / p_{t-1}`
- **Log Returns**: `log_r_t = log(p_t / p_{t-1}) = log(p_t) - log(p_{t-1})` (more stable for crypto)
- **Rolling Volatility**: Standard deviation of returns over window
- **Price Momentum**: Mean, min, max returns over window

**Market Microstructure Features:**
- **Bid-ask Spread**: Absolute and basis points (bps)
- **Spread Volatility**: Standard deviation of spreads over window

**Volume/Activity Features:**
- **Trade Intensity**: Tick count per window
- **Time Since Last Trade**: Seconds since previous tick

#### Complete Feature List

| Feature Name | Formula | Window | Aggregation | Missing Value Handling |
|--------------|---------|--------|-------------|------------------------|
| `return_mean_{window}s` | `mean((p_t - p_{t-1}) / p_{t-1})` | 30s, 60s, 300s | Mean | 0.0 |
| `return_std_{window}s` | `std((p_t - p_{t-1}) / p_{t-1})` | 30s, 60s, 300s | Std Dev | 0.0 |
| `return_min_{window}s` | `min((p_t - p_{t-1}) / p_{t-1})` | 30s, 60s, 300s | Min | 0.0 |
| `return_max_{window}s` | `max((p_t - p_{t-1}) / p_{t-1})` | 30s, 60s, 300s | Max | 0.0 |
| `log_return_mean_{window}s` | `mean(log(p_t) - log(p_{t-1}))` | 30s, 60s, 300s | Mean | 0.0 |
| `log_return_std_{window}s` | `std(log(p_t) - log(p_{t-1}))` | 30s, 60s, 300s | Std Dev | 0.0 |
| `price_mean_{window}s` | `mean(p_t)` | 30s, 60s, 300s | Mean | 0.0 |
| `price_std_{window}s` | `std(p_t)` | 30s, 60s, 300s | Std Dev | 0.0 |
| `tick_count_{window}s` | Count of ticks | 30s, 60s, 300s | Count | 0 |
| `spread_std_{window}s` | `std(spread_t)` | 30s, 60s, 300s | Std Dev | 0.0 |
| `spread_mean_{window}s` | `mean(spread_t)` | 30s, 60s, 300s | Mean | 0.0 |
| `time_since_last_trade` | `t_current - t_previous` (seconds) | N/A | Difference | 0.0 |
| `gap_seconds` | Time gap between consecutive ticks | N/A | Difference | 0.0 |

#### Features Used in Model (10 features)

The current model uses a reduced feature set to minimize multicollinearity. Features were selected based on separation analysis and correlation reduction:

| Feature Name | Description | Window | Rationale |
|--------------|-------------|--------|-----------|
| `log_return_std_30s` | 30-second log return volatility | 30s | More stable for crypto, excellent separation |
| `log_return_std_60s` | 60-second log return volatility | 60s | Best separation (0.569 std dev), more stable than simple returns |
| `log_return_std_300s` | 300-second log return volatility | 300s | Longer-term context, more stable than simple returns |
| `return_mean_60s` | 1-minute return mean | 60s | Good separation (0.74 std dev) |
| `return_mean_300s` | 5-minute return mean | 300s | Moderate separation (0.51 std dev) |
| `return_min_30s` | Minimum return in 30s | 30s | Good separation (0.64 std dev), downside risk indicator |
| `spread_std_300s` | 300-second spread volatility | 300s | Market microstructure indicator |
| `spread_mean_60s` | 60-second spread mean | 60s | Market liquidity indicator |
| `tick_count_60s` | Trading intensity | 60s | Moderate separation (0.21 std dev) |
| `return_range_60s` | Return range (max - min) | 60s | Derived feature, volatility proxy |

**Note:** Removed perfectly correlated features (r=1.0) to improve Logistic Regression performance:
- Removed `return_std_30s`, `return_std_60s`, `return_std_300s` (duplicates of log_return_std_*)
- Removed `log_return_mean_30s`, `log_return_mean_60s` (duplicates of return_mean_*)

This reduction improved Logistic Regression PR-AUC by +6.6% (0.2298 → 0.2449).

#### Baseline Model Features (8 features)

The baseline model uses a composite z-score approach across 8 features (matching `BaselineVolatilityDetector.DEFAULT_FEATURES`):

| Feature Name | Description | Window | Usage |
|--------------|-------------|--------|-------|
| `log_return_std_30s` | 30-second log return volatility | 30s | Composite z-score |
| `log_return_std_60s` | 60-second log return volatility | 60s | Composite z-score |
| `log_return_std_300s` | 300-second log return volatility | 300s | Composite z-score |
| `return_mean_60s` | 1-minute return mean | 60s | Composite z-score |
| `return_mean_300s` | 5-minute return mean | 300s | Composite z-score |
| `return_min_30s` | Minimum return in 30s | 30s | Composite z-score |
| `spread_std_300s` | 300-second spread volatility | 300s | Composite z-score |
| `spread_mean_60s` | 60-second spread mean | 60s | Composite z-score |
| `tick_count_60s` | Trading intensity | 60s | Composite z-score |

**Baseline Method:**
1. Standardize each feature using training mean/std
2. Compute per-feature z-scores
3. Calculate composite score as mean of z-scores
4. Apply threshold (default: 2.0) to composite z-score
5. Predict spike if composite z-score >= threshold

### 3.3 Feature Engineering Rationale

**Why these features?**

1. **Multiple time windows** capture both short-term noise and longer-term trends
2. **Return statistics** directly measure price movement patterns
3. **Spread metrics** indicate market liquidity and stress
4. **Tick intensity** proxies for trading activity and information flow
5. **Log returns** preferred over simple returns for crypto (more stable, symmetric)

**What we're NOT using (yet):**
- Order book imbalance (complexity vs benefit trade-off)
- Volume-weighted features (not available in ticker channel)
- Cross-asset correlations (single-pair focus for MVP)

**Performance Impact:**
- **XGBoost (Stratified):** PR-AUC 0.7815 with 10-feature set
- **Feature reduction:** Removing perfectly correlated features improved Logistic Regression PR-AUC by +6.6%
- **Stratified splitting:** Balancing spike rates across splits improved XGBoost PR-AUC from 0.7359 to 0.7815

---

## 4. Reproducibility & Determinism

### 4.1 Deterministic Computations
- ✅ No randomness in feature computation logic
- ✅ Fixed window boundaries (time-based, not tick-based)
- ✅ Deterministic aggregation functions (mean, std, min, max)

### 4.2 Timestamp Handling
- **Timezone:** UTC (explicitly enforced)
- **Format:** ISO 8601 or Unix timestamp (auto-detected)
- **Consistency:** All timestamps converted to UTC timezone-aware
- **Validation:** Timestamp ordering checked (warns on backward jumps >1s)

### 4.3 Replay Verification
- **Script:** `scripts/replay.py` verifies reproducibility
- **Method:** Re-process raw data through feature pipeline
- **Verification:** Compare replayed features with original (within 1e-6 tolerance)
- **Usage:** `python scripts/replay.py --raw data/raw/*.ndjson --compare data/processed/features.parquet`

### 4.4 Window Boundaries
- **Type:** Fixed-size sliding windows (time-based)
- **Boundaries:** `[current_time - window_seconds, current_time]` (inclusive)
- **Example:** At t=15:01:00, 60s window includes [15:00:00, 15:01:00]
- **Documentation:** Window logic documented in code comments

## 5. Data Processing Pipeline

### 5.1 Real-Time Pipeline
```
Coinbase WebSocket → Kafka (ticks.raw) → Featurizer → Kafka (ticks.features) → Parquet
```

### 5.2 Replay Pipeline (for reproducibility)
```
NDJSON files → replay.py → FeatureComputer → FeaturePipeline._add_labels_to_dataframe → Parquet
```

**Validation:** Replay and live features must match exactly (verified via `scripts/replay.py`)

**Label Creation:**
- Labels (`volatility_spike`) are created using `FeaturePipeline._add_labels_to_dataframe`
- Chunk-aware calculation ensures labels respect data collection gaps
- Can be added during feature generation (`--add-labels` flag) or separately via `scripts/add_labels.py`

---

## 5. Data Quality Considerations

### 5.1 Missing Data Handling

**Strategy:** Set features to `0.0` for consistency (not `None` or `NaN`)

- **Midprice missing:** Skip tick (requires both bid and ask)
- **Insufficient window data:** Features set to `0.0` (occurs for first few ticks)
- **Timestamp issues:** Use current UTC time as fallback
- **NaN/Infinite values:** Detected and replaced with `0.0` (logged as warning)

**Rationale:** 
- Consistent handling downstream (train.py fills NaN with 0)
- Prevents downstream errors from None values
- Missing data is rare (< 0.01% in practice)

### 5.2 Gap Handling

**Gap Detection:**
- Feature `gap_seconds` tracks time between consecutive ticks
- Large gaps (>10s) logged as warnings
- Gaps are natural in crypto markets (24/7 trading but brief pauses)

**Chunk Detection (v1.1):**
- Data collection gaps (>300s default) define chunk boundaries
- Chunks are detected automatically during label creation
- Volatility calculation respects chunk boundaries (no calculation across gaps)
- Prevents artificial volatility spikes from connecting unrelated data segments

**Current Strategy:**
- No forward-fill implemented (gaps preserved in features)
- Windowed features naturally handle gaps (fewer ticks = lower tick_count)
- Chunk-aware label creation ensures forward-looking volatility only uses ticks within same chunk
- Future enhancement: Forward-fill short gaps (<10s) with last known price

**Gap Tolerance:**
- Windows with >10% missing ticks may have reduced signal quality
- Documented in feature statistics but not filtered
- Model learns to handle variable tick density
- Chunk boundaries prevent cross-gap calculations that could introduce bias

### 5.3 Data Quality Checks

**Implemented Checks:**
- ✅ NaN detection and replacement
- ✅ Infinite value detection and replacement
- ✅ Timestamp ordering validation (warns on >1s backward jumps)
- ✅ Gap detection and logging

**Monitoring:**
- Periodic statistics logged (min, max, mean) during processing
- Quality issues logged as warnings
- Feature distributions tracked in Evidently reports

### 5.4 Edge Cases

- **Market reconnections:** Feature windows reset when buffer empty
- **Extreme outliers:** Not filtered in feature computation (model's job)
- **Time gaps:** Preserved in features (no interpolation)
- **Out-of-order timestamps:** Validated and logged (small backward jumps allowed)

### 5.5 Known Limitations

- Features lag reality by ~100-500ms (typical Kafka + compute latency)
- Window sizes fixed (not adaptive to market regime)
- No handling of trading halts or circuit breakers
- Volume-based features not available (ticker channel doesn't provide volume data)
- Forward-fill not implemented (gaps preserved)

---

## 6. Feature Statistics

**From EDA (`notebooks/eda.ipynb`) and processed data:**

| Metric | Value |
|--------|-------|
| Total samples | ~9,629 (after filtering) |
| Time range | 2025-11-08 15:12:31 to 2025-11-09 01:25:17 (~10.2 hours) |
| Positive class % | ~10.00% (varies by split method) |
| Missing data % | 0.01% |
| Avg ticks/second | 1.43 |
| Data chunks | Multiple (gaps >300s define boundaries) |

**Data Split Statistics:**

**Time-Based Split:**
- Training: ~6,740 samples (6.60% spike rate)
- Validation: ~1,444 samples (2.42% spike rate)
- Test: ~1,445 samples (33.43% spike rate)
- *Note: Temporal clustering causes test set to have much higher spike rate*

**Stratified Split (Recommended):**
- Training: ~6,740 samples (10.0% spike rate)
- Validation: ~1,444 samples (10.0% spike rate)
- Test: ~1,445 samples (10.0% spike rate)
- *Note: Balanced spike rates improve model performance*

**Feature Statistics (mean, std):**
- `return_mean_60s`: mean≈0.0, std≈0.000002
- `return_mean_300s`: mean≈0.0, std≈0.000001
- `log_return_std_60s`: Best separation (0.569 std dev between classes)
- `spread`: mean≈0.27, std≈0.95
- `spread_bps`: mean≈0.027, std≈0.093

---

## 7. Model Performance with These Features

**Best Model: XGBoost (Stratified Split)**
- PR-AUC: 0.7815 (Test)
- Recall: 97.31%
- Precision: 52.87%
- F1-Score: 0.6851

**Key Findings:**
- **Stratified splitting** significantly improves performance (XGBoost PR-AUC: 0.7359 → 0.7815)
- **Chunk-aware label creation** ensures correct forward-looking volatility calculation
- **10-feature set** provides good balance between information and model complexity
- **Log returns** preferred over simple returns for crypto volatility modeling

**Feature Importance (XGBoost):**
- Top features: `log_return_std_60s`, `log_return_std_300s`, `return_range_60s`
- Spread features (`spread_std_300s`, `spread_mean_60s`) contribute to model performance
- Trade intensity (`tick_count_60s`) provides additional signal

## 8. Next Steps & Future Enhancements

1. ✅ **Train models** using these features - Complete
2. ✅ **Evaluate** using PR-AUC (primary metric) - Complete
3. ✅ **Monitor drift** between train and test distributions - Complete
4. ✅ **Iterate** on features based on model performance - Complete
5. **Future:** Explore additional features (order book depth, volume-weighted metrics)
6. **Future:** Adaptive window sizes based on market regime
7. **Future:** Multi-asset features for cross-market signals

---

## Appendix: Feature Correlation & Importance

**Correlation with target variable (`volatility_spike`):**

Top features correlated with future volatility:
1. `log_return_std_60s`: Best separation (0.569 std dev between classes)
2. `return_mean_60s`: Good separation (0.74 std dev)
3. `return_min_30s`: Good separation (0.64 std dev), downside risk indicator
4. `return_mean_300s`: Moderate separation (0.51 std dev)
5. `tick_count_60s`: Moderate separation (0.21 std dev)

**XGBoost Feature Importance:**
- `log_return_std_60s`: Highest importance (60-second volatility is key predictor)
- `log_return_std_300s`: High importance (longer-term context)
- `return_range_60s`: High importance (derived feature captures volatility range)
- Spread features: Moderate importance (market microstructure signals)
- Trade intensity: Lower but non-zero importance

**Interpretation:**
- **Log return volatility** (especially 60s window) is the strongest predictor of future volatility spikes
- **Return statistics** (mean, min) provide additional signal about price movement patterns
- **Spread features** contribute to model performance, indicating market liquidity stress
- **Derived features** (`return_range_60s`) effectively capture volatility proxies
- **Multiple time windows** (30s, 60s, 300s) capture different aspects of market dynamics

**Model Performance by Feature Set:**
- **10-feature set:** XGBoost PR-AUC 0.7815 (best performance)
- **Full feature set:** Higher multicollinearity, lower Logistic Regression performance
- **Baseline (8 features):** Composite z-score approach, PR-AUC 0.2295 (stratified) to 0.2881 (time-based)