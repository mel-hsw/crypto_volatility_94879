# Runbook: Crypto Volatility Detection API

## Table of Contents

1. [Startup Procedures](#startup-procedures)
2. [Health Checks](#health-checks)
3. [Common Issues & Troubleshooting](#common-issues--troubleshooting)
4. [Recovery Procedures](#recovery-procedures)
5. [Model Rollback](#model-rollback)
6. [Monitoring & Alerts](#monitoring--alerts)

---

## Startup Procedures

### Initial Setup

```bash
# 1. Clone repository
git clone <repository-url>
cd operationaliseai

# 2. Set up Python environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Copy environment variables
cp .env.example .env
# Edit .env with your configuration

# 4. Start infrastructure
cd docker
docker compose up -d

# 5. Verify services
docker compose ps
# Expected: All services "Up"
```

### Service Startup Order

1. **Zookeeper** (if using Zookeeper mode)
2. **Kafka**
3. **MLflow**
4. **Prometheus**
5. **Grafana**
6. **API**

### Verify Services

```bash
# Check API health
curl http://localhost:8000/health

# Check Prometheus
curl http://localhost:9090/-/healthy

# Check Grafana
curl http://localhost:3000/api/health

# Check MLflow
curl http://localhost:5001/health
```

---

## Health Checks

### API Health Endpoint

```bash
curl http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-24T12:00:00Z",
  "model_loaded": true,
  "kafka_connected": true
}
```

### Component Health Checks

**Kafka:**
```bash
docker exec -it kafka kafka-broker-api-versions --bootstrap-server localhost:9092
```

**Model Loading:**
```bash
curl http://localhost:8000/version
# Should return model version and path
```

**Metrics Endpoint:**
```bash
curl http://localhost:8000/metrics
# Should return Prometheus metrics
```

---

## Common Issues & Troubleshooting

### Issue 1: API Returns 503 "Model not loaded"

**Symptoms:**
- `/predict` endpoint returns 503
- `/health` shows `model_loaded: false`

**Diagnosis:**
```bash
# Check model file exists
ls -lh models/artifacts/random_forest/model.pkl

# Check API logs
docker logs volatility-api

# Check MODEL_PATH environment variable
docker exec volatility-api env | grep MODEL_PATH
```

**Resolution:**
1. Verify model file exists at expected path
2. Check file permissions (should be readable)
3. Restart API service:
   ```bash
   docker compose restart api
   ```

### Issue 2: Kafka Connection Failed

**Symptoms:**
- API logs show "Failed to connect to Kafka"
- Health check shows `kafka_connected: false`

**Diagnosis:**
```bash
# Check Kafka is running
docker compose ps kafka

# Test Kafka connectivity (try to find correct command path)
KAFKA_CMD="kafka-topics"
if ! docker exec kafka which kafka-topics >/dev/null 2>&1; then
    if docker exec kafka test -f /usr/bin/kafka-topics; then
        KAFKA_CMD="/usr/bin/kafka-topics"
    elif docker exec kafka test -f /usr/bin/kafka-topics.sh; then
        KAFKA_CMD="/usr/bin/kafka-topics.sh"
    fi
fi
docker exec -it kafka $KAFKA_CMD --list --bootstrap-server localhost:9092

# Check network connectivity
docker exec volatility-api ping kafka
```

**Resolution:**
1. Restart Kafka:
   ```bash
   docker compose restart kafka
   ```
2. Wait 30 seconds for Kafka to fully start
3. Restart API:
   ```bash
   docker compose restart api
   ```

### Issue 3: High Latency (p95 > 800ms)

**Symptoms:**
- Grafana dashboard shows p95 latency > 800ms
- Predictions are slow

**Diagnosis:**
```bash
# Check API resource usage
docker stats volatility-api

# Check for errors in logs
docker logs volatility-api | grep -i error

# Run load test
python scripts/load_test.py --requests 100
```

**Resolution:**
1. Check system resources (CPU, memory)
2. Review recent code changes
3. Consider scaling API instances
4. Check for memory leaks in logs

### Issue 4: Prometheus Not Scraping Metrics

**Symptoms:**
- Grafana shows no data
- Prometheus targets show "DOWN"

**Diagnosis:**
```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Check API metrics endpoint
curl http://localhost:8000/metrics

# Check Prometheus config
docker exec prometheus cat /etc/prometheus/prometheus.yml
```

**Resolution:**
1. Verify API is accessible from Prometheus network
2. Check Prometheus configuration
3. Restart Prometheus:
   ```bash
   docker compose restart prometheus
   ```

### Issue 5: Model Predictions Are Inaccurate

**Symptoms:**
- Predictions seem wrong
- Performance metrics degraded

**Diagnosis:**
```bash
# Check model version
curl http://localhost:8000/version

# Generate drift report
python scripts/generate_evidently_report.py \
  --features data/processed/features_replay.parquet \
  --output reports/evidently/data_drift_report.html \
  --report_type data_drift

# Check model performance
# (Review MLflow experiments)
```

**Resolution:**
1. Review drift report for data quality issues
2. Check if model needs retraining
3. Consider model rollback (see below)

---

## Recovery Procedures

### Full System Restart

```bash
# 1. Stop all services
cd docker
docker compose down

# 2. Clean up (optional - removes volumes)
# WARNING: This deletes data
# docker compose down -v

# 3. Start services
docker compose up -d

# 4. Wait for services to be healthy
sleep 30

# 5. Verify health
curl http://localhost:8000/health
```

### API Service Recovery

```bash
# Restart API only
docker compose restart api

# Check logs
docker logs -f volatility-api

# Verify health
curl http://localhost:8000/health
```

### Kafka Recovery

```bash
# Restart Kafka
docker compose restart kafka

# Wait for Kafka to be ready
sleep 20

# Verify topics exist (using detected command path)
docker exec -it kafka $KAFKA_CMD --list --bootstrap-server localhost:9092

# Restart dependent services
docker compose restart api
```

### Data Recovery

If data is lost or corrupted:

1. **Check backups:**
   ```bash
   ls -lh data/raw/
   ls -lh data/processed/
   ```

2. **Replay data:**
   ```bash
   python scripts/replay_to_kafka.py \
     --input data/raw/ticks_BTCUSD_20251109_130539.ndjson \
     --topic ticks.raw
   ```

3. **Regenerate features:**
   ```bash
   python scripts/replay.py \
     --raw "data/raw/ticks_BTCUSD_*.ndjson" \
     --out data/processed/features_replay.parquet \
     --add-labels
   ```

---

## Model Rollback

### Switch to Baseline Model

```bash
# 1. Update docker-compose.yaml or set environment variable
# MODEL_VARIANT=baseline

# 2. Restart API
docker compose restart api

# 3. Verify model loaded
curl http://localhost:8000/version
# Should show "baseline" model
```

### Switch Back to ML Model

```bash
# 1. Set MODEL_VARIANT=ml

# 2. Restart API
docker compose restart api

# 3. Verify
curl http://localhost:8000/version
```

### Rollback via Environment Variable

```bash
# Set environment variable in docker-compose.yaml
environment:
  - MODEL_VARIANT=baseline  # or 'ml'

# Restart
docker compose restart api
```

---

## Monitoring & Alerts

### Key Metrics to Monitor

1. **Latency:** p50, p95, p99 (target: p95 ≤ 800ms)
2. **Error Rate:** 4xx/5xx responses (target: < 1%)
3. **Success Rate:** 2xx responses (target: ≥ 99%)
4. **Model Load Status:** Should be 1 (loaded)
5. **Request Rate:** Requests per second

### Grafana Dashboard

Access: `http://localhost:3000`
- Username: `admin`
- Password: `admin`

### Prometheus Queries

**Check p95 latency:**
```promql
histogram_quantile(0.95, rate(prediction_latency_seconds_bucket[5m])) * 1000
```

**Check error rate:**
```promql
sum(rate(http_requests_total{status=~"4..|5.."}[5m])) / sum(rate(http_requests_total[5m]))
```

**Check model loaded:**
```promql
model_loaded
```

### Alert Thresholds

- **p95 Latency > 1000ms** for 5 minutes → Investigate
- **Error Rate > 2%** for 5 minutes → Alert team
- **Model Not Loaded** → Immediate action required
- **Success Rate < 98%** for 5 minutes → Investigate

---

## Emergency Contacts

- **On-Call Engineer:** [To be filled]
- **Team Lead:** [To be filled]
- **Slack Channel:** #crypto-volatility-alerts

---

**Last Updated:** November 2025  
**Version:** 1.0

