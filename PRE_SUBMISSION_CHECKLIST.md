# Pre-Submission Checklist: Crypto Volatility Detection AI Service

**Last Updated:** November 24, 2025

This checklist guides you through the final steps to ensure your project is complete, functional, and meets all assignment requirements before submission.

---

## ✅ Week 4 Deliverables (Complete)

- [x] `docker/compose.yaml` - Infrastructure setup (Kafka, Zookeeper, MLflow, API, Prometheus, Grafana)
- [x] `docker/Dockerfile.api` - API containerization
- [x] `docs/architecture_diagram.md` - System architecture
- [x] Working `/predict` endpoint
- [x] `/health`, `/version`, `/metrics` endpoints
- [x] Model selection rationale (Random Forest selected)

---

## ✅ Week 5 Deliverables (Complete)

- [x] `.github/workflows/ci.yml` - CI/CD pipeline (Black, Ruff, pytest)
- [x] `tests/test_api_integration.py` - Integration tests
- [x] `scripts/load_test.py` - Load testing script
- [x] `docs/load_test_results.json` - Load test results
- [x] Kafka resilience (reconnection, retry, graceful shutdown in `scripts/ws_ingest.py`)
- [x] `.env.example` - Environment configuration template (created)
- [x] `README.md` - Updated with ≤10-line quick start

---

## ✅ Week 6 Deliverables (Complete)

- [x] `docker/compose.yaml` - Prometheus and Grafana added
- [x] `docker/prometheus/prometheus.yml` - Prometheus configuration
- [x] `docker/grafana/dashboards/crypto-volatility.json` - Pre-configured dashboard
- [x] `docker/grafana/datasources/prometheus.yml` - Datasource provisioning
- [x] `docs/grafana_dashboard_screenshot.png` - Dashboard screenshot
- [x] `docs/slo.md` - Service Level Objectives
- [x] `docs/drift_summary.md` - Data drift tracking
- [x] `docs/runbook.md` - Operations runbook
- [x] Model rollback via `MODEL_VARIANT` environment variable

---

## ✅ Week 7 Deliverables (Complete)

- [x] `docs/runbook.md` - Complete runbook (startup, troubleshooting, recovery)
- [x] `docs/performance_summary.md` - Performance metrics summary
- [x] `docs/demo_checklist.md` - Demo script and checklist

---

## 📋 Final Steps Before Submission

### 1. Update Performance Summary

- [x] Fill in actual load test results
- [x] Add latency metrics
- [x] Add model performance comparison (PR-AUC values)
- [ ] Add uptime statistics (if available from Prometheus)

**Update command:**
```bash
# Check MLflow for model metrics
# Visit http://localhost:5001
# Update docs/performance_summary.md with actual PR-AUC values
```

---

### 2. Verify All Services Work

```bash
# Test one-command startup
cd docker
docker compose up -d

# Verify all services
docker compose ps
# Should show: kafka, zookeeper, mlflow, prometheus, grafana, api all "Up"

# Test API
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":{"log_return_300s":0.001,"spread_mean_300s":0.5,"trade_intensity_300s":100,"order_book_imbalance_300s":0.6,"spread_mean_60s":0.3,"order_book_imbalance_60s":0.55,"price_velocity_300s":0.0001,"realized_volatility_300s":0.002,"order_book_imbalance_30s":0.52,"realized_volatility_60s":0.0015}}'

# Verify Grafana
# Visit http://localhost:3000 (credentials from .env file)
# Dashboard should show data

# Verify Prometheus
# Visit http://localhost:9090
# Check targets are "UP"
```

---

### 3. Documentation Review

- [x] README has ≤10-line quick start at top
- [x] All required docs exist (slo.md, runbook.md, performance_summary.md, etc.)
- [x] Architecture diagram exists
- [x] Performance summary has load test results
- [x] Performance summary has model PR-AUC values
- [ ] Review all docs for completeness and accuracy

---

### 4. Code Quality

```bash
# Run CI checks locally
source .venv/bin/activate
black --check .
ruff check .
pytest tests/ -v
```

---

### 5. Git Commit & Tag

```bash
# Review changes
git status

# Stage all files
git add .

# Commit with descriptive message
git commit -m "Complete weeks 5-7: CI/CD, monitoring, SLOs, runbook, performance summary"

# Create release tag
git tag v1.0.0

# Push to remote (when ready)
git push origin main
git push origin v1.0.0
```

---

### 6. Demo Video (Required for Week 7)

- [ ] **SKIPPED** - Demo video not required for this submission
- [ ] Record 8-minute demo video following `docs/demo_checklist.md` (if needed later)
- [ ] Show: startup, prediction, monitoring, failure recovery, model rollback
- [ ] Upload to YouTube/Loom (unlisted)
- [ ] Add link to README or submission platform

**Note:** Demo video skipped per user request. Can be added later if needed.

---

### 7. Final Verification

- [ ] All services start with `docker compose up -d`
- [ ] API responds at `http://localhost:8000/health`
- [ ] Grafana dashboard shows data (after generating traffic)
- [ ] Prometheus metrics are being scraped
- [ ] Load test passes (p95 ≤ 800ms)
- [ ] CI pipeline would pass (if pushed to GitHub)

---

## 📊 Current Performance Metrics

### Load Test Results (100 requests)

- **Success Rate:** 100%
- **p95 Latency:** 465.73ms (✓ PASS - well under 800ms target)
- **p50 Latency:** 148.39ms
- **Mean Latency:** 187.67ms
- **Requests/Second:** 51.5

### SLO Compliance

- ✅ **p95 ≤ 800ms:** PASS (465.73ms)
- ✅ **Success Rate ≥ 99%:** PASS (100%)
- ✅ **Error Rate < 1%:** PASS (0%)

### Model Performance (Test Set)

- **Random Forest:** 
  - PR-AUC: 0.9859
  - ROC-AUC: 0.9983
  - Accuracy: 0.9888 (98.88%)
  - Precision: 0.9572 (95.72%)
  - Recall: 0.9372 (93.72%)
  - F1-Score: 0.9471 (94.71%)
- **Baseline:** PR-AUC 0.1039 (test), 0.0861 (validation)
- **Improvement:** +849% PR-AUC improvement (9.5x better)

---

## 🎯 Submission Requirements

According to the assignment, submit via GitHub (tagged release):

- [x] Source code
- [x] `docker/compose.yaml`
- [x] Documentation (README, SLOs, runbook, etc.)
- [x] README with ≤10-line setup guide
- [ ] Demo video link (YouTube/Loom)
- [ ] Tagged release (v1.0.0)

---

## 📝 Quick Commands Reference

```bash
# One-command startup (from project root)
cd docker && docker compose up -d

# Test API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":{"log_return_300s":0.001,"spread_mean_300s":0.5,"trade_intensity_300s":100,"order_book_imbalance_300s":0.6,"spread_mean_60s":0.3,"order_book_imbalance_60s":0.55,"price_velocity_300s":0.0001,"realized_volatility_300s":0.002,"order_book_imbalance_30s":0.52,"realized_volatility_60s":0.0015}}'

# Run load test
python scripts/load_test.py --requests 100 --output docs/load_test_results.json

# Generate drift report
python scripts/generate_evidently_report.py \
  --features data/processed/features_replay.parquet \
  --output reports/evidently/train_test_drift_report.html \
  --report_type train_test

# Monitor data collection
./scripts/monitor_collection.sh

# Process and retrain after data collection
./scripts/process_and_retrain_after_collection.sh
```

---

## ⚠️ Critical Items Before Submission

1. ~~**Demo Video**~~ - SKIPPED per user request
2. **Git Tag** - Create v1.0.0 release tag
3. **Final Test** - Verify one-command startup works
4. **Documentation** - Review all docs for completeness

---

**You're almost ready!** Just need to:
1. ~~Record demo video~~ (SKIPPED)
2. Create git tag
3. Final verification test
4. Submit!
