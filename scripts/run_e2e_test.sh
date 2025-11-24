#!/bin/bash
# End-to-end pipeline test - runs live for 5 minutes

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

echo "=========================================="
echo "End-to-End Pipeline Test (5 minutes)"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}✗ Docker is not running. Please start Docker Desktop.${NC}"
    exit 1
fi

# Check if services are running
echo -e "${YELLOW}Step 1: Checking services...${NC}"
cd docker
if ! docker compose ps | grep -q "volatility-api.*Up"; then
    echo -e "${YELLOW}Starting services...${NC}"
    docker compose up -d
    echo "Waiting for services to be ready..."
    sleep 10
fi

# Verify services
echo -e "${YELLOW}Verifying services...${NC}"
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo -e "${RED}✗ API is not responding${NC}"
    exit 1
fi
echo -e "${GREEN}✓ API is healthy${NC}"

if ! curl -s http://localhost:9090/-/healthy > /dev/null; then
    echo -e "${RED}✗ Prometheus is not responding${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Prometheus is healthy${NC}"

# Check Kafka topics exist
echo -e "${YELLOW}Step 2: Checking Kafka topics...${NC}"
if ! docker exec kafka kafka-topics --list --bootstrap-server localhost:9092 | grep -q "ticks.raw"; then
    echo -e "${YELLOW}Creating Kafka topics...${NC}"
    docker exec kafka kafka-topics --create \
        --topic ticks.raw \
        --bootstrap-server localhost:9092 \
        --partitions 3 \
        --replication-factor 1 \
        --if-not-exists 2>/dev/null || true
    
    docker exec kafka kafka-topics --create \
        --topic ticks.features \
        --bootstrap-server localhost:9092 \
        --partitions 3 \
        --replication-factor 1 \
        --if-not-exists 2>/dev/null || true
fi
echo -e "${GREEN}✓ Kafka topics ready${NC}"

cd "$PROJECT_DIR"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Create log directory
mkdir -p logs
LOG_DIR="$PROJECT_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo ""
echo -e "${YELLOW}Step 3: Starting data ingestion (5 minutes)...${NC}"
echo "Ingesting from Coinbase WebSocket → Kafka (ticks.raw)"
python scripts/ws_ingest.py --pair BTC-USD --minutes 5 --save-disk > "$LOG_DIR/ingest_${TIMESTAMP}.log" 2>&1 &
INGEST_PID=$!
echo -e "${GREEN}✓ Ingestion started (PID: $INGEST_PID)${NC}"

# Wait a bit for data to start flowing
sleep 5

echo ""
echo -e "${YELLOW}Step 4: Starting feature pipeline...${NC}"
echo "Consuming from Kafka (ticks.raw) → Computing features → Kafka (ticks.features)"
python features/featurizer.py \
    --topic_in ticks.raw \
    --topic_out ticks.features \
    --bootstrap_servers localhost:9092 \
    --output_file "data/processed/features_e2e_${TIMESTAMP}.parquet" \
    --windows 30 60 300 \
    --add-labels > "$LOG_DIR/featurizer_${TIMESTAMP}.log" 2>&1 &
FEATURIZER_PID=$!
echo -e "${GREEN}✓ Feature pipeline started (PID: $FEATURIZER_PID)${NC}"

# Wait a bit for features to start being generated
sleep 10

echo ""
echo -e "${YELLOW}Step 5: Generating API predictions...${NC}"
echo "Making predictions every 2 seconds for 5 minutes"
python -c "
import time
import requests
import json
from datetime import datetime

url = 'http://localhost:8000/predict'
features = {
    'log_return_300s': 0.001,
    'spread_mean_300s': 0.5,
    'trade_intensity_300s': 100,
    'order_book_imbalance_300s': 0.6,
    'spread_mean_60s': 0.3,
    'order_book_imbalance_60s': 0.55,
    'price_velocity_300s': 0.0001,
    'realized_volatility_300s': 0.002,
    'order_book_imbalance_30s': 0.52,
    'realized_volatility_60s': 0.0015
}

start_time = time.time()
duration = 300  # 5 minutes
request_count = 0

print(f'Starting predictions at {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')
while time.time() - start_time < duration:
    try:
        response = requests.post(url, json={'features': features}, timeout=5)
        if response.status_code == 200:
            request_count += 1
            if request_count % 10 == 0:
                result = response.json()
                print(f'[{request_count}] Prediction: {result[\"prediction\"]}, Probability: {result[\"probability\"]:.3f}, Latency: {result[\"inference_time_ms\"]:.2f}ms')
        else:
            print(f'Error: {response.status_code}')
    except Exception as e:
        print(f'Request failed: {e}')
    time.sleep(2)

print(f'Completed {request_count} predictions')
" > "$LOG_DIR/predictions_${TIMESTAMP}.log" 2>&1 &
PREDICTIONS_PID=$!
echo -e "${GREEN}✓ Prediction generator started (PID: $PREDICTIONS_PID)${NC}"

echo ""
echo -e "${GREEN}=========================================="
echo "Pipeline is running!"
echo "==========================================${NC}"
echo ""
echo "Components:"
echo "  • WebSocket Ingestion: PID $INGEST_PID"
echo "  • Feature Pipeline: PID $FEATURIZER_PID"
echo "  • API Predictions: PID $PREDICTIONS_PID"
echo ""
echo "Monitoring:"
echo "  • Grafana: http://localhost:3000 (credentials from .env or docker/compose.yaml defaults)"
echo "  • Prometheus: http://localhost:9090"
echo "  • API Health: http://localhost:8000/health"
echo "  • API Metrics: http://localhost:8000/metrics"
echo ""
echo "Logs:"
echo "  • Ingestion: $LOG_DIR/ingest_${TIMESTAMP}.log"
echo "  • Featurizer: $LOG_DIR/featurizer_${TIMESTAMP}.log"
echo "  • Predictions: $LOG_DIR/predictions_${TIMESTAMP}.log"
echo ""
echo -e "${YELLOW}Running for 5 minutes...${NC}"
echo "Press Ctrl+C to stop early"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Stopping processes...${NC}"
    kill $INGEST_PID 2>/dev/null || true
    kill $FEATURIZER_PID 2>/dev/null || true
    kill $PREDICTIONS_PID 2>/dev/null || true
    echo -e "${GREEN}✓ Processes stopped${NC}"
}

trap cleanup EXIT INT TERM

# Wait for 5 minutes (300 seconds)
sleep 300

echo ""
echo -e "${GREEN}=========================================="
echo "Test Complete!"
echo "==========================================${NC}"
echo ""
echo "Summary:"
echo "  • Check Grafana dashboard for metrics"
echo "  • Review logs in $LOG_DIR/"
echo "  • Features saved to: data/processed/features_e2e_${TIMESTAMP}.parquet"
echo ""

# Show quick stats
echo "Quick Stats:"
echo "  • Ingestion messages: $(grep -c "Processed.*messages" "$LOG_DIR/ingest_${TIMESTAMP}.log" 2>/dev/null || echo "0")"
echo "  • Features computed: $(grep -c "Computed features" "$LOG_DIR/featurizer_${TIMESTAMP}.log" 2>/dev/null || echo "0")"
echo "  • Predictions made: $(grep -c "Prediction:" "$LOG_DIR/predictions_${TIMESTAMP}.log" 2>/dev/null || echo "0")"
echo ""

cleanup

