#!/bin/bash

# Script to verify GPU usage and Triton server status

echo "=========================================="
echo "Triton GPU Verification Script"
echo "=========================================="
echo ""

# 1. Check if container is running
echo "1. Checking Triton container status..."
if docker ps | grep -q triton-indictrans-v2; then
    echo "✓ Container is running"
    CONTAINER_ID=$(docker ps | grep triton-indictrans-v2 | awk '{print $1}')
    echo "  Container ID: $CONTAINER_ID"
else
    echo "✗ Container is not running"
    exit 1
fi
echo ""

# 2. Check GPU visibility from container
echo "2. Checking GPU visibility from container..."
docker exec triton-indictrans-v2 nvidia-smi --list-gpus 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ GPU is visible inside container"
else
    echo "✗ GPU not accessible from container"
fi
echo ""

# 3. Check Triton server health
echo "3. Checking Triton server health..."
HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/v2/health/ready)
if [ "$HEALTH" = "200" ]; then
    echo "✓ Server is ready"
else
    echo "✗ Server is not ready (HTTP $HEALTH)"
fi
echo ""

# 4. Check model configuration for GPU instance
echo "4. Checking model instance configuration..."
INSTANCE_INFO=$(curl -s http://localhost:8000/v2/models/nmt/config | python3 -c "
import sys, json
data = json.load(sys.stdin)
instances = data.get('instance_group', [])
for inst in instances:
    print(f\"  Instance: {inst.get('name', 'N/A')}\")
    print(f\"  Kind: {inst.get('kind', 'N/A')}\")
    print(f\"  GPUs: {inst.get('gpus', [])}\")
    print(f\"  Count: {inst.get('count', 'N/A')}\")
" 2>/dev/null)

if echo "$INSTANCE_INFO" | grep -q "KIND_GPU"; then
    echo "✓ Model is configured to use GPU"
    echo "$INSTANCE_INFO"
else
    echo "✗ Model is not configured for GPU"
    echo "$INSTANCE_INFO"
fi
echo ""

# 5. Check current GPU utilization
echo "5. Current GPU utilization (before inference):"
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader
echo ""

# 6. Run inference and monitor GPU
echo "6. Running inference and monitoring GPU utilization..."
echo "   (Watch GPU utilization increase during inference)"
echo ""

# Start GPU monitoring in background
(
    for i in {1..10}; do
        nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        sleep 0.5
    done
) > /tmp/gpu_monitor.log &

MONITOR_PID=$!

# Run inference
echo "   Sending inference request..."
curl -s -X POST http://localhost:8000/v2/models/nmt/infer \
  -H 'Content-Type: application/json' \
  -d '{"inputs":[{"name":"INPUT_TEXT","datatype":"BYTES","shape":[1,1],"data":["Hello, how are you? This is a test to verify GPU utilization during translation."]},{"name":"INPUT_LANGUAGE_ID","datatype":"BYTES","shape":[1,1],"data":["en"]},{"name":"OUTPUT_LANGUAGE_ID","datatype":"BYTES","shape":[1,1],"data":["hi"]}]}' > /dev/null

# Wait for monitoring to finish
wait $MONITOR_PID 2>/dev/null

echo "   GPU utilization during inference:"
cat /tmp/gpu_monitor.log | awk -F', ' '{printf "  Time %2d: GPU: %3s%% | Memory: %3s%% | Mem Used: %s MB\n", NR, $1, $2, $3}'
rm -f /tmp/gpu_monitor.log
echo ""

# 7. Check Triton metrics
echo "7. Checking Triton metrics endpoint..."
METRICS=$(curl -s http://localhost:8002/metrics 2>/dev/null)
if [ ! -z "$METRICS" ]; then
    echo "✓ Metrics endpoint is accessible"
    echo ""
    echo "  GPU metrics (if available):"
    echo "$METRICS" | grep -i gpu | head -5
    echo ""
    echo "  Model inference metrics:"
    echo "$METRICS" | grep -i "nv_inference" | head -5
else
    echo "✗ Metrics endpoint not accessible"
fi
echo ""

# 8. Check container logs for GPU-related messages
echo "8. Checking container logs for GPU initialization..."
docker logs triton-indictrans-v2 2>&1 | grep -i "gpu\|cuda" | tail -5
echo ""

echo "=========================================="
echo "Verification Complete"
echo "=========================================="
echo ""
echo "Quick GPU check: nvidia-smi"
echo "Monitor continuously: watch -n 1 nvidia-smi"
echo "Triton metrics: curl http://localhost:8002/metrics | grep gpu"

