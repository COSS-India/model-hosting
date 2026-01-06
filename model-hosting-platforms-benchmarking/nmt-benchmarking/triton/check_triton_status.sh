#!/bin/bash

# Comprehensive Triton server status check

echo "=========================================="
echo "Triton Server Status Check"
echo "=========================================="
echo ""

# 1. Container Status
echo "1. DOCKER CONTAINER STATUS"
echo "---------------------------"
docker ps --filter "name=triton-indictrans-v2" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""

# 2. Server Health
echo "2. SERVER HEALTH"
echo "---------------------------"
echo -n "Live:   "
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/v2/health/live && echo " ✓" || echo " ✗"

echo -n "Ready:  "
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/v2/health/ready && echo " ✓" || echo " ✗"
echo ""

# 3. Server Metadata
echo "3. SERVER METADATA"
echo "---------------------------"
curl -s http://localhost:8000/v2 | python3 -m json.tool 2>/dev/null | head -10
echo ""

# 4. Model Status
echo "4. MODEL STATUS"
echo "---------------------------"
echo "Model: nmt"
curl -s http://localhost:8000/v2/models/nmt | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"  Name: {data.get('name', 'N/A')}\")
print(f\"  Versions: {data.get('versions', [])}\")
print(f\"  Platform: {data.get('platform', 'N/A')}\")
print(f\"  Inputs: {len(data.get('inputs', []))}\")
print(f\"  Outputs: {len(data.get('outputs', []))}\")
" 2>/dev/null
echo ""

# 5. Model Instance Configuration
echo "5. MODEL INSTANCE CONFIGURATION"
echo "---------------------------"
curl -s http://localhost:8000/v2/models/nmt/config | python3 -c "
import sys, json
data = json.load(sys.stdin)
instances = data.get('instance_group', [])
for i, inst in enumerate(instances):
    print(f\"Instance {i+1}:\")
    print(f\"  Name: {inst.get('name', 'N/A')}\")
    print(f\"  Kind: {inst.get('kind', 'N/A')}\")
    if inst.get('kind') == 'KIND_GPU':
        print(f\"  ✓ Running on GPU\")
        print(f\"  GPU IDs: {inst.get('gpus', [])}\")
    else:
        print(f\"  ✗ Running on CPU\")
    print(f\"  Count: {inst.get('count', 'N/A')}\")
" 2>/dev/null
echo ""

# 6. GPU Status
echo "6. GPU STATUS"
echo "---------------------------"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
    echo ""
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "Total GPUs detected: $GPU_COUNT"
else
    echo "nvidia-smi not available"
fi
echo ""

# 7. Port Status
echo "7. PORT STATUS"
echo "---------------------------"
echo -n "Port 8000 (HTTP):  "
nc -z localhost 8000 2>/dev/null && echo "✓ Open" || echo "✗ Closed"

echo -n "Port 8001 (gRPC):  "
nc -z localhost 8001 2>/dev/null && echo "✓ Open" || echo "✗ Closed"

echo -n "Port 8002 (Metrics): "
nc -z localhost 8002 2>/dev/null && echo "✓ Open" || echo "✗ Closed"
echo ""

# 8. Metrics Sample
echo "8. SAMPLE METRICS"
echo "---------------------------"
METRICS=$(curl -s http://localhost:8002/metrics 2>/dev/null)
if [ ! -z "$METRICS" ]; then
    echo "$METRICS" | grep -E "nv_inference_request|nv_inference_exec|nv_inference_request_success" | head -3
else
    echo "Metrics not available"
fi
echo ""

echo "=========================================="
echo "Status Check Complete"
echo "=========================================="

