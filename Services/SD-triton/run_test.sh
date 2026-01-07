#!/bin/bash
# Run test script for SD-triton

cd /home/ubuntu/incubalm/SD-triton

# Check if server is ready
echo "Checking server health..."
if curl -s http://localhost:8700/v2/health/ready > /dev/null 2>&1; then
    echo "✅ Server is ready!"
    echo ""
    echo "Running test with audio file..."
    python3 test_client.py ../ALD-triton/ta2.wav
else
    echo "❌ Server is not ready."
    echo ""
    echo "Please start the server first:"
    echo "  export HUGGING_FACE_HUB_TOKEN='your_token_here'"
    echo "  docker run -d --gpus all -p 8700:8000 -p 8701:8001 -p 8702:8002 \\"
    echo "    -e HUGGING_FACE_HUB_TOKEN=\"\${HUGGING_FACE_HUB_TOKEN}\" \\"
    echo "    --name sd-triton-server sd-triton:latest"
    echo ""
    echo "Then wait 1-2 minutes and check: curl http://localhost:8700/v2/health/ready"
    exit 1
fi












