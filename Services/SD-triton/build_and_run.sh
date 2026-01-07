#!/bin/bash
# Build and run script for SD-triton

set -e

# Check if HuggingFace token is set
if [ -z "${HUGGING_FACE_HUB_TOKEN}" ]; then
    echo "ERROR: HUGGING_FACE_HUB_TOKEN environment variable is not set!"
    echo "Please set it with: export HUGGING_FACE_HUB_TOKEN='your_token_here'"
    echo "Get your token from: https://huggingface.co/settings/tokens"
    exit 1
fi

echo "Building Docker image..."
docker build --build-arg HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" -t sd-triton:latest .

echo ""
echo "Stopping and removing old container if exists..."
docker stop sd-triton-server 2>/dev/null || true
docker rm sd-triton-server 2>/dev/null || true

echo ""
echo "Starting Triton server..."
docker run -d --gpus all \
  -p 8700:8000 \
  -p 8701:8001 \
  -p 8702:8002 \
  -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
  --name sd-triton-server \
  sd-triton:latest

echo ""
echo "Waiting for server to start..."
sleep 10

echo ""
echo "Checking server status..."
docker logs sd-triton-server 2>&1 | tail -20

echo ""
echo "Server should be running on port 8700"
echo "Check health: curl http://localhost:8700/v2/health/ready"
echo "Test with: python3 test_client.py ../ALD-triton/ta2.wav"

