#!/bin/bash
# Start SD-triton server

# Check if HuggingFace token is set
if [ -z "${HUGGING_FACE_HUB_TOKEN}" ]; then
    echo "ERROR: HUGGING_FACE_HUB_TOKEN environment variable is not set!"
    echo "Please set it with: export HUGGING_FACE_HUB_TOKEN='your_token_here'"
    echo "Get your token from: https://huggingface.co/settings/tokens"
    exit 1
fi

echo "Stopping old container if exists..."
docker stop sd-triton-server 2>/dev/null || true
docker rm sd-triton-server 2>/dev/null || true

echo "Starting Triton server on port 8700..."
docker run -d --gpus all \
  -p 8700:8000 \
  -p 8701:8001 \
  -p 8702:8002 \
  -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
  --name sd-triton-server \
  sd-triton:latest

echo ""
echo "Server starting... This may take 1-2 minutes for model download."
echo "Check logs with: docker logs -f sd-triton-server"
echo "Check health with: curl http://localhost:8700/v2/health/ready"

