#!/bin/bash
# Run BentoML Docker container in foreground mode (for debugging)
# Use run_docker_background.sh to run in background

CONTAINER_NAME="bentoml-asr-service"
IMAGE_NAME="indic_conformer_asr:latest"

# Stop and remove existing container if running
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping existing container..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
fi

echo "Starting BentoML ASR service container in foreground..."
echo "Service will be available at http://localhost:3000"
echo "Press CTRL+C to stop"
echo ""

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN environment variable is not set."
    echo "If the model requires authentication, set it with:"
    echo "  export HF_TOKEN=your_token_here"
    echo "Or pass it directly: HF_TOKEN=your_token_here ./run_docker.sh"
    echo ""
fi

docker run --rm \
  --name "$CONTAINER_NAME" \
  --gpus all \
  -p 3000:3000 \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e MODEL_ID="${MODEL_ID:-ai4bharat/indic-conformer-600m-multilingual}" \
  "$IMAGE_NAME"

