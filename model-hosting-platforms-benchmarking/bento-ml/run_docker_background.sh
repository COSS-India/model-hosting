#!/bin/bash
# Run BentoML Docker container in background mode

CONTAINER_NAME="bentoml-asr-service"
IMAGE_NAME="indic_conformer_asr:latest"
PORT=3000

# Stop and remove existing container if running
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping existing container..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
fi

# Run container in background
echo "Starting BentoML ASR service container in background..."
echo "Container name: $CONTAINER_NAME"
echo "Service will be available at http://localhost:$PORT"
echo ""

# Check if HF_TOKEN is set, prompt if not
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN environment variable is not set."
    echo "If the model requires authentication, set it with:"
    echo "  export HF_TOKEN=your_token_here"
    echo "Or pass it directly: HF_TOKEN=your_token_here ./run_docker_background.sh"
    echo ""
    read -p "Continue without HF_TOKEN? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus all \
  -p ${PORT}:3000 \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e MODEL_ID="${MODEL_ID:-ai4bharat/indic-conformer-600m-multilingual}" \
  "$IMAGE_NAME"

if [ $? -eq 0 ]; then
    echo "✅ Container started successfully!"
    echo ""
    echo "To view logs: docker logs -f $CONTAINER_NAME"
    echo "To stop: docker stop $CONTAINER_NAME"
    echo "To remove: docker rm $CONTAINER_NAME"
    echo ""
    echo "Waiting for service to initialize..."
    sleep 5
    docker logs "$CONTAINER_NAME" | tail -10
else
    echo "❌ Failed to start container"
    exit 1
fi

