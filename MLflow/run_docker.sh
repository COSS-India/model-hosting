#!/bin/bash
# Run MLflow ASR service in Docker container

set -e

CONTAINER_NAME="mlflow-asr-service"
IMAGE_NAME="mlflow-asr-service"
IMAGE_TAG="${1:-latest}"
PORT="${2:-5000}"
MODEL_ID="${3:-m-8f33614a5aeb46f6a4f4c8b0c64b9cf7}"

# Stop and remove existing container if running
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping existing container..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
fi

# Check if image exists
if ! docker image inspect "${IMAGE_NAME}:${IMAGE_TAG}" >/dev/null 2>&1; then
    echo "⚠️  Image ${IMAGE_NAME}:${IMAGE_TAG} not found. Building..."
    ./build_docker.sh "$IMAGE_TAG" "$MODEL_ID"
fi

echo "Starting MLflow ASR service container..."
echo "Container name: $CONTAINER_NAME"
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Port: $PORT"
echo "Model ID: $MODEL_ID"
echo ""

# Check for GPU
GPU_FLAG=""
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    GPU_FLAG="--gpus all"
    echo "✅ GPU detected - using GPU support"
else
    echo "⚠️  No GPU detected - running on CPU"
fi

# Run container
docker run -d \
    --name "$CONTAINER_NAME" \
    $GPU_FLAG \
    -p "${PORT}:5000" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e MODEL_PATH="mlruns/0/models/${MODEL_ID}/artifacts" \
    -e PYTHONUNBUFFERED=1 \
    "$IMAGE_NAME:$IMAGE_TAG"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Container started successfully!"
    echo ""
    echo "Service will be available at: http://localhost:${PORT}"
    echo ""
    echo "Useful commands:"
    echo "  View logs:    docker logs -f $CONTAINER_NAME"
    echo "  Stop:         docker stop $CONTAINER_NAME"
    echo "  Remove:       docker rm $CONTAINER_NAME"
    echo "  Test:         ./test_curl.sh ta ctc"
    echo ""
    echo "Waiting for service to initialize..."
    sleep 5
    
    # Check health
    if curl -f http://localhost:${PORT}/health >/dev/null 2>&1; then
        echo "✅ Service is healthy!"
    else
        echo "⚠️  Service is starting... (may take a minute for first request)"
        echo "   Check logs: docker logs $CONTAINER_NAME"
    fi
else
    echo ""
    echo "❌ Failed to start container"
    exit 1
fi

