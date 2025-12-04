#!/bin/bash
# Script to run BentoML containerized service and test with curl

set -e

CONTAINER_NAME="bentoml-asr-service"
IMAGE_NAME="indic_conformer_asr:latest"
PORT=3000
AUDIO_FILE="/home/ubuntu/Benchmarking/ta2.wav"

echo "=========================================="
echo "BentoML ASR Service - Containerized Test"
echo "=========================================="
echo ""

# Check if audio file exists
if [ ! -f "$AUDIO_FILE" ]; then
    echo "Error: Audio file not found at $AUDIO_FILE"
    echo "Please provide a valid audio file path."
    exit 1
fi

# Stop and remove existing container if running
echo "Checking for existing container..."
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping and removing existing container..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
fi

# Check if image exists
if ! docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "^${IMAGE_NAME}$"; then
    echo "Error: Docker image '$IMAGE_NAME' not found!"
    echo "Please build the image first using:"
    echo "  cd /home/ubuntu/Benchmarking/Frameworks/bento-ml"
    echo "  ./bento/bin/bentoml containerize indic_conformer_asr:latest"
    exit 1
fi

# Run the container
echo "Starting Docker container '$CONTAINER_NAME'..."
docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus all \
  -p ${PORT}:3000 \
  -e PYTHONPATH=/home/bentoml/bento/src/bento/lib/python3.10/site-packages:$PYTHONPATH \
  "$IMAGE_NAME"

echo "Container started. Waiting for service to be ready..."
echo ""

# Wait for service to be ready (check health endpoint or just wait)
sleep 10

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Error: Container failed to start!"
    echo "Checking logs..."
    docker logs "$CONTAINER_NAME" | tail -20
    exit 1
fi

# Wait a bit more for the service to fully initialize
echo "Waiting for service initialization..."
for i in {1..30}; do
    if curl -s -f http://localhost:${PORT}/healthz > /dev/null 2>&1 || \
       curl -s -f http://localhost:${PORT}/docs > /dev/null 2>&1; then
        echo "Service is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "Warning: Service may not be fully ready, but proceeding with test..."
    else
        sleep 2
    fi
done

echo ""
echo "=========================================="
echo "Testing ASR Inference with curl"
echo "=========================================="
echo ""

# Test inference with curl
echo "Sending request to /asr endpoint..."
echo "Audio file: $AUDIO_FILE"
echo "Language: ta (Tamil)"
echo "Strategy: ctc"
echo ""

RESPONSE=$(curl -X POST http://localhost:${PORT}/asr \
  -F "file=@${AUDIO_FILE}" \
  -F "lang=ta" \
  -F "strategy=ctc" \
  -w "\nHTTP_STATUS:%{http_code}" \
  -s)

HTTP_STATUS=$(echo "$RESPONSE" | grep -o "HTTP_STATUS:[0-9]*" | cut -d: -f2)
BODY=$(echo "$RESPONSE" | sed 's/HTTP_STATUS:[0-9]*$//')

echo "Response:"
echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
echo ""
echo "HTTP Status: $HTTP_STATUS"
echo ""

if [ "$HTTP_STATUS" = "200" ]; then
    echo "✅ Inference successful!"
else
    echo "❌ Inference failed with status $HTTP_STATUS"
    echo ""
    echo "Container logs (last 20 lines):"
    docker logs "$CONTAINER_NAME" | tail -20
fi

echo ""
echo "=========================================="
echo "Container Status"
echo "=========================================="
docker ps --filter "name=$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""
echo "To view logs: docker logs -f $CONTAINER_NAME"
echo "To stop container: docker stop $CONTAINER_NAME"
echo "To remove container: docker rm $CONTAINER_NAME"

