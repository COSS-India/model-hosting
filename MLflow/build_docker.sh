#!/bin/bash
# Build Docker image for MLflow ASR service

set -e

IMAGE_NAME="mlflow-asr-service"
IMAGE_TAG="${1:-latest}"
MODEL_ID="${2:-m-8f33614a5aeb46f6a4f4c8b0c64b9cf7}"

echo "Building MLflow ASR Docker image..."
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Model ID: ${MODEL_ID}"
echo ""

# Check if model exists
if [ ! -d "mlruns/0/models/${MODEL_ID}" ]; then
    echo "⚠️  Warning: Model ${MODEL_ID} not found in mlruns/0/models/"
    echo "   Please run 'python log_model.py' first to log the model."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Build Docker image
docker build \
    --tag "${IMAGE_NAME}:${IMAGE_TAG}" \
    --build-arg MODEL_ID="${MODEL_ID}" \
    -f Dockerfile \
    .

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Docker image built successfully!"
    echo ""
    echo "To run the container:"
    echo "  ./run_docker.sh"
    echo ""
    echo "Or manually:"
    echo "  docker run --rm -p 5000:5000 --gpus all -e HF_TOKEN=\$HF_TOKEN ${IMAGE_NAME}:${IMAGE_TAG}"
else
    echo ""
    echo "❌ Docker build failed!"
    exit 1
fi






