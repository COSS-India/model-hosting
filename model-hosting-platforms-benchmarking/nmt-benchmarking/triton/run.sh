#!/bin/bash

# Script to run the Triton IndicTrans v2 model container

echo "Starting Triton IndicTrans v2 container..."

docker run -d \
  --name triton-indictrans-v2 \
  --gpus all \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  --shm-size=1g \
  --restart unless-stopped \
  ai4bharat/triton-indictrans-v2:latest

echo "Container started. Waiting for server to be ready..."
sleep 5

# Check if server is ready
max_attempts=30
attempt=0
while [ $attempt -lt $max_attempts ]; do
  if curl -f http://localhost:8000/v2/health/ready > /dev/null 2>&1; then
    echo "Server is ready!"
    echo "HTTP endpoint: http://localhost:8000"
    echo "gRPC endpoint: localhost:8001"
    echo "Metrics endpoint: http://localhost:8002/metrics"
    exit 0
  fi
  attempt=$((attempt + 1))
  echo "Waiting for server... ($attempt/$max_attempts)"
  sleep 2
done

echo "Warning: Server may not be ready yet. Check logs with: docker logs triton-indictrans-v2"

