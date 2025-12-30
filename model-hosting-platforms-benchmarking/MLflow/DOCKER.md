# Docker Deployment Guide

This guide explains how to build and run the MLflow ASR service using Docker.

## Prerequisites

- Docker installed (version 20.10+)
- Docker Compose (optional, for easier deployment)
- NVIDIA Docker runtime (for GPU support, optional)
- Model already logged to MLflow (run `python log_model.py` first)

## Quick Start

### 1. Build the Docker Image

```bash
./build_docker.sh
```

Or manually:

```bash
docker build -t mlflow-asr-service:latest .
```

### 2. Run the Container

**Background mode** (recommended):

```bash
./run_docker.sh
```

**Foreground mode** (for debugging):

```bash
./run_docker_foreground.sh
```

Or manually:

```bash
docker run -d \
  --name mlflow-asr-service \
  --gpus all \
  -p 5000:5000 \
  -e HF_TOKEN="${HF_TOKEN}" \
  mlflow-asr-service:latest
```

### 3. Test the Service

```bash
./test_curl.sh ta ctc
```

## Docker Compose

### Using Docker Compose

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Docker Compose with Custom Settings

Edit `docker-compose.yml` to customize:
- Port mapping
- Environment variables
- GPU settings
- Volume mounts

## Building the Image

### Basic Build

```bash
docker build -t mlflow-asr-service:latest .
```

### Build with Custom Tag

```bash
docker build -t mlflow-asr-service:v1.0 .
```

### Build with Custom Model ID

```bash
./build_docker.sh latest m-<your-model-id>
```

## Running the Container

### Basic Run (CPU only)

```bash
docker run -d \
  --name mlflow-asr-service \
  -p 5000:5000 \
  -e HF_TOKEN="${HF_TOKEN}" \
  mlflow-asr-service:latest
```

### Run with GPU Support

```bash
docker run -d \
  --name mlflow-asr-service \
  --gpus all \
  -p 5000:5000 \
  -e HF_TOKEN="${HF_TOKEN}" \
  mlflow-asr-service:latest
```

### Run with Custom Port

```bash
docker run -d \
  --name mlflow-asr-service \
  --gpus all \
  -p 8080:5000 \
  -e HF_TOKEN="${HF_TOKEN}" \
  mlflow-asr-service:latest
```

Then access at `http://localhost:8080`

### Run with Custom Model Path

```bash
docker run -d \
  --name mlflow-asr-service \
  --gpus all \
  -p 5000:5000 \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e MODEL_PATH="mlruns/0/models/m-<model-id>/artifacts" \
  mlflow-asr-service:latest
```

## Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `HF_TOKEN` | HuggingFace authentication token | No* | - |
| `MODEL_PATH` | Path to MLflow model artifacts | No | `mlruns/0/models/m-8f33614a5aeb46f6a4f4c8b0c64b9cf7/artifacts` |
| `PYTHONUNBUFFERED` | Python output buffering | No | `1` |

*Required if the model requires authentication

## Container Management

### View Logs

```bash
# Follow logs
docker logs -f mlflow-asr-service

# Last 100 lines
docker logs --tail 100 mlflow-asr-service
```

### Stop Container

```bash
docker stop mlflow-asr-service
```

### Remove Container

```bash
docker rm mlflow-asr-service
```

### Restart Container

```bash
docker restart mlflow-asr-service
```

### Execute Commands in Container

```bash
# Open shell
docker exec -it mlflow-asr-service /bin/bash

# Check Python version
docker exec mlflow-asr-service python --version

# Check GPU
docker exec mlflow-asr-service python -c "import torch; print(torch.cuda.is_available())"
```

## Health Checks

The container includes a health check that verifies the service is responding:

```bash
# Check health status
docker inspect --format='{{.State.Health.Status}}' mlflow-asr-service

# Manual health check
curl http://localhost:5000/health
```

## Troubleshooting

### Issue: Container exits immediately

**Check logs**:
```bash
docker logs mlflow-asr-service
```

**Common causes**:
- Model path incorrect
- Missing dependencies
- Port already in use

### Issue: GPU not available in container

**Verify NVIDIA Docker runtime**:
```bash
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

**Check container GPU access**:
```bash
docker exec mlflow-asr-service python -c "import torch; print(torch.cuda.is_available())"
```

### Issue: Model loading fails

**Check HF_TOKEN**:
```bash
docker exec mlflow-asr-service env | grep HF_TOKEN
```

**Verify model path**:
```bash
docker exec mlflow-asr-service ls -la /app/mlruns/0/models/
```

### Issue: Port already in use

**Find process using port**:
```bash
sudo lsof -i :5000
# or
sudo netstat -tlnp | grep 5000
```

**Use different port**:
```bash
docker run -d -p 8080:5000 ... mlflow-asr-service:latest
```

### Issue: Out of memory

**Check container resource usage**:
```bash
docker stats mlflow-asr-service
```

**Limit memory**:
```bash
docker run -d --memory="4g" ... mlflow-asr-service:latest
```

## Building for Production

### Multi-stage Build (Optional)

For smaller images, you can use a multi-stage build:

```dockerfile
# Build stage
FROM python:3.10-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Runtime stage
FROM python:3.10-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY mlflow_asr.py log_model.py ./
COPY mlruns/ ./mlruns/
ENV PATH=/root/.local/bin:$PATH
EXPOSE 5000
CMD ["mlflow", "models", "serve", "-m", "mlruns/0/models/m-8f33614a5aeb46f6a4f4c8b0c64b9cf7/artifacts", "--no-conda", "--host", "0.0.0.0", "--port", "5000"]
```

### Security Best Practices

1. **Don't commit secrets**: Use environment variables or secrets management
2. **Use non-root user**:
   ```dockerfile
   RUN useradd -m -u 1000 appuser
   USER appuser
   ```
3. **Scan images**: Use `docker scan` or tools like Trivy
4. **Keep base images updated**: Regularly rebuild with latest base images

## Image Size Optimization

### Current Image Size

```bash
docker images mlflow-asr-service
```

### Reduce Image Size

1. Use `python:3.10-slim` (already used)
2. Remove build dependencies after installation
3. Use multi-stage builds
4. Clean apt cache: `rm -rf /var/lib/apt/lists/*` (already done)

## Networking

### Expose to All Interfaces

The container binds to `0.0.0.0:5000` by default, making it accessible from:
- `localhost:5000` (host)
- `<host-ip>:5000` (network)

### Custom Network

```bash
# Create network
docker network create mlflow-network

# Run container on network
docker run -d --network mlflow-network --name mlflow-asr-service ...
```

## Volume Mounts

### Persistent Model Storage

```bash
docker run -d \
  -v $(pwd)/mlruns:/app/mlruns:ro \
  ... \
  mlflow-asr-service:latest
```

### Logs Directory

```bash
docker run -d \
  -v $(pwd)/logs:/app/logs \
  ... \
  mlflow-asr-service:latest
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Build and Push Docker Image

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build Docker image
        run: docker build -t mlflow-asr-service:${{ github.sha }} .
      - name: Push to registry
        run: |
          echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u "${{ secrets.DOCKER_USERNAME }}" --password-stdin
          docker push mlflow-asr-service:${{ github.sha }}
```

## Registry Deployment

### Push to Docker Hub

```bash
docker tag mlflow-asr-service:latest username/mlflow-asr-service:latest
docker push username/mlflow-asr-service:latest
```

### Pull and Run

```bash
docker pull username/mlflow-asr-service:latest
docker run -d -p 5000:5000 username/mlflow-asr-service:latest
```

## Monitoring

### Resource Usage

```bash
# Real-time stats
docker stats mlflow-asr-service

# One-time check
docker stats --no-stream mlflow-asr-service
```

### Logs Analysis

```bash
# Error logs only
docker logs mlflow-asr-service 2>&1 | grep -i error

# Count requests
docker logs mlflow-asr-service 2>&1 | grep -c "POST /asr"
```

## Next Steps

- Read [README.md](README.md) for general usage
- Check [USAGE.md](USAGE.md) for API examples
- See [SETUP.md](SETUP.md) for local setup

---

**Need Help?** Check the troubleshooting section above or review container logs.

