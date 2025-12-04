# BentoML ASR Service - Containerized Deployment

This guide explains how to containerize and run the BentoML ASR service for inference.

## Prerequisites

1. **Docker** with NVIDIA Container Toolkit installed
2. **BentoML** service built and ready
3. **Audio file** for testing (e.g., `ta2.wav`)

## Quick Start

### 1. Build the Docker Image

First, ensure the BentoML service is built:

```bash
cd /home/ubuntu/Benchmarking/Frameworks/bento-ml
./bento/bin/bentoml containerize indic_conformer_asr:latest
```

This will create a Docker image tagged as `indic_conformer_asr:latest`.

### 2. Run Containerized Service

#### Option A: Automated Script (Recommended)

Run the container and test inference automatically:

```bash
./run_containerized.sh
```

This script will:
- Start the Docker container with GPU support
- Wait for the service to be ready
- Test inference with curl
- Display results and container status

#### Option B: Manual Run

Run the container in foreground (for debugging):

```bash
./run_docker.sh
```

Or run in background:

```bash
docker run -d \
  --name bentoml-asr-service \
  --gpus all \
  -p 3000:3000 \
  -e PYTHONPATH=/home/bentoml/bento/src/bento/lib/python3.10/site-packages:$PYTHONPATH \
  indic_conformer_asr:latest
```

### 3. Test Inference with curl

Once the container is running, test inference:

```bash
# Using the test script
./test_curl.sh

# Or manually
curl -X POST http://localhost:3000/asr \
  -F "file=@/home/ubuntu/Benchmarking/ta2.wav" \
  -F "lang=ta" \
  -F "strategy=ctc" \
  -w "\nHTTP Status: %{http_code}\n"
```

## Container Management

### View Logs

```bash
docker logs -f bentoml-asr-service
```

### Stop Container

```bash
docker stop bentoml-asr-service
```

### Remove Container

```bash
docker rm bentoml-asr-service
```

### Check Container Status

```bash
docker ps --filter "name=bentoml-asr-service"
```

## API Endpoint

- **Endpoint**: `POST /asr`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file`: Audio file (WAV format)
  - `lang`: Language code (e.g., `ta`, `hi`, `en`)
  - `strategy`: Decoding strategy (e.g., `ctc`)

### Example Response

```json
{
  "text": "transcribed text here",
  "language": "ta",
  "confidence": 0.95
}
```

## Troubleshooting

### Container fails to start

1. Check if GPU is available:
   ```bash
   nvidia-smi
   ```

2. Verify NVIDIA Container Toolkit is installed:
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

3. Check container logs:
   ```bash
   docker logs bentoml-asr-service
   ```

### Module not found errors

The container includes a `PYTHONPATH` environment variable to ensure all dependencies are found. If you still encounter import errors, check:

1. The image was built correctly
2. All dependencies are in `requirements.txt`
3. The `bento.yml` includes all necessary packages

### GPU not utilized

1. Ensure `--gpus all` flag is used when running the container
2. Check GPU utilization:
   ```bash
   nvidia-smi
   ```
3. Verify `onnxruntime-gpu` is installed (not `onnxruntime`)

## Files

- `run_containerized.sh`: Automated script to run container and test
- `run_docker.sh`: Manual script to run container in foreground
- `test_curl.sh`: Script to test inference with curl
- `service.py`: BentoML service definition
- `bento.yml`: BentoML configuration
- `requirements.txt`: Python dependencies

