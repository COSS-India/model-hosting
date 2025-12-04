# Running BentoML ASR Service in Docker

## Quick Start

### Run in Background (Recommended)

```bash
./run_docker_background.sh
```

This will:
- Start the container in detached mode (background)
- Map port 3000
- Enable GPU support
- Set up environment variables

### Run in Foreground (for debugging)

```bash
./run_docker.sh
```

This will show live logs. Press `CTRL+C` to stop.

### Manual Docker Run

```bash
# Background mode (no live logs)
docker run -d \
  --name bentoml-asr-service \
  --gpus all \
  -p 3000:3000 \
  -e HF_TOKEN="your_huggingface_token" \
  indic_conformer_asr:latest

# Foreground mode (shows live logs)
docker run --rm \
  --gpus all \
  -p 3000:3000 \
  -e HF_TOKEN="your_huggingface_token" \
  indic_conformer_asr:latest
```

## Why Live Logs Appear

When you run `docker run` without the `-d` flag, Docker runs the container in **foreground mode**, which means:
- The container's stdout/stderr are displayed in your terminal
- The terminal is attached to the container
- You see live logs as they happen
- Press `CTRL+C` to stop the container

To run in **background mode** (detached), add the `-d` flag:
```bash
docker run -d --gpus all -p 3000:3000 indic_conformer_asr:latest
```

## Environment Variables

- `HF_TOKEN`: Your HuggingFace token (required if model is gated)
- `MODEL_ID`: Model ID (default: `ai4bharat/indic-conformer-600m-multilingual`)

## Testing the Service

Once the container is running:

```bash
curl -X POST http://localhost:3000/asr \
  -F "file=@/path/to/audio.wav" \
  -F "lang=ta" \
  -F "strategy=ctc"
```

Or use the test script:
```bash
./test_curl.sh
```

## Container Management

```bash
# View logs
docker logs -f bentoml-asr-service

# Stop container
docker stop bentoml-asr-service

# Remove container
docker rm bentoml-asr-service

# Check status
docker ps --filter "name=bentoml-asr-service"
```

## Troubleshooting

### Soundfile Not Found
The service has a fallback to `scipy.io.wavfile` if `soundfile` is not available. Both are included in the container.

### Model Not Found in BentoML Store
This is expected in Docker. The service will:
1. Try to load model config from BentoML store
2. Fall back to environment variables if not found
3. Load the model directly from HuggingFace at runtime

Make sure `HF_TOKEN` is set if the model requires authentication.

### GPU Not Utilized
- Ensure `--gpus all` flag is used
- Check GPU availability: `nvidia-smi`
- Verify NVIDIA Container Toolkit is installed

