# Docker Deployment Guide

This guide explains how to build and run the ASR server using Docker.

## Prerequisites

- Docker installed
- For GPU support: NVIDIA Docker runtime (nvidia-docker2)
- HuggingFace token (HF_TOKEN)

## Quick Start

### GPU Version (Recommended)

1. **Build the image:**
   ```bash
   docker build -t asr-server:latest .
   ```

2. **Run with GPU:**
   ```bash
   docker run -d \
     --name asr-server \
     --gpus all \
     -p 8000:8000 \
     -e HF_TOKEN=your_huggingface_token_here \
     -e DEVICE=cuda \
     asr-server:latest
   ```

3. **Or use docker-compose:**
   ```bash
   export HF_TOKEN=your_huggingface_token_here
   docker-compose up -d
   ```

### CPU Version

1. **Build the CPU image:**
   ```bash
   docker build -f Dockerfile.cpu -t asr-server:cpu .
   ```

2. **Run without GPU:**
   ```bash
   docker run -d \
     --name asr-server \
     -p 8000:8000 \
     -e HF_TOKEN=your_huggingface_token_here \
     -e DEVICE=cpu \
     asr-server:cpu
   ```

## Environment Variables

- `HF_TOKEN` (required): HuggingFace authentication token
- `DEVICE` (optional): `cuda` or `cpu` (default: auto-detect)
- `MODEL_NAME` (optional): Model name (default: ai4bharat/indic-conformer-600m-multilingual)

## Docker Compose

The `docker-compose.yml` file includes:
- GPU support configuration
- Health checks
- Volume mounting for HuggingFace cache (persists model downloads)
- Auto-restart policy

### Using Docker Compose:

```bash
# Set environment variables
export HF_TOKEN=your_token_here
export DEVICE=cuda  # or cpu

# Start the service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

## Verify Deployment

1. **Check health:**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Test transcription:**
   ```bash
   curl -X POST "http://localhost:8000/asr" \
     -F "audio=@test.wav" \
     -F "lang=en"
   ```

3. **View API docs:**
   Open http://localhost:8000/docs in your browser

## Troubleshooting

### GPU Not Available
- Ensure NVIDIA Docker runtime is installed: `nvidia-docker2`
- Check GPU access: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`
- Use CPU version if GPU is not available

### Model Download Issues
- Ensure HF_TOKEN is set correctly
- Check network connectivity
- Model will be cached in `~/.cache/huggingface` (mounted as volume in docker-compose)

### Port Already in Use
- Change port mapping: `-p 8001:8000` (host:container)
- Or stop existing service: `docker stop asr-server`

## Building for Production

For production deployment, consider:
- Using specific version tags instead of `latest`
- Setting resource limits
- Using multi-stage builds to reduce image size
- Setting up proper logging and monitoring

## Image Sizes

- GPU version: ~8-10 GB (includes CUDA runtime)
- CPU version: ~2-3 GB (smaller, no CUDA)




