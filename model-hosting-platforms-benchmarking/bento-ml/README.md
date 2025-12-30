# BentoML ASR Service

A production-ready BentoML service for Automatic Speech Recognition (ASR) using the AI4Bharat IndicConformer-600M-Multilingual model.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Setup](#model-setup)
- [Building the Service](#building-the-service)
- [Running the Service](#running-the-service)
- [API Documentation](#api-documentation)
- [Docker Deployment](#docker-deployment)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)
- [Additional Resources](#additional-resources)

## 🎯 Overview

This BentoML service provides a scalable, containerized ASR (Automatic Speech Recognition) solution that:

- Hosts the AI4Bharat IndicConformer-600M-Multilingual model
- Supports multiple Indian languages (Tamil, Hindi, English, etc.)
- Provides RESTful API for audio transcription
- Optimized for GPU inference with ONNX Runtime
- Production-ready with Docker containerization

## ✨ Features

- **Multi-language Support**: Supports 20+ Indian languages
- **GPU Acceleration**: Optimized for NVIDIA GPUs with ONNX Runtime
- **Docker Ready**: Fully containerized with GPU support
- **Production Grade**: Built with BentoML for easy deployment
- **Flexible Audio Input**: Accepts WAV audio files via multipart form data
- **Custom Decoding Strategies**: Supports CTC and other decoding methods
- **Lazy Model Loading**: Models loaded on-demand for efficient resource usage

## 📋 Prerequisites

### System Requirements

- **Python**: 3.10 or higher
- **BentoML**: 1.4.0 or higher
- **Docker**: 20.10+ (for containerized deployment)
- **NVIDIA GPU**: Optional but recommended (for GPU acceleration)
- **NVIDIA Container Toolkit**: Required for GPU support in Docker

### Software Dependencies

- **CUDA**: 11.8+ (for GPU support)
- **Docker**: With NVIDIA Container Toolkit
- **HuggingFace Account**: With access token

## 🔧 Installation

### 1. Install BentoML

```bash
pip install bentoml>=1.4.0
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up HuggingFace Token

```bash
export HF_TOKEN=your_huggingface_token_here
```

Get your token from: https://huggingface.co/settings/tokens

## 🚀 Quick Start

### Step 1: Download Model

```bash
python download_model.py --token $HF_TOKEN
```

This saves the model configuration to BentoML's model store.

### Step 2: Build BentoML Service

```bash
bentoml build
```

This creates a BentoML bundle with all dependencies.

### Step 3: Containerize

```bash
bentoml containerize indic_conformer_asr:latest -t indic_conformer_asr:latest
```

### Step 4: Run Service

```bash
HF_TOKEN=$HF_TOKEN ./run_docker_background.sh
```

### Step 5: Test

```bash
./test_curl.sh
```

## 📦 Model Setup

### Downloading the Model

The model is downloaded and saved to BentoML's model store using `download_model.py`:

```bash
python download_model.py --token $HF_TOKEN
```

**What this does:**
- Downloads model configuration from HuggingFace
- Saves model metadata to BentoML model store
- Model weights are loaded at runtime (not stored in bundle)

**Note**: The actual model weights are loaded from HuggingFace at runtime, so the bundle only contains configuration.

### Model Information

- **Model ID**: `ai4bharat/indic-conformer-600m-multilingual`
- **Framework**: ONNX Runtime (GPU-accelerated)
- **Size**: ~600M parameters
- **Languages**: 20+ Indian languages
- **Input**: 16kHz mono WAV audio
- **Output**: Transcribed text

## 🏗️ Building the Service

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Download model
python download_model.py --token $HF_TOKEN

# Build BentoML bundle
bentoml build

# Run locally
bentoml serve service:svc --reload
```

### Production Build

```bash
# Build bundle
bentoml build

# Containerize
bentoml containerize indic_conformer_asr:latest -t indic_conformer_asr:latest

# Verify
docker images | grep indic_conformer_asr
```

## 🎮 Running the Service

### Option 1: Local Development

```bash
bentoml serve service:svc --reload
```

Service will be available at: `http://localhost:3000`

### Option 2: Docker (Background)

```bash
HF_TOKEN=$HF_TOKEN ./run_docker_background.sh
```

### Option 3: Docker (Foreground)

```bash
HF_TOKEN=$HF_TOKEN ./run_docker.sh
```

### Option 4: Manual Docker Run

```bash
docker run -d \
  --name bentoml-asr-service \
  --gpus all \
  -p 3000:3000 \
  -e HF_TOKEN=$HF_TOKEN \
  indic_conformer_asr:latest
```

## 📡 API Documentation

### Endpoint: `POST /asr`

Transcribe audio to text.

#### Request

**Content-Type**: `multipart/form-data`

**Parameters**:
- `file` (required): Audio file (WAV format)
- `lang` (optional): Language code (default: `"hi"`)
  - Supported: `ta`, `hi`, `en`, `te`, `kn`, `ml`, `mr`, `gu`, `pa`, `bn`, `or`, `as`, `ne`, `ur`, `si`, `my`, `ks`, `sd`, `sa`
- `strategy` (optional): Decoding strategy (default: `"ctc"`)

#### Example Request

```bash
curl -X POST http://localhost:3000/asr \
  -F "file=@audio.wav" \
  -F "lang=ta" \
  -F "strategy=ctc"
```

#### Response

**Success (200 OK)**:
```json
{
  "text": "transcribed text in the specified language",
  "lang": "ta",
  "strategy": "ctc"
}
```

**Error (500 Internal Server Error)**:
```json
{
  "error": "Error message describing what went wrong"
}
```

#### Response Codes

- `200 OK`: Successful transcription
- `400 Bad Request`: Invalid request (missing file, invalid format)
- `500 Internal Server Error`: Server error (model loading, inference failure)

## 🐳 Docker Deployment

### Building Docker Image

```bash
# Build BentoML bundle first
bentoml build

# Containerize
bentoml containerize indic_conformer_asr:latest -t indic_conformer_asr:latest
```

### Running Container

#### With GPU Support

```bash
docker run -d \
  --name bentoml-asr-service \
  --gpus all \
  -p 3000:3000 \
  -e HF_TOKEN=$HF_TOKEN \
  indic_conformer_asr:latest
```

#### Without GPU (CPU only)

```bash
docker run -d \
  --name bentoml-asr-service \
  -p 3000:3000 \
  -e HF_TOKEN=$HF_TOKEN \
  -e DEVICE=cpu \
  indic_conformer_asr:latest
```

### Environment Variables

- `HF_TOKEN` (required): HuggingFace authentication token
- `MODEL_ID` (optional): Model ID (default: `ai4bharat/indic-conformer-600m-multilingual`)
- `DEVICE` (optional): Device to use (`cuda` or `cpu`, default: `cuda` if GPU available)

### Container Management

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

For detailed Docker usage, see [DOCKER_USAGE.md](DOCKER_USAGE.md).

## 🏛️ Architecture

### Service Components

1. **Service Definition** (`service.py`):
   - BentoML service class with API endpoint
   - Model loading and inference logic
   - Audio preprocessing (WAV loading, resampling)

2. **Model Runner**:
   - Lazy loading of ASR model
   - GPU/CPU device management
   - ONNX Runtime inference

3. **Middleware**:
   - Multipart form data parsing
   - Request preprocessing

### Request Flow

```
Client Request (multipart/form-data)
    ↓
MultipartMiddleware (parse form data)
    ↓
predict() method
    ↓
Audio Loading (Python wave module)
    ↓
Audio Preprocessing (resample to 16kHz, mono)
    ↓
Model Inference (ONNX Runtime)
    ↓
Response (JSON with transcription)
```

### Model Loading Strategy

- **Configuration**: Stored in BentoML model store
- **Weights**: Loaded from HuggingFace at runtime
- **Caching**: Model loaded once and reused for subsequent requests
- **Device**: Automatically uses GPU if available

## 🔍 Troubleshooting

### Common Issues

#### 1. Model Not Found

**Error**: `no Models with name 'indic_conformer_600m_model' exist`

**Solution**:
```bash
# Download model to BentoML store
python download_model.py --token $HF_TOKEN

# Verify
bentoml models list
```

#### 2. GPU Not Available

**Error**: `CUDA not available` or GPU utilization is 0%

**Solution**:
- Verify GPU: `nvidia-smi`
- Check Docker GPU support: `docker run --rm --gpus all nvidia/cuda:11.8.0-base nvidia-smi`
- Ensure `--gpus all` flag is used when running container

#### 3. Audio Loading Errors

**Error**: `Failed to load audio`

**Solution**:
- Ensure audio is in WAV format
- Check audio file is not corrupted
- Verify audio is readable (permissions)

#### 4. HF_TOKEN Not Set

**Error**: `HF_TOKEN not set` or `401 Unauthorized`

**Solution**:
```bash
export HF_TOKEN=your_token_here
# Or pass when running Docker
docker run -e HF_TOKEN=$HF_TOKEN ...
```

#### 5. Port Already in Use

**Error**: `Address already in use`

**Solution**:
```bash
# Find process using port
sudo lsof -i :3000

# Kill process or use different port
docker run -p 3001:3000 ...
```

#### 6. Build Failures

**Error**: `ModuleNotFoundError` during build

**Solution**:
- Check `requirements.txt` includes all dependencies
- Verify `bento.yml` configuration
- Rebuild: `bentoml delete indic_conformer_asr --yes && bentoml build`

### Getting Help

1. Check [DOCKER_USAGE.md](DOCKER_USAGE.md) for Docker-specific issues
2. Review [ISSUES_FIXED.md](ISSUES_FIXED.md) for known issues and solutions
3. Check container logs: `docker logs bentoml-asr-service`
4. Verify environment: `env | grep HF_TOKEN`

## 📚 Additional Resources

### Documentation Files

- [DOCKER_USAGE.md](DOCKER_USAGE.md) - Detailed Docker usage guide
- [README_CONTAINER.md](README_CONTAINER.md) - Container deployment guide
- [ISSUES_FIXED.md](ISSUES_FIXED.md) - Known issues and fixes

### Scripts

- `download_model.py` - Download and save model to BentoML
- `run_docker_background.sh` - Run container in background
- `run_docker.sh` - Run container in foreground
- `test_curl.sh` - Test the service with curl

### Configuration Files

- `bento.yml` - BentoML service configuration
- `requirements.txt` - Python dependencies
- `service.py` - Main service implementation

## 🔗 Related Links

- [BentoML Documentation](https://docs.bentoml.com/)
- [AI4Bharat IndicConformer Model](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual)
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)

## 📝 Notes

- The service uses Python's `wave` module for audio loading (no FFmpeg required)
- Model weights are loaded from HuggingFace at runtime, not stored in the bundle
- GPU acceleration requires NVIDIA GPU with CUDA support
- The service automatically falls back to CPU if GPU is not available

---

**Last Updated**: December 2024
