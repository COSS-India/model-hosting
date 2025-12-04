# Setup Guide

Complete setup instructions for the ASR Framework Benchmarking Suite.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Initial Setup](#initial-setup)
3. [HuggingFace Token Setup](#huggingface-token-setup)
4. [FastAPI Setup](#fastapi-setup)
5. [BentoML Setup](#bentoml-setup)
6. [Docker Setup](#docker-setup)
7. [GPU Setup](#gpu-setup)
8. [Verification](#verification)
9. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.10 or higher
- **RAM**: 16GB
- **Disk**: 20GB free space
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional but recommended)

### Recommended Requirements
- **OS**: Ubuntu 22.04 LTS
- **Python**: 3.10 or 3.11
- **RAM**: 32GB
- **Disk**: 50GB free space
- **GPU**: NVIDIA GPU with 16GB+ VRAM (RTX 3090, A100, etc.)

## Initial Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd Benchmarking
```

### 2. Install System Dependencies

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    docker.io \
    docker-compose \
    git \
    curl \
    wget
```

#### Install NVIDIA Container Toolkit (for GPU support)
```bash
# Add NVIDIA package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker

# Verify installation
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### 3. Set Up Python Environment

```bash
# Create virtual environment (optional but recommended)
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

## HuggingFace Token Setup

### 1. Get Your HuggingFace Token

1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create a new token with "Read" permissions
3. Copy the token

### 2. Set Environment Variable

```bash
# Add to ~/.bashrc or ~/.zshrc for persistence
export HF_TOKEN=your_token_here

# Or set temporarily
export HF_TOKEN=your_token_here
```

### 3. Verify Token

```bash
# Test token access
huggingface-cli login --token $HF_TOKEN
```

## FastAPI Setup

### Option 1: Docker (Recommended)

```bash
cd Frameworks/fastapi

# Build Docker image
docker build -t asr-server:latest .

# Run container
docker run -d \
  --name asr-fastapi \
  --gpus all \
  -p 8000:8000 \
  -e HF_TOKEN=$HF_TOKEN \
  -e DEVICE=cuda \
  asr-server:latest

# Check logs
docker logs -f asr-fastapi
```

### Option 2: Local Installation

```bash
cd Frameworks/fastapi

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export HF_TOKEN=your_token_here
export DEVICE=cuda  # or "cpu"

# Run server
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Verify FastAPI Service

```bash
# Test endpoint
curl -X POST http://localhost:8000/asr \
  -F "audio=@../ta2.wav" \
  -F "lang=ta"
```

## BentoML Setup

### 1. Install BentoML

```bash
cd Frameworks/bento-ml

# Install BentoML and dependencies
pip install -r requirements.txt
```

### 2. Download Model to BentoML

```bash
# Download and save model to BentoML model store
python download_model.py --token $HF_TOKEN

# Verify model is saved
bentoml models list
```

### 3. Build BentoML Bundle

```bash
# Build the BentoML service bundle
bentoml build

# Verify bundle
bentoml list
```

### 4. Containerize (Docker)

```bash
# Containerize the bundle
bentoml containerize indic_conformer_asr:latest -t indic_conformer_asr:latest

# Verify Docker image
docker images | grep indic_conformer_asr
```

### 5. Run BentoML Service

```bash
# Run in background
HF_TOKEN=$HF_TOKEN ./run_docker_background.sh

# Or run manually
docker run -d \
  --name bentoml-asr-service \
  --gpus all \
  -p 3000:3000 \
  -e HF_TOKEN=$HF_TOKEN \
  indic_conformer_asr:latest
```

### Verify BentoML Service

```bash
# Test endpoint
curl -X POST http://localhost:3000/asr \
  -F "file=@../ta2.wav" \
  -F "lang=ta" \
  -F "strategy=ctc"
```

## Docker Setup

### Verify Docker Installation

```bash
# Check Docker version
docker --version

# Check Docker Compose version
docker-compose --version

# Test Docker
docker run hello-world
```

### Verify GPU Support in Docker

```bash
# Test NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

If this fails, see [GPU Setup](#gpu-setup) section.

## GPU Setup

### 1. Install NVIDIA Drivers

```bash
# Check current driver
nvidia-smi

# If not installed, install drivers
sudo apt-get update
sudo apt-get install -y nvidia-driver-535  # or latest version
sudo reboot
```

### 2. Install CUDA Toolkit

```bash
# Download CUDA 11.8 (or latest compatible version)
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run

# Install (follow prompts)
sudo sh cuda_11.8.0_520.61.05_linux.run
```

### 3. Set CUDA Environment Variables

```bash
# Add to ~/.bashrc
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# Reload
source ~/.bashrc
```

### 4. Verify GPU Setup

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA
nvcc --version

# Test PyTorch GPU
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Verification

### 1. Test FastAPI Service

```bash
# Start service
cd Frameworks/fastapi
docker run -d --gpus all -p 8000:8000 \
  -e HF_TOKEN=$HF_TOKEN \
  asr-server:latest

# Wait for service to start
sleep 10

# Test
curl -X POST http://localhost:8000/asr \
  -F "audio=@../ta2.wav" \
  -F "lang=ta"
```

### 2. Test BentoML Service

```bash
# Start service
cd Frameworks/bento-ml
HF_TOKEN=$HF_TOKEN ./run_docker_background.sh

# Wait for service to start
sleep 15

# Test
curl -X POST http://localhost:3000/asr \
  -F "file=@../ta2.wav" \
  -F "lang=ta" \
  -F "strategy=ctc"
```

### 3. Run Benchmark

```bash
cd Frameworks

# Benchmark FastAPI
python benchmark_asr.py \
  --endpoint http://localhost:8000/asr \
  --audio ../ta2.wav \
  --lang_id ta \
  --outputdir ../bench_results_fastapi

# Benchmark BentoML
python benchmark_asr.py \
  --endpoint http://localhost:3000/asr \
  --audio ../ta2.wav \
  --lang_id ta \
  --outputdir ../bench_results_bento
```

## Troubleshooting

### Common Issues

#### 1. GPU Not Available in Docker

**Error**: `docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]`

**Solution**:
```bash
# Install NVIDIA Container Toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### 2. Model Download Fails

**Error**: `401 Unauthorized` or `Model not found`

**Solution**:
- Verify HF_TOKEN is set: `echo $HF_TOKEN`
- Check token has read permissions
- Accept model terms on HuggingFace if model is gated

#### 3. Out of Memory

**Error**: `CUDA out of memory`

**Solution**:
- Reduce batch size in service code
- Use CPU mode: `export DEVICE=cpu`
- Use a smaller model or reduce audio length

#### 4. Port Already in Use

**Error**: `Address already in use`

**Solution**:
```bash
# Find process using port
sudo lsof -i :8000  # or :3000

# Kill process or use different port
docker run -p 8001:8000 ...
```

#### 5. BentoML Build Fails

**Error**: `ModuleNotFoundError` or build errors

**Solution**:
- Ensure all dependencies in `requirements.txt`
- Check `bento.yml` configuration
- Rebuild: `bentoml delete indic_conformer_asr --yes && bentoml build`

#### 6. Audio Loading Errors

**Error**: `FFmpeg not found` or audio loading fails

**Solution**:
- BentoML uses Python's `wave` module (no FFmpeg needed)
- FastAPI may need FFmpeg for some formats
- Ensure audio is in WAV format

### Getting Help

1. Check framework-specific documentation
2. Review error logs: `docker logs <container-name>`
3. Verify environment variables: `env | grep HF_TOKEN`
4. Check GPU: `nvidia-smi`
5. Open an issue on GitHub with:
   - Error message
   - System information
   - Steps to reproduce

## Next Steps

After setup is complete:

1. **Run Benchmarks**: See [README.md](README.md#benchmarking)
2. **Compare Results**: Check Excel files in output directories
3. **Optimize**: Adjust parameters based on results
4. **Scale**: Test with higher concurrency and rates

---

**Need Help?** Open an issue on GitHub or check the framework-specific documentation.

