# ASR Framework Benchmarking Suite

A comprehensive benchmarking suite for comparing Automatic Speech Recognition (ASR) performance across different frameworks: **FastAPI**, **BentoML**, and **NVIDIA Triton**.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Usage](#usage)
- [Benchmarking](#benchmarking)
- [Framework-Specific Guides](#framework-specific-guides)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project provides:
- **ASR Model**: AI4Bharat IndicConformer-600M-Multilingual model
- **Multiple Framework Implementations**: FastAPI, BentoML
- **Comprehensive Benchmarking**: Latency (p50, p95, p99), QPS, GPU/CPU utilization, Memory usage, Throughput
- **Docker Support**: Containerized deployments for all frameworks
- **Excel Reports**: Detailed performance metrics in spreadsheet format

## ✨ Features

- **Multi-Framework Support**: Compare performance across FastAPI and BentoML
- **Comprehensive Metrics**: 
  - Latency percentiles (p50, p95, p99)
  - Queries Per Second (QPS)
  - GPU Utilization and Memory
  - CPU and System Memory Usage
  - Throughput vs Latency analysis
- **Dockerized Deployments**: Easy-to-deploy containerized services
- **Framework-Independent Benchmarking**: Single benchmark script works with all frameworks
- **Excel Export**: All metrics exported to Excel for easy comparison

## 📁 Project Structure

```
Benchmarking/
├── README.md                 # This file
├── SETUP.md                  # Detailed setup instructions
├── .gitignore               # Git ignore rules
│
├── Frameworks/
│   ├── benchmark_asr.py     # Main benchmark script (framework-independent)
│   │
│   ├── fastapi/             # FastAPI implementation
│   │   ├── app.py           # FastAPI ASR service
│   │   ├── Dockerfile       # Docker configuration
│   │   ├── requirements.txt # Python dependencies
│   │   └── README.md        # FastAPI-specific docs
│   │
│   └── bento-ml/            # BentoML implementation
│       ├── service.py       # BentoML ASR service
│       ├── download_model.py # Model download script
│       ├── bento.yml        # BentoML configuration
│       ├── requirements.txt # Python dependencies
│       ├── run_docker_background.sh # Docker run script
│       └── DOCKER_USAGE.md  # Docker usage guide
│
└── ta2.wav                   # Sample audio file for testing
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd Benchmarking
```

### 2. Set Up HuggingFace Token

```bash
export HF_TOKEN=your_huggingface_token_here
```

### 3. Choose a Framework

#### FastAPI (Docker)
```bash
cd Frameworks/fastapi
docker build -t asr-server:latest .
docker run --gpus all -p 8000:8000 -e HF_TOKEN=$HF_TOKEN asr-server:latest
```

#### BentoML (Docker)
```bash
cd Frameworks/bento-ml
# Download model to BentoML
python download_model.py --token $HF_TOKEN

# Build and containerize
bentoml build
bentoml containerize indic_conformer_asr:latest -t indic_conformer_asr:latest

# Run
HF_TOKEN=$HF_TOKEN ./run_docker_background.sh
```

### 4. Run Benchmark

```bash
cd Frameworks
python benchmark_asr.py \
  --endpoint http://localhost:8000/asr \
  --audio ../ta2.wav \
  --lang_id ta \
  --outputdir ../bench_results
```

## 📋 Prerequisites

- **Python 3.10+**
- **Docker** with NVIDIA Container Toolkit (for GPU support)
- **NVIDIA GPU** (recommended) with CUDA support
- **HuggingFace Account** with access token
- **Git**

### System Requirements

- **OS**: Linux (Ubuntu 20.04+ recommended)
- **GPU**: NVIDIA GPU with CUDA 11.8+ (for GPU acceleration)
- **RAM**: 16GB+ recommended
- **Disk**: 20GB+ free space (for models and containers)

## 🔧 Setup

See [SETUP.md](SETUP.md) for detailed setup instructions including:
- Environment setup
- Dependency installation
- Model download
- Docker configuration
- GPU setup

## 💻 Usage

### Starting a Service

#### FastAPI
```bash
cd Frameworks/fastapi
docker run --gpus all -p 8000:8000 \
  -e HF_TOKEN=$HF_TOKEN \
  -e DEVICE=cuda \
  asr-server:latest
```

#### BentoML
```bash
cd Frameworks/bento-ml
HF_TOKEN=$HF_TOKEN ./run_docker_background.sh
```

### Testing a Service

```bash
# FastAPI
curl -X POST http://localhost:8000/asr \
  -F "audio=@../ta2.wav" \
  -F "lang=ta"

# BentoML
curl -X POST http://localhost:3000/asr \
  -F "file=@../ta2.wav" \
  -F "lang=ta" \
  -F "strategy=ctc"
```

## 📊 Benchmarking

The benchmark script is framework-independent and works with any ASR endpoint.

### Basic Usage

```bash
python benchmark_asr.py \
  --endpoint http://localhost:8000/asr \
  --audio /path/to/audio.wav \
  --lang_id ta \
  --outputdir /path/to/results
```

### Advanced Options

```bash
python benchmark_asr.py \
  --endpoint http://localhost:8000/asr \
  --audio /path/to/audio.wav \
  --lang_id ta \
  --outputdir /path/to/results \
  --duration 60 \
  --rate 10 \
  --concurrency 5
```

### Parameters

- `--endpoint`: ASR service endpoint URL
- `--audio`: Path to audio file (WAV format)
- `--lang_id`: Language code (e.g., `ta`, `hi`, `en`)
- `--outputdir`: Output directory for results
- `--duration`: Benchmark duration in seconds (default: 60)
- `--rate`: Target requests per second (default: 10)
- `--concurrency`: Number of concurrent requests (default: 5)

### Output

The benchmark generates:
- **`benchmark_results.xlsx`**: Excel file with all metrics in separate columns
- **`requests.csv`**: Detailed request-level data
- **`gpu_samples.csv`**: GPU utilization samples
- **`sys_samples.csv`**: System resource samples

### Metrics Collected

- **Latency**: p50, p95, p99 percentiles (ms)
- **QPS**: Queries Per Second
- **GPU Utilization**: Average and peak (%)
- **GPU Memory**: Used and peak (MB)
- **CPU Usage**: Average and peak (%)
- **Memory Usage**: Average and peak (MB)
- **Throughput**: Requests processed per second
- **Success Rate**: Percentage of successful requests

## 📚 Framework-Specific Guides

### FastAPI

- [FastAPI README](Frameworks/fastapi/README.md)
- [Docker Setup](Frameworks/fastapi/README_DOCKER.md)
- [GPU Setup](Frameworks/fastapi/README_GPU_SETUP.md)

### BentoML

- [BentoML Docker Guide](Frameworks/bento-ml/DOCKER_USAGE.md)
- [Container Setup](Frameworks/bento-ml/README_CONTAINER.md)

## 📈 Results

Benchmark results are exported to Excel with the following columns:

| Metric | Description |
|--------|-------------|
| Latency p50 | 50th percentile latency (ms) |
| Latency p95 | 95th percentile latency (ms) |
| Latency p99 | 99th percentile latency (ms) |
| QPS | Queries Per Second |
| GPU Util Avg | Average GPU utilization (%) |
| GPU Util Peak | Peak GPU utilization (%) |
| GPU Mem Used | GPU memory used (MB) |
| GPU Mem Peak | Peak GPU memory (MB) |
| CPU Avg | Average CPU usage (%) |
| CPU Peak | Peak CPU usage (%) |
| Memory Avg | Average system memory (MB) |
| Memory Peak | Peak system memory (MB) |
| Throughput | Requests/second |
| Success Rate | Percentage of successful requests |


## 🙏 Acknowledgments

- [AI4Bharat](https://ai4bharat.iitm.ac.in/) for the IndicConformer model
- [HuggingFace](https://huggingface.co/) for model hosting
- FastAPI and BentoML communities



## 🔗 Related Links

- [AI4Bharat IndicConformer Model](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [BentoML Documentation](https://docs.bentoml.com/)

---

**Last Updated**: December 2024

