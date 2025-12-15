# ASR Framework Benchmarking Suite

A comprehensive benchmarking suite for comparing Automatic Speech Recognition (ASR) performance across different frameworks: **FastAPI**, **BentoML**, and **MLflow**.

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
- **Multiple Framework Implementations**: FastAPI, BentoML, MLflow
- **Comprehensive Benchmarking**: Latency (p50, p95, p99), QPS, GPU/CPU utilization, Memory usage, Throughput
- **Docker Support**: Containerized deployments for all frameworks
- **Excel Reports**: Detailed performance metrics in spreadsheet format

## ✨ Features

- **Multi-Framework Support**: Compare performance across FastAPI, BentoML, and MLflow
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
│   ├── benchmark_asr.py     # Main benchmark script (BentoML)
│   ├── benchmark_asr_fastapi.py # FastAPI-specific benchmark script
│   ├── benchmark_asr_mlflow.py # MLflow-specific benchmark script
│   │
│   ├── fastapi/             # FastAPI implementation
│   │   ├── app.py           # FastAPI ASR service
│   │   ├── Dockerfile       # Docker configuration
│   │   ├── requirements.txt # Python dependencies
│   │   └── README.md        # FastAPI-specific docs
│   │
│   ├── bento-ml/            # BentoML implementation
│   │   ├── service.py       # BentoML ASR service
│   │   ├── download_model.py # Model download script
│   │   ├── bento.yml        # BentoML configuration
│   │   ├── requirements.txt # Python dependencies
│   │   ├── run_docker_background.sh # Docker run script
│   │   └── DOCKER_USAGE.md  # Docker usage guide
│   │
│   └── MLflow/              # MLflow implementation
│       ├── mlflow_asr.py    # MLflow PyFunc model wrapper
│       ├── log_model.py     # Model logging script
│       ├── server.py        # Custom MLflow server
│       ├── Dockerfile       # Docker configuration
│       ├── requirements.txt # Python dependencies
│       └── README.md        # MLflow-specific docs
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

#### MLflow (Docker)
```bash
cd Frameworks/MLflow
# Log model to MLflow (one-time setup)
source mlflow/bin/activate
python log_model.py

# Build and run Docker container
./build_docker.sh
./run_docker.sh
```

### 4. Run Benchmark

```bash
cd Frameworks

# For FastAPI (uses 'audio' field name)
python benchmark_asr_fastapi.py \
  --endpoint http://localhost:8000/asr \
  --audio ../ta2.wav \
  --lang_id ta \
  --outputdir ../bench_results

# For BentoML (uses 'file' field name)
python benchmark_asr.py \
  --endpoint http://localhost:3000/asr \
  --audio ../ta2.wav \
  --lang_id ta \
  --outputdir ../bench_results

# For MLflow (uses different endpoint format)
python benchmark_asr_mlflow.py \
  --endpoint http://localhost:5000/asr \
  --audio ../ta2.wav \
  --lang_id ta \
  --decoding ctc \
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

#### MLflow
```bash
cd Frameworks/MLflow
./run_docker.sh
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

# MLflow
curl -X POST http://localhost:5000/asr \
  -H "Content-Type: application/json" \
  -d "$(python3 -c "import base64, json; audio_b64 = base64.b64encode(open('../ta2.wav', 'rb').read()).decode('utf-8'); print(json.dumps({'audio_base64': audio_b64, 'lang': 'ta', 'decoding': 'ctc'}))")" \
  | python3 -m json.tool
```

## 📊 Benchmarking

Each framework has its own benchmark script optimized for the specific API format.

### Basic Usage

**For FastAPI:**
```bash
python benchmark_asr_fastapi.py \
  --endpoint http://localhost:8000/asr \
  --audio /path/to/audio.wav \
  --lang_id ta \
  --outputdir /path/to/results
```

**For BentoML:**
```bash
python benchmark_asr.py \
  --endpoint http://localhost:3000/asr \
  --audio /path/to/audio.wav \
  --lang_id ta \
  --outputdir /path/to/results
```

**For MLflow:**
```bash
python benchmark_asr_mlflow.py \
  --endpoint http://localhost:5000/asr \
  --audio /path/to/audio.wav \
  --lang_id ta \
  --decoding ctc \
  --outputdir /path/to/results
```

### Advanced Options

**For FastAPI:**
```bash
python benchmark_asr_fastapi.py \
  --endpoint http://localhost:8000/asr \
  --audio /path/to/audio.wav \
  --lang_id ta \
  --outputdir /path/to/results \
  --duration 60 \
  --rate 10 \
  --sample_interval 0.5
```

**For BentoML:**
```bash
python benchmark_asr.py \
  --endpoint http://localhost:3000/asr \
  --audio /path/to/audio.wav \
  --lang_id ta \
  --outputdir /path/to/results \
  --duration 60 \
  --rate 10 \
  --sample_interval 0.5
```

**For MLflow:**
```bash
python benchmark_asr_mlflow.py \
  --endpoint http://localhost:5000/asr \
  --audio /path/to/audio.wav \
  --lang_id ta \
  --decoding ctc \
  --outputdir /path/to/results \
  --duration 60 \
  --rate 10 \
  --sample_interval 0.5
```

### Parameters

**Common Parameters (all scripts):**
- `--endpoint`: ASR service endpoint URL
- `--audio`: Path to audio file (WAV format)
- `--lang_id`: Language code (e.g., `ta`, `hi`, `en`)
- `--outputdir`: Output directory for results
- `--duration`: Benchmark duration in seconds (default: 30)
- `--rate`: Target requests per second (default: 10.0)
- `--sample_interval`: Sampling interval for GPU/CPU in seconds (default: 0.5)

**MLflow specific:**
- `--decoding`: Decoding strategy - `ctc` or `greedy` (default: ctc)

### Framework-Specific Notes

- **FastAPI** (`benchmark_asr_fastapi.py`): Uses `audio` field name in multipart form data
- **BentoML** (`benchmark_asr.py`): Uses `file` field name and includes `strategy` parameter
- **MLflow** (`benchmark_asr_mlflow.py`): Uses JSON format with `audio_base64` field

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

### MLflow

- [MLflow README](Frameworks/MLflow/README.md)
- [MLflow Quick Start](Frameworks/MLflow/QUICKSTART.md)
- [MLflow Docker Guide](Frameworks/MLflow/DOCKER.md)
- [MLflow Docker Quick Start](Frameworks/MLflow/DOCKER_QUICKSTART.md)
- [MLflow API Reference](Frameworks/MLflow/API_REFERENCE.md)

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
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

---

**Last Updated**: December 2024

