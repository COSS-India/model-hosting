# NMT Framework Benchmarking Suite

A comprehensive benchmarking suite for comparing Neural Machine Translation (NMT) performance across different frameworks: **Triton**, **FastAPI**, **BentoML**, and **MLflow**.

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
- **NMT Model**: AI4Bharat IndicTrans2 Neural Machine Translation model
- **Multiple Framework Implementations**: Triton, FastAPI, BentoML, MLflow
- **Comprehensive Benchmarking**: Latency (p50, p95, p99), QPS, GPU/CPU utilization, Memory usage, Throughput
- **Docker Support**: Containerized deployments for all frameworks
- **Excel Reports**: Detailed performance metrics in spreadsheet format

## ✨ Features

- **Multi-Framework Support**: Compare performance across Triton, FastAPI, BentoML, and MLflow
- **Comprehensive Metrics**: 
  - Latency percentiles (p50, p95, p99)
  - Queries Per Second (QPS)
  - GPU Utilization and Memory
  - CPU and System Memory Usage
  - Throughput vs Latency analysis
- **Dockerized Deployments**: Easy-to-deploy containerized services
- **Framework-Independent Benchmarking**: Benchmark scripts for each framework
- **Excel Export**: All metrics exported to Excel for easy comparison
- **Multi-Language Support**: English ↔ Indic and Indic ↔ Indic translation

## 📁 Project Structure

```
nmt-benchmarking/
├── README.md                 # This file
│
├── triton/                   # Triton implementation
│   ├── docker-compose.yml   # Docker Compose configuration
│   ├── run.sh               # Service startup script
│   ├── benchmark_nmt.py     # Benchmark script
│   ├── test_curl.sh         # Test curl commands
│   └── README.md            # Triton-specific docs
│
├── fastapi/                  # FastAPI implementation
│   ├── app.py               # FastAPI NMT service
│   ├── requirements.txt     # Python dependencies
│   ├── start_service.sh     # Service startup script
│   ├── benchmark_nmt.py     # Benchmark script
│   ├── test_curl.sh         # Test curl commands
│   └── README.md            # FastAPI-specific docs
│
├── bento-ml/                 # BentoML implementation
│   ├── service.py           # BentoML NMT service
│   ├── bentofile.yaml       # BentoML configuration
│   ├── requirements.txt     # Python dependencies
│   ├── start_service.sh     # Service startup script
│   ├── benchmark_nmt.py     # Benchmark script
│   ├── test_curl.sh         # Test curl commands
│   └── README.md            # BentoML-specific docs
│
└── mlflow/                   # MLflow implementation
    ├── mlflow_nmt_model.py  # MLflow PyFunc model wrapper
    ├── register_model.py    # Model registration script
    ├── serve_model.sh       # Service startup script
    ├── benchmark_nmt.py     # Benchmark script
    ├── test_curl.sh         # Test curl commands
    ├── requirements.txt     # Python dependencies
    └── README.md            # MLflow-specific docs
```

## 🚀 Quick Start

### 1. Set Up HuggingFace Token

```bash
export HF_TOKEN=your_huggingface_token_here
```

**Important**: You need to request access to the IndicTrans2 models:
- [En-Indic Model](https://huggingface.co/ai4bharat/indictrans2-en-indic-1B)
- [Indic-En Model](https://huggingface.co/ai4bharat/indictrans2-indic-en-1B)
- [Indic-Indic Model](https://huggingface.co/ai4bharat/indictrans2-indic-indic-1B)

### 2. Choose a Framework

#### Triton

```bash
cd nmt-benchmarking/triton

# Pull the Docker image
docker pull ai4bharat/triton-indictrans-v2:latest

# Start the service
docker-compose up -d
# OR
./run.sh

# Service will be available at http://localhost:8000
```

#### FastAPI

```bash
cd nmt-benchmarking/fastapi

# Create virtual environment
python3 -m venv fastapi
source fastapi/bin/activate

# Install dependencies
pip install -r requirements.txt

# Clone IndicTrans2 repository (if not already cloned)
git clone https://github.com/AI4Bharat/IndicTrans2
git clone https://github.com/VarunGumma/IndicTransToolkit
cd IndicTransToolkit && pip install -e . && cd ..

# Start the service
export HF_TOKEN=your_token_here
./start_service.sh 8000

# Service will be available at http://localhost:8000
```

#### BentoML

```bash
cd nmt-benchmarking/bento-ml

# Create virtual environment
python3 -m venv bento
source bento/bin/activate

# Install dependencies
pip install -r requirements.txt

# Clone IndicTrans2 repository (if not already cloned)
git clone https://github.com/AI4Bharat/IndicTrans2
git clone https://github.com/VarunGumma/IndicTransToolkit
cd IndicTransToolkit && pip install -e . && cd ..

# Build Bento
bentoml build

# Start the service
export HF_TOKEN=your_token_here
./start_service.sh

# Service will be available at http://localhost:3000
```

#### MLflow

```bash
cd nmt-benchmarking/mlflow

# Create virtual environment
python3 -m venv mlflow
source mlflow/bin/activate

# Install dependencies
pip install -r requirements.txt

# Clone IndicTrans2 repository (if not already cloned)
git clone https://github.com/AI4Bharat/IndicTrans2
git clone https://github.com/VarunGumma/IndicTransToolkit
cd IndicTransToolkit && pip install -e . && cd ..

# Register the model (one-time setup)
export HF_TOKEN=your_token_here
python3 register_model.py

# Start the service
./serve_model.sh 5000

# Service will be available at http://localhost:5000
```

### 3. Run Benchmark

```bash
# For Triton
cd nmt-benchmarking/triton
python3 benchmark_nmt.py \
  --endpoint http://localhost:8000/v2/models/nmt/infer \
  --input_text "Hello, how are you?" \
  --src_lang en \
  --tgt_lang hi \
  --outputdir ./bench_results \
  --rate 5.0 \
  --duration 30

# For FastAPI
cd nmt-benchmarking/fastapi
python3 benchmark_nmt.py \
  --endpoint http://localhost:8000/translate \
  --input_text "Hello, how are you?" \
  --src_lang eng_Latn \
  --tgt_lang hin_Deva \
  --outputdir ./bench_results \
  --rate 5.0 \
  --duration 30

# For BentoML
cd nmt-benchmarking/bento-ml
python3 benchmark_nmt.py \
  --endpoint http://localhost:3000/translate \
  --input_text "Hello, how are you?" \
  --src_lang eng_Latn \
  --tgt_lang hin_Deva \
  --outputdir ./bench_results \
  --rate 5.0 \
  --duration 30

# For MLflow
cd nmt-benchmarking/mlflow
python3 benchmark_nmt.py \
  --endpoint http://localhost:5000/invocations \
  --input_text "Hello, how are you?" \
  --src_lang eng_Latn \
  --tgt_lang hin_Deva \
  --outputdir ./bench_results \
  --rate 5.0 \
  --duration 30
```

## 📋 Prerequisites

- **Python 3.10+**
- **Docker** with NVIDIA Container Toolkit (for GPU support)
- **NVIDIA GPU** (recommended) with CUDA support
- **HuggingFace Account** with access token and model access
- **Git**

### System Requirements

- **OS**: Linux (Ubuntu 20.04+ recommended)
- **GPU**: NVIDIA GPU with CUDA 11.8+ (for GPU acceleration)
- **RAM**: 16GB+ recommended
- **Disk**: 30GB+ free space (for models and containers)

## 🔧 Setup

See framework-specific READMEs for detailed setup instructions:
- [Triton README](triton/README.md)
- [FastAPI README](fastapi/README.md)
- [BentoML README](bento-ml/README.md)
- [MLflow README](mlflow/README.md)

## 💻 Usage

### Starting a Service

#### Triton
```bash
cd nmt-benchmarking/triton
docker-compose up -d
# OR
./run.sh
```

#### FastAPI
```bash
cd nmt-benchmarking/fastapi
source fastapi/bin/activate
export HF_TOKEN=your_token_here
./start_service.sh 8000
```

#### BentoML
```bash
cd nmt-benchmarking/bento-ml
source bento/bin/activate
export HF_TOKEN=your_token_here
./start_service.sh
```

#### MLflow
```bash
cd nmt-benchmarking/mlflow
source mlflow/bin/activate
export HF_TOKEN=your_token_here
./serve_model.sh 5000
```

### Testing a Service

#### Triton
```bash
curl -X POST http://localhost:8000/v2/models/nmt/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "INPUT_TEXT",
        "datatype": "BYTES",
        "shape": [1, 1],
        "data": ["Hello"]
      },
      {
        "name": "INPUT_LANGUAGE_ID",
        "datatype": "BYTES",
        "shape": [1, 1],
        "data": ["en"]
      },
      {
        "name": "OUTPUT_LANGUAGE_ID",
        "datatype": "BYTES",
        "shape": [1, 1],
        "data": ["hi"]
      }
    ]
  }' | python3 -m json.tool
```

#### FastAPI
```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?",
    "src_lang": "eng_Latn",
    "tgt_lang": "hin_Deva"
  }' | python3 -m json.tool
```

#### BentoML
```bash
curl -X POST http://localhost:3000/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?",
    "src_lang": "eng_Latn",
    "tgt_lang": "hin_Deva"
  }' | python3 -m json.tool
```

#### MLflow
```bash
curl -X POST http://localhost:5000/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": {
      "text": "Hello, how are you?",
      "src_lang": "eng_Latn",
      "tgt_lang": "hin_Deva"
    }
  }' | python3 -m json.tool
```

## 📊 Benchmarking

Each framework has its own benchmark script optimized for the specific API format.

### Basic Usage

**For Triton:**
```bash
python3 benchmark_nmt.py \
  --endpoint http://localhost:8000/v2/models/nmt/infer \
  --input_text "Hello, how are you?" \
  --src_lang en \
  --tgt_lang hi \
  --outputdir ./bench_results
```

**For FastAPI:**
```bash
python3 benchmark_nmt.py \
  --endpoint http://localhost:8000/translate \
  --input_text "Hello, how are you?" \
  --src_lang eng_Latn \
  --tgt_lang hin_Deva \
  --outputdir ./bench_results
```

**For BentoML:**
```bash
python3 benchmark_nmt.py \
  --endpoint http://localhost:3000/translate \
  --input_text "Hello, how are you?" \
  --src_lang eng_Latn \
  --tgt_lang hin_Deva \
  --outputdir ./bench_results
```

**For MLflow:**
```bash
python3 benchmark_nmt.py \
  --endpoint http://localhost:5000/invocations \
  --input_text "Hello, how are you?" \
  --src_lang eng_Latn \
  --tgt_lang hin_Deva \
  --outputdir ./bench_results
```

### Advanced Options

**Common parameters for all frameworks:**
```bash
python3 benchmark_nmt.py \
  --endpoint <endpoint_url> \
  --input_text "Your text here" \
  --src_lang <source_language> \
  --tgt_lang <target_language> \
  --outputdir ./bench_results \
  --duration 60 \
  --rate 10 \
  --sample_interval 0.5
```

### Parameters

**Common Parameters (all scripts):**
- `--endpoint`: NMT service endpoint URL
- `--input_text`: Text to translate
- `--src_lang`: Source language code
- `--tgt_lang`: Target language code
- `--outputdir`: Output directory for results
- `--duration`: Benchmark duration in seconds (default: 30)
- `--rate`: Target requests per second (default: 10.0)
- `--sample_interval`: Sampling interval for GPU/CPU in seconds (default: 0.5)

**Language Codes:**
- **Triton**: Uses simple codes (`en`, `hi`, `ta`, etc.)
- **FastAPI/BentoML/MLflow**: Uses FLORES codes (`eng_Latn`, `hin_Deva`, `tam_Taml`, etc.)

### Framework-Specific Notes

- **Triton**: Uses Triton Inference Server format with tensor inputs. Language codes are simplified (e.g., `en`, `hi`)
- **FastAPI**: Uses JSON with Pydantic validation. FLORES language codes required
- **BentoML**: Uses JSON with BentoML service API. FLORES language codes required
- **MLflow**: Uses MLflow 2.0+ format with `inputs` wrapper. FLORES language codes required

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
- **Throughput**: Bytes/second, MB/second
- **Success Rate**: Percentage of successful requests

## 📚 Framework-Specific Guides

### Triton

- [Triton README](triton/README.md) - Comprehensive guide for Triton setup and usage
- [Triton CURL Examples](triton/CURL_EXAMPLES.md) - API usage examples
- [GPU Verification](triton/GPU_VERIFICATION.md) - How to verify GPU usage

### FastAPI

- [FastAPI README](fastapi/README.md) - FastAPI-specific documentation
- [FastAPI Test Script](fastapi/test_curl.sh) - Test examples

### BentoML

- [BentoML README](bento-ml/README.md) - BentoML-specific documentation
- [BentoML CURL Examples](bento-ml/CURL_EXAMPLES.md) - API usage examples
- [Model Access Guide](bento-ml/MODEL_ACCESS.md) - HuggingFace model access instructions

### MLflow

- [MLflow README](mlflow/README.md) - MLflow-specific documentation
- [MLflow Quick Start](mlflow/QUICK_START.md) - Quick start guide

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
| Throughput | Requests/second or MB/second |
| Success Rate | Percentage of successful requests |

## 🔍 Framework Comparison

This section compares benchmark results across all frameworks (Triton, FastAPI, BentoML, and MLflow) for the IndicTrans2 NMT model. Results are based on actual benchmark runs with the specified configurations. **Note**: Benchmarks were run with different target rates, so results should be interpreted in context of their test configurations.

### Quick Comparison Summary

| Aspect | Winner | Details |
|--------|--------|---------|
| **Lowest Latency (p50)** | 🥇 Triton | 93.83 ms (2.1x faster) |
| **Highest Throughput** | 🥇 Triton | 10.0 QPS (2x higher) |
| **Lowest GPU Memory** | 🥇 FastAPI | 2.7 GB (2.2x less) |
| **Lowest CPU Usage** | 🥇 MLflow | 4.9% avg |
| **Best Reliability** | 🥇 Triton/FastAPI/BentoML | 100% success rate |
| **Best for Production** | 🥇 Triton | Optimized inference server |
| **Easiest Development** | 🥇 FastAPI | Simple setup, auto docs |
| **Best Model Management** | 🥇 BentoML/MLflow | Built-in versioning/tracking |

### Performance Comparison Table

| Framework | Latency p50 (ms) | Latency p95 (ms) | Latency p99 (ms) | QPS | Success Rate | GPU Util Avg (%) | GPU Memory (MB) | CPU Avg (%) | Requests |
|-----------|------------------|------------------|------------------|-----|--------------|------------------|-----------------|-------------|----------|
| **Triton** | **93.83** ⚡ | **98.81** ⚡ | **152.54** ⚡ | **10.0** | 100% | 53.36 | 6,020 | 13.53 | 300 |
| **FastAPI** | 204.38 | 210.78 | 214.46 | 5.0 | 100% | 38.46 | **2,744** 💾 | 13.62 | 150 |
| **BentoML** | 199.12 | 209.05 | 214.50 | 5.0 | 100% | 37.87 | 5,052 | 13.44 | 150 |
| **MLflow** | 222.02 | 223.68 | 224.52 | 1.67 | 99.33% | 12.98 | 4,716 | **4.90** 🟢 | 150 |

**Legend:**
- ⚡ Best performance in category
- 💾 Lowest resource usage
- 🟢 Most efficient CPU usage

### Detailed Benchmark Results

#### Triton Inference Server
- **Test Configuration**: 30s duration, 10 req/s target rate
- **Latency Performance**: 
  - p50: 93.83 ms (fastest)
  - p95: 98.81 ms
  - p99: 152.54 ms
- **Throughput**: 10.0 QPS (matched target)
- **Resource Usage**: 
  - GPU: 53.36% avg utilization, 6,020 MB memory
  - CPU: 13.53% avg usage
- **Reliability**: 100% success rate (300/300 requests)
- **Strengths**: 
  - Lowest latency across all frameworks
  - Optimized for high-throughput inference
  - Efficient GPU utilization
- **Best For**: Production deployments requiring maximum performance

#### FastAPI
- **Test Configuration**: 30s duration, 5 req/s target rate
- **Latency Performance**: 
  - p50: 204.38 ms
  - p95: 210.78 ms
  - p99: 214.46 ms
- **Throughput**: 5.0 QPS (matched target)
- **Resource Usage**: 
  - GPU: 38.46% avg utilization, 2,744 MB memory (lowest)
  - CPU: 13.62% avg usage
- **Reliability**: 100% success rate (150/150 requests)
- **Strengths**: 
  - Low GPU memory footprint
  - Simple deployment and development
  - Automatic API documentation
  - Easy to customize and extend
- **Best For**: Development, prototyping, and small to medium-scale deployments

#### BentoML
- **Test Configuration**: 30s duration, 5 req/s target rate
- **Latency Performance**: 
  - p50: 199.12 ms (second fastest at 5 req/s)
  - p95: 209.05 ms
  - p99: 214.50 ms
- **Throughput**: 5.0 QPS (matched target)
- **Resource Usage**: 
  - GPU: 37.87% avg utilization, 5,052 MB memory
  - CPU: 13.44% avg usage
- **Reliability**: 100% success rate (150/150 requests)
- **Strengths**: 
  - Good balance of performance and ease of use
  - Model versioning and management built-in
  - Easy containerization
  - Production-ready features
- **Best For**: Teams requiring model management and versioning capabilities

#### MLflow
- **Test Configuration**: 30s duration, 5 req/s target rate
- **Latency Performance**: 
  - p50: 222.02 ms
  - p95: 223.68 ms
  - p99: 224.52 ms
- **Throughput**: 1.67 QPS (below target, had timeout issues)
- **Resource Usage**: 
  - GPU: 12.98% avg utilization (lowest), 4,716 MB memory
  - CPU: 4.90% avg usage (lowest)
- **Reliability**: 99.33% success rate (149/150 requests, 1 timeout)
- **Strengths**: 
  - Model tracking and experiment management
  - Model registry capabilities
  - Low CPU usage
- **Limitations**: 
  - Some timeout issues observed
  - Lower throughput capability
- **Best For**: ML lifecycle management and experiment tracking

### Key Insights

1. **Latency Performance**:
   - **Triton** achieves the lowest latency (~94 ms p50) - **2.1x faster** than other frameworks, due to its optimized inference server architecture
   - **BentoML** has slightly better latency than FastAPI (~199 ms vs ~204 ms) at 5 req/s
   - **FastAPI** and **BentoML** show similar latency patterns (~199-214 ms) at 5 req/s
   - **MLflow** has the highest latency (~222 ms) but with more consistent p95/p99 values (less variance)

2. **Throughput**:
   - **Triton** demonstrated highest throughput (10 req/s) with stable performance - **2x** the throughput of other frameworks
   - **FastAPI** and **BentoML** both maintained target rate of 5 req/s reliably (100% success)
   - **MLflow** struggled to maintain target rate, achieving only 1.67 req/s (33% of target)

3. **Resource Efficiency**:
   - **FastAPI** uses the least GPU memory (2.7 GB) - **2.2x less** than Triton, making it ideal for resource-constrained environments
   - **Triton** has highest GPU utilization (53%) indicating efficient use of compute resources
   - **MLflow** shows lowest CPU usage (4.9%) but also lowest throughput
   - **BentoML** and **MLflow** use similar GPU memory (~4.7-5.1 GB)

4. **Reliability**:
   - **Triton**, **FastAPI**, and **BentoML** achieved 100% success rates across all requests
   - **MLflow** had 1 timeout failure (99.33% success rate) - may need timeout tuning

5. **Performance Ranking (at tested rates)**:
   - **Best Overall Performance**: Triton (lowest latency + highest throughput)
   - **Best Resource Efficiency**: FastAPI (lowest GPU memory)
   - **Best for Model Management**: BentoML or MLflow (built-in versioning/tracking)
   - **Most Consistent Latency**: MLflow (tight p95/p99 distribution)

6. **Use Case Recommendations**:
   - **Maximum Performance/High Throughput Production**: Triton
   - **Development/Prototyping**: FastAPI (easiest setup, low memory)
   - **Model Management & Versioning**: BentoML or MLflow
   - **Resource Constrained Environments**: FastAPI (lowest GPU memory footprint)
   - **Experiment Tracking & ML Lifecycle**: MLflow
   - **Balanced Performance & Features**: BentoML

### Benchmark Configuration Notes

- **Test Input**: "Hello, how are you?" (19 bytes)
- **Translation**: English → Hindi
- **Triton**: Tested at 10 req/s for 30s (300 requests)
- **FastAPI/BentoML/MLflow**: Tested at 5 req/s for 30s (150 requests)
- All tests used the same hardware: NVIDIA Tesla T4 GPU, 8 vCPUs, 30GB RAM
- Results may vary based on hardware, model warm-up, and system load

### Generating Your Own Comparison

To generate updated comparisons, run benchmarks for all frameworks with the same configuration:

```bash
# Run benchmarks with identical parameters
# For 5 req/s, 30s duration:

# Triton
cd nmt-benchmarking/triton
python3 benchmark_nmt.py \
  --endpoint http://localhost:8000/v2/models/nmt/infer \
  --input_text "Hello, how are you?" \
  --src_lang en --tgt_lang hi \
  --outputdir ./bench_results \
  --rate 5.0 --duration 30

# FastAPI
cd nmt-benchmarking/fastapi
python3 benchmark_nmt.py \
  --endpoint http://localhost:8000/translate \
  --input_text "Hello, how are you?" \
  --src_lang eng_Latn --tgt_lang hin_Deva \
  --outputdir ./bench_results \
  --rate 5.0 --duration 30

# BentoML
cd nmt-benchmarking/bento-ml
python3 benchmark_nmt.py \
  --endpoint http://localhost:3000/translate \
  --input_text "Hello, how are you?" \
  --src_lang eng_Latn --tgt_lang hin_Deva \
  --outputdir ./bench_results \
  --rate 5.0 --duration 30

# MLflow
cd nmt-benchmarking/mlflow
python3 benchmark_nmt.py \
  --endpoint http://localhost:5000/invocations \
  --input_text "Hello, how are you?" \
  --src_lang eng_Latn --tgt_lang hin_Deva \
  --outputdir ./bench_results \
  --rate 5.0 --duration 30
```

Then compare the `benchmark_results.xlsx` files from each framework's `bench_results/` directory.

## 🌐 Supported Languages

### Translation Directions

The IndicTrans2 models support three types of translation:

1. **English → Indic**: Translate from English to any Indic language
   - Model: `ai4bharat/indictrans2-en-indic-1B`
   - Examples: English → Hindi, English → Tamil, English → Telugu

2. **Indic → English**: Translate from any Indic language to English
   - Model: `ai4bharat/indictrans2-indic-en-1B`
   - Examples: Hindi → English, Tamil → English, Telugu → English

3. **Indic → Indic**: Translate between Indic languages
   - Model: `ai4bharat/indictrans2-indic-indic-1B`
   - Examples: Hindi → Marathi, Tamil → Telugu, Bengali → Gujarati

### Language Codes

**Triton** uses simple language codes:
- `en` - English
- `hi` - Hindi
- `ta` - Tamil
- `te` - Telugu
- `mr` - Marathi
- `gu` - Gujarati
- And more...

**FastAPI/BentoML/MLflow** use FLORES codes:
- `eng_Latn` - English
- `hin_Deva` - Hindi (Devanagari)
- `tam_Taml` - Tamil
- `tel_Telu` - Telugu
- `mar_Deva` - Marathi
- `guj_Gujr` - Gujarati
- And more...

For the complete list of supported languages, see:
- [IndicTrans2 GitHub](https://github.com/AI4Bharat/IndicTrans2#supported-languages)

## 🙏 Acknowledgments

- [AI4Bharat](https://ai4bharat.iitm.ac.in/) for the IndicTrans2 model
- [HuggingFace](https://huggingface.co/) for model hosting
- FastAPI, BentoML, MLflow, and NVIDIA Triton communities

## 🔗 Related Links

- [AI4Bharat IndicTrans2 GitHub](https://github.com/AI4Bharat/IndicTrans2)
- [IndicTrans2 En-Indic Model](https://huggingface.co/ai4bharat/indictrans2-en-indic-1B)
- [IndicTrans2 Indic-En Model](https://huggingface.co/ai4bharat/indictrans2-indic-en-1B)
- [IndicTrans2 Indic-Indic Model](https://huggingface.co/ai4bharat/indictrans2-indic-indic-1B)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [BentoML Documentation](https://docs.bentoml.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server)

---

**Last Updated**: January 2025

