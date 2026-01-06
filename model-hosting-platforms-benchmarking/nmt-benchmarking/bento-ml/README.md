# BentoML NMT Service - IndicTrans2

This directory contains a comprehensive BentoML implementation for hosting the AI4Bharat IndicTrans2 Neural Machine Translation model.

## 📋 Table of Contents

- [What is BentoML?](#what-is-bentoml)
- [Overview](#overview)
- [The IndicTrans2 Model](#the-indictrans2-model)
- [How We Got the Model](#how-we-got-the-model)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Bento Build Process](#bento-build-process)
- [Serving the Model](#serving-the-model)
- [Usage](#usage)
- [Benchmarking](#benchmarking)
- [API Reference](#api-reference)
- [Docker Deployment](#docker-deployment)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## 🤔 What is BentoML?

**BentoML** is an open-source framework for building and deploying machine learning services. It provides:

1. **Service Definition**: Simple Python decorators to define ML services
2. **Automatic API Generation**: REST APIs generated from your Python functions
3. **Model Packaging**: Package models, code, and dependencies into deployable artifacts (Bentos)
4. **Production Deployment**: Easy deployment to cloud, Kubernetes, or Docker
5. **Type Safety**: Automatic request/response validation using Pydantic
6. **Async Support**: Native async/await for high-performance serving

### Why BentoML for NMT?

- **Easy Service Definition**: Simple Python decorators to expose translation APIs
- **Automatic Validation**: Type-safe API with Pydantic models
- **Production Ready**: Built-in support for production deployments
- **Docker Support**: One command to containerize your service
- **Model Management**: Built-in model versioning and storage
- **Async Support**: High-performance async inference
- **Developer Friendly**: Fast development and iteration cycle

### BentoML Components Used

1. **BentoML Service**: Core service definition with `@bentoml.api` decorators
2. **Bento Package**: Deployable artifact containing model, code, and dependencies
3. **Bento Build**: Packaging system that creates the Bento
4. **Bento Serve**: Development server for testing
5. **Bento Containerize**: Docker image generation from Bento

## 🎯 Overview

This project implements a BentoML-based service for the **AI4Bharat IndicTrans2** Neural Machine Translation model, providing a REST API for text translation between:

- **English ↔ Indic** languages (e.g., English to Hindi, Hindi to English)
- **Indic ↔ Indic** languages (e.g., Hindi to Marathi, Tamil to Telugu)

### Supported Translation Directions

| Type | Source | Target | Model Used |
|------|--------|--------|------------|
| En-Indic | English | Any Indic language | `ai4bharat/indictrans2-en-indic-1B` |
| Indic-En | Any Indic language | English | `ai4bharat/indictrans2-indic-en-1B` |
| Indic-Indic | Any Indic language | Any other Indic language | `ai4bharat/indictrans2-indic-indic-1B` |

### Models Used

- **En-Indic**: `ai4bharat/indictrans2-en-indic-1B` (1B parameters)
- **Indic-En**: `ai4bharat/indictrans2-indic-en-1B` (1B parameters)
- **Indic-Indic**: `ai4bharat/indictrans2-indic-indic-1B` (1B parameters)

**Source**: [AI4Bharat IndicTrans2 GitHub Repository](https://github.com/AI4Bharat/IndicTrans2)

## 📦 The IndicTrans2 Model

### Model Architecture

IndicTrans2 is a neural machine translation model based on the **IndicBART** architecture, which is a multilingual sequence-to-sequence model specifically designed for Indic languages. The model uses:

- **Transformer-based architecture** with encoder-decoder structure
- **Multilingual training** on large-scale parallel corpora
- **FLORES language codes** for standardized language identification
- **IndicTransToolkit** for preprocessing and postprocessing

### Features

- **22+ Indic languages** supported
- **High-quality translations** across multiple language pairs
- **Robust handling** of script variations and transliterations
- **Sentence-level** and **document-level** translation support

### Model Access

The models are hosted on **HuggingFace Hub** and require:
1. **HuggingFace account** with access token
2. **Access approval** for gated models (request access on model pages)
3. **Valid HF_TOKEN** environment variable

**Request Access:**
- [En-Indic Model](https://huggingface.co/ai4bharat/indictrans2-en-indic-1B)
- [Indic-En Model](https://huggingface.co/ai4bharat/indictrans2-indic-en-1B)
- [Indic-Indic Model](https://huggingface.co/ai4bharat/indictrans2-indic-indic-1B)

## 🔍 How We Got the Model

### Step 1: Discovering IndicTrans2

The IndicTrans2 model was identified from:
- **AI4Bharat Research**: Published research on multilingual Indic translation
- **HuggingFace Hub**: Official model releases by AI4Bharat
- **GitHub Repository**: [AI4Bharat/IndicTrans2](https://github.com/AI4Bharat/IndicTrans2)

### Step 2: Model Repository Structure

```
IndicTrans2/
├── huggingface_interface/     # HuggingFace-compatible model code
│   ├── modeling_indictrans.py # Custom model architecture
│   ├── configuration_indictrans.py # Model configuration
│   └── convert_indictrans_checkpoint_to_pytorch.py
├── inference/                 # Inference scripts
│   ├── engine.py             # Inference engine
│   └── requirements.txt      # Dependencies
└── README.md                 # Documentation
```

### Step 3: Cloning the Repository

```bash
cd /home/ubuntu/nmt-benchmarking/bento-ml
git clone https://github.com/AI4Bharat/IndicTrans2
```

This gives us:
- Model architecture code (custom transformers implementation)
- Inference utilities
- Documentation and examples

### Step 4: Installing IndicTransToolkit

The preprocessing/postprocessing toolkit:

```bash
git clone https://github.com/VarunGumma/IndicTransToolkit
cd IndicTransToolkit
pip install -e .
cd ..
```

**IndicTransToolkit provides:**
- Sentence tokenization for Indic languages
- Script normalization
- Preprocessing for model input
- Postprocessing for model output

### Step 5: Model Access Setup

1. **Create HuggingFace account** (if not exists)
2. **Generate access token**: https://huggingface.co/settings/tokens
3. **Request model access** on each model page:
   - Click "Request access" button
   - Wait for approval (usually quick)
4. **Set environment variable**:
   ```bash
   export HF_TOKEN=your_token_here
   ```

**Important**: Never hardcode the token. Always use environment variables.

### Step 6: Model Loading

Models are loaded on-demand using HuggingFace Transformers with lazy loading:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Lazy loading - model loaded only when needed
model = AutoModelForSeq2SeqLM.from_pretrained(
    "ai4bharat/indictrans2-en-indic-1B",
    trust_remote_code=True,
    token=HF_TOKEN
)
```

The `trust_remote_code=True` is required because IndicTrans2 uses custom model code.

## 🏗️ How We're Hosting in BentoML

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  BentoML Architecture                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │ service.py   │      │ BentoML      │                    │
│  │ (Service     │ ───> │ Service      │                    │
│  │  Definition) │      │ Framework    │                    │
│  └──────────────┘      └──────────────┘                    │
│         │                     │                             │
│         │                     │                             │
│         v                     v                             │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │ IndicTrans   │      │ BentoML      │                    │
│  │ NMTModel     │      │ Model Store  │                    │
│  │ (Runner)     │      │ (~/.bentoml) │                    │
│  └──────────────┘      └──────────────┘                    │
│         │                                                     │
│         v                                                     │
│  ┌──────────────┐                                            │
│  │ bentoml      │                                            │
│  │ build        │  Creates Bento package                    │
│  └──────────────┘                                            │
│         │                                                     │
│         v                                                     │
│  ┌──────────────┐                                            │
│  │ Bento        │  indictrans_nmt:<tag>                     │
│  │ Package      │  (contains model, code, deps)             │
│  └──────────────┘                                            │
│         │                                                     │
│         v                                                     │
│  ┌──────────────┐                                            │
│  │ bentoml      │                                            │
│  │ serve        │  Development server                       │
│  └──────────────┘  OR                                        │
│         │                                                     │
│         v                                                     │
│  ┌──────────────┐                                            │
│  │ bentoml      │                                            │
│  │ containerize │  Docker image                             │
│  └──────────────┘                                            │
│         │                                                     │
│         v                                                     │
│  ┌──────────────┐                                            │
│  │ REST API     │  POST /translate                          │
│  │ (uvicorn)    │  {"text": "...", ...}                     │
│  └──────────────┘                                            │
└─────────────────────────────────────────────────────────────┘
```

### Step-by-Step BentoML Integration

#### Step 1: Define the Service

We created `service.py` with:

**a) Model Runner Class**:
```python
class IndicTransNMTModel:
    """Model runner for IndicTrans2"""
    def __init__(self):
        # Initialize lazy-loaded models
        self.en_indic_model = None
        self.indic_en_model = None
        self.indic_indic_model = None
    
    async def translate_text(self, text: str, src_lang: str, tgt_lang: str):
        # Load appropriate model based on language pair
        # Perform translation
        # Return result
```

**Key Features:**
- **Lazy Model Loading**: Models loaded only when needed (En-Indic, Indic-En, or Indic-Indic)
- **Automatic Model Selection**: Chooses correct model based on language pair
- **Async Support**: Uses `asyncio` and `ThreadPoolExecutor` for non-blocking inference
- **Batch Processing**: Handles multiple sentences efficiently

**b) Pydantic Models for Type Safety**:
```python
class TranslationInput(bentoml.BaseModel):
    text: str
    src_lang: str
    tgt_lang: str

class TranslationOutput(bentoml.BaseModel):
    translation: str
    src_lang: str
    tgt_lang: str
```

**c) BentoML Service Definition**:
```python
svc = Service("indictrans_nmt", inner=IndicTransNMTModel())

@svc.api(
    input=JSON(pydantic_model=TranslationInput),
    output=JSON(pydantic_model=TranslationOutput),
    route="/translate"
)
async def translate(input_data: TranslationInput) -> TranslationOutput:
    """Translate text from source language to target language"""
    runner = svc.inner
    result = await runner.translate_text(
        input_data.text, input_data.src_lang, input_data.tgt_lang
    )
    return TranslationOutput(**result)
```

#### Step 2: Configure Bento Build

We created `bentofile.yaml`:

```yaml
service: service:svc
include:
  - service.py
  - requirements.txt
  - IndicTrans2/              # Model code
  - IndicTransToolkit/        # Preprocessing toolkit

python:
  requirements_txt: requirements.txt
```

**What this does:**
- Specifies the service entry point (`service:svc`)
- Includes all necessary files and directories
- Lists Python dependencies from `requirements.txt`

#### Step 3: Build the Bento

```bash
bentoml build
```

**What happens during build:**

1. **Validates service**: Checks if `service:svc` is valid
2. **Resolves dependencies**: Installs packages from `requirements.txt`
3. **Packages files**: Copies all files listed in `include`
4. **Creates Bento**: Generates a versioned Bento package
5. **Stores locally**: Saves to `~/.bentoml/bentos/<name>/<tag>/`

**Bento Structure:**
```
indictrans_nmt:latest/
├── bento.yaml              # Bento metadata
├── api/                    # API definitions
├── env/                    # Environment files
├── models/                 # Model artifacts (if stored)
├── src/                    # Source code
│   ├── service.py
│   ├── IndicTrans2/
│   └── IndicTransToolkit/
└── README.md               # Auto-generated README
```

#### Step 4: Serve the Bento

**Development mode:**
```bash
bentoml serve indictrans_nmt:latest --port 3000 --host 0.0.0.0
```

**Production mode:**
```bash
bentoml serve indictrans_nmt:latest --production --port 3000
```

**What happens:**
- BentoML loads the Bento package
- Initializes the service and model runner
- Starts **uvicorn** server (ASGI/WSGI server)
- Creates REST API endpoints from `@svc.api` decorators
- Handles requests with automatic validation

### BentoML Features Used

1. **Service Definition**: `Service()` and `@svc.api()` decorators
2. **Type Safety**: Pydantic models for request/response validation
3. **Async Support**: Native async/await for high performance
4. **Model Runner**: Separates model logic from API logic
5. **Bento Packaging**: Versioned, reproducible deployments
6. **Automatic API Generation**: REST APIs from Python functions

## 📁 Project Structure

```
bento-ml/
├── service.py              # BentoML service definition
├── requirements.txt        # Python dependencies
├── bentofile.yaml         # BentoML build configuration
├── start_service.sh        # Service startup script
├── test_curl.sh           # Test curl commands
├── benchmark_nmt.py       # Benchmark script for performance testing
├── README.md              # This file
├── CURL_EXAMPLES.md       # Curl command examples
├── MODEL_ACCESS.md        # Model access instructions
├── QUICK_START.md         # Quick start guide
│
├── IndicTrans2/           # Cloned IndicTrans2 repository
│   ├── huggingface_interface/
│   ├── inference/
│   └── ...
│
├── IndicTransToolkit/     # Cloned IndicTransToolkit
│   └── ...
│
└── bento/                 # Python virtual environment
    └── ...
```

## 🔧 Setup and Installation

### Prerequisites

- **Python 3.10+**
- **NVIDIA GPU** with CUDA support (recommended)
- **HuggingFace account** with access to IndicTrans2 models
- **Git** for cloning repositories

### Step 1: Create Virtual Environment

```bash
cd /home/ubuntu/nmt-benchmarking/bento-ml
python3 -m venv bento
source bento/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Key Dependencies:**
- `bentoml>=1.4.0` - BentoML framework
- `torch>=2.0.0` - PyTorch for model inference
- `transformers>=4.30.0` - HuggingFace Transformers
- `indictranstoolkit` - Preprocessing/postprocessing
- `nltk`, `sacremoses`, `mosestokenizer` - Text processing
- `pydantic` - Type validation

### Step 3: Clone IndicTrans2 Repository

```bash
git clone https://github.com/AI4Bharat/IndicTrans2
```

### Step 4: Install IndicTransToolkit

```bash
git clone https://github.com/VarunGumma/IndicTransToolkit
cd IndicTransToolkit
pip install -e .
cd ..
```

### Step 5: Download NLTK Data

The service will automatically download NLTK data on first run, but you can pre-download:

```bash
python3 -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt')"
```

### Step 6: Set HuggingFace Token

```bash
export HF_TOKEN=your_huggingface_token_here
```

**Important**: Never hardcode the token. Always use environment variables.

### Step 7: Verify Installation

```bash
# Check BentoML installation
bentoml --version

# Check if service can be imported
python3 -c "from service import svc; print('Service loaded successfully')"
```

## 📦 Bento Build Process

### What is a Bento?

A **Bento** is a deployable artifact that contains:
- Your service code
- Model definitions and configurations
- Python dependencies
- System dependencies (if specified)
- Metadata and version information

### Building the Bento

```bash
cd /home/ubuntu/nmt-benchmarking/bento-ml
source bento/bin/activate
bentoml build
```

### Build Process Details

**Step 1: Service Validation**
- BentoML imports your service module
- Validates the service definition
- Checks for common errors

**Step 2: Dependency Resolution**
- Reads `requirements.txt`
- Resolves Python package versions
- Checks for conflicts

**Step 3: File Packaging**
- Copies files listed in `bentofile.yaml` `include` section
- Preserves directory structure
- Validates file existence

**Step 4: Bento Creation**
- Creates Bento directory structure
- Generates `bento.yaml` with metadata
- Packages all files into the Bento

**Step 5: Versioning**
- Assigns a tag (e.g., `latest`, or timestamp-based)
- Stores in `~/.bentoml/bentos/`

### Build Output

After successful build, you'll see:

```
Successfully built Bento(tag="indictrans_nmt:latest").
```

**Bento Location:**
```bash
~/.bentoml/bentos/indictrans_nmt/<tag>/
```

**List all Bentos:**
```bash
bentoml list
```

**Inspect a Bento:**
```bash
bentoml get indictrans_nmt:latest
```

### Bento Versioning

BentoML automatically versions your Bentos:

- **Tags**: Use descriptive tags like `latest`, `v1`, `production`
- **Automatic Tags**: Timestamp-based tags for each build
- **Metadata**: Each Bento stores build time, dependencies, and configuration

**Example:**
```bash
indictrans_nmt:latest
indictrans_nmt:20250115_123456
indictrans_nmt:v1.0.0
```

## 🚀 Serving the Model

### BentoML Serving

BentoML provides built-in serving using **uvicorn** (ASGI server) which creates a FastAPI-based REST API.

### Step 1: Start the Server

**Using the startup script (Recommended):**
```bash
export HF_TOKEN=your_token_here
./start_service.sh
```

**Manually:**
```bash
export HF_TOKEN=your_token_here
source bento/bin/activate
bentoml serve indictrans_nmt:latest --port 3000 --host 0.0.0.0
```

**In background:**
```bash
nohup env HF_TOKEN="$HF_TOKEN" bentoml serve indictrans_nmt:latest \
  --port 3000 --host 0.0.0.0 > bentoml.log 2>&1 &
```

### Step 2: Server Startup Process

When you start the server:

1. **BentoML loads** the Bento package from `~/.bentoml/bentos/`
2. **Initializes** the service (`IndicTransNMTModel`)
3. **Starts uvicorn** server on specified port
4. **Creates** REST API endpoints from `@svc.api` decorators
5. **Ready** to accept requests

### Step 3: Verify Server is Running

```bash
# Check if server is running
ps aux | grep "bentoml serve"

# Check logs (if running in background)
tail -f bentoml.log

# Test the API
curl -X POST http://localhost:3000/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello","src_lang":"eng_Latn","tgt_lang":"hin_Deva"}'
```

### Development vs Production Mode

**Development Mode** (default):
```bash
bentoml serve indictrans_nmt:latest --port 3000
```
- Single worker
- Auto-reload on code changes
- Detailed error messages
- Suitable for testing

**Production Mode**:
```bash
bentoml serve indictrans_nmt:latest --production --port 3000
```
- Multiple workers (based on CPU cores)
- No auto-reload
- Optimized for performance
- Suitable for production

## 💻 Usage

### API Endpoint

**Endpoint**: `POST /translate`

**Base URL**: `http://localhost:3000`

### Request Format

```json
{
  "text": "Hello, how are you?",
  "src_lang": "eng_Latn",
  "tgt_lang": "hin_Deva"
}
```

### Response Format

```json
{
  "translation": "नमस्ते, आप कैसे हैं?",
  "src_lang": "eng_Latn",
  "tgt_lang": "hin_Deva"
}
```

### Supported Language Codes

The service uses **FLORES language codes**. Common examples:

| Language | FLORES Code | Script |
|----------|-------------|--------|
| English | `eng_Latn` | Latin |
| Hindi | `hin_Deva` | Devanagari |
| Marathi | `mar_Deva` | Devanagari |
| Gujarati | `guj_Gujr` | Gujarati |
| Tamil | `tam_Taml` | Tamil |
| Telugu | `tel_Telu` | Telugu |
| Kannada | `kan_Knda` | Kannada |
| Malayalam | `mal_Mlym` | Malayalam |
| Bengali | `ben_Beng` | Bengali |
| Punjabi | `pan_Guru` | Gurmukhi |
| Odia | `ory_Orya` | Odia |
| Urdu | `urd_Arab` | Arabic |

For full list, see [IndicTrans2 README](https://github.com/AI4Bharat/IndicTrans2#supported-languages)

### Translation Examples

#### English to Hindi
```bash
curl -X POST http://localhost:3000/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?",
    "src_lang": "eng_Latn",
    "tgt_lang": "hin_Deva"
  }'
```

#### Hindi to English
```bash
curl -X POST http://localhost:3000/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "नमस्ते, आप कैसे हैं?",
    "src_lang": "hin_Deva",
    "tgt_lang": "eng_Latn"
  }'
```

#### Hindi to Marathi (Indic-Indic)
```bash
curl -X POST http://localhost:3000/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "नमस्ते, आप कैसे हैं?",
    "src_lang": "hin_Deva",
    "tgt_lang": "mar_Deva"
  }'
```

### Using the Test Script

```bash
chmod +x test_curl.sh
./test_curl.sh
```

### BentoML API Documentation

BentoML automatically generates API documentation:

**Swagger UI**: `http://localhost:3000/docs`
**OpenAPI Spec**: `http://localhost:3000/openapi.json`

## 📊 Benchmarking

### Benchmark Script

The `benchmark_nmt.py` script performs comprehensive performance testing:

- **Latency metrics**: p50, p95, p99 percentiles
- **QPS**: Queries Per Second
- **GPU utilization**: Average, max, min
- **GPU memory**: Usage and peak
- **CPU and system memory**: Usage statistics
- **Throughput**: Bytes/second, MB/second
- **Success rate**: Percentage of successful requests

### Running Benchmark

```bash
python3 benchmark_nmt.py \
  --endpoint http://localhost:3000/translate \
  --input_text "Hello, how are you?" \
  --src_lang eng_Latn \
  --tgt_lang hin_Deva \
  --outputdir ./bench_results \
  --rate 5.0 \
  --duration 30
```

### Parameters

- `--endpoint`: BentoML service endpoint URL
- `--input_text`: Text to translate
- `--src_lang`: Source language code (FLORES format)
- `--tgt_lang`: Target language code (FLORES format)
- `--outputdir`: Output directory for results
- `--rate`: Target requests per second (default: 10.0)
- `--duration`: Test duration in seconds (default: 30)
- `--sample_interval`: Sampling interval for GPU/CPU (default: 0.5)

### Output Files

- **`benchmark_results.xlsx`**: Excel file with all metrics
- **`requests.csv`**: Detailed request-level data (timestamp, latency, status)
- **`gpu_samples.csv`**: GPU utilization samples over time
- **`sys_samples.csv`**: CPU and memory samples over time

### Example Output

```
============================================================
NMT Benchmark Tool - BentoML
============================================================
Endpoint: http://localhost:3000/translate
Input Text: Hello, how are you?
Source Language: eng_Latn
Target Language: hin_Deva
Rate: 5.0 req/s
Duration: 30s
============================================================

--- Latency Metrics ---
  p50: 1250.45 ms
  p95: 1800.23 ms
  p99: 2100.56 ms

--- QPS Metrics ---
  QPS: 4.98 req/s
  Success Rate: 100.00%

--- GPU Metrics ---
  Avg Utilization: 85.32%
  Max Memory: 10240.00 MB
```

## 📖 API Reference

### Endpoint

**POST** `/translate`

### Request Headers

```
Content-Type: application/json
```

### Request Body

```json
{
  "text": "string",          // Required: Text to translate
  "src_lang": "string",      // Required: Source language (FLORES code)
  "tgt_lang": "string"       // Required: Target language (FLORES code)
}
```

### Response

**Success (200 OK):**
```json
{
  "translation": "string",
  "src_lang": "string",
  "tgt_lang": "string"
}
```

**Error (400/500):**
```json
{
  "error": {
    "detail": "string"
  }
}
```

### Validation Errors

BentoML automatically validates requests using Pydantic:

**Missing field:**
```json
{
  "error": {
    "detail": [
      {
        "loc": ["body", "text"],
        "msg": "field required",
        "type": "value_error.missing"
      }
    ]
  }
}
```

**Invalid type:**
```json
{
  "error": {
    "detail": [
      {
        "loc": ["body", "src_lang"],
        "msg": "str type expected",
        "type": "type_error.str"
      }
    ]
  }
}
```

## 🐳 Docker Deployment

### Containerizing the Bento

BentoML makes it easy to containerize your service:

```bash
bentoml containerize indictrans_nmt:latest -t indictrans-nmt:latest
```

**What this does:**
- Creates a Docker image from the Bento
- Includes all dependencies
- Sets up the entrypoint
- Optimizes image size

### Running the Container

**Basic run:**
```bash
docker run -d \
  --name indictrans-nmt \
  --gpus all \
  -p 3000:3000 \
  -e HF_TOKEN=your_token_here \
  -e DEVICE=cuda \
  indictrans-nmt:latest
```

**With volume mounts (for logs, etc.):**
```bash
docker run -d \
  --name indictrans-nmt \
  --gpus all \
  -p 3000:3000 \
  -v $(pwd)/logs:/var/log \
  -e HF_TOKEN=your_token_here \
  -e DEVICE=cuda \
  indictrans-nmt:latest
```

**View logs:**
```bash
docker logs -f indictrans-nmt
```

**Stop container:**
```bash
docker stop indictrans-nmt
docker rm indictrans-nmt
```

### Docker Compose

You can also use Docker Compose for easier management:

```yaml
version: '3.8'

services:
  indictrans-nmt:
    image: indictrans-nmt:latest
    container_name: indictrans-nmt
    ports:
      - "3000:3000"
    environment:
      - HF_TOKEN=${HF_TOKEN}
      - DEVICE=cuda
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

## 🔧 Troubleshooting

### 403 Forbidden Error

**Error**: `403 Client Error: Cannot access gated repo`

**Solution:**
1. Request access to models on HuggingFace:
   - https://huggingface.co/ai4bharat/indictrans2-en-indic-1B
   - https://huggingface.co/ai4bharat/indictrans2-indic-en-1B
   - https://huggingface.co/ai4bharat/indictrans2-indic-indic-1B
2. Wait for approval (usually quick)
3. Verify `HF_TOKEN` is set correctly
4. Check logs: `tail -f bentoml.log`

### Connection Refused

**Error**: `Connection refused` on curl requests

**Solution:**
1. Check server is running: `ps aux | grep "bentoml serve"`
2. View logs: `tail -f bentoml.log`
3. Verify port is correct: `netstat -tulpn | grep 3000`
4. Check firewall settings
5. Ensure `--host 0.0.0.0` is used for external access

### Model Loading Takes Long Time

**Solution:**
- First request loads the model (can take 30-60 seconds)
- Subsequent requests are faster
- Models are cached in GPU memory
- Check logs to see model loading progress

### Validation Errors

**Error**: `Validation error for Input`

**Solution:**
- Ensure all required fields are provided: `text`, `src_lang`, `tgt_lang`
- Check field types (all should be strings)
- Verify JSON format is correct
- Check API documentation: `http://localhost:3000/docs`

### Bento Build Fails

**Error**: `Failed to build Bento`

**Solution:**
1. Check service syntax: `python3 -c "from service import svc"`
2. Verify all files in `bentofile.yaml` exist
3. Check dependencies in `requirements.txt`
4. Ensure virtual environment is activated
5. Check build logs for specific errors

### Import Errors

**Error**: `ModuleNotFoundError` or `ImportError`

**Solution:**
1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Verify IndicTrans2 and IndicTransToolkit are cloned
3. Check Python path includes IndicTrans2
4. Ensure virtual environment is activated

### GPU Not Available

**Error**: Model runs on CPU instead of GPU

**Solution:**
1. Check GPU availability: `nvidia-smi`
2. Verify CUDA is installed: `python3 -c "import torch; print(torch.cuda.is_available())"`
3. Set device explicitly: `export DEVICE=cuda`
4. Check logs for device information

## 📚 References

### Model and Research

- [IndicTrans2 GitHub](https://github.com/AI4Bharat/IndicTrans2)
- [IndicTrans2 Paper](https://arxiv.org/abs/2305.16307)
- [AI4Bharat Website](https://ai4bharat.iitm.ac.in/)

### Tools and Libraries

- [IndicTransToolkit](https://github.com/VarunGumma/IndicTransToolkit)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [BentoML Documentation](https://docs.bentoml.com/)
- [BentoML API Reference](https://docs.bentoml.com/en/latest/reference/index.html)

### Model Access

- [En-Indic Model](https://huggingface.co/ai4bharat/indictrans2-en-indic-1B)
- [Indic-En Model](https://huggingface.co/ai4bharat/indictrans2-indic-en-1B)
- [Indic-Indic Model](https://huggingface.co/ai4bharat/indictrans2-indic-indic-1B)

## 📝 Quick Reference

### Build Bento
```bash
source bento/bin/activate
bentoml build
```

### Serve Model
```bash
export HF_TOKEN=your_token_here
./start_service.sh
```

### Test Service
```bash
curl -X POST http://localhost:3000/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello","src_lang":"eng_Latn","tgt_lang":"hin_Deva"}'
```

### Run Benchmark
```bash
python3 benchmark_nmt.py \
  --endpoint http://localhost:3000/translate \
  --input_text "Hello, how are you?" \
  --src_lang eng_Latn \
  --tgt_lang hin_Deva \
  --outputdir ./bench_results \
  --rate 5.0 \
  --duration 30
```

### Containerize
```bash
bentoml containerize indictrans_nmt:latest -t indictrans-nmt:latest
```

### View API Docs
```bash
# Open in browser
http://localhost:3000/docs
```

---

**Last Updated**: January 2025
