# FastAPI NMT Service - IndicTrans2

This directory contains a comprehensive FastAPI implementation for hosting the AI4Bharat IndicTrans2 Neural Machine Translation model.

## 📋 Table of Contents

- [What is FastAPI?](#what-is-fastapi)
- [Overview](#overview)
- [The IndicTrans2 Model](#the-indictrans2-model)
- [How We Got the Model](#how-we-got-the-model)
- [How We're Hosting in FastAPI](#how-were-hosting-in-fastapi)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Serving the Model](#serving-the-model)
- [Usage](#usage)
- [Benchmarking](#benchmarking)
- [API Reference](#api-reference)
- [Docker Deployment](#docker-deployment)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## 🤔 What is FastAPI?

**FastAPI** is a modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints. It provides:

1. **High Performance**: One of the fastest Python frameworks available, comparable to NodeJS and Go
2. **Easy to Use**: Intuitive API design with automatic interactive API documentation
3. **Standards Based**: Built on open standards like OpenAPI (formerly Swagger) and JSON Schema
4. **Automatic Documentation**: Interactive API docs (Swagger UI) and ReDoc generated automatically
5. **Type Safety**: Based on Python type hints for automatic validation and serialization
6. **Async Support**: Native support for async/await for handling concurrent requests

### Why FastAPI for NMT?

- **High Performance**: Fast request handling for high-throughput translation services
- **Easy Development**: Simple, intuitive API development with automatic validation
- **Built-in Documentation**: Auto-generated API docs make it easy for users to test
- **Type Safety**: Pydantic models ensure request/response validation
- **Production Ready**: Easy to deploy with uvicorn, gunicorn, or Docker
- **Async Support**: Native async/await for handling multiple concurrent translation requests
- **Developer Friendly**: Fast development and iteration cycle

### FastAPI Components Used

1. **FastAPI Application**: Main application framework
2. **Pydantic Models**: Request/response validation and serialization
3. **Uvicorn ASGI Server**: High-performance ASGI server for serving the API
4. **CORS Middleware**: Cross-origin resource sharing support
5. **Async Route Handlers**: Non-blocking request handling

## 🎯 Overview

This project implements a FastAPI-based service for the **AI4Bharat IndicTrans2** Neural Machine Translation model, providing a REST API for text translation between:

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
cd /home/ubuntu/nmt-benchmarking/fastapi
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

## 🏗️ How We're Hosting in FastAPI

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI Architecture                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │ app.py       │      │ FastAPI      │                    │
│  │ (Application │ ───> │ Framework    │                    │
│  │  Definition) │      │              │                    │
│  └──────────────┘      └──────────────┘                    │
│         │                     │                             │
│         │                     │                             │
│         v                     v                             │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │ IndicTrans   │      │ Pydantic     │                    │
│  │ NMTModel     │      │ Models       │                    │
│  │ (Model       │      │ (Validation) │                    │
│  │  Runner)     │      │              │                    │
│  └──────────────┘      └──────────────┘                    │
│         │                                                     │
│         v                                                     │
│  ┌──────────────┐                                            │
│  │ Uvicorn      │                                            │
│  │ ASGI Server  │  http://localhost:8000                     │
│  └──────────────┘                                            │
│         │                                                     │
│         v                                                     │
│  ┌──────────────┐                                            │
│  │ REST API     │  POST /translate                           │
│  │ Endpoints    │  GET /health                               │
│  │              │  GET /docs (Swagger)                       │
│  └──────────────┘                                            │
└─────────────────────────────────────────────────────────────┘
```

### Step-by-Step FastAPI Integration

#### Step 1: Create Model Runner Class

We created `IndicTransNMTModel` class in `app.py`:

```python
class IndicTransNMTModel:
    """Model runner for IndicTrans2"""
    def __init__(self):
        # Initialize lazy-loaded models
        self.en_indic_model = None
        self.indic_en_model = None
        self.indic_indic_model = None
    
    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        # Load appropriate model based on language pair
        # Perform translation
        # Return result
```

**Key Features:**
- **Lazy Model Loading**: Models loaded only when needed (En-Indic, Indic-En, or Indic-Indic)
- **Automatic Model Selection**: Chooses correct model based on language pair
- **Batch Processing**: Handles multiple sentences efficiently
- **Thread-Safe**: Can be used in async context with ThreadPoolExecutor

#### Step 2: Define Pydantic Models

We use Pydantic for request/response validation:

```python
class TranslationRequest(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str

class TranslationResponse(BaseModel):
    translation: str
    src_lang: str
    tgt_lang: str
```

**Benefits:**
- Automatic validation of request data
- Type checking and conversion
- Clear error messages for invalid inputs
- Automatic OpenAPI schema generation

#### Step 3: Create FastAPI Application

We create the FastAPI app with endpoints:

```python
app = FastAPI(
    title="IndicTrans2 NMT Service",
    description="FastAPI service for AI4Bharat IndicTrans2 Neural Machine Translation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(CORSMiddleware, ...)

# Translation endpoint
@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    # Run translation in executor to avoid blocking
    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    translation = await loop.run_in_executor(
        executor,
        model_runner.translate,
        request.text,
        request.src_lang,
        request.tgt_lang
    )
    return TranslationResponse(...)
```

#### Step 4: Serve with Uvicorn

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

**What happens:**
- Uvicorn loads the FastAPI application
- Initializes the model runner
- Starts ASGI server on specified port
- Creates REST API endpoints
- Generates automatic API documentation

### FastAPI Features Used

1. **FastAPI Application**: Main application framework with automatic OpenAPI generation
2. **Pydantic Models**: Request/response validation and serialization
3. **Async Route Handlers**: Non-blocking request handling with `async def`
4. **CORS Middleware**: Cross-origin resource sharing support
5. **ThreadPoolExecutor**: Run blocking model inference in executor
6. **Automatic Documentation**: Swagger UI at `/docs` and ReDoc at `/redoc`

## 📁 Project Structure

```
fastapi/
├── app.py                    # FastAPI application and model runner
├── requirements.txt          # Python dependencies
├── start_service.sh          # Service startup script
├── test_curl.sh             # Test curl commands
├── benchmark_nmt.py         # Benchmark script for performance testing
├── README.md                # This file
│
├── IndicTrans2/             # Cloned IndicTrans2 repository
│   ├── huggingface_interface/
│   ├── inference/
│   └── ...
│
├── IndicTransToolkit/       # Cloned IndicTransToolkit
│   └── ...
│
└── fastapi/                 # Python virtual environment
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
cd /home/ubuntu/nmt-benchmarking/fastapi
python3 -m venv fastapi
source fastapi/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Key Dependencies:**
- `fastapi>=0.104.0` - FastAPI framework
- `uvicorn[standard]>=0.24.0` - ASGI server
- `pydantic>=2.0.0` - Data validation
- `torch>=2.0.0` - PyTorch for model inference
- `transformers>=4.30.0` - HuggingFace Transformers
- `indictranstoolkit` - Preprocessing/postprocessing
- `nltk`, `sacremoses`, `mosestokenizer` - Text processing

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

```bash
python3 -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt')"
```

Or the service will automatically download it on first run.

### Step 6: Set HuggingFace Token

```bash
export HF_TOKEN=your_huggingface_token_here
```

**Important**: Never hardcode the token. Always use environment variables.

### Step 7: Verify Installation

```bash
# Check FastAPI installation
python3 -c "import fastapi; print(fastapi.__version__)"

# Check if app can be imported
python3 -c "from app import app; print('App loaded successfully')"
```

## 🚀 Serving the Model

### FastAPI Serving

FastAPI uses **uvicorn** as the ASGI server to serve the application.

### Step 1: Start the Server

**Using the startup script (Recommended):**
```bash
export HF_TOKEN=your_token_here
./start_service.sh [PORT]
```

**Manually:**
```bash
export HF_TOKEN=your_token_here
source fastapi/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8000
```

**In background:**
```bash
nohup env HF_TOKEN="$HF_TOKEN" uvicorn app:app \
  --host 0.0.0.0 --port 8000 > fastapi.log 2>&1 &
```

### Step 2: Server Startup Process

When you start the server:

1. **Uvicorn loads** the FastAPI application from `app.py`
2. **Initializes** the `IndicTransNMTModel` instance
3. **Starts ASGI server** on specified port
4. **Creates** REST API endpoints:
   - `POST /translate` - Translation endpoint
   - `GET /health` - Health check endpoint
   - `GET /` - Root endpoint with API info
   - `GET /docs` - Swagger UI documentation
   - `GET /redoc` - ReDoc documentation
   - `GET /openapi.json` - OpenAPI schema

### Step 3: Verify Server is Running

```bash
# Check if server is running
ps aux | grep "uvicorn app:app"

# Check logs (if running in background)
tail -f fastapi.log

# Test health endpoint
curl http://localhost:8000/health

# Test translation endpoint
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello","src_lang":"eng_Latn","tgt_lang":"hin_Deva"}'
```

### Development vs Production Mode

**Development Mode** (default):
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
- Single worker
- Auto-reload on code changes (with `--reload`)
- Detailed error messages
- Suitable for testing

**Production Mode**:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```
- Multiple workers (based on CPU cores)
- No auto-reload
- Optimized for performance
- Suitable for production

**With Gunicorn** (for production):
```bash
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

## 💻 Usage

### API Endpoints

**Base URL**: `http://localhost:8000`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/translate` | POST | Translate text |
| `/health` | GET | Health check |
| `/` | GET | API information |
| `/docs` | GET | Swagger UI documentation |
| `/redoc` | GET | ReDoc documentation |
| `/openapi.json` | GET | OpenAPI schema |

### Translation Endpoint

**Endpoint**: `POST /translate`

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
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?",
    "src_lang": "eng_Latn",
    "tgt_lang": "hin_Deva"
  }'
```

#### Hindi to English
```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "नमस्ते, आप कैसे हैं?",
    "src_lang": "hin_Deva",
    "tgt_lang": "eng_Latn"
  }'
```

#### Hindi to Marathi (Indic-Indic)
```bash
curl -X POST http://localhost:8000/translate \
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

### Interactive API Documentation

FastAPI automatically generates interactive API documentation:

**Swagger UI**: `http://localhost:8000/docs`
- Interactive API testing interface
- Try out endpoints directly from browser
- See request/response schemas

**ReDoc**: `http://localhost:8000/redoc`
- Alternative documentation format
- Clean, readable API documentation

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
  --endpoint http://localhost:8000/translate \
  --input_text "Hello, how are you?" \
  --src_lang eng_Latn \
  --tgt_lang hin_Deva \
  --outputdir ./bench_results \
  --rate 5.0 \
  --duration 30
```

### Parameters

- `--endpoint`: FastAPI service endpoint URL
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
NMT Benchmark Tool - FastAPI
============================================================
Endpoint: http://localhost:8000/translate
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

**Error (422 Validation Error):**
```json
{
  "detail": [
    {
      "loc": ["body", "text"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**Error (500 Internal Server Error):**
```json
{
  "detail": "Error message"
}
```

### Health Check

**GET** `/health`

**Response:**
```json
{
  "status": "healthy",
  "device": "cuda",
  "cuda_available": true
}
```

## 🐳 Docker Deployment

### Create Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone repositories
RUN git clone https://github.com/AI4Bharat/IndicTrans2 && \
    git clone https://github.com/VarunGumma/IndicTransToolkit && \
    cd IndicTransToolkit && pip install -e . && cd ..

# Copy application
COPY app.py .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build Docker Image

```bash
docker build -t fastapi-indictrans-nmt:latest .
```

### Run Container

```bash
docker run -d \
  --name fastapi-indictrans-nmt \
  --gpus all \
  -p 8000:8000 \
  -e HF_TOKEN=your_token_here \
  -e DEVICE=cuda \
  fastapi-indictrans-nmt:latest
```

### Docker Compose

```yaml
version: '3.8'

services:
  fastapi-indictrans-nmt:
    build: .
    container_name: fastapi-indictrans-nmt
    ports:
      - "8000:8000"
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
4. Check logs: `tail -f fastapi.log`

### Connection Refused

**Error**: `Connection refused` on curl requests

**Solution:**
1. Check server is running: `ps aux | grep "uvicorn app:app"`
2. View logs: `tail -f fastapi.log`
3. Verify port is correct: `netstat -tulpn | grep 8000`
4. Check firewall settings
5. Ensure `--host 0.0.0.0` is used for external access

### Validation Errors

**Error**: `422 Unprocessable Entity` with validation details

**Solution:**
- Ensure all required fields are provided: `text`, `src_lang`, `tgt_lang`
- Check field types (all should be strings)
- Verify JSON format is correct
- Check API documentation: `http://localhost:8000/docs`

### Model Loading Takes Long Time

**Solution:**
- First request loads the model (can take 30-60 seconds)
- Subsequent requests are faster
- Models are cached in GPU memory
- Check logs to see model loading progress

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
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Uvicorn Documentation](https://www.uvicorn.org/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

### Model Access

- [En-Indic Model](https://huggingface.co/ai4bharat/indictrans2-en-indic-1B)
- [Indic-En Model](https://huggingface.co/ai4bharat/indictrans2-indic-en-1B)
- [Indic-Indic Model](https://huggingface.co/ai4bharat/indictrans2-indic-indic-1B)

## 📝 Quick Reference

### Start Service
```bash
export HF_TOKEN=your_token_here
./start_service.sh 8000
```

### Test Service
```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello","src_lang":"eng_Latn","tgt_lang":"hin_Deva"}'
```

### View API Docs
```bash
# Open in browser
http://localhost:8000/docs
```

### Run Benchmark
```bash
python3 benchmark_nmt.py \
  --endpoint http://localhost:8000/translate \
  --input_text "Hello, how are you?" \
  --src_lang eng_Latn \
  --tgt_lang hin_Deva \
  --outputdir ./bench_results \
  --rate 5.0 \
  --duration 30
```

---

**Last Updated**: January 2025

