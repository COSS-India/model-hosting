# MLflow NMT Service - IndicTrans2

This directory contains a comprehensive MLflow implementation for hosting the AI4Bharat IndicTrans2 Neural Machine Translation model.

## 📋 Table of Contents

- [What is MLflow?](#what-is-mlflow)
- [Overview](#overview)
- [The IndicTrans2 Model](#the-indictrans2-model)
- [How We Got the Model](#how-we-got-the-model)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Model Registration](#model-registration)
- [Serving the Model](#serving-the-model)
- [Usage](#usage)
- [Benchmarking](#benchmarking)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## 🤔 What is MLflow?

**MLflow** is an open-source platform for managing the machine learning lifecycle. It provides tools for:

1. **MLflow Tracking**: Logging parameters, metrics, and artifacts during model development
2. **MLflow Projects**: Packaging ML code for reproducible runs
3. **MLflow Models**: Standardizing model packaging and deployment
4. **Model Registry**: Centralized model store for collaboration

### Why MLflow for NMT?

- **Standardized Deployment**: MLflow provides a consistent API across different ML frameworks
- **Model Versioning**: Track and manage different versions of your models
- **Easy Serving**: Simple REST API serving with `mlflow models serve`
- **Flexibility**: Support for custom Python models via PyFunc interface
- **Production Ready**: Built-in support for production deployments

### MLflow Components Used

1. **MLflow Tracking**: Stores model metadata, parameters, and artifacts
2. **MLflow Models**: Package our IndicTrans2 model as a PyFunc model
3. **Model Registry**: Register and version our translation models
4. **Model Serving**: REST API endpoint for inference

## 🎯 Overview

This project implements an MLflow-based service for the **AI4Bharat IndicTrans2** Neural Machine Translation model, providing a REST API for text translation between:

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
cd /home/ubuntu/nmt-benchmarking/mlflow
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

### Step 6: Model Loading

Models are loaded on-demand using HuggingFace Transformers:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained(
    "ai4bharat/indictrans2-en-indic-1B",
    trust_remote_code=True,
    token=HF_TOKEN
)
```

The `trust_remote_code=True` is required because IndicTrans2 uses custom model code.

## 🏗️ How We're Hosting in MLflow

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    MLflow Architecture                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │ Model        │      │ MLflow       │                    │
│  │ Registration │ ───> │ Model        │                    │
│  │ Script       │      │ Registry     │                    │
│  └──────────────┘      └──────────────┘                    │
│         │                     │                             │
│         │                     │                             │
│         v                     v                             │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │ PyFunc       │      │ MLflow       │                    │
│  │ Model        │      │ Tracking     │                    │
│  │ Wrapper      │      │ (mlruns/)    │                    │
│  └──────────────┘      └──────────────┘                    │
│         │                                                     │
│         v                                                     │
│  ┌──────────────┐                                            │
│  │ MLflow       │                                            │
│  │ Model Server │  http://localhost:5000/invocations        │
│  │ (uvicorn)    │                                            │
│  └──────────────┘                                            │
│         │                                                     │
│         v                                                     │
│  ┌──────────────┐                                            │
│  │ REST API     │  POST /invocations                        │
│  │ Endpoint     │  {"inputs": {...}}                        │
│  └──────────────┘                                            │
└─────────────────────────────────────────────────────────────┘
```

### Step-by-Step MLflow Integration

#### Step 1: Create PyFunc Model Wrapper

We created `mlflow_nmt_model.py` that implements MLflow's `PythonModel` interface:

```python
class IndicTransNMTModel(PythonModel):
    def load_context(self, context):
        # Initialize models and processors
        # Lazy loading: models loaded on first use
    
    def predict(self, context, model_input):
        # Handle different input formats (DataFrame, dict, etc.)
        # Route to appropriate model (En-Indic, Indic-En, Indic-Indic)
        # Perform translation
        # Return results
```

**Key Features:**
- **Lazy Model Loading**: Models are loaded only when needed
- **Automatic Model Selection**: Chooses correct model based on language pair
- **Input Format Handling**: Supports DataFrame (MLflow default), dict, JSON string
- **Batch Processing**: Handles multiple sentences efficiently

#### Step 2: Register Model with MLflow

We use `register_model.py` to:

1. **Authenticate** with HuggingFace using HF_TOKEN
2. **Create MLflow run** to track the registration
3. **Log the model** using `mlflow.pyfunc.log_model()`
4. **Register** the model in MLflow Model Registry

```python
with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="indictrans_nmt_model",
        python_model=IndicTransNMTModel(),
        pip_requirements="requirements.txt",
        registered_model_name="IndicTransNMT"
    )
```

**What Happens:**
- MLflow packages the model code
- Stores it in `mlruns/` directory
- Creates version in Model Registry
- Records metadata (parameters, model paths, etc.)

#### Step 3: Serve the Model

Using MLflow's built-in serving:

```bash
mlflow models serve \
  -m "runs:/<run_id>/indictrans_nmt_model" \
  --port 5000 \
  --host 0.0.0.0 \
  --no-conda
```

**What Happens:**
- MLflow loads the registered model
- Starts a **uvicorn** server (FastAPI-based)
- Creates REST API endpoint at `/invocations` (MLflow's standard inference endpoint)
- Handles requests in MLflow 2.0+ format

**Note on `/invocations` endpoint**: This is MLflow's **default and standard** endpoint name for all model inference requests. MLflow automatically creates this endpoint when you use `mlflow models serve`, regardless of model type. The name "invocations" refers to invoking/calling the model for predictions. You cannot customize this endpoint name when using MLflow's built-in serving - it's part of MLflow's standardized API.

#### Step 4: MLflow Model Format

The registered model structure:

```
mlruns/
└── 0/
    └── <run_id>/
        └── artifacts/
            └── indictrans_nmt_model/
                ├── MLmodel                    # Model metadata
                ├── python_env.yaml            # Python environment
                ├── requirements.txt           # Dependencies
                └── python_model.pkl           # Serialized PythonModel
```

### MLflow Tracking

MLflow tracks:
- **Model parameters**: Model names, device, batch size
- **Model artifacts**: The actual model code and dependencies
- **Model metadata**: Registration timestamp, version number
- **Model versions**: Multiple versions for A/B testing

## 📁 Project Structure

```
mlflow/
├── mlflow_nmt_model.py      # MLflow PyFunc model wrapper
├── register_model.py         # Script to register model with MLflow
├── serve_model.sh            # Script to serve the model
├── benchmark_nmt.py          # Benchmark script for performance testing
├── test_curl.sh              # Test curl commands
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── QUICK_START.md            # Quick start guide
│
├── IndicTrans2/              # Cloned IndicTrans2 repository
│   ├── huggingface_interface/
│   ├── inference/
│   └── ...
│
├── IndicTransToolkit/        # Cloned IndicTransToolkit
│   └── ...
│
├── mlflow/                   # Python virtual environment
│   └── ...
│
└── mlruns/                   # MLflow tracking data
    └── 0/
        └── <run_ids>/
```

## 🔧 Setup and Installation

### Prerequisites

- **Python 3.10+**
- **NVIDIA GPU** with CUDA support (recommended)
- **HuggingFace account** with access to IndicTrans2 models
- **Git** for cloning repositories

### Step 1: Create Virtual Environment

```bash
cd /home/ubuntu/nmt-benchmarking/mlflow
python3 -m venv mlflow
source mlflow/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Key Dependencies:**
- `mlflow>=2.10.0` - MLflow framework
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

### Step 6: Set HuggingFace Token

```bash
export HF_TOKEN=your_huggingface_token_here
```

**Important**: Never hardcode the token. Always use environment variables.

## 📝 Model Registration

### What is Model Registration?

Model registration in MLflow:
- **Stores** the model code and metadata
- **Versions** the model (v1, v2, etc.)
- **Tracks** model lineage and metadata
- **Enables** easy model serving and deployment

### Registering the Model

```bash
cd /home/ubuntu/nmt-benchmarking/mlflow
source mlflow/bin/activate
export HF_TOKEN=your_token_here
python3 register_model.py
```

**What the script does:**

1. **Sets MLflow tracking URI**:
   ```python
   mlflow.set_tracking_uri("file:///home/ubuntu/nmt-benchmarking/mlflow/mlruns")
   ```

2. **Authenticates with HuggingFace**:
   ```python
   from huggingface_hub import login
   login(token=HF_TOKEN)
   ```

3. **Starts MLflow run**:
   ```python
   with mlflow.start_run():
       # Log model and metadata
   ```

4. **Logs the model**:
   ```python
   mlflow.pyfunc.log_model(
       artifact_path="indictrans_nmt_model",
       python_model=IndicTransNMTModel(),
       pip_requirements="requirements.txt",
       registered_model_name="IndicTransNMT"
   )
   ```

5. **Logs metadata**:
   - Model type
   - Model checkpoint paths
   - Device configuration

### Registration Output

After successful registration, you'll see:

```
✓ Model registered successfully!
  Model name: IndicTransNMT
  Run ID: 5c7c8011642c4282954c44cc49ed05f0
  Artifact path: indictrans_nmt_model

To serve the model, use:
  mlflow models serve -m runs:/5c7c8011642c4282954c44cc49ed05f0/indictrans_nmt_model --port 5000
```

## 🚀 Serving the Model

### MLflow Model Serving

MLflow provides built-in model serving using **uvicorn** (ASGI server) which creates a FastAPI-based REST API.

### Step 1: Start the Server

**Using the startup script:**
```bash
export HF_TOKEN=your_token_here
./serve_model.sh 5000
```

**Manually:**
```bash
export HF_TOKEN=your_token_here
export MLFLOW_TRACKING_URI="file:///home/ubuntu/nmt-benchmarking/mlflow/mlruns"
mlflow models serve \
  -m "runs:/<run_id>/indictrans_nmt_model" \
  --port 5000 \
  --host 0.0.0.0 \
  --no-conda
```

**Important flags:**
- `--no-conda`: Use current Python environment (avoids pyenv requirement)
- `--host 0.0.0.0`: Allow external connections
- `--port 5000`: Server port

### Step 2: Server Startup Process

When you start the server:

1. **MLflow loads** the registered model from `mlruns/`
2. **Unpacks** model artifacts and dependencies
3. **Initializes** the PythonModel (`load_context()` is called)
4. **Starts uvicorn** server on specified port
5. **Creates** `/invocations` endpoint

### Step 3: Verify Server is Running

```bash
# Check if server is running
ps aux | grep "mlflow models serve"

# Test health (MLflow doesn't have explicit health endpoint, but you can test the API)
curl -X POST http://localhost:5000/invocations \
  -H "Content-Type: application/json" \
  -d '{"inputs":{"text":"Hello","src_lang":"eng_Latn","tgt_lang":"hin_Deva"}}'
```

## 💻 Usage

### API Endpoint

**Endpoint**: `POST /invocations`

**Base URL**: `http://localhost:5000`

**Why `/invocations`?** This is MLflow's **standard default endpoint** for model inference. When you run `mlflow models serve`, MLflow automatically creates this endpoint. The name "invocations" refers to invoking/calling the model for predictions. This is consistent across all MLflow-served models, regardless of the model type (PyFunc, sklearn, etc.).

### Request Format (MLflow 2.0+)

MLflow 2.0+ uses a standardized input format:

```json
{
  "inputs": {
    "text": "Hello, how are you?",
    "src_lang": "eng_Latn",
    "tgt_lang": "hin_Deva"
  }
}
```

### Response Format

```json
{
  "predictions": {
    "translation": "नमस्ते, आप कैसे हैं?",
    "src_lang": "eng_Latn",
    "tgt_lang": "hin_Deva"
  }
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
curl -X POST http://localhost:5000/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": {
      "text": "Hello, how are you?",
      "src_lang": "eng_Latn",
      "tgt_lang": "hin_Deva"
    }
  }'
```

#### Hindi to English
```bash
curl -X POST http://localhost:5000/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": {
      "text": "नमस्ते, आप कैसे हैं?",
      "src_lang": "hin_Deva",
      "tgt_lang": "eng_Latn"
    }
  }'
```

#### Hindi to Marathi (Indic-Indic)
```bash
curl -X POST http://localhost:5000/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": {
      "text": "नमस्ते, आप कैसे हैं?",
      "src_lang": "hin_Deva",
      "tgt_lang": "mar_Deva"
    }
  }'
```

### Using the Test Script

```bash
chmod +x test_curl.sh
./test_curl.sh
```

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
  --endpoint http://localhost:5000/invocations \
  --input_text "Hello, how are you?" \
  --src_lang eng_Latn \
  --tgt_lang hin_Deva \
  --outputdir ./bench_results \
  --rate 5.0 \
  --duration 30
```

### Parameters

- `--endpoint`: MLflow service endpoint URL
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
NMT Benchmark Tool - MLflow
============================================================
Endpoint: http://localhost:5000/invocations
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

**POST** `/invocations`

### Request Headers

```
Content-Type: application/json
```

### Request Body

```json
{
  "inputs": {
    "text": "string",          // Required: Text to translate
    "src_lang": "string",      // Required: Source language (FLORES code)
    "tgt_lang": "string"       // Required: Target language (FLORES code)
  }
}
```

### Response

**Success (200 OK):**
```json
{
  "predictions": {
    "translation": "string",
    "src_lang": "string",
    "tgt_lang": "string"
  }
}
```

**Error (400/500):**
```json
{
  "error_code": "string",
  "message": "string",
  "stack_trace": "string"
}
```

## 🔧 Troubleshooting

### Model Not Found

**Error**: `No model found`

**Solution:**
1. Ensure model is registered: `python3 register_model.py`
2. Check MLflow runs: `ls mlruns/0/`
3. Use correct run ID in serve command

### 403 Forbidden Error

**Error**: `403 Client Error: Cannot access gated repo`

**Solution:**
1. Request access to models on HuggingFace
2. Verify `HF_TOKEN` is set correctly
3. Check token has access permissions

### pyenv Error

**Error**: `Could not find the pyenv binary`

**Solution:**
Use `--no-conda` flag:
```bash
mlflow models serve ... --no-conda
```

### Connection Refused

**Error**: `Connection refused` on curl requests

**Solution:**
1. Check server is running: `ps aux | grep "mlflow models serve"`
2. Verify port is correct: `netstat -tulpn | grep 5000`
3. Check firewall settings
4. Ensure `--host 0.0.0.0` is used for external access

### Input Format Error

**Error**: `The input must be a JSON dictionary with exactly one of the input fields`

**Solution:**
Use MLflow 2.0+ format:
```json
{
  "inputs": {
    "text": "...",
    "src_lang": "...",
    "tgt_lang": "..."
  }
}
```

### Model Loading Takes Long Time

**Solution:**
- First request loads the model (can take 30-60 seconds)
- Subsequent requests are faster
- Models are cached in GPU memory

## 📚 References

### Model and Research

- [IndicTrans2 GitHub](https://github.com/AI4Bharat/IndicTrans2)
- [IndicTrans2 Paper](https://arxiv.org/abs/2305.16307)
- [AI4Bharat Website](https://ai4bharat.iitm.ac.in/)

### Tools and Libraries

- [IndicTransToolkit](https://github.com/VarunGumma/IndicTransToolkit)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow PyFunc Models](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html)

### Model Access

- [En-Indic Model](https://huggingface.co/ai4bharat/indictrans2-en-indic-1B)
- [Indic-En Model](https://huggingface.co/ai4bharat/indictrans2-indic-en-1B)
- [Indic-Indic Model](https://huggingface.co/ai4bharat/indictrans2-indic-indic-1B)

## 📝 Quick Reference

### Register Model
```bash
export HF_TOKEN=your_token_here
python3 register_model.py
```

### Serve Model
```bash
export HF_TOKEN=your_token_here
./serve_model.sh 5000
```

### Test Service
```bash
curl -X POST http://localhost:5000/invocations \
  -H "Content-Type: application/json" \
  -d '{"inputs":{"text":"Hello","src_lang":"eng_Latn","tgt_lang":"hin_Deva"}}'
```

### Run Benchmark
```bash
python3 benchmark_nmt.py \
  --endpoint http://localhost:5000/invocations \
  --input_text "Hello, how are you?" \
  --src_lang eng_Latn \
  --tgt_lang hin_Deva \
  --outputdir ./bench_results \
  --rate 5.0 \
  --duration 30
```

---

**Last Updated**: January 2025
