# MLflow ASR Service

Automatic Speech Recognition (ASR) service using MLflow for model serving, built with the `ai4bharat/indic-conformer-600m-multilingual` model.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Documentation](#documentation)

## 🎯 Overview

This project implements an ASR service using MLflow's PyFunc model flavor. The service can transcribe audio in multiple Indic languages using the Indic Conformer 600M multilingual model from AI4Bharat.

**Model**: `ai4bharat/indic-conformer-600m-multilingual`  
**Framework**: MLflow PyFunc  
**Supported Languages**: Multiple Indic languages (Hindi, Tamil, Telugu, etc.)

## ✨ Features

- 🎤 **Multilingual ASR**: Supports multiple Indic languages
- 🚀 **MLflow Integration**: Easy model versioning and serving
- 🔄 **Flexible Decoding**: Supports CTC and greedy decoding strategies
- 📦 **Container Ready**: Can be deployed using MLflow's serving capabilities
- 🎯 **REST API**: Standard MLflow scoring server endpoint
- 🔧 **GPU Support**: Automatically uses GPU if available

## 📦 Prerequisites

- Python 3.10+
- CUDA-capable GPU (optional, but recommended)
- HuggingFace account with access to `ai4bharat/indic-conformer-600m-multilingual`
- MLflow 3.7.0+
- PyTorch 2.0+
- Transformers library

## 🚀 Installation

### 1. Create Virtual Environment

```bash
cd /home/ubuntu/Benchmarking/Frameworks/MLflow
python3 -m venv mlflow
source mlflow/bin/activate
```

### 2. Install Dependencies

```bash
pip install mlflow torch torchaudio transformers soundfile pandas
```

Or install from requirements (if available):

```bash
pip install -r requirements.txt
```

### 3. Set Up HuggingFace Token (Optional)

If the model requires authentication:

```bash
export HF_TOKEN="your_huggingface_token_here"
```

Or use HuggingFace CLI:

```bash
huggingface-cli login
```

## 📝 Usage

### Step 1: Log the Model

First, log the model to MLflow's model registry:

```bash
source mlflow/bin/activate
python log_model.py
```

This will:
- Create a PyFunc wrapper around the Indic Conformer model
- Save the model to `mlruns/0/models/`
- Register dependencies and environment configuration

**Output**: Model will be logged with a unique model ID (e.g., `m-8f33614a5aeb46f6a4f4c8b0c64b9cf7`)

### Step 2: Start the MLflow Model Server

```bash
source mlflow/bin/activate

# Get your model path from mlruns directory
mlflow models serve \
  -m "mlruns/0/models/m-8f33614a5aeb46f6a4f4c8b0c64b9cf7/artifacts" \
  --no-conda \
  --host 0.0.0.0 \
  --port 5000
```

**Note**: Replace `m-8f33614a5aeb46f6a4f4c8b0c64b9cf7` with your actual model ID.

The server will start on `http://0.0.0.0:5000`

### Step 3: Test the Service

Use the provided test script:

```bash
./test_curl.sh ta ctc
```

Or use curl directly (see [Testing](#testing) section).

## 📚 API Documentation

### Endpoint

**POST** `/asr`

### Request Format

The API expects data in the following format:

```json
{
  "audio_base64": "<base64_encoded_audio_bytes>",
  "lang": "ta",
  "decoding": "ctc"
}
```

#### Parameters

- **audio_base64** (required): Base64-encoded audio file (WAV format recommended)
- **lang** (optional, default: "hi"): Language code for transcription
  - Supported: `hi` (Hindi), `ta` (Tamil), `te` (Telugu), etc.
- **decoding** (optional, default: "ctc"): Decoding strategy
  - Options: `ctc`, `greedy`

### Response Format

```json
{
  "transcription": "transcribed text in the specified language",
  "lang": "ta",
  "decoding": "ctc"
}
```

### Example Request

```bash
curl -X POST http://localhost:5000/asr \
  -H "Content-Type: application/json" \
  -d '{
    "audio_base64": "<base64_encoded_audio>",
    "lang": "ta",
    "decoding": "ctc"
  }'
```

### API Documentation UI

Once the server is running, visit:
- **Swagger UI**: http://localhost:5000/docs
- **ReDoc**: http://localhost:5000/redoc

## 🧪 Testing

### Using the Test Script

```bash
# Test with Tamil audio, CTC decoding
./test_curl.sh ta ctc

# Test with Hindi audio, greedy decoding
./test_curl.sh hi greedy
```

### Using curl Directly

**One-liner**:

```bash
curl -X POST http://localhost:5000/asr \
  -H "Content-Type: application/json" \
  -d "$(python3 -c "import base64, json; audio_b64 = base64.b64encode(open('/home/ubuntu/Benchmarking/ta2.wav', 'rb').read()).decode('utf-8'); print(json.dumps({'audio_base64': audio_b64, 'lang': 'ta', 'decoding': 'ctc'}))")" \
  | python3 -m json.tool
```

**Two-step process**:

```bash
# Step 1: Create payload
python3 -c "
import base64, json
audio_b64 = base64.b64encode(open('/home/ubuntu/Benchmarking/ta2.wav', 'rb').read()).decode('utf-8')
payload = {'audio_base64': audio_b64, 'lang': 'ta', 'decoding': 'ctc'}
json.dump(payload, open('/tmp/mlflow_payload.json', 'w'))
"

# Step 2: Send request
curl -X POST http://localhost:5000/asr \
  -H "Content-Type: application/json" \
  -d @/tmp/mlflow_payload.json \
  | python3 -m json.tool
```

### Expected Response

```json
{
  "transcription": "நாடகமாடுவதென்றால் பாட்டி இல்லாமல் உதவாது என்று அபிப்பிராயப்படுவோர் இதைச் சற்று கவனிப்பாராக",
  "lang": "ta",
  "decoding": "ctc"
}
```

## 🔧 Troubleshooting

### Issue: `mlflow: command not found`

**Solution**: Activate the virtual environment:
```bash
source mlflow/bin/activate
```

### Issue: `RuntimeError: Failed to open the input`

**Cause**: Invalid or corrupted audio data, or unsupported audio format.

**Solution**: 
- Ensure audio is in WAV format
- Check that base64 encoding is correct
- Verify audio file is not empty

### Issue: `ModuleNotFoundError: No module named 'torch'`

**Solution**: Install dependencies:
```bash
source mlflow/bin/activate
pip install torch torchaudio transformers soundfile pandas
```

### Issue: Model loading fails with authentication error

**Solution**: Set HuggingFace token:
```bash
export HF_TOKEN="your_token_here"
```

Or use HuggingFace CLI:
```bash
huggingface-cli login
```

### Issue: GPU not being used

**Check GPU availability**:
```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

If CUDA is available but not used, the model will automatically use GPU when available.

### Issue: Server not responding

**Check if server is running**:
```bash
ps aux | grep mlflow
netstat -tlnp | grep 5000
```

**Check server logs** for detailed error messages.

## 📁 Project Structure

```
MLflow/
├── README.md                 # This file
├── log_model.py              # Script to log model to MLflow
├── mlflow_asr.py             # PyFunc model wrapper implementation
├── test_curl.sh              # Test script for API endpoint
├── curl_command.txt          # Reference curl commands
├── mlflow/                   # Virtual environment (not in git)
│   ├── bin/
│   ├── lib/
│   └── ...
├── mlruns/                   # MLflow runs and models (not in git)
│   └── 0/
│       └── models/
│           └── m-<model_id>/
│               └── artifacts/
│                   ├── MLmodel
│                   ├── python_model.pkl
│                   ├── requirements.txt
│                   └── ...
└── mlflow.db                 # MLflow tracking database (not in git)
```

## 🔑 Key Files

- **`mlflow_asr.py`**: Contains `IndicConformerWrapper` class that implements the MLflow PyFunc interface
- **`log_model.py`**: Script to register the model with MLflow
- **`test_curl.sh`**: Convenient test script for the API

## 📖 Model Details

- **Model Name**: `ai4bharat/indic-conformer-600m-multilingual`
- **Architecture**: Conformer-based ASR model
- **Parameters**: 600M
- **Sample Rate**: 16kHz
- **Supported Languages**: Multiple Indic languages
- **Decoding**: CTC or greedy decoding

## 🚢 Deployment

### Using Custom Server (Local)

```bash
source mlflow/bin/activate
python server.py
```

Or with environment variables:

```bash
MODEL_PATH="mlruns/0/models/<model_id>/artifacts" python server.py
```

### Using Docker (Recommended)

**Quick Start**:

```bash
# Build image
./build_docker.sh

# Run container
./run_docker.sh

# Test
./test_curl.sh ta ctc
```

**Using Docker Compose**:

```bash
docker-compose up -d
```

For detailed Docker instructions, see [DOCKER.md](DOCKER.md).

## 📝 Notes

- The model is loaded on first request (lazy loading)
- Audio is automatically resampled to 16kHz if needed
- Multi-channel audio is converted to mono by averaging channels
- GPU is used automatically if available

## 🤝 Contributing

1. Ensure all tests pass
2. Follow Python PEP 8 style guidelines
3. Update documentation for any API changes

## 📄 License

This project uses the `ai4bharat/indic-conformer-600m-multilingual` model. Please refer to the model's license for usage terms.

## 📚 Documentation

Comprehensive documentation is available:

- **[README.md](README.md)** - This file, main project documentation
- **[QUICKSTART.md](QUICKSTART.md)** - Quick start guide (5 minutes)
- **[SETUP.md](SETUP.md)** - Detailed setup instructions
- **[USAGE.md](USAGE.md)** - Usage guide with examples
- **[DOCKER.md](DOCKER.md)** - Docker deployment guide
- **[DOCKER_QUICKSTART.md](DOCKER_QUICKSTART.md)** - Docker quick start
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture and design
- **[API_REFERENCE.md](API_REFERENCE.md)** - Complete API reference
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and changes

## 🔗 Related Links

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow PyFunc Models](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html)
- [AI4Bharat Indic Conformer](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual)
- [Transformers Library](https://huggingface.co/docs/transformers)

## 💡 Tips

- Use `--no-conda` flag if you're using a virtual environment
- Monitor GPU usage: `nvidia-smi -l 1`
- Check server health: `curl http://localhost:5000/health`
- View API docs: `http://localhost:5000/docs`

---

**Last Updated**: December 2024

