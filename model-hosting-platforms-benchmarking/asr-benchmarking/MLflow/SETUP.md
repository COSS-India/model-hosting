# MLflow ASR Service - Setup Guide

This guide provides detailed setup instructions for the MLflow ASR service.

## Prerequisites

### System Requirements

- **OS**: Linux (Ubuntu 20.04+ recommended) or macOS
- **Python**: 3.10 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **Disk Space**: At least 10GB free (for model and dependencies)

### Software Requirements

- Python 3.10+
- pip (Python package manager)
- Git (for cloning repository)

### GPU Setup (Optional)

If you have an NVIDIA GPU:

1. **Install NVIDIA Drivers**:
   ```bash
   sudo apt update
   sudo apt install nvidia-driver-535  # or latest version
   ```

2. **Install CUDA Toolkit** (if not already installed):
   ```bash
   # Check CUDA version
   nvidia-smi
   ```

3. **Verify GPU**:
   ```bash
   python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

## Installation Steps

### Step 1: Clone Repository

```bash
cd /home/ubuntu/Benchmarking/Frameworks/MLflow
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv mlflow
source mlflow/bin/activate
```

**Note**: The virtual environment directory `mlflow/` is already created. If you need to recreate it:

```bash
# Remove old environment (if needed)
rm -rf mlflow/

# Create new environment
python3 -m venv mlflow
source mlflow/bin/activate
```

### Step 3: Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install mlflow torch torchaudio transformers soundfile pandas
```

### Step 5: Set Up HuggingFace Authentication

The model `ai4bharat/indic-conformer-600m-multilingual` may require authentication.

**Option 1: Using Environment Variable**

```bash
export HF_TOKEN="your_huggingface_token_here"
```

Add to `~/.bashrc` or `~/.zshrc` for persistence:

```bash
echo 'export HF_TOKEN="your_huggingface_token_here"' >> ~/.bashrc
source ~/.bashrc
```

**Option 2: Using HuggingFace CLI**

```bash
pip install huggingface_hub
huggingface-cli login
```

Enter your token when prompted.

**Get your token**: https://huggingface.co/settings/tokens

### Step 6: Verify Installation

```bash
# Check MLflow
mlflow --version

# Check PyTorch and CUDA
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Check Transformers
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

## Model Logging

### Step 1: Log the Model

```bash
source mlflow/bin/activate
python log_model.py
```

**Expected Output**:
```
Model loaded successfully!
Model logged to mlruns/ (check mlflow ui or mlruns directory)
```

### Step 2: Verify Model is Logged

```bash
# List models
mlflow models list

# Or check directory
ls -la mlruns/0/models/
```

You should see a directory like `m-<model_id>/` with artifacts.

### Step 3: Note Your Model ID

```bash
# Get the model ID
ls mlruns/0/models/
```

Example output: `m-8f33614a5aeb46f6a4f4c8b0c64b9cf7`

## Starting the Server

### Basic Server Start

```bash
source mlflow/bin/activate

# Start custom server with /asr endpoint
python server.py

# Or with custom model path
MODEL_PATH="mlruns/0/models/m-<model_id>/artifacts" python server.py
```

### Server Options

- `--no-conda`: Use current environment instead of creating conda environment
- `--host 0.0.0.0`: Make server accessible from all network interfaces
- `--port 5000`: Specify port (default is 5000)
- `--workers 1`: Number of worker processes (default is 1)

### Running in Background

```bash
# Using nohup
nohup mlflow models serve \
  -m "mlruns/0/models/m-<model_id>/artifacts" \
  --no-conda \
  --host 0.0.0.0 \
  --port 5000 \
  > mlflow_server.log 2>&1 &

# Check if running
ps aux | grep mlflow
```

### Using screen or tmux

```bash
# Using screen
screen -S mlflow
source mlflow/bin/activate
mlflow models serve -m "mlruns/0/models/m-<model_id>/artifacts" --no-conda --host 0.0.0.0 --port 5000
# Press Ctrl+A then D to detach

# Reattach later
screen -r mlflow
```

## Testing the Setup

### Test 1: Health Check

```bash
curl http://localhost:5000/health
```

### Test 2: API Documentation

Open in browser:
- http://localhost:5000/docs (Swagger UI)
- http://localhost:5000/redoc (ReDoc)

### Test 3: Send Test Request

```bash
./test_curl.sh ta ctc
```

## Common Setup Issues

### Issue: Virtual Environment Not Activating

**Symptoms**: `source mlflow/bin/activate` doesn't work

**Solution**:
```bash
# Check if virtual environment exists
ls -la mlflow/bin/activate

# If missing, recreate
python3 -m venv mlflow
```

### Issue: Permission Denied

**Symptoms**: Permission errors when installing packages

**Solution**:
```bash
# Don't use sudo with pip in virtual environment
# If needed, fix permissions
chmod -R u+w mlflow/
```

### Issue: CUDA Out of Memory

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solution**:
- Reduce batch size in model wrapper
- Use CPU instead: The model will automatically fall back to CPU if CUDA is not available
- Close other GPU processes

### Issue: Model Download Fails

**Symptoms**: Timeout or authentication errors when loading model

**Solution**:
1. Check internet connection
2. Verify HF_TOKEN is set: `echo $HF_TOKEN`
3. Try downloading manually:
   ```bash
   python3 -c "from transformers import AutoModel; AutoModel.from_pretrained('ai4bharat/indic-conformer-600m-multilingual', token='your_token')"
   ```

## Next Steps

After successful setup:

1. Read [README.md](README.md) for usage instructions
2. Test the API with `./test_curl.sh`
3. Explore the API documentation at http://localhost:5000/docs
4. Integrate the service into your application

## Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)

---

**Need Help?** Check the [Troubleshooting](#common-setup-issues) section or review the main [README.md](README.md).

