# Indiclid-triton Service Guide

## 📖 What is Indiclid-triton?

**Indiclid** stands for **Indic Language Identification**. This service can read text and automatically identify which Indian language it is written in. It supports all **22 official Indian languages** in both their native scripts (like Devanagari, Tamil, Telugu) and romanized scripts (English letters).

### Real-World Use Cases
- **Content Moderation**: Automatically categorize user-generated content by language
- **Translation Services**: Route text to appropriate translation models
- **Search Engines**: Improve search results by understanding language
- **Social Media**: Tag posts with their language for better content discovery
- **Customer Support**: Route support tickets to language-specific teams

---

## 🎯 What You Need Before Starting

### For Everyone (Non-Technical)

Before you can use this service, you need:
1. **A computer with Linux** (Ubuntu recommended)
2. **An NVIDIA graphics card** (GPU) - This makes the service run much faster
3. **Internet connection** - To download the necessary software and models

### For Technical Users

**System Requirements:**
- **OS**: Linux (Ubuntu 20.04+)
- **GPU**: NVIDIA GPU with CUDA support
- **Docker**: Version 20.10+
- **Docker Compose**: Version 1.29+
- **NVIDIA Container Toolkit**: For GPU access in Docker

**Software Installation:**
```bash
# Install Docker
sudo apt-get update
sudo apt-get install -y docker.io docker-compose

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

---

## 🏗️ Understanding the Service Structure

The service uses a smart two-stage approach:
1. **First**, it checks if text is in native script or roman script
2. **Then**, it uses the appropriate model to identify the language

```
indiclid-triton/
├── Dockerfile              # Recipe to build the container
├── README.md               # Documentation
├── test_client.py          # Test script
└── model_repository/       # Model storage
    └── indiclid/
        ├── config.pbtxt    # Service configuration
        └── 1/
            └── model.py    # Processing logic
            ├── indiclid-ftn.bin    # Native script model
            ├── indiclid-ftr.bin    # Roman script model
            └── indiclid-bert.pt    # BERT model for low confidence cases
```

---

## 🔨 Step 1: Building the Docker Image

### What is Building?

Building a Docker image packages everything needed to run the service: the code, the AI models, and all dependencies. Once built, you can run it anywhere.

### Step-by-Step Build Instructions

#### Option A: Simple Build (Recommended for Beginners)

1. **Open a terminal** (command line window)

2. **Navigate to the indiclid-triton folder:**
   ```bash
   cd indiclid-triton
   ```

3. **Build the image:**
   ```bash
   docker build -t indiclid-triton:latest .
   ```
   
   **What this does:**
   - `docker build` = Start building
   - `-t indiclid-triton:latest` = Name the image "indiclid-triton" with tag "latest"
   - `.` = Use the current directory (where Dockerfile is located)

4. **Wait for it to complete** (this may take 10-20 minutes the first time)
   - Downloads the base Triton server
   - Installs Python packages (transformers, fasttext, etc.)
   - Downloads IndicBERT model components
   - Downloads IndicLID models from GitHub
   - Sets everything up

#### Option B: Understanding What Happens During Build

The Dockerfile does these steps:
1. **Starts with Triton Server base image** - Pre-configured server
2. **Installs Python packages** - Transformers, FastText, and other ML libraries
3. **Downloads IndicBERT** - The BERT model for Indian languages
4. **Downloads IndicLID models from GitHub** - Three models:
   - `indiclid-ftn.bin` - FastText Native (for native scripts)
   - `indiclid-ftr.bin` - FastText Roman (for roman scripts)
   - `indiclid-bert.pt` - BERT model (for low-confidence cases)
   - Source: [https://github.com/AI4Bharat/IndicLID](https://github.com/AI4Bharat/IndicLID)
5. **Copies the model code** - Your custom processing logic

**Expected Output:**
```
Step 1/6 : FROM nvcr.io/nvidia/tritonserver:24.01-py3
...
Downloading IndicLID models...
Extracting models...
Model files prepared
...
Successfully built abc123def456
Successfully tagged indiclid-triton:latest
```

**Troubleshooting Build Issues:**
- **"Cannot connect to Docker daemon"**: Run `sudo systemctl start docker`
- **"No space left on device"**: Free up disk space (models are ~500MB)
- **"Network timeout"**: Check internet connection, models download from GitHub
- **"Model download failed"**: Retry the build, GitHub releases can be slow

---

## 📥 How the IndicLID Model Was Obtained from GitHub

### Model Source

The Indiclid service uses the **IndicLID** (Indic Language Identification) model from AI4Bharat:
- **Repository**: [https://github.com/AI4Bharat/IndicLID](https://github.com/AI4Bharat/IndicLID)
- **Model Type**: Language Identification for Indian Languages
- **License**: MIT License
- **Download Link**: [https://github.com/AI4Bharat/IndicLID/releases/tag/v1.0](https://github.com/AI4Bharat/IndicLID/releases/tag/v1.0)

### About the Model

IndicLID is a language identification system specifically designed for Indian languages that:

- **Supports 22 official Indian languages** in both native and roman scripts:
  - Native scripts: Hindi (Devanagari), Tamil, Telugu, Bengali, Gujarati, Kannada, Malayalam, Marathi, Punjabi, Odia, Assamese, Urdu, and more
  - Roman scripts: All 22 languages in romanized form (e.g., "main bharat se hoon" for Hindi)
  - Plus English and "Other" category

- **Uses a two-stage detection pipeline**:
  1. **Script Detection**: Determines if text is in native script or roman script
  2. **Language Identification**: Uses appropriate model based on script type
     - FastText models for fast inference (IndicLID-FTN for native, IndicLID-FTR for roman)
     - BERT model (IndicLID-BERT) for low-confidence cases in roman script

- **Performance**:
  - FastText models: ~30,000 sentences/second
  - High accuracy: 98% F1-score on native scripts
  - BERT model: ~3 sentences/second but more accurate for ambiguous cases

### How the Model is Downloaded

During the Docker build process, the models are automatically downloaded from GitHub releases:

```bash
# The Dockerfile downloads models from GitHub releases
wget https://github.com/AI4Bharat/IndicLID/releases/download/v1.0/indiclid-ftn.bin
wget https://github.com/AI4Bharat/IndicLID/releases/download/v1.0/indiclid-ftr.bin
wget https://github.com/AI4Bharat/IndicLID/releases/download/v1.0/indiclid-bert.pt
```

**What happens:**
1. Docker build process connects to GitHub releases
2. Downloads three model files:
   - `indiclid-ftn.bin` - FastText Native model (~200MB)
   - `indiclid-ftr.bin` - FastText Roman model (~200MB)
   - `indiclid-bert.pt` - BERT model (~100MB)
3. Models are cached in the Docker image
4. Models are ready to use when the container starts

### Model Files Structure

The downloaded models include:
- **FastText Native (FTN)**: Pre-trained FastText model for native script identification
- **FastText Roman (FTR)**: Pre-trained FastText model for roman script identification
- **BERT Model**: Fine-tuned IndicBERT model for low-confidence roman script cases
- **IndicBERT**: Base BERT model for Indian languages (downloaded from HuggingFace)

### Two-Stage Detection Process

1. **Script Detection**:
   - Checks percentage of roman characters in text
   - If < 50% roman → Use IndicLID-FTN (native script model)
   - If ≥ 50% roman → Use IndicLID-FTR (roman script model)

2. **Roman Script Path** (if needed):
   - **Stage 1**: FastText Roman model (very fast)
   - If confidence > 0.6: Return result
   - If confidence ≤ 0.6: Proceed to Stage 2
   - **Stage 2**: BERT model (slower but more accurate)

### References

- **GitHub Repository**: [https://github.com/AI4Bharat/IndicLID](https://github.com/AI4Bharat/IndicLID)
- **Model Releases**: [https://github.com/AI4Bharat/IndicLID/releases/tag/v1.0](https://github.com/AI4Bharat/IndicLID/releases/tag/v1.0)
- **Contributors**: Yash Madhani, Mitesh M. Khapra, Anoop Kunchukuttan (AI4Bharat, IITM)

---

## 🚀 Step 2: Running the Service

### What is Running?

Running the service means starting it up so it can accept requests. It's like opening a restaurant for business.

### Step-by-Step Run Instructions

#### Basic Run (For Testing)

```bash
docker run --gpus all \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  --name indiclid-server \
  indiclid-triton:latest
```

**What each part means:**
- `docker run` = Start a container
- `--gpus all` = Use all available GPUs (makes it faster)
- `-p 8000:8000` = Map port 8000 on your computer to port 8000 in container
- `-p 8001:8001` = Map gRPC port
- `-p 8002:8002` = Map metrics port
- `--name indiclid-server` = Name the container "indiclid-server"
- `indiclid-triton:latest` = Use the image we built

#### Run in Background (Recommended for Production)

```bash
docker run -d --gpus all \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  --name indiclid-server \
  indiclid-triton:latest
```

The `-d` flag runs it in the background (detached mode).

#### Using Docker Compose (Easiest Method)

If you're using the main docker-compose.yml file:

```bash
docker-compose up -d indiclid-server
```

This automatically handles all the configuration.

### Understanding Ports

- **Port 8000 (HTTP)**: Main entrance for web requests
- **Port 8001 (gRPC)**: Fast lane for program-to-program communication
- **Port 8002 (Metrics)**: Monitoring room for checking service health

---

## ✅ Step 3: Verifying the Service is Running

### Check if Container is Running

```bash
docker ps
```

You should see `indiclid-server` in the list with status "Up".

### Check Service Health

```bash
curl http://localhost:8000/v2/health/ready
```

**Expected Response:** `{"status":"ready"}`

If you see this, the service is ready to accept requests!

### List Available Models

```bash
curl http://localhost:8000/v2/models
```

**Expected Response:**
```json
{
  "models": [
    {
      "name": "indiclid",
      "platform": "python",
      "versions": ["1"]
    }
  ]
}
```

### View Logs

```bash
docker logs indiclid-server
```

Look for:
- `"Server is ready to receive inference requests"` = Success!
- Any error messages = Something went wrong

---

## 🧪 Step 4: Testing the Service

### Method 1: Using the Provided Test Script (Easiest)

```bash
cd indiclid-triton
python3 test_client.py
```

This script tests all supported languages in both native and roman scripts.

**Expected Output:**
```
Testing Hindi (Native Script): मैं भारत से हूं
Result: {"langCode": "hin_Deva", "confidence": 0.9999, "model": "IndicLID-FTN"}

Testing Hindi (Roman Script): main bharat se hoon
Result: {"langCode": "hin_Latn", "confidence": 0.9850, "model": "IndicLID-FTR"}
```

### Method 2: Manual Testing with curl

#### Native Script Example (Hindi)

```bash
curl -X POST http://localhost:8000/v2/models/indiclid/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "INPUT_TEXT",
        "shape": [1, 1],
        "datatype": "BYTES",
        "data": [["मैं भारत से हूं"]]
      }
    ],
    "outputs": [
      {
        "name": "OUTPUT_TEXT"
      }
    ]
  }'
```

**Expected Response:**
```json
{
  "outputs": [
    {
      "name": "OUTPUT_TEXT",
      "datatype": "BYTES",
      "shape": [1, 1],
      "data": ["{\"input\": \"मैं भारत से हूं\", \"langCode\": \"hin_Deva\", \"confidence\": 0.9999, \"model\": \"IndicLID-FTN\"}"]
    }
  ]
}
```

#### Roman Script Example (Hindi)

```bash
curl -X POST http://localhost:8000/v2/models/indiclid/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "INPUT_TEXT",
        "shape": [1, 1],
        "datatype": "BYTES",
        "data": [["main bharat se hoon"]]
      }
    ],
    "outputs": [
      {
        "name": "OUTPUT_TEXT"
      }
    ]
  }'
```

### Method 3: Python Test Script

Create a file `test_my_text.py`:

```python
import requests
import json

# Your text to test
text = "मैं भारत से हूं"  # Hindi in Devanagari script

# Prepare request
url = "http://localhost:8000/v2/models/indiclid/infer"
payload = {
    "inputs": [
        {
            "name": "INPUT_TEXT",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [[text]]
        }
    ],
    "outputs": [
        {"name": "OUTPUT_TEXT"}
    ]
}

# Send request
response = requests.post(url, json=payload)
result = response.json()

# Parse and display results
output_text = result["outputs"][0]["data"][0]
result_json = json.loads(output_text)

print(f"Input: {result_json['input']}")
print(f"Detected Language: {result_json['langCode']}")
print(f"Confidence: {result_json['confidence']:.4f}")
print(f"Model Used: {result_json['model']}")
```

Run it:
```bash
python3 test_my_text.py
```

---

## 📊 Understanding the API

### Input Format

**INPUT_TEXT** (Required)
- **Type**: String (text)
- **What to send**: The text you want to identify
- **Supports**: Native script and roman script text

### Output Format

The service returns a JSON string with:

```json
{
  "input": "original input text",
  "langCode": "detected language code",
  "confidence": 0.9999,
  "model": "model used for detection"
}
```

**Fields:**
- **input**: The text you sent
- **langCode**: Language code (e.g., `hin_Deva` for Hindi in Devanagari, `hin_Latn` for Hindi in Roman)
- **confidence**: How sure the model is (0.0 to 1.0)
- **model**: Which model was used (`IndicLID-FTN`, `IndicLID-FTR`, or `IndicLID-BERT`)

### Supported Languages

The service supports all 22 official Indian languages:

| Language | Native Script Code | Roman Script Code |
|----------|-------------------|-------------------|
| Hindi | hin_Deva | hin_Latn |
| Bengali | ben_Beng | ben_Latn |
| Tamil | tam_Tamil | tam_Latn |
| Telugu | tel_Telu | tel_Latn |
| Gujarati | guj_Gujr | guj_Latn |
| Kannada | kan_Knda | kan_Latn |
| Malayalam | mal_Mlym | mal_Latn |
| Marathi | mar_Deva | mar_Latn |
| Punjabi | pan_Guru | pan_Latn |
| Odia | ori_Orya | ori_Latn |
| Assamese | asm_Beng | asm_Latn |
| Urdu | urd_Arab | urd_Latn |
| And 10 more languages... |

Plus English (`eng_Latn`) and Other (`other`).

---

## 🧠 How It Works (Technical Details)

### Two-Stage Detection Pipeline

1. **Script Detection**: 
   - Checks if text is primarily roman script or native script
   - Threshold: 50% roman characters = roman script

2. **Native Script Path** (if < 50% roman):
   - Uses **IndicLID-FTN** (FastText Native) model
   - Very fast (~30,000 sentences/second)
   - High accuracy (98% F1-score)

3. **Roman Script Path** (if ≥ 50% roman):
   - **Stage 1**: Uses **IndicLID-FTR** (FastText Roman) model
   - If confidence > 0.6: Returns FTR result
   - If confidence ≤ 0.6: Proceeds to Stage 2
   - **Stage 2**: Uses **IndicLID-BERT** model
   - Slower (~3 sentences/second) but more accurate
   - Used only for low-confidence cases

### Performance Characteristics

- **FastText Models**: Very fast, good for high-throughput scenarios
- **BERT Model**: Slower but more accurate, used only when needed
- **Overall**: Optimized for both speed and accuracy

---

## ⚙️ Configuration Options

### Adjusting Performance

You can modify `model_repository/indiclid/config.pbtxt` to change:

1. **Max Batch Size**: How many requests to process together
   - Current: 64
   - Increase for more throughput (needs more GPU memory)
   - Decrease if running out of memory

2. **Dynamic Batching**: Automatically groups requests
   - Current: Enabled with sizes [1, 2, 4, 8, 16, 32, 64]
   - Adjust based on your workload

3. **GPU Instances**: Number of model copies
   - Current: 1
   - Increase for higher throughput (uses more GPU memory)

**After changing config.pbtxt, rebuild the image:**
```bash
docker build -t indiclid-triton:latest .
docker stop indiclid-server
docker rm indiclid-server
docker run -d --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  --name indiclid-server indiclid-triton:latest
```

---

## 🛠️ Troubleshooting

### Problem: Service Won't Start

**Symptoms**: Container exits immediately or shows errors

**Solutions**:
1. **Check logs**: `docker logs indiclid-server`
2. **Verify GPU**: `nvidia-smi` should show your GPU
3. **Check port conflicts**: `sudo lsof -i :8000` (should be empty)
4. **Verify Docker GPU access**: 
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

### Problem: Model Files Not Found

**Symptoms**: Errors about missing model files (ftn.bin, ftr.bin, bert.pt)

**Solutions**:
1. **Check if models downloaded during build**: Review build logs
2. **Rebuild the image**: Models should download automatically
3. **Check disk space**: `df -h` (models need ~500MB)
4. **Verify internet connection**: Models download from GitHub releases

### Problem: "Out of Memory" Errors

**Symptoms**: CUDA out of memory errors in logs

**Solutions**:
1. **Reduce batch size** in config.pbtxt
2. **Use shorter text inputs**
3. **Check GPU memory**: `nvidia-smi`
4. **Close other GPU applications**

### Problem: Wrong Language Detected

**Symptoms**: Service detects incorrect language

**Solutions**:
1. **Use longer text** (at least 10-20 characters)
2. **Check if language is supported** (22 Indian languages + English)
3. **For roman script**: Ensure text is clearly romanized
4. **Try multiple text samples** for consistency
5. **Check confidence score**: Low confidence (< 0.6) may indicate ambiguous text

### Problem: Slow Inference

**Symptoms**: Requests take a long time to process

**Solutions**:
1. **Verify GPU is being used**: Check logs for "Using device: cuda"
2. **Check GPU utilization**: `nvidia-smi -l 1` (should show activity)
3. **Increase batch size** (if memory allows)
4. **Note**: BERT model is slower but only used for low-confidence cases

### Problem: Cannot Connect to Service

**Symptoms**: Connection refused or timeout errors

**Solutions**:
1. **Verify service is running**: `docker ps`
2. **Check port mapping**: `docker ps` should show port mappings
3. **Test from container**: 
   ```bash
   docker exec indiclid-server curl http://localhost:8000/v2/health/ready
   ```
4. **Check firewall**: `sudo ufw status`

---

## 📈 Monitoring and Metrics

### Health Check

```bash
curl http://localhost:8000/v2/health/ready
```

### Metrics Endpoint

```bash
curl http://localhost:8002/metrics
```

This provides Prometheus-compatible metrics including:
- Request count
- Inference latency
- GPU utilization
- Error rates

### View Real-Time Logs

```bash
docker logs -f indiclid-server
```

Press `Ctrl+C` to stop viewing logs.

---

## 🔄 Common Operations

### Stop the Service

```bash
docker stop indiclid-server
```

### Start the Service

```bash
docker start indiclid-server
```

### Restart the Service

```bash
docker restart indiclid-server
```

### Remove the Service

```bash
docker stop indiclid-server
docker rm indiclid-server
```

### Update the Service

```bash
# Rebuild with latest changes
docker build -t indiclid-triton:latest .

# Stop and remove old container
docker stop indiclid-server
docker rm indiclid-server

# Start new container
docker run -d --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  --name indiclid-server indiclid-triton:latest
```

---

## 📚 Additional Resources

### Service Documentation
- **Detailed README**: `indiclid-triton/README.md`
- **Model Source (GitHub)**: [https://github.com/AI4Bharat/IndicLID](https://github.com/AI4Bharat/IndicLID)
  - This is where the IndicLID models were obtained from
- **Model Downloads**: [https://github.com/AI4Bharat/IndicLID/releases/tag/v1.0](https://github.com/AI4Bharat/IndicLID/releases/tag/v1.0)

### Technical References
- **Triton Server Docs**: https://docs.nvidia.com/deeplearning/triton-inference-server/
- **Docker Docs**: https://docs.docker.com/

### Getting Help

If you encounter issues:
1. Check the logs: `docker logs indiclid-server`
2. Review this guide's troubleshooting section
3. Check the service README: `indiclid-triton/README.md`
4. Review IndicLID documentation on GitHub

---

## 📝 Quick Reference

### Essential Commands

```bash
# Build
cd indiclid-triton
docker build -t indiclid-triton:latest .

# Run
docker run -d --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  --name indiclid-server indiclid-triton:latest

# Check status
docker ps
curl http://localhost:8000/v2/health/ready

# Test
cd indiclid-triton
python3 test_client.py

# View logs
docker logs -f indiclid-server

# Stop
docker stop indiclid-server
```

### Port Information

- **HTTP API**: `http://localhost:8000`
- **gRPC API**: `localhost:8001`
- **Metrics**: `http://localhost:8002/metrics`

### Model Information

- **Model Name**: `indiclid`
- **Backend**: Python
- **Max Batch Size**: 64
- **GPU Required**: Yes
- **Supported Languages**: 22 Indian languages + English

---

## ✅ Summary

You've learned how to:
1. ✅ Build the Indiclid-triton Docker image
2. ✅ Run the service
3. ✅ Verify it's working
4. ✅ Test language identification
5. ✅ Use the API
6. ✅ Understand how the two-stage detection works
7. ✅ Troubleshoot common issues

The Indiclid-triton service is now ready to identify Indian languages in text! For production use, consider setting up monitoring, load balancing, and proper security measures.



