# NMT-triton Service Guide

## 📖 What is NMT-triton?

**NMT** stands for **Neural Machine Translation**. This service can automatically translate text between **22 scheduled Indian languages** and English. It uses IndicTrans2, a state-of-the-art multilingual translation model developed by AI4Bharat.

### Real-World Use Cases
- **Content Translation**: Translate websites, documents, and applications
- **Customer Support**: Provide multilingual customer service
- **Education**: Translate educational content to regional languages
- **Government Services**: Translate official documents and communications
- **Media**: Translate news articles, subtitles, and media content
- **E-commerce**: Translate product descriptions and reviews
- **Communication**: Enable communication between speakers of different languages

---

## 🎯 What You Need Before Starting

### For Everyone (Non-Technical)

Before you can use this service, you need:
1. **A computer with Linux** (Ubuntu recommended)
2. **An NVIDIA graphics card** (GPU) - This makes the service run much faster
3. **Internet connection** - To download the Docker image and model
4. **Docker** - Software to run the service in a container

### For Technical Users

**System Requirements:**
- **OS**: Linux (Ubuntu 20.04+)
- **GPU**: NVIDIA GPU with CUDA support
- **Docker**: Version 20.10+
- **Docker Compose**: Version 1.29+ (optional, but recommended)
- **NVIDIA Container Toolkit**: For GPU access in Docker
- **Shared Memory**: At least 64MB (`shm_size`)
- **Hardware Specifications**: May vary depending on the scale of your application
- **Tested Machine**: g4dn.2xlarge (For detailed specifications and pricing, check [AWS EC2 g4dn.2xlarge](https://instances.vantage.sh/aws/ec2/g4dn.2xlarge?currency=USD))

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

Think of this service like a translation office:
- **Docker Image** = Pre-built package containing everything needed
- **Model Repository** = The translation desk where AI models work
- **Triton Server** = The coordinator managing translation requests
- **SentencePiece Models** = Language dictionaries for encoding/decoding text
- **Translation Model** = The AI translator that converts between languages

```
NMT Service/
├── Docker Image: ai4bharat/triton-indictrans-v2:latest
├── Model: nmt (single translation model)
├── Input: Source text + source language + target language
├── Output: Translated text
└── Ports:
    ├── 8000 (HTTP API)
    ├── 8001 (gRPC API)
    └── 8002 (Metrics)
```

---

## 🐳 Step 1: Pulling the Docker Image

### What is Pulling?

Pulling a Docker image means downloading a pre-built package from Docker Hub (a repository of container images). The NMT service image is already built and ready to use - you just need to download it.

### Step-by-Step Pull Instructions

#### Option A: Simple Pull (Recommended)

1. **Open a terminal** (command line window)

2. **Pull the image:**
   ```bash
   docker pull ai4bharat/triton-indictrans-v2:latest
   ```
   
   **What this does:**
   - `docker pull` = Download the image
   - `ai4bharat/triton-indictrans-v2:latest` = Image name and version tag
   - `latest` = Most recent version
   - `v2` = IndicTrans2 model version

3. **Wait for it to complete** (this may take 5-20 minutes depending on internet speed)
   - The image is large (several GB) because it contains:
     - Triton Inference Server
     - Python runtime and libraries
     - Pre-trained translation models
     - SentencePiece tokenizers for all supported languages
     - All dependencies

**Expected Output:**
```
latest: Pulling from ai4bharat/triton-indictrans-v2
...
Status: Downloaded newer image for ai4bharat/triton-indictrans-v2:latest
docker.io/ai4bharat/triton-indictrans-v2:latest
```

#### Verify Image is Downloaded

```bash
docker images | grep triton-indictrans
```

You should see the image listed with its size.

**Troubleshooting Pull Issues:**
- **"Cannot connect to Docker daemon"**: Run `sudo systemctl start docker`
- **"Network timeout"**: Check internet connection, retry the pull
- **"No space left on device"**: Free up disk space (`docker system prune` to clean old images)

---

## 🚀 Step 2: Running the Service

### What is Running?

Running the service means starting it up so it can accept requests. The service will load the translation models into GPU memory and be ready to translate text.

### Step-by-Step Run Instructions

#### Option A: Using Docker Run (For Testing)

```bash
docker run -d --gpus all \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  --shm-size=64mb \
  --name indictrans \
  ai4bharat/triton-indictrans-v2:latest \
  tritonserver --model-repository=/models
```

**What each part means:**
- `docker run` = Start a container
- `-d` = Run in background (detached mode)
- `--gpus all` = Use all available GPUs (makes it faster)
- `-p 8000:8000` = Map port 8000 on your computer to port 8000 in container (HTTP)
- `-p 8001:8001` = Map port 8001 for gRPC API
- `-p 8002:8002` = Map port 8002 for metrics
- `--shm-size=64mb` = Allocate 64MB shared memory (smaller than ASR/TTS)
- `--name indictrans` = Name the container "indictrans"
- `ai4bharat/triton-indictrans-v2:latest` = Use the image we pulled
- `tritonserver --model-repository=/models` = Command to start the server

#### Option B: Using Docker Compose (Recommended for Production)

Create or use `docker-compose.yml`:

```yaml
version: '3.8'

services:
  indictrans:
    image: ai4bharat/triton-indictrans-v2:latest
    container_name: indictrans
    ports:
      - "8000:8000"  # HTTP API
      - "8001:8001"  # GRPC API
      - "8002:8002"  # Metrics
    command: tritonserver --model-repository=/models
    shm_size: 64mb
    runtime: nvidia
    restart: always
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
```

Then run:
```bash
docker-compose up -d indictrans
```

### Understanding Ports

Think of ports like different doors to the same building:
- **Port 8000 (HTTP)**: Main entrance for web requests (REST API)
- **Port 8001 (gRPC)**: Fast lane for program-to-program communication
- **Port 8002 (Metrics)**: Monitoring room for checking service health and performance

**Note**: If you're running multiple services, make sure port 8000 is not already in use by another service. NMT uses ports 8000-8002.

---

## ✅ Step 3: Verifying the Service is Running

### Check if Container is Running

```bash
docker ps
```

You should see `indictrans` in the list with status "Up".

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
      "name": "nmt",
      "platform": "python",
      "versions": ["1"]
    }
  ]
}
```

### View Logs

```bash
docker logs indictrans
```

Or follow logs in real-time:
```bash
docker logs -f indictrans
```

Look for:
- `"Server is ready to receive inference requests"` = Success!
- `"successfully loaded 'nmt' version 1"` = Model loaded successfully
- `"Initializing sentencepiece model for SRC and TGT"` = Language models initializing
- Any error messages = Something went wrong

---

## 🧪 Step 4: Testing the Service

> **Note**: If you're accessing the service from a remote machine, replace `localhost` with your server's IP address. For example, if your server IP is `192.168.1.100`, use `http://192.168.1.100:8000` instead of `http://localhost:8000`.

### Method 1: Using a Python Script (Recommended)

Create a file `test_nmt.py`:

```python
#!/usr/bin/env python3
"""
Script to translate text using NMT model via Triton HTTP API
"""
import json
import requests
import sys

def test_nmt(text, source_lang, target_lang, endpoint="http://localhost:8000/v2/models/nmt/infer"):
    """
    Send translation request to NMT service
    """
    print(f"Translating from {source_lang} to {target_lang}")
    print(f"Source text: {text}")
    
    # Create Triton request payload
    payload = {
        "inputs": [
            {
                "name": "INPUT_TEXT",
                "shape": [1],
                "datatype": "BYTES",
                "data": [text]
            },
            {
                "name": "INPUT_LANGUAGE_ID",
                "shape": [1],
                "datatype": "BYTES",
                "data": [source_lang]
            },
            {
                "name": "OUTPUT_LANGUAGE_ID",
                "shape": [1],
                "datatype": "BYTES",
                "data": [target_lang]
            }
        ]
    }
    
    print(f"\nSending request to: {endpoint}")
    
    # Send request
    response = requests.post(endpoint, json=payload)
    response.raise_for_status()
    
    result = response.json()
    
    print("\n=== Response ===")
    print(json.dumps(result, indent=2))
    
    # Extract translated text
    if "outputs" in result:
        for output in result["outputs"]:
            if output.get("name") == "OUTPUT_TEXT":
                if "data" in output:
                    translated_text = output["data"][0]
                    if isinstance(translated_text, bytes):
                        translated_text = translated_text.decode('utf-8')
                    print(f"\n=== Translated Text ===")
                    print(translated_text)
                    return translated_text
    
    return None

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 test_nmt.py <text> <source_lang> <target_lang>")
        print("Example: python3 test_nmt.py 'Hello world' en hi")
        sys.exit(1)
    
    text = sys.argv[1]
    source_lang = sys.argv[2]
    target_lang = sys.argv[3]
    
    test_nmt(text, source_lang, target_lang)
```

**Run the script:**
```bash
python3 test_nmt.py "Hello, how are you?" en hi
```

**Expected Output:**
```
Translating from en to hi
Source text: Hello, how are you?

=== Translated Text ===
नमस्ते, आप कैसे हैं?
```

### Method 2: Using curl (Manual Testing)

```bash
curl -X POST http://localhost:8000/v2/models/nmt/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "INPUT_TEXT",
        "shape": [1],
        "datatype": "BYTES",
        "data": ["Hello, how are you?"]
      },
      {
        "name": "INPUT_LANGUAGE_ID",
        "shape": [1],
        "datatype": "BYTES",
        "data": ["en"]
      },
      {
        "name": "OUTPUT_LANGUAGE_ID",
        "shape": [1],
        "datatype": "BYTES",
        "data": ["hi"]
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
      "shape": [1],
      "data": ["नमस्ते, आप कैसे हैं?"]
    }
  ]
}
```

### Method 3: Batch Translation Example

You can translate multiple sentences by sending multiple requests or using batch processing:

```python
import requests

def translate_batch(texts, source_lang, target_lang):
    """Translate multiple texts"""
    results = []
    for text in texts:
        payload = {
            "inputs": [
                {"name": "INPUT_TEXT", "shape": [1], "datatype": "BYTES", "data": [text]},
                {"name": "INPUT_LANGUAGE_ID", "shape": [1], "datatype": "BYTES", "data": [source_lang]},
                {"name": "OUTPUT_LANGUAGE_ID", "shape": [1], "datatype": "BYTES", "data": [target_lang]}
            ]
        }
        response = requests.post("http://localhost:8000/v2/models/nmt/infer", json=payload)
        result = response.json()
        translated = result["outputs"][0]["data"][0]
        if isinstance(translated, bytes):
            translated = translated.decode('utf-8')
        results.append(translated)
    return results

# Example usage
texts = ["Hello", "How are you?", "Thank you"]
translations = translate_batch(texts, "en", "hi")
for original, translated in zip(texts, translations):
    print(f"{original} -> {translated}")
```

---

## 📊 Understanding the API

### Model Information

- **Model Name**: `nmt`
- **Type**: Python backend model
- **Backend**: Triton Inference Server with Python backend
- **Max Batch Size**: 512
- **Dynamic Batching**: Enabled

### Input Format

**INPUT_TEXT** (Required)
- **Type**: BYTES (string)
- **Shape**: `[1]` (single text string)
- **Description**: The text to be translated
- **Format**: UTF-8 encoded string

**INPUT_LANGUAGE_ID** (Required)
- **Type**: BYTES (string)
- **Shape**: `[1]`
- **Description**: Source language code (language of INPUT_TEXT)
- **Supported Codes**: See language list below

**OUTPUT_LANGUAGE_ID** (Required)
- **Type**: BYTES (string)
- **Shape**: `[1]`
- **Description**: Target language code (desired translation language)
- **Supported Codes**: See language list below

### Output Format

**OUTPUT_TEXT**
- **Type**: BYTES (string)
- **Shape**: `[1]`
- **Description**: Translated text in the target language
- **Format**: UTF-8 encoded string

### Supported Languages

The service supports **22 scheduled Indian languages** plus English:

#### Indic Languages:
- **Assamese** (as)
- **Bengali** (bn)
- **Bodo** (brx)
- **Dogri** (doi)
- **Gujarati** (gu)
- **Hindi** (hi)
- **Kannada** (kn)
- **Kashmiri** (ks)
- **Konkani** (kok)
- **Maithili** (mai)
- **Malayalam** (ml)
- **Manipuri** (mni)
- **Marathi** (mr)
- **Nepali** (ne)
- **Odia** (or)
- **Punjabi** (pa)
- **Sanskrit** (sa)
- **Santali** (sat)
- **Sindhi** (sd)
- **Tamil** (ta)
- **Telugu** (te)
- **Urdu** (ur)

#### English:
- **English** (en)

**Translation Directions:**
- Any Indic language ↔ English
- Any Indic language ↔ Any other Indic language

---

## 📝 Text Format Requirements

### Input Text Guidelines

- **Encoding**: UTF-8
- **Length**: No strict limit, but longer texts are processed sentence by sentence
- **Format**: Plain text (no special formatting required)
- **Scripts**: Supports all Indic scripts (Devanagari, Tamil, Telugu, etc.)

### Tips for Best Results

1. **Complete sentences** translate better than fragments
2. **Clear punctuation** helps the model understand sentence boundaries
3. **Domain-specific terms** may need post-processing for accuracy
4. **Proper nouns** (names, places) may be transliterated rather than translated
5. **Mixed language text** should be separated by language

---

## ⚙️ Configuration Options

### Docker Run Options

You can customize the container with additional options:

```bash
docker run -d --gpus all \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  --shm-size=64mb \
  --name indictrans \
  --restart=always \
  -e NVIDIA_VISIBLE_DEVICES=all \
  ai4bharat/triton-indictrans-v2:latest \
  tritonserver --model-repository=/models
```

**Options explained:**
- `--restart=always` = Automatically restart container if it crashes
- `-e NVIDIA_VISIBLE_DEVICES=all` = Use all GPUs (can specify specific GPU like "0" or "0,1")
- `--shm-size=64mb` = Shared memory size (64MB is sufficient for NMT)

### Resource Allocation

- **GPU Memory**: Model requires several GB of GPU memory
- **System Memory**: At least 4GB RAM recommended
- **Shared Memory**: 64MB minimum (much smaller than ASR/TTS)
- **Batch Size**: Supports up to 512 items per batch

### Performance Optimization

The model supports **dynamic batching**, which means:
- Multiple requests can be batched together automatically
- Better GPU utilization
- Higher throughput for multiple requests
- Preferred batch size: 512

---

## 🛠️ Troubleshooting

### Problem: Service Won't Start

**Symptoms**: Container exits immediately or shows errors

**Solutions**:
1. **Check logs**: `docker logs indictrans`
2. **Verify GPU**: `nvidia-smi` should show your GPU
3. **Check port conflicts**: `sudo lsof -i :8000` (should be empty)
4. **Verify Docker GPU access**: 
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```
5. **Check shared memory**: Ensure `--shm-size=64mb` is set

### Problem: "Out of Memory" Errors

**Symptoms**: CUDA out of memory errors in logs

**Solutions**:
1. **Check GPU memory**: `nvidia-smi`
2. **Process shorter texts** or reduce batch size
3. **Close other GPU applications**
4. **Use CPU fallback** (if configured, though slower)

### Problem: Slow Inference

**Symptoms**: Requests take a long time to process

**Solutions**:
1. **Verify GPU is being used**: Check logs for GPU usage
2. **Check GPU utilization**: `nvidia-smi -l 1` (should show activity)
3. **Use batch processing** for multiple translations
4. **Check text length** - longer texts take more time

### Problem: Wrong Translation or Poor Quality

**Symptoms**: Translation doesn't make sense

**Solutions**:
1. **Verify language codes** are correct (case-sensitive)
2. **Check if language pair is supported**
3. **Use complete sentences** for better context
4. **Pre-process text** (remove extra whitespace, fix encoding)
5. **Check source text quality** - garbage in, garbage out

### Problem: Language Code Error

**Symptoms**: Error about unsupported language code

**Solutions**:
1. **Use correct language codes** from the supported list
2. **Check code format** - should be lowercase 2-letter codes (e.g., "hi", not "HI" or "hindi")
3. **Verify both source and target languages** are in supported list

### Problem: Cannot Connect to Service

**Symptoms**: Connection refused or timeout errors

**Solutions**:
1. **Verify service is running**: `docker ps`
2. **Check port mapping**: `docker ps` should show port mappings
3. **Test from container**: 
   ```bash
   docker exec indictrans curl http://localhost:8000/v2/health/ready
   ```
4. **Check firewall**: `sudo ufw status`
5. **Verify port 8000 is not in use** by another service

### Problem: Container Keeps Restarting

**Symptoms**: Container status shows "Restarting"

**Solutions**:
1. **Check logs**: `docker logs indictrans`
2. **Verify GPU availability**: `nvidia-smi`
3. **Check system resources**: `free -h` and `df -h`
4. **Verify Docker runtime**: Ensure `runtime: nvidia` in docker-compose

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
- Batch processing statistics

### View Real-Time Logs

```bash
docker logs -f indictrans
```

Press `Ctrl+C` to stop viewing logs.

### Check Container Stats

```bash
docker stats indictrans
```

Shows CPU, memory, GPU, and network usage in real-time.

---

## 🔄 Common Operations

### Stop the Service

```bash
docker stop indictrans
```

### Start the Service

```bash
docker start indictrans
```

### Restart the Service

```bash
docker restart indictrans
```

### Remove the Service

```bash
docker stop indictrans
docker rm indictrans
```

### Update the Service

```bash
# Pull latest image
docker pull ai4bharat/triton-indictrans-v2:latest

# Stop and remove old container
docker stop indictrans
docker rm indictrans

# Start new container (use your preferred method)
docker-compose up -d indictrans
```

---

## 📚 Additional Resources

### Service Documentation
- **Docker Hub**: https://hub.docker.com/r/ai4bharat/triton-indictrans-v2
- **IndicTrans2 GitHub**: https://github.com/ai4bharat/IndicTrans2
- **AI4Bharat**: https://ai4bharat.iitm.ac.in/

### Technical References
- **Triton Server Docs**: https://docs.nvidia.com/deeplearning/triton-inference-server/
- **Docker Docs**: https://docs.docker.com/

### Getting Help

If you encounter issues:
1. Check the logs: `docker logs indictrans`
2. Review this guide's troubleshooting section
3. Check AI4Bharat GitHub repositories
4. Review IndicTrans2 documentation
5. Review Triton Server documentation

---

## 📝 Quick Reference

### Essential Commands

```bash
# Pull image
docker pull ai4bharat/triton-indictrans-v2:latest

# Run with docker-compose (recommended)
docker-compose up -d indictrans

# Or run directly
docker run -d --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  --shm-size=64mb --name indictrans \
  ai4bharat/triton-indictrans-v2:latest \
  tritonserver --model-repository=/models

# Check status
docker ps
curl http://localhost:8000/v2/health/ready

# Test translation
curl -X POST http://localhost:8000/v2/models/nmt/infer \
  -H "Content-Type: application/json" \
  -d '{"inputs":[{"name":"INPUT_TEXT","shape":[1],"datatype":"BYTES","data":["Hello"]},{"name":"INPUT_LANGUAGE_ID","shape":[1],"datatype":"BYTES","data":["en"]},{"name":"OUTPUT_LANGUAGE_ID","shape":[1],"datatype":"BYTES","data":["hi"]}]}'

# View logs
docker logs -f indictrans

# Stop
docker stop indictrans
```

### Port Information

- **HTTP API**: `http://localhost:8000`
- **gRPC API**: `localhost:8001`
- **Metrics**: `http://localhost:8002/metrics`

### Model Information

- **Model Name**: `nmt`
- **Type**: Python backend
- **Max Batch Size**: 512
- **GPU Required**: Yes (recommended)
- **Shared Memory**: 64MB minimum

### Language Codes Quick Reference

Common pairs:
- **English ↔ Hindi**: en ↔ hi
- **English ↔ Tamil**: en ↔ ta
- **English ↔ Telugu**: en ↔ te
- **English ↔ Bengali**: en ↔ bn
- **Hindi ↔ Tamil**: hi ↔ ta

---

## ✅ Summary

You've learned how to:
1. ✅ Pull the NMT Docker image from Docker Hub
2. ✅ Run the NMT service container
3. ✅ Verify it's working correctly
4. ✅ Test translation with text input
5. ✅ Use the API for translating between languages
6. ✅ Troubleshoot common issues

The NMT-triton service is now ready to translate text between 22 Indian languages and English! For production use, consider setting up monitoring, load balancing, and proper security measures.

