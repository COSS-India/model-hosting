# Indic-xlit-triton Service Guide

## 📖 What is Indic-xlit-triton?

**Indic-xlit** stands for **Indic Transliteration**. This service can convert text written in English (Roman script) into various Indian language scripts (like Devanagari, Tamil, Telugu, Bengali, etc.) and vice versa. It helps bridge the gap between Romanized text and native Indic scripts.

### Real-World Use Cases
- **Social Media**: Convert "namaste" to "नमस्ते" for Hindi speakers
- **Search Engines**: Enable users to search in their native script using Roman input
- **Messaging Apps**: Allow users to type in English and get text in their preferred script
- **Content Creation**: Convert English text to Indic scripts for localized content
- **Language Learning**: Help learners see how words are written in native scripts
- **Accessibility**: Enable typing in Roman script for users more comfortable with English keyboards

---

## 🎯 What You Need Before Starting

### For Everyone (Non-Technical)

Before you can use this service, you need:
1. **A computer with Linux** (Ubuntu recommended)
2. **An NVIDIA graphics card** (GPU) - This makes the service run much faster
3. **Internet connection** - To download the Docker image
4. **Docker installed** - Software to run the service

### For Technical Users

**System Requirements:**
- **OS**: Linux (Ubuntu 20.04+)
- **GPU**: NVIDIA GPU with CUDA support (recommended for performance)
- **Docker**: Version 20.10+
- **Docker Compose**: Version 1.29+ (optional, for easier management)
- **NVIDIA Container Toolkit**: For GPU access in Docker
- **Shared Memory**: At least 2GB (`shm_size`)
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

The Indic-xlit service uses a pre-built Docker image from AI4Bharat. Unlike other services that need to be built, this one is ready to use directly.

```
indic-xlit-server/
├── Docker Image: ai4bharat/triton-indic-xlit:latest (pre-built)
├── Model: AI4Bharat Indic-Xlit model
└── Ports: 8200 (HTTP), 8201 (gRPC), 8202 (Metrics)
```

**Key Features:**
- **Word-level transliteration**: Can transliterate individual words or full sentences
- **Multiple candidates**: Returns top-K transliteration options
- **Bidirectional**: Can transliterate from English to Indic scripts and vice versa
- **10 Indic languages**: Supports Hindi, Bengali, Gujarati, Punjabi, Odia, Marathi, Kannada, Telugu, Malayalam, and Tamil

---

## 🔨 Step 1: Pulling the Docker Image

### What is Pulling?

Pulling a Docker image downloads a pre-built container image from a registry (like Docker Hub). Since this service uses a pre-built image, you don't need to build it yourself - just download it!

### Step-by-Step Pull Instructions

#### Option A: Simple Pull (Recommended for Beginners)

1. **Open a terminal** (command line window)

2. **Pull the image:**
   ```bash
   docker pull ai4bharat/triton-indic-xlit:latest
   ```
   
   **What this does:**
   - `docker pull` = Download the image
   - `ai4bharat/triton-indic-xlit:latest` = The image name and version
   - Downloads the complete service with all models and dependencies

3. **Wait for it to complete** (this may take 5-15 minutes depending on internet speed)
   - Downloads the Triton server base image
   - Downloads the Indic-Xlit model
   - Sets everything up

**Expected Output:**
```
latest: Pulling from ai4bharat/triton-indic-xlit
...
Status: Downloaded newer image for ai4bharat/triton-indic-xlit:latest
```

**Troubleshooting Pull Issues:**
- **"Cannot connect to Docker daemon"**: Run `sudo systemctl start docker`
- **"Network timeout"**: Check internet connection, image is large (~1-2GB)
- **"No space left on device"**: Free up disk space (image needs several GB)

---

## 📥 How the Indic-Xlit Model Works

### Model Source

The Indic-xlit service uses the **Indic-Xlit** model from AI4Bharat:
- **Repository**: [https://github.com/AI4Bharat/Indic-Xlit](https://github.com/AI4Bharat/Indic-Xlit)
- **Model Type**: Neural Machine Translation for Transliteration
- **License**: MIT License
- **Documentation**: [https://indicnlp.ai4bharat.org/indic-xlit/](https://indicnlp.ai4bharat.org/indic-xlit/)

### About the Model

Indic-Xlit is a transliteration system specifically designed for Indian languages that:

- **Supports 10 Indic languages**:
  - Hindi (Devanagari script)
  - Bengali (Bengali script)
  - Gujarati (Gujarati script)
  - Punjabi (Gurmukhi script)
  - Odia (Odia script)
  - Marathi (Devanagari script)
  - Kannada (Kannada script)
  - Telugu (Telugu script)
  - Malayalam (Malayalam script)
  - Tamil (Tamil script)

- **Bidirectional transliteration**:
  - English (Roman) → Indic script
  - Indic script → English (Roman)

- **Features**:
  - Word-level transliteration
  - Sentence-level transliteration
  - Multiple candidate generation (top-K)
  - High accuracy (85%+ top-1, 95%+ top-5)

### How Transliteration Works

1. **Input Processing**: Takes text in source language/script
2. **Model Inference**: Uses neural network to predict target script
3. **Candidate Generation**: Generates multiple transliteration options
4. **Output**: Returns top-K transliteration candidates

### Performance Characteristics

- **Accuracy**: 85.2% top-1 accuracy, 95.5% top-5 accuracy (on standard datasets)
- **Speed**: Fast inference on GPU
- **Memory**: ~990 MB VRAM usage

---

## 🚀 Step 2: Running the Service

### What is Running?

Running the service means starting it up so it can accept requests. It's like opening a restaurant for business.

### Step-by-Step Run Instructions

#### Basic Run (For Testing)

```bash
docker run --gpus all \
  -p 8200:8000 \
  -p 8201:8001 \
  -p 8202:8002 \
  --shm-size=2gb \
  --name indic-xlit-server \
  ai4bharat/triton-indic-xlit:latest \
  tritonserver --model-repository=/models --log-verbose=1 --strict-readiness=false
```

**What each part means:**
- `docker run` = Start a container
- `--gpus all` = Use all available GPUs (makes it faster)
- `-p 8200:8000` = Map port 8200 on your computer to port 8000 in container (HTTP)
- `-p 8201:8001` = Map port 8201 to 8001 (gRPC)
- `-p 8202:8002` = Map port 8202 to 8002 (Metrics)
- `--shm-size=2gb` = Allocate 2GB shared memory (required for the model)
- `--name indic-xlit-server` = Name the container "indic-xlit-server"
- `ai4bharat/triton-indic-xlit:latest` = Use the image we pulled
- `tritonserver ...` = Start the Triton server with specified options

#### Run in Background (Recommended for Production)

```bash
docker run -d --gpus all \
  -p 8200:8000 \
  -p 8201:8001 \
  -p 8202:8002 \
  --shm-size=2gb \
  --name indic-xlit-server \
  ai4bharat/triton-indic-xlit:latest \
  tritonserver --model-repository=/models --log-verbose=1 --strict-readiness=false
```

The `-d` flag runs it in the background (detached mode).

#### Using Docker Compose (Recommended for Production)

Create or use `docker-compose.yml`:

```yaml
version: '3.8'

services:
  indic-xlit-server:
    image: ai4bharat/triton-indic-xlit:latest
    container_name: indic-xlit-server
    ports:
      - "8200:8000"  # HTTP API
      - "8201:8001"  # GRPC API
      - "8202:8002"  # Metrics
    command: tritonserver --model-repository=/models --log-verbose=1 --strict-readiness=false
    shm_size: 2gb
    runtime: nvidia
    restart: always
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
```

Then run:
```bash
docker-compose up -d indic-xlit-server
```

### Understanding Ports

- **Port 8200 (HTTP)**: Main entrance for web requests
- **Port 8201 (gRPC)**: Fast lane for program-to-program communication
- **Port 8202 (Metrics)**: Monitoring room for checking service health

---

## ✅ Step 3: Verifying the Service is Running

### Check if Container is Running

```bash
docker ps
```

You should see `indic-xlit-server` in the list with status "Up".

### Check Service Health

```bash
curl http://localhost:8200/v2/health/ready
```

**Expected Response:** 
- HTTP Status: `200` 
- Body: Empty (this is correct - a 200 status code means the service is ready)

Alternatively, check the status code only:
```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:8200/v2/health/ready
# Should output: 200
```

If you see HTTP 200, the service is ready to accept requests!

### List Available Models

**Note**: The `/v2/models` endpoint may return a 400 Bad Request in some Triton versions. Instead, check the specific model:

```bash
curl http://localhost:8200/v2/models/transliteration
```

**Expected Response:**
```json
{
  "name": "transliteration",
  "platform": "python",
  "versions": ["1"],
  "inputs": [...],
  "outputs": [...]
}
```

Or check server information:
```bash
curl http://localhost:8200/v2 | python3 -m json.tool
```

### View Logs

```bash
docker logs indic-xlit-server
```

Look for:
- `"Server is ready to receive inference requests"` = Success!
- `"successfully loaded 'transliteration' version 1"` = Model loaded successfully
- `"Initializing transliteration model"` = Model initializing
- Any error messages = Something went wrong

---

## 🧪 Step 4: Testing the Service

> **Note**: If you're accessing the service from a remote machine, replace `localhost` with your server's IP address. For example, if your server IP is `192.168.1.100`, use `http://192.168.1.100:8200` instead of `http://localhost:8200`.

### Method 1: Manual Testing with curl

#### Example 1: English to Hindi (Simple)

```bash
curl -X POST http://localhost:8200/v2/models/transliteration/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "INPUT_TEXT",
        "shape": [1],
        "datatype": "BYTES",
        "data": ["namaste"]
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
      },
      {
        "name": "IS_WORD_LEVEL",
        "shape": [1],
        "datatype": "BOOL",
        "data": [true]
      },
      {
        "name": "TOP_K",
        "shape": [1],
        "datatype": "UINT8",
        "data": [5]
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
  "model_name": "transliteration",
  "model_version": "1",
  "outputs": [
    {
      "name": "OUTPUT_TEXT",
      "datatype": "BYTES",
      "shape": [5],
      "data": ["नमस्ते", "नमसते", "नमस्त", "नमस्ते", "नमस्त"]
    }
  ]
}
```

#### Example 2: English to Bengali

```bash
curl -X POST http://localhost:8200/v2/models/transliteration/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "INPUT_TEXT",
        "shape": [1],
        "datatype": "BYTES",
        "data": ["bhalo"]
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
        "data": ["bn"]
      },
      {
        "name": "IS_WORD_LEVEL",
        "shape": [1],
        "datatype": "BOOL",
        "data": [true]
      },
      {
        "name": "TOP_K",
        "shape": [1],
        "datatype": "UINT8",
        "data": [3]
      }
    ],
    "outputs": [
      {
        "name": "OUTPUT_TEXT"
      }
    ]
  }'
```

#### Example 3: Sentence-level Transliteration (Hindi)

```bash
curl -X POST http://localhost:8200/v2/models/transliteration/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "INPUT_TEXT",
        "shape": [1],
        "datatype": "BYTES",
        "data": ["main bharat se hoon"]
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
      },
      {
        "name": "IS_WORD_LEVEL",
        "shape": [1],
        "datatype": "BOOL",
        "data": [false]
      },
      {
        "name": "TOP_K",
        "shape": [1],
        "datatype": "UINT8",
        "data": [1]
      }
    ],
    "outputs": [
      {
        "name": "OUTPUT_TEXT"
      }
    ]
  }'
```

### Method 2: Batch Transliteration Example

You can transliterate multiple words by sending multiple requests:

```python
import requests

def transliterate_batch(words, input_lang, output_lang, word_level=True, top_k=5):
    """Transliterate multiple words"""
    results = []
    for word in words:
        payload = {
            "inputs": [
                {"name": "INPUT_TEXT", "shape": [1], "datatype": "BYTES", "data": [word]},
                {"name": "INPUT_LANGUAGE_ID", "shape": [1], "datatype": "BYTES", "data": [input_lang]},
                {"name": "OUTPUT_LANGUAGE_ID", "shape": [1], "datatype": "BYTES", "data": [output_lang]},
                {"name": "IS_WORD_LEVEL", "shape": [1], "datatype": "BOOL", "data": [word_level]},
                {"name": "TOP_K", "shape": [1], "datatype": "UINT8", "data": [top_k]}
            ],
            "outputs": [{"name": "OUTPUT_TEXT"}]
        }
        response = requests.post("http://localhost:8200/v2/models/transliteration/infer", json=payload)
        result = response.json()
        transliterated = result["outputs"][0]["data"][0]
        if isinstance(transliterated, bytes):
            transliterated = transliterated.decode('utf-8')
        results.append(transliterated)
    return results

# Example usage
words = ["namaste", "dhanyawad", "kripya"]
transliterations = transliterate_batch(words, "en", "hi")
for original, transliterated in zip(words, transliterations):
    print(f"{original} -> {transliterated}")
```

### Method 3: Python Test Script

Create a file `test_xlit.py`:

```python
import requests
import json

def test_transliteration(text, input_lang, output_lang, word_level=True, top_k=5):
    """
    Test transliteration service.
    
    Args:
        text: Input text to transliterate
        input_lang: Source language code (e.g., "en", "hi")
        output_lang: Target language code (e.g., "hi", "bn", "ta")
        word_level: Whether to perform word-level transliteration
        top_k: Number of candidates to return
    """
    url = "http://localhost:8200/v2/models/transliteration/infer"
    
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
                "data": [input_lang]
            },
            {
                "name": "OUTPUT_LANGUAGE_ID",
                "shape": [1],
                "datatype": "BYTES",
                "data": [output_lang]
            },
            {
                "name": "IS_WORD_LEVEL",
                "shape": [1],
                "datatype": "BOOL",
                "data": [word_level]
            },
            {
                "name": "TOP_K",
                "shape": [1],
                "datatype": "UINT8",
                "data": [top_k]
            }
        ],
        "outputs": [
            {
                "name": "OUTPUT_TEXT"
            }
        ]
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        output_data = result["outputs"][0]["data"]
        
        return output_data
    
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Error parsing response: {e}")
        print(f"Response: {response.text if 'response' in locals() else 'N/A'}")
        return None


# Test cases
if __name__ == "__main__":
    print("=" * 80)
    print("Indic-Xlit Transliteration Test")
    print("=" * 80)
    print()
    
    # Test 1: English to Hindi
    print("Test 1: English to Hindi")
    print("Input: 'namaste'")
    result = test_transliteration("namaste", "en", "hi", word_level=True, top_k=5)
    if result:
        print(f"Output: {result}")
        print(f"Top candidate: {result[0]}")
    print()
    
    # Test 2: English to Tamil
    print("Test 2: English to Tamil")
    print("Input: 'vanakkam'")
    result = test_transliteration("vanakkam", "en", "ta", word_level=True, top_k=3)
    if result:
        print(f"Output: {result}")
        print(f"Top candidate: {result[0]}")
    print()
    
    # Test 3: Sentence-level
    print("Test 3: Sentence-level (English to Hindi)")
    print("Input: 'main bharat se hoon'")
    result = test_transliteration("main bharat se hoon", "en", "hi", word_level=False, top_k=1)
    if result:
        print(f"Output: {result[0]}")
    print()
```

Run it:
```bash
python3 test_xlit.py
```

---

## 📝 Text Format Requirements

### Input Text Guidelines

- **Encoding**: UTF-8
- **Format**: Plain text (no special formatting required)
- **Scripts**: Supports English (Roman script) and all Indic scripts (Devanagari, Tamil, Telugu, etc.)
- **Length**: No strict limit, but word-level transliteration works best with single words or short phrases

### Tips for Best Results

1. **Word-level transliteration** works best for single words
2. **Sentence-level transliteration** is better for phrases and sentences
3. **Use appropriate language codes** - check the supported languages list
4. **Try multiple TOP_K values** to see different transliteration candidates
5. **For reverse transliteration** (Indic to English), use the Indic language code as INPUT_LANGUAGE_ID and "en" as OUTPUT_LANGUAGE_ID

---

## 📊 Understanding the API

### Input Format

The API requires 5 inputs:

1. **INPUT_TEXT** (Required)
   - **Type**: String (BYTES)
   - **What to send**: The text you want to transliterate
   - **Example**: `"namaste"` or `"main bharat se hoon"`

2. **INPUT_LANGUAGE_ID** (Required)
   - **Type**: String (BYTES)
   - **What to send**: Source language code
   - **Supported values**: `"en"` (English) or language codes like `"hi"`, `"bn"`, etc.

3. **OUTPUT_LANGUAGE_ID** (Required)
   - **Type**: String (BYTES)
   - **What to send**: Target language code
   - **Supported values**: 
     - `"hi"` - Hindi
     - `"bn"` - Bengali
     - `"gu"` - Gujarati
     - `"pa"` - Punjabi
     - `"or"` - Odia
     - `"mr"` - Marathi
     - `"kn"` - Kannada
     - `"te"` - Telugu
     - `"ml"` - Malayalam
     - `"ta"` - Tamil
     - `"en"` - English (for reverse transliteration)

4. **IS_WORD_LEVEL** (Required)
   - **Type**: Boolean (BOOL)
   - **What to send**: `true` for word-level, `false` for sentence-level
   - **Word-level**: Transliterates each word separately
   - **Sentence-level**: Transliterates the entire sentence as a unit

5. **TOP_K** (Required)
   - **Type**: Integer (UINT8)
   - **What to send**: Number of transliteration candidates to return (1-10 recommended)
   - **Example**: `5` returns top 5 transliteration options

### Output Format

The service returns a JSON response with:

```json
{
  "model_name": "transliteration",
  "model_version": "1",
  "outputs": [
    {
      "name": "OUTPUT_TEXT",
      "datatype": "BYTES",
      "shape": [K],
      "data": ["candidate1", "candidate2", "candidate3", ...]
    }
  ]
}
```

**Fields:**
- **data**: Array of transliteration candidates, ordered by confidence (best first)
- **shape**: Number of candidates returned (equals TOP_K)

### Supported Languages

| Language | Code | Script |
|----------|------|--------|
| Hindi | hi | Devanagari |
| Bengali | bn | Bengali |
| Gujarati | gu | Gujarati |
| Punjabi | pa | Gurmukhi |
| Odia | or | Odia |
| Marathi | mr | Devanagari |
| Kannada | kn | Kannada |
| Telugu | te | Telugu |
| Malayalam | ml | Malayalam |
| Tamil | ta | Tamil |
| English | en | Latin/Roman |

### Language Code Reference

- **Input language**: Use `"en"` for English/Roman script, or the target language code for reverse transliteration
- **Output language**: Use the 2-letter ISO code for the target Indic language

---

## 🧠 How It Works (Technical Details)

### Transliteration Process

1. **Input Processing**: 
   - Text is tokenized (split into words if word-level)
   - Characters are encoded for the model

2. **Model Inference**: 
   - Neural network processes the input
   - Generates transliteration candidates
   - Scores each candidate

3. **Candidate Ranking**: 
   - Candidates are ranked by confidence/score
   - Top-K candidates are selected

4. **Output Formatting**: 
   - Results are formatted and returned
   - Best candidate is first in the array

### Word-level vs Sentence-level

- **Word-level** (`IS_WORD_LEVEL=true`):
  - Each word is transliterated independently
  - Better for single words or when you want multiple options per word
  - Example: "namaste" → ["नमस्ते", "नमसते", ...]

- **Sentence-level** (`IS_WORD_LEVEL=false`):
  - Entire sentence is transliterated as a unit
  - Better for phrases and sentences
  - Considers context between words
  - Example: "main bharat se hoon" → "मैं भारत से हूं"

### Performance Characteristics

- **Accuracy**: 85.2% top-1 accuracy, 95.5% top-5 accuracy
- **Speed**: Fast inference on GPU
- **Memory**: ~990 MB VRAM usage
- **Throughput**: Can handle multiple requests efficiently

---

## ⚙️ Configuration Options

### Docker Run Options

You can customize the container with additional options:

```bash
docker run -d --gpus all \
  -p 8200:8000 \
  -p 8201:8001 \
  -p 8202:8002 \
  --shm-size=2gb \
  --name indic-xlit-server \
  --restart=always \
  -e NVIDIA_VISIBLE_DEVICES=all \
  ai4bharat/triton-indic-xlit:latest \
  tritonserver --model-repository=/models --log-verbose=1 --strict-readiness=false
```

You can modify the docker run command to adjust:

1. **Shared Memory Size**: 
   - Current: `--shm-size=2gb`
   - Increase if you encounter memory issues
   - Decrease if you have limited system memory

2. **Port Mapping**: 
   - Change `-p 8200:8000` to use different host ports
   - Example: `-p 9200:8000` to use port 9200 instead

3. **GPU Selection**: 
   - Use `--gpus '"device=0"'` to use specific GPU
   - Use `--gpus all` to use all available GPUs

**Options explained:**
- `--restart=always` = Automatically restart container if it crashes
- `-e NVIDIA_VISIBLE_DEVICES=all` = Use all GPUs (can specify specific GPU like "0" or "0,1")
- `--shm-size=2gb` = Shared memory size (2GB is recommended for this model)

### Resource Allocation

- **GPU Memory**: Model requires ~1GB of GPU memory
- **System Memory**: At least 4GB RAM recommended
- **Shared Memory**: 2GB minimum (required for the model)
- **Batch Processing**: Supports multiple requests efficiently

### Performance Optimization

- **Word-level transliteration** is faster than sentence-level
- **Lower TOP_K values** process faster
- **GPU acceleration** significantly improves speed
- The model uses dynamic batching for better throughput

### Environment Variables

The service doesn't require environment variables, but you can set:

- `CUDA_VISIBLE_DEVICES`: Control which GPUs are visible
- `TORCH_DEVICE`: Force CPU or CUDA (usually auto-detected)

---

## 🛠️ Troubleshooting

### Problem: Service Won't Start

**Symptoms**: Container exits immediately or shows errors

**Solutions**:
1. **Check logs**: `docker logs indic-xlit-server`
2. **Verify GPU**: `nvidia-smi` should show your GPU
3. **Check port conflicts**: `sudo lsof -i :8200` (should be empty)
4. **Verify Docker GPU access**: 
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```
5. **Check shared memory**: Ensure `--shm-size=2gb` is set

### Problem: "Out of Memory" Errors

**Symptoms**: CUDA out of memory errors in logs

**Solutions**:
1. **Increase shared memory**: `--shm-size=4gb`
2. **Check GPU memory**: `nvidia-smi`
3. **Close other GPU applications**
4. **Use CPU mode** (if GPU memory is insufficient):
   ```bash
   docker run -p 8200:8000 -p 8201:8001 -p 8202:8002 \
     --shm-size=2gb --name indic-xlit-server \
     -e TORCH_DEVICE=cpu \
     ai4bharat/triton-indic-xlit:latest \
     tritonserver --model-repository=/models --log-verbose=1 --strict-readiness=false
   ```

### Problem: Wrong Transliteration

**Symptoms**: Service returns incorrect transliteration

**Solutions**:
1. **Check language codes**: Ensure correct ISO codes are used
2. **Try word-level**: Use `IS_WORD_LEVEL=true` for single words
3. **Check top-K**: Increase TOP_K to see more candidates
4. **Verify input**: Ensure input text is properly formatted
5. **Try different candidates**: The first result may not always be best

### Problem: Slow Inference

**Symptoms**: Requests take a long time to process

**Solutions**:
1. **Verify GPU is being used**: Check logs for GPU usage
2. **Check GPU utilization**: `nvidia-smi -l 1` (should show activity)
3. **Reduce TOP_K**: Lower values process faster
4. **Use word-level**: Sentence-level can be slower

### Problem: Cannot Connect to Service

**Symptoms**: Connection refused or timeout errors

**Solutions**:
1. **Verify service is running**: `docker ps`
2. **Check port mapping**: `docker ps` should show port mappings
3. **Test from container**: 
   ```bash
   docker exec indic-xlit-server curl http://localhost:8000/v2/health/ready
   ```
4. **Check firewall**: `sudo ufw status`
5. **Verify port number**: Use 8200, not 8000 (host port)

### Problem: "Model Not Found" Errors

**Symptoms**: Errors about missing model files

**Solutions**:
1. **Verify image was pulled correctly**: `docker images | grep indic-xlit`
2. **Re-pull the image**: `docker pull ai4bharat/triton-indic-xlit:latest`
3. **Check container logs**: `docker logs indic-xlit-server`
4. **Restart the container**: `docker restart indic-xlit-server`

---

## 📈 Monitoring and Metrics

### Health Check

```bash
curl -w "\nHTTP Status: %{http_code}\n" http://localhost:8200/v2/health/ready
```

**Note**: The endpoint returns HTTP 200 with an empty body when ready. Check the status code (should be 200).

### Metrics Endpoint

```bash
curl http://localhost:8202/metrics
```

This provides Prometheus-compatible metrics including:
- Request count
- Inference latency
- GPU utilization
- Error rates
- Batch processing statistics

### View Real-Time Logs

```bash
docker logs -f indic-xlit-server
```

Press `Ctrl+C` to stop viewing logs.

---

## 🔄 Common Operations

### Stop the Service

```bash
docker stop indic-xlit-server
```

### Start the Service

```bash
docker start indic-xlit-server
```

### Restart the Service

```bash
docker restart indic-xlit-server
```

### Remove the Service

```bash
docker stop indic-xlit-server
docker rm indic-xlit-server
```

### Update the Service

```bash
# Pull latest image
docker pull ai4bharat/triton-indic-xlit:latest

# Stop and remove old container
docker stop indic-xlit-server
docker rm indic-xlit-server

# Start new container (use your preferred method)
docker-compose up -d indic-xlit-server
```

Or manually:
```bash
docker run -d --gpus all -p 8200:8000 -p 8201:8001 -p 8202:8002 \
  --shm-size=2gb --name indic-xlit-server \
  ai4bharat/triton-indic-xlit:latest \
  tritonserver --model-repository=/models --log-verbose=1 --strict-readiness=false
```

---

## 📚 Additional Resources

### Service Documentation
- **Model Source (GitHub)**: [https://github.com/AI4Bharat/Indic-Xlit](https://github.com/AI4Bharat/Indic-Xlit)
- **Indic-Xlit Page**: [https://indicnlp.ai4bharat.org/indic-xlit/](https://indicnlp.ai4bharat.org/indic-xlit/)
- **AI4Bharat Transliteration**: [https://ai4bharat.iitm.ac.in/areas/xlit](https://ai4bharat.iitm.ac.in/areas/xlit)

### Technical References
- **Triton Server Docs**: https://docs.nvidia.com/deeplearning/triton-inference-server/
- **Docker Docs**: https://docs.docker.com/

### Getting Help

If you encounter issues:
1. Check the logs: `docker logs indic-xlit-server`
2. Review this guide's troubleshooting section
3. Check the AI4Bharat Indic-Xlit GitHub repository
4. Review Triton Server documentation

---

## 📝 Quick Reference

### Essential Commands

```bash
# Pull image
docker pull ai4bharat/triton-indic-xlit:latest

# Run
docker run -d --gpus all -p 8200:8000 -p 8201:8001 -p 8202:8002 \
  --shm-size=2gb --name indic-xlit-server \
  ai4bharat/triton-indic-xlit:latest \
  tritonserver --model-repository=/models --log-verbose=1 --strict-readiness=false

# Check status
docker ps
curl -w "\nHTTP Status: %{http_code}\n" http://localhost:8200/v2/health/ready
# Note: Returns HTTP 200 with empty body when ready

# Test (English to Hindi)
curl -X POST http://localhost:8200/v2/models/transliteration/infer \
  -H "Content-Type: application/json" \
  -d '{"inputs":[{"name":"INPUT_TEXT","shape":[1],"datatype":"BYTES","data":["namaste"]},{"name":"INPUT_LANGUAGE_ID","shape":[1],"datatype":"BYTES","data":["en"]},{"name":"OUTPUT_LANGUAGE_ID","shape":[1],"datatype":"BYTES","data":["hi"]},{"name":"IS_WORD_LEVEL","shape":[1],"datatype":"BOOL","data":[true]},{"name":"TOP_K","shape":[1],"datatype":"UINT8","data":[5]}]}'

# View logs
docker logs -f indic-xlit-server

# Stop
docker stop indic-xlit-server
```

### Port Information

- **HTTP API**: `http://localhost:8200`
- **gRPC API**: `localhost:8201`
- **Metrics**: `http://localhost:8202/metrics`

### Model Information

- **Model Name**: `transliteration`
- **Type**: Python backend
- **GPU Required**: Recommended (can run on CPU)
- **Supported Languages**: 10 Indic languages + English
- **Shared Memory**: 2GB minimum
- **Max Batch Size**: Supports efficient batch processing

### Language Codes Quick Reference

Common pairs:
- **English ↔ Hindi**: en ↔ hi
- **English ↔ Tamil**: en ↔ ta
- **English ↔ Telugu**: en ↔ te
- **English ↔ Bengali**: en ↔ bn
- **English ↔ Gujarati**: en ↔ gu

Full list:
- **hi** - Hindi
- **bn** - Bengali
- **gu** - Gujarati
- **pa** - Punjabi
- **or** - Odia
- **mr** - Marathi
- **kn** - Kannada
- **te** - Telugu
- **ml** - Malayalam
- **ta** - Tamil
- **en** - English

---

## ✅ Summary

You've learned how to:
1. ✅ Pull the Indic-xlit Docker image from Docker Hub
2. ✅ Run the Indic-xlit service container
3. ✅ Verify it's working correctly
4. ✅ Test transliteration with text input
5. ✅ Use the API for transliterating between scripts
6. ✅ Understand word-level vs sentence-level transliteration
7. ✅ Troubleshoot common issues

The Indic-xlit-triton service is now ready to transliterate text between English and Indic scripts! For production use, consider setting up monitoring, load balancing, and proper security measures.

---

