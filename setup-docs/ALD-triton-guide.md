# ALD-triton Service Guide

## 📖 What is ALD-triton?

**ALD** stands for **Audio Language Detection**. This service can listen to audio recordings and automatically identify which language is being spoken. It supports **107 different languages** including English, Hindi, Tamil, Spanish, French, and many more.

### Real-World Use Cases
- **Call Centers**: Automatically route calls based on the language spoken
- **Content Moderation**: Identify the language of user-uploaded audio content
- **Media Processing**: Automatically tag audio files with their language
- **Accessibility**: Provide language-specific services based on detected language

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
- **Hardware Specifications**: May vary depending on the scale of your application
- **Tested Machine**: g4dn.2xlarge (For detailed specifications and pricing, check [AWS EC2 g4dn.2xlarge](https://instances.vantage.sh/aws/ec2/g4dn.2xlarge?currency=USD))

> **Note**: The model used in this service is provided as a reference implementation. You can replace it with your own trained model or use different model variants based on your specific requirements, performance needs, and use case.

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

Think of this service like a restaurant:
- **Dockerfile** = The recipe for building the service
- **model_repository** = The kitchen where the AI model lives
- **config.pbtxt** = The menu (what inputs/outputs are available)
- **model.py** = The chef (the code that processes requests)

```
ALD-triton/
├── Dockerfile              # Recipe to build the container
├── README.md               # Documentation
├── test_client.py          # Test script
└── model_repository/       # Model storage
    └── ald/
        ├── config.pbtxt    # Service configuration
        └── 1/
            └── model.py    # Processing logic
```

---

## 🔨 Step 1: Building the Docker Image

### What is Building?

Building a Docker image is like creating a package that contains everything needed to run the service: the code, the AI model, and all dependencies. Once built, you can run it anywhere.

### Step-by-Step Build Instructions

#### Option A: Simple Build (Recommended for Beginners)

1. **Open a terminal** (command line window)

2. **Navigate to the ALD-triton folder:**
   ```bash
   cd ALD-triton
   ```

3. **Build the image:**
   ```bash
   docker build -t ald-triton:latest .
   ```
   
   **What this does:**
   - `docker build` = Start building
   - `-t ald-triton:latest` = Name the image "ald-triton" with tag "latest"
   - `.` = Use the current directory (where Dockerfile is located)

4. **Wait for it to complete** (this may take 5-15 minutes the first time)
   - It will download the base Triton server
   - Install Python packages
   - Download the language detection model
   - Set everything up

#### Option B: Understanding What Happens During Build

The Dockerfile does these steps:

1. **Starts with Triton Server base image** (`FROM nvcr.io/nvidia/tritonserver:24.01-py3`)
   - Uses NVIDIA's pre-configured Triton Inference Server
   - Includes Python 3 and CUDA support
   - Version 24.01 provides stable PyTorch support

2. **Installs system dependencies** (`RUN apt-get install`)
   - `libsndfile1` - Library for reading/writing audio files (WAV, FLAC, etc.)
   - Required for audio processing in Python

3. **Installs Python packages** (`RUN pip install`)
   - `torch==2.1.0` - PyTorch deep learning framework
   - `torchaudio==2.1.0` - Audio processing for PyTorch
   - `speechbrain==1.0.3` - SpeechBrain library (contains the language detection model)
   - `huggingface_hub<0.20.0` - For downloading models from HuggingFace (compatible version)
   - `protobuf==3.20.3` - Protocol buffers for Triton communication
   - `requests` - HTTP library for API calls
   - `soundfile` - Python wrapper for libsndfile

4. **Copies model repository** (`COPY model_repository /models`)
   - Copies your custom model code (`model.py`) and configuration (`config.pbtxt`)
   - This is where the ALD processing logic lives

5. **Pre-downloads the VoxLingua107 model** (`RUN python3 -c ...`)
   - Downloads the language detection model from HuggingFace
   - Model: `speechbrain/lang-id-voxlingua107-ecapa`
   - Source: [https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)
   - This is a pre-trained model that can detect 107 languages
   - Uses ECAPA-TDNN architecture for spoken language identification
   - Pre-downloading speeds up first request (model is cached in the image)
   - If download fails during build, model will download on first run

6. **Exposes ports** (`EXPOSE 8000 8001 8002`)
   - Port 8000: HTTP API
   - Port 8001: gRPC API
   - Port 8002: Metrics endpoint

7. **Sets startup command** (`CMD ["tritonserver", ...]`)
   - Starts Triton server when container runs
   - Points to `/models` as the model repository
   - Enables verbose logging

**Expected Output:**
```
Step 1/7 : FROM nvcr.io/nvidia/tritonserver:24.01-py3
Step 2/7 : WORKDIR /workspace
Step 3/7 : RUN apt-get update && apt-get install -y libsndfile1
Step 4/7 : RUN pip install --no-cache-dir torch==2.1.0 torchaudio==2.1.0 ...
Step 5/7 : COPY model_repository /models
Step 6/7 : RUN python3 -c "from speechbrain.inference.classifiers..."
Downloading VoxLingua107 ECAPA-TDNN model...
Model downloaded successfully
Step 7/7 : CMD ["tritonserver", "--model-repository=/models", "--log-verbose=1"]
Successfully built abc123def456
Successfully tagged ald-triton:latest
```

**Troubleshooting Build Issues:**
- **"Cannot connect to Docker daemon"**: Run `sudo systemctl start docker`
- **"No space left on device"**: Free up disk space
- **"Network timeout"**: Check internet connection, the build downloads large files

---

## 📥 How the ALD Model Was Obtained from HuggingFace

### Model Source

The ALD service uses the **VoxLingua107 ECAPA-TDNN** model from HuggingFace:
- **Model Repository**: [speechbrain/lang-id-voxlingua107-ecapa](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)
- **Model Type**: Spoken Language Identification
- **Architecture**: ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation)
- **License**: Apache 2.0

### About the Model

The VoxLingua107 model is a state-of-the-art spoken language identification system that:

- **Supports 107 languages** including:
  - English, Hindi, Tamil, Telugu, Spanish, French, German, Chinese, Japanese, and 99 more
  - See the [full list on HuggingFace](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)
  
- **Trained on VoxLingua107 dataset**:
  - 6,628 hours of training data
  - Automatically extracted from YouTube videos
  - Validated development set with 1,609 segments from 33 languages
  
- **Performance**:
  - Error rate: 6.7% on the VoxLingua107 development dataset
  - Processes audio at 16kHz (single channel)
  - Automatically normalizes audio (resampling + mono channel conversion)

### How the Model is Downloaded

During the Docker build process, the model is automatically downloaded from HuggingFace using SpeechBrain:

```python
from speechbrain.inference.classifiers import EncoderClassifier

# This downloads the model from HuggingFace
model = EncoderClassifier.from_hparams(
    source='speechbrain/lang-id-voxlingua107-ecapa',
    savedir='tmp_ald_model'
)
```

**What happens:**
1. SpeechBrain connects to HuggingFace Hub
2. Downloads the model files (weights, configuration, etc.)
3. Caches them in the Docker image
4. Model is ready to use when the container starts

### Model Files Structure

The downloaded model includes:
- **Model weights** - Pre-trained neural network parameters
- **Configuration files** - Hyperparameters and model settings
- **Label encoder** - Maps language codes to language names
- **Preprocessing pipeline** - Audio normalization settings

### Using the Model Directly (Optional)

You can also use the model directly in Python without Docker:

```python
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier

# Load the model from HuggingFace
language_id = EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-voxlingua107-ecapa",
    savedir="tmp"
)

# Load and classify audio
signal = language_id.load_audio("your_audio.wav")
prediction = language_id.classify_batch(signal)

# Get results
language_code = prediction[3][0]  # e.g., 'en', 'hi', 'ta'
confidence = prediction[1].exp().item()  # Confidence score
```

### Model Limitations

As noted in the [HuggingFace model card](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa), the model has some limitations:

- **Smaller languages**: Accuracy may be limited for less common languages
- **Gender bias**: Works better on male speech (YouTube data has more male speakers)
- **Accents**: May not work well on speech with foreign accents
- **Special cases**: May struggle with children's speech or speech disorders

### References

- **HuggingFace Model**: [https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)
- **SpeechBrain Paper**: [arXiv:2106.04624](https://arxiv.org/abs/2106.04624)
- **VoxLingua107 Dataset**: [IEEE SLT Workshop 2021](https://arxiv.org/abs/2106.04624)

---

## 🚀 Step 2: Running the Service

### What is Running?

Running the service means starting it up so it can accept requests. It's like opening a restaurant for business - the kitchen is ready, and customers can now place orders.

### Step-by-Step Run Instructions

#### Basic Run (For Testing)

```bash
docker run --gpus all \
  -p 8100:8000 \
  -p 8101:8001 \
  -p 8102:8002 \
  --name ald-server \
  ald-triton:latest
```

**What each part means:**
- `docker run` = Start a container
- `--gpus all` = Use all available GPUs (makes it faster)
- `-p 8100:8000` = Map port 8100 on your computer to port 8000 in container
- `-p 8101:8001` = Map gRPC port
- `-p 8102:8002` = Map metrics port
- `--name ald-server` = Name the container "ald-server"
- `ald-triton:latest` = Use the image we built

#### Run in Background (Recommended for Production)

```bash
docker run -d --gpus all \
  -p 8100:8000 \
  -p 8101:8001 \
  -p 8102:8002 \
  --name ald-server \
  ald-triton:latest
```

The `-d` flag runs it in the background (detached mode), so you can use your terminal for other things.

#### Using Docker Compose (Easiest Method)

If you're using the main docker-compose.yml file:

```bash
docker-compose up -d ald-server
```

This automatically handles all the configuration.

### Understanding Ports

Think of ports like different doors to the same building:
- **Port 8100 (HTTP)**: Main entrance for web requests
- **Port 8101 (gRPC)**: Fast lane for program-to-program communication
- **Port 8102 (Metrics)**: Monitoring room for checking service health

---

## ✅ Step 3: Verifying the Service is Running

### Check if Container is Running

```bash
docker ps
```

You should see `ald-server` in the list with status "Up".

### Check Service Health

```bash
curl http://localhost:8100/v2/health/ready
```

**Expected Response:** `{"status":"ready"}`

If you see this, the service is ready to accept requests!

### List Available Models

```bash
curl http://localhost:8100/v2/models
```

**Expected Response:**
```json
{
  "models": [
    {
      "name": "ald",
      "platform": "python",
      "versions": ["1"]
    }
  ]
}
```

### View Logs

```bash
docker logs ald-server
```

This shows what the service is doing. Look for:
- `"Server is ready to receive inference requests"` = Success!
- Any error messages = Something went wrong

---

## 🧪 Step 4: Testing the Service

> **Note**: If you're accessing the service from a remote machine, replace `localhost` with your server's IP address. For example, if your server IP is `192.168.1.100`, use `http://192.168.1.100:8100` instead of `http://localhost:8100`.

### Method 1: Manual Testing with curl

#### Step 1: Prepare Your Audio File

Convert your audio to base64:
```bash
AUDIO_B64=$(base64 -w 0 your_audio.wav)
```

#### Step 2: Send Request

```bash
curl -X POST http://localhost:8100/v2/models/ald/infer \
  -H "Content-Type: application/json" \
  -d "{
    \"inputs\": [
      {
        \"name\": \"AUDIO_DATA\",
        \"shape\": [1, 1],
        \"datatype\": \"BYTES\",
        \"data\": [[\"$AUDIO_B64\"]]
      }
    ],
    \"outputs\": [
      {\"name\": \"LANGUAGE_CODE\"},
      {\"name\": \"CONFIDENCE\"},
      {\"name\": \"ALL_SCORES\"}
    ]
  }"
```

**Expected Response:**
```json
{
  "outputs": [
    {
      "name": "LANGUAGE_CODE",
      "datatype": "BYTES",
      "shape": [1, 1],
      "data": ["en"]
    },
    {
      "name": "CONFIDENCE",
      "datatype": "FP32",
      "shape": [1, 1],
      "data": [0.9850]
    },
    {
      "name": "ALL_SCORES",
      "datatype": "BYTES",
      "shape": [1, 1],
      "data": ["{\"predicted_language\": \"en\", \"confidence\": 0.9850, ...}"]
    }
  ]
}
```

### Method 2: Python Test Script

Create a file `test_my_audio.py`:

```python
import requests
import json
import base64

# Read your audio file
with open("your_audio.wav", "rb") as f:
    audio_bytes = f.read()
    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

# Prepare request
url = "http://localhost:8100/v2/models/ald/infer"
payload = {
    "inputs": [
        {
            "name": "AUDIO_DATA",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [[audio_b64]]
        }
    ],
    "outputs": [
        {"name": "LANGUAGE_CODE"},
        {"name": "CONFIDENCE"},
        {"name": "ALL_SCORES"}
    ]
}

# Send request
response = requests.post(url, json=payload)
result = response.json()

# Display results
language = result["outputs"][0]["data"][0]
confidence = result["outputs"][1]["data"][0]
all_scores = json.loads(result["outputs"][2]["data"][0])

print(f"Detected Language: {language}")
print(f"Confidence: {confidence:.4f}")
print(f"Details: {all_scores}")
```

Run it:
```bash
python3 test_my_audio.py
```

**Expected Output:**
```
Detected Language: en
Confidence: 0.9850
Details: {'predicted_language': 'en', 'confidence': 0.9850, 'all_scores': {...}}
```

---

## 📊 Understanding the API

### Input Format

**AUDIO_DATA** (Required)
- **Type**: Base64-encoded string
- **Format**: WAV, MP3, FLAC, or other audio formats
- **What to send**: Your audio file converted to base64

### Output Format

The service returns three things:

1. **LANGUAGE_CODE**: The detected language (e.g., "en", "hi", "ta")
2. **CONFIDENCE**: How sure the model is (0.0 to 1.0, where 1.0 = 100% sure)
3. **ALL_SCORES**: Detailed breakdown with top predictions

### Supported Languages

The service supports 107 languages including:
- **English** (en)
- **Hindi** (hi)
- **Tamil** (ta)
- **Spanish** (es)
- **French** (fr)
- **And 102 more languages!**

See the full list in `ALD-triton/README.md`.

---

## 🎵 Audio Format Requirements

### What Audio Formats Work?

- **WAV** (recommended)
- **MP3**
- **FLAC**
- **Any format supported by torchaudio**

### Audio Specifications

- **Sample Rate**: Any (automatically converted to 16kHz)
- **Channels**: Mono or Stereo (automatically converted to mono)
- **Duration**: No strict limit, but longer audio takes more time

### Tips for Best Results

1. **Clear audio** works better than noisy audio
2. **At least 1-2 seconds** of speech for reliable detection
3. **Single language** per audio file (mixed languages may confuse the model)

---

## ⚙️ Configuration Options

### Adjusting Performance

You can modify `model_repository/ald/config.pbtxt` to change:

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
docker build -t ald-triton:latest .
docker stop ald-server
docker rm ald-server
docker run -d --gpus all -p 8100:8000 -p 8101:8001 -p 8102:8002 \
  --name ald-server ald-triton:latest
```

---

## 🛠️ Troubleshooting

### Problem: Service Won't Start

**Symptoms**: Container exits immediately or shows errors

**Solutions**:
1. **Check logs**: `docker logs ald-server`
2. **Verify GPU**: `nvidia-smi` should show your GPU
3. **Check port conflicts**: `sudo lsof -i :8100` (should be empty)
4. **Verify Docker GPU access**: 
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

### Problem: "Out of Memory" Errors

**Symptoms**: CUDA out of memory errors in logs

**Solutions**:
1. **Reduce batch size** in config.pbtxt
2. **Use smaller audio files**
3. **Check GPU memory**: `nvidia-smi`
4. **Close other GPU applications**

### Problem: Slow Inference

**Symptoms**: Requests take a long time to process

**Solutions**:
1. **Verify GPU is being used**: Check logs for "Using device: cuda"
2. **Check GPU utilization**: `nvidia-smi -l 1` (should show activity)
3. **Increase batch size** (if memory allows)
4. **Use shorter audio files** for faster processing

### Problem: Wrong Language Detected

**Symptoms**: Service detects incorrect language

**Solutions**:
1. **Use longer audio** (at least 2-3 seconds)
2. **Ensure clear audio** (reduce background noise)
3. **Check if language is supported** (107 languages supported)
4. **Try multiple audio samples** for consistency

### Problem: Cannot Connect to Service

**Symptoms**: Connection refused or timeout errors

**Solutions**:
1. **Verify service is running**: `docker ps`
2. **Check port mapping**: `docker ps` should show port mappings
3. **Test from container**: 
   ```bash
   docker exec ald-server curl http://localhost:8000/v2/health/ready
   ```
4. **Check firewall**: `sudo ufw status`

### Problem: Model Download Fails

**Symptoms**: Build fails when downloading model

**Solutions**:
1. **Check internet connection**
2. **Retry the build** (network issues can be temporary)
3. **Check disk space**: `df -h`
4. **Build without pre-download** (model will download on first run)

---

## 📈 Monitoring and Metrics

### Health Check

```bash
curl http://localhost:8100/v2/health/ready
```

### Metrics Endpoint

```bash
curl http://localhost:8102/metrics
```

This provides Prometheus-compatible metrics including:
- Request count
- Inference latency
- GPU utilization
- Error rates

### View Real-Time Logs

```bash
docker logs -f ald-server
```

Press `Ctrl+C` to stop viewing logs.

---

## 🔄 Common Operations

### Stop the Service

```bash
docker stop ald-server
```

### Start the Service

```bash
docker start ald-server
```

### Restart the Service

```bash
docker restart ald-server
```

### Remove the Service

```bash
docker stop ald-server
docker rm ald-server
```

### Update the Service

```bash
# Rebuild with latest changes
docker build -t ald-triton:latest .

# Stop and remove old container
docker stop ald-server
docker rm ald-server

# Start new container
docker run -d --gpus all -p 8100:8000 -p 8101:8001 -p 8102:8002 \
  --name ald-server ald-triton:latest
```

---

## 📚 Additional Resources

### Service Documentation
- **Detailed README**: `ALD-triton/README.md`
- **Model Source (HuggingFace)**: [https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)
  - This is where the ALD model was obtained from
  - Contains model documentation, usage examples, and performance metrics

### Technical References
- **Triton Server Docs**: https://docs.nvidia.com/deeplearning/triton-inference-server/
- **Docker Docs**: https://docs.docker.com/

### Getting Help

If you encounter issues:
1. Check the logs: `docker logs ald-server`
2. Review this guide's troubleshooting section
3. Check the service README: `ALD-triton/README.md`
4. Review Triton Server documentation

---

## 📝 Quick Reference

### Essential Commands

```bash
# Build
cd ALD-triton
docker build -t ald-triton:latest .

# Run
docker run -d --gpus all -p 8100:8000 -p 8101:8001 -p 8102:8002 \
  --name ald-server ald-triton:latest

# Check status
docker ps
curl http://localhost:8100/v2/health/ready

# Test
cd ALD-triton
python3 test_client.py

# View logs
docker logs -f ald-server

# Stop
docker stop ald-server
```

### Port Information

- **HTTP API**: `http://localhost:8100`
- **gRPC API**: `localhost:8101`
- **Metrics**: `http://localhost:8102/metrics`

### Model Information

- **Model Name**: `ald`
- **Backend**: Python
- **Max Batch Size**: 64
- **GPU Required**: Yes

---

## ✅ Summary

You've learned how to:
1. ✅ Build the ALD-triton Docker image
2. ✅ Run the service
3. ✅ Verify it's working
4. ✅ Test language detection
5. ✅ Use the API
6. ✅ Troubleshoot common issues

The ALD-triton service is now ready to detect languages in audio files! For production use, consider setting up monitoring, load balancing, and proper security measures.



