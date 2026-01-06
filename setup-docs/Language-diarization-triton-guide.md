# Language-diarization-triton Service Guide

## 📖 What is Language-diarization-triton?

**Language Diarization** is a service that analyzes audio recordings and identifies **when different languages are spoken** throughout the audio. Unlike simple language detection (which tells you one language for the whole audio), diarization creates a timeline showing which language is spoken at each moment.

### Real-World Use Cases
- **Multilingual Meetings**: Track when participants switch between languages
- **Call Centers**: Identify language changes during customer calls
- **Content Analysis**: Analyze multilingual video/audio content
- **Language Learning Apps**: Track language usage in conversations
- **Media Production**: Automatically segment content by language

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

The service works by:
1. **Segmenting** the audio into small chunks (2 seconds each)
2. **Detecting** the language in each segment
3. **Combining** results into a timeline

```
Language-diarization-triton/
├── Dockerfile              # Recipe to build the container
├── README.md               # Documentation
├── test_client.py          # Test script
└── model_repository/       # Model storage
    └── lang_diarization/
        ├── config.pbtxt    # Service configuration
        └── 1/
            └── model.py    # Processing logic
```

---

## 🔨 Step 1: Building the Docker Image

### What is Building?

Building a Docker image packages everything needed to run the service: the code, the AI model, and all dependencies. Once built, you can run it anywhere.

### Step-by-Step Build Instructions

#### Option A: Simple Build (Recommended for Beginners)

1. **Open a terminal** (command line window)

2. **Navigate to the Language-diarization-triton folder:**
   ```bash
   cd Language-diarization-triton
   ```

3. **Build the image:**
   ```bash
   docker build -t lang-diarization-triton:latest .
   ```
   
   **What this does:**
   - `docker build` = Start building
   - `-t lang-diarization-triton:latest` = Name the image
   - `.` = Use the current directory

4. **Wait for it to complete** (this may take 5-15 minutes the first time)
   - Downloads the base Triton server
   - Installs audio processing libraries
   - Downloads the VoxLingua107 language detection model
   - Sets everything up

#### Option B: Understanding What Happens During Build

The Dockerfile does these steps:
1. **Starts with Triton Server base image** - Pre-configured server
2. **Installs audio libraries** - Tools to handle audio files (libsndfile1)
3. **Installs Python packages** - PyTorch, SpeechBrain, torchaudio
4. **Copies the model code** - Your custom processing logic
5. **Pre-downloads the model** - Gets the VoxLingua107 language detection model ready
   - Uses SpeechBrain's implementation from the W2V-E2E-Language-Diarization repository
   - Base model source: [https://github.com/jagabandhumishra/W2V-E2E-Language-Diarization](https://github.com/jagabandhumishra/W2V-E2E-Language-Diarization)

**Expected Output:**
```
Step 1/7 : FROM nvcr.io/nvidia/tritonserver:24.01-py3
...
Downloading VoxLingua107 ECAPA-TDNN model...
Model downloaded successfully
...
Successfully built abc123def456
Successfully tagged lang-diarization-triton:latest
```

**Troubleshooting Build Issues:**
- **"Cannot connect to Docker daemon"**: Run `sudo systemctl start docker`
- **"No space left on device"**: Free up disk space
- **"Network timeout"**: Check internet connection, the build downloads large files

---

## 📥 How the Language Diarization Model Was Obtained from GitHub

### Model Source

The Language-diarization service uses the **W2V-E2E-Language-Diarization** model from GitHub:
- **Repository**: [https://github.com/jagabandhumishra/W2V-E2E-Language-Diarization](https://github.com/jagabandhumishra/W2V-E2E-Language-Diarization)
- **Model Type**: End-to-End Language Diarization
- **Base Model**: VoxLingua107 ECAPA-TDNN (from SpeechBrain)

### About the Model

The W2V-E2E-Language-Diarization system:

- **Creates language timelines** from audio:
  - Segments audio into 2-second chunks
  - Detects language in each segment
  - Combines results into a continuous timeline
  - Shows when different languages are spoken

- **Uses VoxLingua107 model**:
  - Pre-trained on 107 languages
  - ECAPA-TDNN architecture for spoken language identification
  - Processes audio at 16kHz (single channel)
  - Automatically normalizes audio (resampling + mono conversion)

- **Segmentation Process**:
  - Segment duration: 2.0 seconds
  - Overlap: 0.5 seconds between segments
  - Ensures no language changes are missed
  - Adjacent segments with same language are merged

### How the Model is Downloaded

During the Docker build process, the model is automatically downloaded via SpeechBrain:

```python
from speechbrain.inference.classifiers import EncoderClassifier

# Downloads VoxLingua107 ECAPA-TDNN model
model = EncoderClassifier.from_hparams(
    source='speechbrain/lang-id-voxlingua107-ecapa',
    savedir='tmp_ald_model'
)
```

**What happens:**
1. SpeechBrain connects to HuggingFace Hub
2. Downloads the VoxLingua107 ECAPA-TDNN model
3. Model is cached in the Docker image
4. The diarization logic (from W2V-E2E-Language-Diarization) processes audio segments

### Model Architecture

The system combines:
- **VoxLingua107 ECAPA-TDNN**: Language identification model (from SpeechBrain/HuggingFace)
- **Custom Diarization Logic**: Segmentation and timeline creation (from W2V-E2E-Language-Diarization repository)
- **Audio Processing**: Handles various audio formats (WAV, MP3, FLAC)

### Supported Languages

The service supports **107 languages** including:
- **English** (en)
- **Hindi** (hi)
- **Tamil** (ta)
- **Telugu** (te)
- **Kannada** (kn)
- **Malayalam** (ml)
- **Bengali** (bn)
- **And 100 more languages!**

### References

- **Original Repository**: [https://github.com/jagabandhumishra/W2V-E2E-Language-Diarization](https://github.com/jagabandhumishra/W2V-E2E-Language-Diarization)
- **Base Model (HuggingFace)**: [https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)
- **SpeechBrain**: [https://github.com/speechbrain/speechbrain](https://github.com/speechbrain/speechbrain)

---

## 🚀 Step 2: Running the Service

### What is Running?

Running the service means starting it up so it can accept requests. It's like opening a restaurant for business.

### Step-by-Step Run Instructions

#### Basic Run (For Testing)

```bash
docker run --gpus all \
  -p 8600:8000 \
  -p 8601:8001 \
  -p 8602:8002 \
  --name lang-diarization-server \
  lang-diarization-triton:latest
```

**What each part means:**
- `docker run` = Start a container
- `--gpus all` = Use all available GPUs (makes it faster)
- `-p 8600:8000` = Map port 8600 on your computer to port 8000 in container
- `-p 8601:8001` = Map gRPC port
- `-p 8602:8002` = Map metrics port
- `--name lang-diarization-server` = Name the container
- `lang-diarization-triton:latest` = Use the image we built

#### Run in Background (Recommended for Production)

```bash
docker run -d --gpus all \
  -p 8600:8000 \
  -p 8601:8001 \
  -p 8602:8002 \
  --name lang-diarization-server \
  lang-diarization-triton:latest
```

The `-d` flag runs it in the background (detached mode).

#### Using Docker Compose (Easiest Method)

If you're using the main docker-compose.yml file:

```bash
docker-compose up -d lang-diarization-server
```

This automatically handles all the configuration.

### Understanding Ports

- **Port 8600 (HTTP)**: Main entrance for web requests
- **Port 8601 (gRPC)**: Fast lane for program-to-program communication
- **Port 8602 (Metrics)**: Monitoring room for checking service health

---

## ✅ Step 3: Verifying the Service is Running

### Check if Container is Running

```bash
docker ps
```

You should see `lang-diarization-server` in the list with status "Up".

### Check Service Health

```bash
curl http://localhost:8600/v2/health/ready
```

**Expected Response:** `{"status":"ready"}`

If you see this, the service is ready to accept requests!

### List Available Models

```bash
curl http://localhost:8600/v2/models
```

**Expected Response:**
```json
{
  "models": [
    {
      "name": "lang_diarization",
      "platform": "python",
      "versions": ["1"]
    }
  ]
}
```

### View Logs

```bash
docker logs lang-diarization-server
```

Look for:
- `"Server is ready to receive inference requests"` = Success!
- Any error messages = Something went wrong

---

## 🧪 Step 4: Testing the Service

> **Note**: If you're accessing the service from a remote machine, replace `localhost` with your server's IP address. For example, if your server IP is `192.168.1.100`, use `http://192.168.1.100:8600` instead of `http://localhost:8600`.

### Method 1: Manual Testing with curl

#### Step 1: Prepare Your Audio File

Convert your audio to base64:
```bash
AUDIO_B64=$(base64 -w 0 your_audio.wav)
```

#### Step 2: Send Request (All Languages)

```bash
curl -X POST http://localhost:8600/v2/models/lang_diarization/infer \
  -H "Content-Type: application/json" \
  -d "{
    \"inputs\": [
      {
        \"name\": \"AUDIO_DATA\",
        \"shape\": [1, 1],
        \"datatype\": \"BYTES\",
        \"data\": [[\"$AUDIO_B64\"]]
      },
      {
        \"name\": \"LANGUAGE\",
        \"shape\": [1, 1],
        \"datatype\": \"BYTES\",
        \"data\": [[\"\"]]
      }
    ],
    \"outputs\": [
      {
        \"name\": \"DIARIZATION_RESULT\"
      }
    ]
  }"
```

#### Step 3: Send Request (Specific Language - e.g., Tamil)

```bash
curl -X POST http://localhost:8600/v2/models/lang_diarization/infer \
  -H "Content-Type: application/json" \
  -d "{
    \"inputs\": [
      {
        \"name\": \"AUDIO_DATA\",
        \"shape\": [1, 1],
        \"datatype\": \"BYTES\",
        \"data\": [[\"$AUDIO_B64\"]]
      },
      {
        \"name\": \"LANGUAGE\",
        \"shape\": [1, 1],
        \"datatype\": \"BYTES\",
        \"data\": [[\"ta\"]]
      }
    ],
    \"outputs\": [
      {
        \"name\": \"DIARIZATION_RESULT\"
      }
    ]
  }"
```

**Expected Response:**
```json
{
  "outputs": [
    {
      "name": "DIARIZATION_RESULT",
      "datatype": "BYTES",
      "shape": [1, 1],
      "data": ["{\"total_segments\": 5, \"segments\": [{\"start_time\": 0.0, \"end_time\": 2.0, \"duration\": 2.0, \"language\": \"ta: Tamil\", \"confidence\": 0.9850}, ...], \"target_language\": \"all\"}"]
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
url = "http://localhost:8600/v2/models/lang_diarization/infer"
payload = {
    "inputs": [
        {
            "name": "AUDIO_DATA",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [[audio_b64]]
        },
        {
            "name": "LANGUAGE",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [[""]]  # Empty for all languages, or "ta" for Tamil, etc.
        }
    ],
    "outputs": [
        {"name": "DIARIZATION_RESULT"}
    ]
}

# Send request
response = requests.post(url, json=payload)
result = response.json()

# Parse and display results
diarization_result = json.loads(result["outputs"][0]["data"][0])

print(f"Total segments: {diarization_result['total_segments']}")
print(f"Target language: {diarization_result.get('target_language', 'all')}")
print("\nSegments:")
for segment in diarization_result['segments']:
    print(f"  [{segment['start_time']:.2f}s - {segment['end_time']:.2f}s]: "
          f"{segment['language']} (confidence: {segment['confidence']:.4f})")
```

Run it:
```bash
python3 test_my_audio.py
```

**Expected Output:**
```
Total segments: 5
Target language: all

Segments:
  [0.00s - 2.00s]: ta: Tamil (confidence: 0.9850)
  [2.00s - 4.00s]: en: English (confidence: 0.9200)
  [4.00s - 6.00s]: ta: Tamil (confidence: 0.9500)
  [6.00s - 8.00s]: en: English (confidence: 0.9100)
  [8.00s - 10.00s]: ta: Tamil (confidence: 0.9600)
```

---

## 📊 Understanding the API

### Input Format

1. **AUDIO_DATA** (Required)
   - **Type**: Base64-encoded string
   - **Format**: WAV, MP3, FLAC, or other audio formats
   - **What to send**: Your audio file converted to base64

2. **LANGUAGE** (Optional)
   - **Type**: String (language code)
   - **Format**: ISO language code (e.g., "ta", "en", "hi")
   - **What to send**: 
     - Empty string `""` = Detect all languages
     - Specific code like `"ta"` = Only return segments in that language

### Output Format

The service returns a JSON string with:

```json
{
  "total_segments": 5,
  "num_speakers": 2,
  "segments": [
    {
      "start_time": 0.0,
      "end_time": 2.0,
      "duration": 2.0,
      "language": "ta: Tamil",
      "confidence": 0.9850
    },
    {
      "start_time": 2.0,
      "end_time": 4.0,
      "duration": 2.0,
      "language": "en: English",
      "confidence": 0.9200
    }
  ],
  "target_language": "all"
}
```

**Fields:**
- **total_segments**: Number of language segments found
- **segments**: Array of segments with:
  - **start_time**: When the segment starts (in seconds)
  - **end_time**: When the segment ends (in seconds)
  - **duration**: Length of the segment (in seconds)
  - **language**: Detected language with name (e.g., "ta: Tamil")
  - **confidence**: How sure the model is (0.0 to 1.0)
- **target_language**: Which language filter was used ("all" or specific code)

### Supported Languages

The service supports **107 languages** including:
- **English** (en)
- **Hindi** (hi)
- **Tamil** (ta)
- **Telugu** (te)
- **Kannada** (kn)
- **Malayalam** (ml)
- **Bengali** (bn)
- **And 100 more languages!**

See the full list in `ALD-triton/README.md` (uses the same VoxLingua107 model).

---

## 🧠 How It Works (Technical Details)

### Diarization Process

1. **Audio Segmentation**:
   - Audio is divided into overlapping segments
   - Segment duration: 2.0 seconds
   - Overlap: 0.5 seconds between segments
   - This ensures no language changes are missed

2. **Language Detection**:
   - Each segment is analyzed using the VoxLingua107 model
   - The model identifies the language in that segment
   - Confidence scores are calculated

3. **Timeline Creation**:
   - Results from all segments are combined
   - Creates a continuous timeline of language changes
   - Adjacent segments with the same language are merged

4. **Filtering** (if target language specified):
   - Only segments matching the target language are returned
   - Other segments are filtered out

### Performance Characteristics

- **Processing Time**: Scales with audio duration
- **Segment Size**: 2 seconds (configurable in model.py)
- **Accuracy**: High for clear audio with distinct language segments
- **Best For**: Audio with clear language boundaries

---

## ⚙️ Configuration Options

### Adjusting Performance

You can modify `model_repository/lang_diarization/config.pbtxt` to change:

1. **Max Batch Size**: How many requests to process together
   - Current: 32
   - Increase for more throughput (needs more GPU memory)
   - Decrease if running out of memory

2. **Dynamic Batching**: Automatically groups requests
   - Current: Enabled with sizes [1, 2, 4, 8, 16, 32]
   - Adjust based on your workload

3. **GPU Instances**: Number of model copies
   - Current: 1
   - Increase for higher throughput (uses more GPU memory)

### Adjusting Diarization Parameters

You can modify `model_repository/lang_diarization/1/model.py` to change:

1. **Segment Duration**: How long each segment is
   - Current: 2.0 seconds
   - Shorter = More granular but slower
   - Longer = Faster but may miss quick language changes

2. **Overlap**: How much segments overlap
   - Current: 0.5 seconds
   - More overlap = Better accuracy but slower
   - Less overlap = Faster but may miss boundaries

**After changing files, rebuild the image:**
```bash
docker build -t lang-diarization-triton:latest .
docker stop lang-diarization-server
docker rm lang-diarization-server
docker run -d --gpus all -p 8600:8000 -p 8601:8001 -p 8602:8002 \
  --name lang-diarization-server lang-diarization-triton:latest
```

---

## 🛠️ Troubleshooting

### Problem: Service Won't Start

**Symptoms**: Container exits immediately or shows errors

**Solutions**:
1. **Check logs**: `docker logs lang-diarization-server`
2. **Verify GPU**: `nvidia-smi` should show your GPU
3. **Check port conflicts**: `sudo lsof -i :8600` (should be empty)
4. **Verify Docker GPU access**: 
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

### Problem: "Out of Memory" Errors

**Symptoms**: CUDA out of memory errors in logs

**Solutions**:
1. **Reduce batch size** in config.pbtxt
2. **Use shorter audio files**
3. **Check GPU memory**: `nvidia-smi`
4. **Close other GPU applications**

### Problem: Slow Processing

**Symptoms**: Requests take a very long time to process

**Solutions**:
1. **Verify GPU is being used**: Check logs for "Using device: cuda"
2. **Check GPU utilization**: `nvidia-smi -l 1` (should show activity)
3. **Use shorter audio files** for faster processing
4. **Increase segment duration** (if acceptable for your use case)

### Problem: Missing Language Changes

**Symptoms**: Service doesn't detect language switches

**Solutions**:
1. **Reduce segment duration** in model.py (more granular detection)
2. **Increase overlap** between segments
3. **Ensure clear audio** (reduce background noise)
4. **Use longer audio** (at least 5-10 seconds per language)

### Problem: Wrong Languages Detected

**Symptoms**: Service detects incorrect languages

**Solutions**:
1. **Use clear audio** (reduce background noise)
2. **Ensure distinct language segments** (not code-switching within segments)
3. **Check if language is supported** (107 languages supported)
4. **Review confidence scores** - low confidence may indicate unclear audio

### Problem: Cannot Connect to Service

**Symptoms**: Connection refused or timeout errors

**Solutions**:
1. **Verify service is running**: `docker ps`
2. **Check port mapping**: `docker ps` should show port mappings
3. **Test from container**: 
   ```bash
   docker exec lang-diarization-server curl http://localhost:8000/v2/health/ready
   ```
4. **Check firewall**: `sudo ufw status`

---

## 📈 Monitoring and Metrics

### Health Check

```bash
curl http://localhost:8600/v2/health/ready
```

### Metrics Endpoint

```bash
curl http://localhost:8602/metrics
```

This provides Prometheus-compatible metrics including:
- Request count
- Inference latency
- GPU utilization
- Error rates

### View Real-Time Logs

```bash
docker logs -f lang-diarization-server
```

Press `Ctrl+C` to stop viewing logs.

---

## 🔄 Common Operations

### Stop the Service

```bash
docker stop lang-diarization-server
```

### Start the Service

```bash
docker start lang-diarization-server
```

### Restart the Service

```bash
docker restart lang-diarization-server
```

### Remove the Service

```bash
docker stop lang-diarization-server
docker rm lang-diarization-server
```

### Update the Service

```bash
# Rebuild with latest changes
docker build -t lang-diarization-triton:latest .

# Stop and remove old container
docker stop lang-diarization-server
docker rm lang-diarization-server

# Start new container
docker run -d --gpus all -p 8600:8000 -p 8601:8001 -p 8602:8002 \
  --name lang-diarization-server lang-diarization-triton:latest
```

---

## 📚 Additional Resources

### Service Documentation
- **Detailed README**: `Language-diarization-triton/README.md`
- **Model Source (GitHub)**: [https://github.com/jagabandhumishra/W2V-E2E-Language-Diarization](https://github.com/jagabandhumishra/W2V-E2E-Language-Diarization)
  - This is where the language diarization implementation was obtained from
- **Base Model (HuggingFace)**: [https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)

### Technical References
- **Triton Server Docs**: https://docs.nvidia.com/deeplearning/triton-inference-server/
- **Docker Docs**: https://docs.docker.com/

### Getting Help

If you encounter issues:
1. Check the logs: `docker logs lang-diarization-server`
2. Review this guide's troubleshooting section
3. Check the service README: `Language-diarization-triton/README.md`
4. Review Triton Server documentation

---

## 📝 Quick Reference

### Essential Commands

```bash
# Build
cd Language-diarization-triton
docker build -t lang-diarization-triton:latest .

# Run
docker run -d --gpus all -p 8600:8000 -p 8601:8001 -p 8602:8002 \
  --name lang-diarization-server lang-diarization-triton:latest

# Check status
docker ps
curl http://localhost:8600/v2/health/ready

# Test
cd Language-diarization-triton
python3 test_client.py

# View logs
docker logs -f lang-diarization-server

# Stop
docker stop lang-diarization-server
```

### Port Information

- **HTTP API**: `http://localhost:8600`
- **gRPC API**: `localhost:8601`
- **Metrics**: `http://localhost:8602/metrics`

### Model Information

- **Model Name**: `lang_diarization`
- **Backend**: Python
- **Max Batch Size**: 32
- **GPU Required**: Yes
- **Supported Languages**: 107 languages

---

## ✅ Summary

You've learned how to:
1. ✅ Build the Language-diarization-triton Docker image
2. ✅ Run the service
3. ✅ Verify it's working
4. ✅ Test language diarization
5. ✅ Use the API (all languages and filtered)
6. ✅ Understand how segmentation works
7. ✅ Troubleshoot common issues

The Language-diarization-triton service is now ready to create language timelines from audio! For production use, consider setting up monitoring, load balancing, and proper security measures.



