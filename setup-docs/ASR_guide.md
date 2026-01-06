# ASR-triton Service Guide

## 📖 What is ASR-triton?

**ASR** stands for **Automatic Speech Recognition**. This service can listen to audio recordings and automatically convert speech to text. It supports **multiple Indian languages** including Hindi, Tamil, Telugu, Kannada, Malayalam, Gujarati, Marathi, Bengali, and more.

### Real-World Use Cases
- **Voice Assistants**: Convert spoken commands to text for processing
- **Transcription Services**: Automatically transcribe audio recordings
- **Call Centers**: Transcribe customer calls for analysis and record-keeping
- **Content Creation**: Generate captions and subtitles for video content
- **Accessibility**: Provide real-time transcription for hearing-impaired users
- **Voice Notes**: Convert voice memos to searchable text

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
- **Hardware Specifications**: May vary depending on the scale of your application
- **Tested Machine**: g4dn.2xlarge (For detailed specifications and pricing, check [AWS EC2 g4dn.2xlarge](https://instances.vantage.sh/aws/ec2/g4dn.2xlarge?currency=USD))
- **Shared Memory**: At least 2GB (`shm_size`)

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

Think of this service like a factory:
- **Docker Image** = Pre-built package containing everything needed
- **Model Repository** = The factory floor where AI models process requests
- **Triton Server** = The production manager coordinating all operations
- **Model Ensemble** = Multiple AI models working together (preprocessor → acoustic model → decoder)

```
ASR Service/
├── Docker Image: ai4bharat/triton-multilingual-asr:latest
├── Model: asr_am_ensemble (ensemble of multiple models)
│   ├── asr_preprocessor (processes audio)
│   ├── asr_am (acoustic model)
│   └── asr_greedy_decoder (generates text)
└── Ports:
    ├── 5000 (HTTP API)
    ├── 5001 (gRPC API)
    └── 5002 (Metrics)
```

---

## 🐳 Step 1: Pulling the Docker Image

### What is Pulling?

Pulling a Docker image means downloading a pre-built package from Docker Hub (a repository of container images). The ASR service image is already built and ready to use - you just need to download it.

### Step-by-Step Pull Instructions

#### Option A: Simple Pull (Recommended)

1. **Open a terminal** (command line window)

2. **Pull the image:**
   ```bash
   docker pull ai4bharat/triton-multilingual-asr:latest
   ```
   
   **What this does:**
   - `docker pull` = Download the image
   - `ai4bharat/triton-multilingual-asr:latest` = Image name and version tag
   - `latest` = Most recent version

3. **Wait for it to complete** (this may take 5-20 minutes depending on internet speed)
   - The image is large (several GB) because it contains:
     - Triton Inference Server
     - Python runtime and libraries
     - Pre-trained ASR models
     - All dependencies

**Expected Output:**
```
latest: Pulling from ai4bharat/triton-multilingual-asr
...
Status: Downloaded newer image for ai4bharat/triton-multilingual-asr:latest
docker.io/ai4bharat/triton-multilingual-asr:latest
```

#### Verify Image is Downloaded

```bash
docker images | grep triton-multilingual-asr
```

You should see the image listed with its size.

**Troubleshooting Pull Issues:**
- **"Cannot connect to Docker daemon"**: Run `sudo systemctl start docker`
- **"Network timeout"**: Check internet connection, retry the pull
- **"No space left on device"**: Free up disk space (`docker system prune` to clean old images)

---

## 🚀 Step 2: Running the Service

### What is Running?

Running the service means starting it up so it can accept requests. The service will load the AI models into GPU memory and be ready to transcribe audio.

### Step-by-Step Run Instructions

#### Option A: Using Docker Run (For Testing)

```bash
docker run -d --gpus all \
  -p 5000:8000 \
  -p 5001:8001 \
  -p 5002:8002 \
  --shm-size=2g \
  --name asr \
  ai4bharat/triton-multilingual-asr:latest \
  tritonserver --model-repository=/models
```

**What each part means:**
- `docker run` = Start a container
- `-d` = Run in background (detached mode)
- `--gpus all` = Use all available GPUs (makes it faster)
- `-p 5000:8000` = Map port 5000 on your computer to port 8000 in container (HTTP)
- `-p 5001:8001` = Map port 5001 for gRPC API
- `-p 5002:8002` = Map port 5002 for metrics
- `--shm-size=2g` = Allocate 2GB shared memory (required for the models)
- `--name asr` = Name the container "asr"
- `ai4bharat/triton-multilingual-asr:latest` = Use the image we pulled
- `tritonserver --model-repository=/models` = Command to start the server

#### Option B: Using Docker Compose (Recommended for Production)

Create or use `docker-compose.yml`:

```yaml
version: '3.8'

services:
  asr:
    image: ai4bharat/triton-multilingual-asr:latest
    container_name: asr
    ports:
      - "5000:8000"  # HTTP API
      - "5001:8001"  # GRPC API
      - "5002:8002"  # Metrics
    command: tritonserver --model-repository=/models
    shm_size: 2gb
    runtime: nvidia
    restart: always
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
```

Then run:
```bash
docker-compose up -d asr
```

### Understanding Ports

Think of ports like different doors to the same building:
- **Port 5000 (HTTP)**: Main entrance for web requests (REST API)
- **Port 5001 (gRPC)**: Fast lane for program-to-program communication
- **Port 5002 (Metrics)**: Monitoring room for checking service health and performance

---

## ✅ Step 3: Verifying the Service is Running

### Check if Container is Running

```bash
docker ps
```

You should see `asr` in the list with status "Up".

### Check Service Health

```bash
curl http://localhost:5000/v2/health/ready
```

**Expected Response:** `{"status":"ready"}`

If you see this, the service is ready to accept requests!

### List Available Models

```bash
curl http://localhost:5000/v2/models
```

**Expected Response:**
```json
{
  "models": [
    {
      "name": "asr_am_ensemble",
      "platform": "ensemble",
      "versions": ["1"]
    }
  ]
}
```

### View Logs

```bash
docker logs asr
```

Or follow logs in real-time:
```bash
docker logs -f asr
```

Look for:
- `"Server is ready to receive inference requests"` = Success!
- `"successfully loaded 'asr_am_ensemble' version 1"` = Models loaded successfully
- Any error messages = Something went wrong

---

## 🧪 Step 4: Testing the Service

> **Note**: If you're accessing the service from a remote machine, replace `localhost` with your server's IP address. For example, if your server IP is `192.168.1.100`, use `http://192.168.1.100:5000` instead of `http://localhost:5000`.

### Method 1: Using a Python Script (Recommended)

Create a file `test_asr.py`:

```python
#!/usr/bin/env python3
"""
Script to send WAV file to ASR model via Triton HTTP API
"""
import json
import base64
import numpy as np
import wave
import sys
import requests

def test_asr(wav_path, lang_id="hi", endpoint="http://localhost:5000/v2/models/asr_am_ensemble/infer"):
    """
    Convert WAV file to Triton inference request and send
    """
    # Read WAV file
    w = wave.open(wav_path, 'rb')
    frames = w.readframes(-1)
    samples = np.frombuffer(frames, dtype=np.int16)
    sample_rate = w.getframerate()
    
    # Convert to float32 and normalize
    audio_float = samples.astype(np.float32) / 32768.0
    num_samples = len(audio_float)
    
    # If stereo, convert to mono
    if w.getnchannels() > 1:
        audio_float = audio_float.reshape(-1, w.getnchannels()).mean(axis=1)
        num_samples = len(audio_float)
    
    w.close()
    
    print(f"Audio file: {wav_path}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Number of samples: {num_samples}")
    print(f"Duration: {num_samples / sample_rate:.2f} seconds")
    print(f"Language ID: {lang_id}")
    
    # Create Triton request payload
    payload = {
        "inputs": [
            {
                "name": "AUDIO_SIGNAL",
                "shape": [1, num_samples],
                "datatype": "FP32",
                "data": audio_float.tolist()
            },
            {
                "name": "NUM_SAMPLES",
                "shape": [1, 1],
                "datatype": "INT32",
                "data": [num_samples]
            },
            {
                "name": "LANG_ID",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [lang_id]
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
    
    # Extract transcript
    if "outputs" in result:
        for output in result["outputs"]:
            if output.get("name") == "TRANSCRIPTS":
                if "data" in output:
                    transcripts = output["data"]
                    if isinstance(transcripts, list) and len(transcripts) > 0:
                        transcript = transcripts[0]
                        if isinstance(transcript, bytes):
                            transcript = transcript.decode('utf-8')
                        print(f"\n=== Transcript ===")
                        print(transcript)
                        return transcript
    
    return None

if __name__ == "__main__":
    wav_path = sys.argv[1] if len(sys.argv) > 1 else "test_audio.wav"
    lang_id = sys.argv[2] if len(sys.argv) > 2 else "hi"
    
    test_asr(wav_path, lang_id)
```

**Run the script:**
```bash
python3 test_asr.py your_audio.wav hi
```

**Expected Output:**
```
Audio file: your_audio.wav
Sample rate: 16000 Hz
Number of samples: 48000
Duration: 3.00 seconds
Language ID: hi

Sending request to: http://localhost:5000/v2/models/asr_am_ensemble/infer

=== Response ===
{
  "model_name": "asr_am_ensemble",
  "model_version": "1",
  "outputs": [
    {
      "name": "TRANSCRIPTS",
      "datatype": "BYTES",
      "shape": [1],
      "data": ["नमस्ते, मैं कैसे आपकी मदद कर सकता हूं"]
    }
  ]
}

=== Transcript ===
नमस्ते, मैं कैसे आपकी मदद कर सकता हूं
```

### Method 2: Using curl (Manual Testing)

#### Step 1: Prepare Your Audio File

Convert your audio to the required format:
- WAV format
- Mono or Stereo (will be converted to mono)
- Any sample rate (will be processed appropriately)

#### Step 2: Create Request JSON

You'll need to convert your audio to a list of float32 values. Use a Python script for this:

```python
import numpy as np
import wave
import json

w = wave.open("your_audio.wav", 'rb')
frames = w.readframes(-1)
samples = np.frombuffer(frames, dtype=np.int16)
audio_float = (samples.astype(np.float32) / 32768.0).tolist()
num_samples = len(audio_float)
w.close()

payload = {
    "inputs": [
        {"name": "AUDIO_SIGNAL", "shape": [1, num_samples], "datatype": "FP32", "data": audio_float},
        {"name": "NUM_SAMPLES", "shape": [1, 1], "datatype": "INT32", "data": [num_samples]},
        {"name": "LANG_ID", "shape": [1, 1], "datatype": "BYTES", "data": ["hi"]}
    ]
}

with open("request.json", "w") as f:
    json.dump(payload, f)
```

#### Step 3: Send Request

```bash
curl -X POST http://localhost:5000/v2/models/asr_am_ensemble/infer \
  -H "Content-Type: application/json" \
  -d @request.json
```

---

## 📊 Understanding the API

### Model Information

- **Model Name**: `asr_am_ensemble`
- **Type**: Ensemble model (multiple models working together)
- **Backend**: Triton Inference Server
- **Max Batch Size**: 32

### Input Format

**AUDIO_SIGNAL** (Required)
- **Type**: FP32 (float32)
- **Shape**: `[batch_size, num_samples]` or `[1, num_samples]`
- **Description**: Audio samples normalized to range [-1.0, 1.0]
- **Format**: List of floating point numbers

**NUM_SAMPLES** (Required)
- **Type**: INT32
- **Shape**: `[1, 1]`
- **Description**: Number of audio samples in the audio signal

**LANG_ID** (Required)
- **Type**: BYTES (string)
- **Shape**: `[1, 1]`
- **Description**: Language code (e.g., "hi", "ta", "te", "kn", "ml", "gu", "mr", "bn")
- **Supported Languages**: Hindi (hi), Tamil (ta), Telugu (te), Kannada (kn), Malayalam (ml), Gujarati (gu), Marathi (mr), Bengali (bn), and more

### Output Format

**TRANSCRIPTS**
- **Type**: BYTES (string)
- **Shape**: `[batch_size]` or `[1]`
- **Description**: Transcribed text from the audio

### Supported Languages

The service supports multiple Indian languages:
- **Hindi** (hi)
- **Tamil** (ta)
- **Telugu** (te)
- **Kannada** (kn)
- **Malayalam** (ml)
- **Gujarati** (gu)
- **Marathi** (mr)
- **Bengali** (bn)
- **And more Indian languages!**

---

## 🎵 Audio Format Requirements

### What Audio Formats Work?

- **WAV** (recommended)
- **Any format supported by Python wave library**

### Audio Specifications

- **Sample Rate**: Any (automatically processed)
- **Channels**: Mono or Stereo (automatically converted to mono)
- **Bit Depth**: 16-bit recommended
- **Duration**: No strict limit, but longer audio takes more time
- **Format**: PCM (uncompressed) recommended

### Tips for Best Results

1. **Clear audio** works better than noisy audio
2. **At least 1-2 seconds** of speech for reliable transcription
3. **Single language** per audio file (mixed languages may reduce accuracy)
4. **Correct language ID** improves accuracy significantly
5. **Normalize audio levels** for better results

---

## ⚙️ Configuration Options

### Docker Run Options

You can customize the container with additional options:

```bash
docker run -d --gpus all \
  -p 5000:8000 \
  -p 5001:8001 \
  -p 5002:8002 \
  --shm-size=2g \
  --name asr \
  --restart=always \
  -e NVIDIA_VISIBLE_DEVICES=all \
  ai4bharat/triton-multilingual-asr:latest \
  tritonserver --model-repository=/models
```

**Options explained:**
- `--restart=always` = Automatically restart container if it crashes
- `-e NVIDIA_VISIBLE_DEVICES=all` = Use all GPUs (can specify specific GPU like "0" or "0,1")
- `--shm-size=2g` = Shared memory size (2GB required, can increase if needed)

### Resource Allocation

- **GPU Memory**: Model requires several GB of GPU memory
- **System Memory**: At least 4GB RAM recommended
- **Shared Memory**: 2GB minimum (configured via `--shm-size`)

---

## 🛠️ Troubleshooting

### Problem: Service Won't Start

**Symptoms**: Container exits immediately or shows errors

**Solutions**:
1. **Check logs**: `docker logs asr`
2. **Verify GPU**: `nvidia-smi` should show your GPU
3. **Check port conflicts**: `sudo lsof -i :5000` (should be empty)
4. **Verify Docker GPU access**: 
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```
5. **Check shared memory**: Ensure `--shm-size=2g` is set

### Problem: "Out of Memory" Errors

**Symptoms**: CUDA out of memory errors in logs

**Solutions**:
1. **Check GPU memory**: `nvidia-smi`
2. **Use smaller batch size** in requests
3. **Close other GPU applications**
4. **Process shorter audio files**

### Problem: Slow Inference

**Symptoms**: Requests take a long time to process

**Solutions**:
1. **Verify GPU is being used**: Check logs for GPU usage
2. **Check GPU utilization**: `nvidia-smi -l 1` (should show activity)
3. **Use shorter audio files** for faster processing
4. **Check network latency** if calling from remote

### Problem: Wrong Transcript or Poor Accuracy

**Symptoms**: Service detects incorrect text

**Solutions**:
1. **Use correct language ID** (very important!)
2. **Use longer audio** (at least 2-3 seconds)
3. **Ensure clear audio** (reduce background noise)
4. **Check if language is supported**
5. **Normalize audio levels**

### Problem: Language Not Supported Error

**Symptoms**: Error like "'en' is not in list" or "ValueError: 'xx' is not in list"

**Solutions**:
1. **Check supported languages** - Use only Indian language codes
2. **Verify language code format** - Should be lowercase (e.g., "hi", not "HI" or "hindi")
3. **Use correct language codes**: hi, ta, te, kn, ml, gu, mr, bn, etc.

### Problem: Cannot Connect to Service

**Symptoms**: Connection refused or timeout errors

**Solutions**:
1. **Verify service is running**: `docker ps`
2. **Check port mapping**: `docker ps` should show port mappings
3. **Test from container**: 
   ```bash
   docker exec asr curl http://localhost:8000/v2/health/ready
   ```
4. **Check firewall**: `sudo ufw status`

### Problem: Container Keeps Restarting

**Symptoms**: Container status shows "Restarting"

**Solutions**:
1. **Check logs**: `docker logs asr`
2. **Verify GPU availability**: `nvidia-smi`
3. **Check system resources**: `free -h` and `df -h`
4. **Verify Docker runtime**: Ensure `runtime: nvidia` in docker-compose

---

## 📈 Monitoring and Metrics

### Health Check

```bash
curl http://localhost:5000/v2/health/ready
```

### Metrics Endpoint

```bash
curl http://localhost:5002/metrics
```

This provides Prometheus-compatible metrics including:
- Request count
- Inference latency
- GPU utilization
- Error rates

### View Real-Time Logs

```bash
docker logs -f asr
```

Press `Ctrl+C` to stop viewing logs.

### Check Container Stats

```bash
docker stats asr
```

Shows CPU, memory, GPU, and network usage in real-time.

---

## 🔄 Common Operations

### Stop the Service

```bash
docker stop asr
```

### Start the Service

```bash
docker start asr
```

### Restart the Service

```bash
docker restart asr
```

### Remove the Service

```bash
docker stop asr
docker rm asr
```

### Update the Service

```bash
# Pull latest image
docker pull ai4bharat/triton-multilingual-asr:latest

# Stop and remove old container
docker stop asr
docker rm asr

# Start new container (use your preferred method)
docker-compose up -d asr
```

---

## 📚 Additional Resources

### Service Documentation
- **Docker Hub**: https://hub.docker.com/r/ai4bharat/triton-multilingual-asr
- **AI4Bharat ASR**: https://github.com/AI4Bharat/indic-asr

### Technical References
- **Triton Server Docs**: https://docs.nvidia.com/deeplearning/triton-inference-server/
- **Docker Docs**: https://docs.docker.com/

### Getting Help

If you encounter issues:
1. Check the logs: `docker logs asr`
2. Review this guide's troubleshooting section
3. Check AI4Bharat GitHub repositories
4. Review Triton Server documentation

---

## 📝 Quick Reference

### Essential Commands

```bash
# Pull image
docker pull ai4bharat/triton-multilingual-asr:latest

# Run with docker-compose (recommended)
docker-compose up -d asr

# Or run directly
docker run -d --gpus all -p 5000:8000 -p 5001:8001 -p 5002:8002 \
  --shm-size=2g --name asr \
  ai4bharat/triton-multilingual-asr:latest \
  tritonserver --model-repository=/models

# Check status
docker ps
curl http://localhost:5000/v2/health/ready

# View logs
docker logs -f asr

# Stop
docker stop asr
```

### Port Information

- **HTTP API**: `http://localhost:5000`
- **gRPC API**: `localhost:5001`
- **Metrics**: `http://localhost:5002/metrics`

### Model Information

- **Model Name**: `asr_am_ensemble`
- **Type**: Ensemble
- **Max Batch Size**: 32
- **GPU Required**: Yes
- **Shared Memory**: 2GB minimum

---

## ✅ Summary

You've learned how to:
1. ✅ Pull the ASR Docker image from Docker Hub
2. ✅ Run the ASR service container
3. ✅ Verify it's working correctly
4. ✅ Test speech recognition with audio files
5. ✅ Use the API for transcription
6. ✅ Troubleshoot common issues

The ASR-triton service is now ready to transcribe audio files! For production use, consider setting up monitoring, load balancing, and proper security measures.

