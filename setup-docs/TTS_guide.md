# TTS-triton Service Guide

## 📖 What is TTS-triton?

**TTS** stands for **Text-to-Speech**. This service can automatically convert written text into natural-sounding speech audio. It supports **Indo-Aryan languages** including Hindi, Bengali, Gujarati, Marathi, Punjabi, and more, with high-quality neural voice synthesis.

### Real-World Use Cases
- **Voice Assistants**: Generate spoken responses from text
- **Accessibility**: Provide audio versions of text content for visually impaired users
- **Content Creation**: Generate voiceovers for videos and podcasts
- **E-learning**: Create audio lessons from written content
- **Audiobooks**: Convert books to audio format
- **Public Announcements**: Generate automated announcements in multiple languages
- **IVR Systems**: Create natural-sounding voice prompts for phone systems

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
- **Shared Memory**: At least 2GB (`shm_size`)
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

Think of this service like a voice recording studio:
- **Docker Image** = Pre-built package containing everything needed
- **Model Repository** = The studio where AI models generate speech
- **Triton Server** = The producer coordinating all operations
- **TTS Model** = The AI that converts text to speech
- **Vocoder (HiFiGAN)** = The component that generates high-quality audio waveforms

```
TTS Service/
├── Docker Image: ai4bharat/triton-indo-aryan-tts:latest
├── Model: tts (single text-to-speech model)
├── Input: Text + Speaker ID + Language ID
├── Output: Audio waveform (float32 array)
├── Sample Rate: 22050 Hz
└── Ports:
    ├── 9000 (HTTP API)
    ├── 9001 (gRPC API)
    └── 9002 (Metrics)
```

---

## 🐳 Step 1: Pulling the Docker Image

### What is Pulling?

Pulling a Docker image means downloading a pre-built package from Docker Hub (a repository of container images). The TTS service image is already built and ready to use - you just need to download it.

### Step-by-Step Pull Instructions

#### Option A: Simple Pull (Recommended)

1. **Open a terminal** (command line window)

2. **Pull the image:**
   ```bash
   docker pull ai4bharat/triton-indo-aryan-tts:latest
   ```
   
   **What this does:**
   - `docker pull` = Download the image
   - `ai4bharat/triton-indo-aryan-tts:latest` = Image name and version tag
   - `latest` = Most recent version
   - `indo-aryan-tts` = Supports Indo-Aryan language family

3. **Wait for it to complete** (this may take 5-20 minutes depending on internet speed)
   - The image is large (several GB) because it contains:
     - Triton Inference Server
     - Python runtime and libraries
     - Pre-trained TTS models
     - HiFiGAN vocoder for audio generation
     - All dependencies

**Expected Output:**
```
latest: Pulling from ai4bharat/triton-indo-aryan-tts
...
Status: Downloaded newer image for ai4bharat/triton-indo-aryan-tts:latest
docker.io/ai4bharat/triton-indo-aryan-tts:latest
```

#### Verify Image is Downloaded

```bash
docker images | grep triton-indo-aryan-tts
```

You should see the image listed with its size.

**Troubleshooting Pull Issues:**
- **"Cannot connect to Docker daemon"**: Run `sudo systemctl start docker`
- **"Network timeout"**: Check internet connection, retry the pull
- **"No space left on device"**: Free up disk space (`docker system prune` to clean old images)

---

## 🚀 Step 2: Running the Service

### What is Running?

Running the service means starting it up so it can accept requests. The service will load the TTS models into GPU memory and be ready to generate speech from text.

### Step-by-Step Run Instructions

#### Option A: Using Docker Run (For Testing)

```bash
docker run -d --gpus all \
  -p 9000:8000 \
  -p 9001:8001 \
  -p 9002:8002 \
  --shm-size=2g \
  --name indo-aryan-tts \
  ai4bharat/triton-indo-aryan-tts:latest \
  tritonserver \
  --model-repository=/home/triton_repo \
  --log-verbose=2 \
  --strict-model-config=false \
  --http-port=8000 \
  --grpc-port=8001 \
  --metrics-port=8002
```

**What each part means:**
- `docker run` = Start a container
- `-d` = Run in background (detached mode)
- `--gpus all` = Use all available GPUs (makes it faster)
- `-p 9000:8000` = Map port 9000 on your computer to port 8000 in container (HTTP)
- `-p 9001:8001` = Map port 9001 for gRPC API
- `-p 9002:8002` = Map port 9002 for metrics
- `--shm-size=2g` = Allocate 2GB shared memory (required for the models)
- `--name indo-aryan-tts` = Name the container "indo-aryan-tts"
- `ai4bharat/triton-indo-aryan-tts:latest` = Use the image we pulled
- `--model-repository=/home/triton_repo` = Model location (note: different from ASR/NMT)
- `--log-verbose=2` = Detailed logging
- `--strict-model-config=false` = More flexible model configuration

#### Option B: Using Docker Compose (Recommended for Production)

Create or use `docker-compose.yml`:

```yaml
version: '3.8'

services:
  indo-aryan-tts:
    image: ai4bharat/triton-indo-aryan-tts:latest
    container_name: indo-aryan-tts
    ports:
      - "9000:8000"  # HTTP API
      - "9001:8001"  # GRPC API
      - "9002:8002"  # Metrics
    command: >
      tritonserver
      --model-repository=/home/triton_repo
      --log-verbose=2
      --strict-model-config=false
      --http-port=8000
      --grpc-port=8001
      --metrics-port=8002
    shm_size: 2gb
    runtime: nvidia
    restart: always
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
```

Then run:
```bash
docker-compose up -d indo-aryan-tts
```

### Understanding Ports

Think of ports like different doors to the same building:
- **Port 9000 (HTTP)**: Main entrance for web requests (REST API)
- **Port 9001 (gRPC)**: Fast lane for program-to-program communication
- **Port 9002 (Metrics)**: Monitoring room for checking service health and performance

**Note**: TTS uses ports 9000-9002. Make sure these ports are not already in use by another service.

---

## ✅ Step 3: Verifying the Service is Running

### Check if Container is Running

```bash
docker ps
```

You should see `indo-aryan-tts` in the list with status "Up".

### Check Service Health

```bash
curl http://localhost:9000/v2/health/ready
```

**Expected Response:** `{"status":"ready"}`

If you see this, the service is ready to accept requests!

### List Available Models

```bash
curl http://localhost:9000/v2/models
```

**Expected Response:**
```json
{
  "models": [
    {
      "name": "tts",
      "platform": "python",
      "versions": ["1"]
    }
  ]
}
```

### View Logs

```bash
docker logs indo-aryan-tts
```

Or follow logs in real-time:
```bash
docker logs -f indo-aryan-tts
```

Look for:
- `"Server is ready to receive inference requests"` = Success!
- `"successfully loaded 'tts' version 1"` = Model loaded successfully
- Any error messages = Something went wrong

---

## 🧪 Step 4: Testing the Service

> **Note**: If you're accessing the service from a remote machine, replace `localhost` with your server's IP address. For example, if your server IP is `192.168.1.100`, use `http://192.168.1.100:9000` instead of `http://localhost:9000`.

### Method 1: Using a Python Script (Recommended)

Create a file `test_tts.py`:

```python
#!/usr/bin/env python3
"""
Script to generate speech from text using TTS model via Triton HTTP API
"""
import json
import requests
import numpy as np
import wave
from pathlib import Path
import sys

def test_tts(text, speaker_id="female", language_id="hi", 
             output_file="tts_output.wav",
             endpoint="http://localhost:9000/v2/models/tts/infer"):
    """
    Send TTS request and save generated audio
    """
    print(f"Generating speech:")
    print(f"  Text: {text}")
    print(f"  Language: {language_id}")
    print(f"  Speaker: {speaker_id}")
    
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
                "name": "INPUT_SPEAKER_ID",
                "shape": [1],
                "datatype": "BYTES",
                "data": [speaker_id]
            },
            {
                "name": "INPUT_LANGUAGE_ID",
                "shape": [1],
                "datatype": "BYTES",
                "data": [language_id]
            }
        ]
    }
    
    print(f"\nSending request to: {endpoint}")
    
    # Send request
    response = requests.post(endpoint, json=payload)
    response.raise_for_status()
    
    result = response.json()
    
    print("\n=== Response received ===")
    
    # Extract audio data
    audio_data = None
    for output in result.get("outputs", []):
        if output.get("name") == "OUTPUT_GENERATED_AUDIO":
            data = output["data"]
            # Handle different data formats
            if isinstance(data[0], list):
                audio_data = np.array(data[0], dtype=np.float32)
            else:
                audio_data = np.array(data, dtype=np.float32)
            break
    
    if audio_data is None:
        print("ERROR: No audio data in response")
        return None
    
    print(f"Audio samples: {len(audio_data)}")
    
    # Sample rate for HiFiGAN vocoder
    sample_rate = 22050
    
    # Convert float32 [-1, 1] to int16
    scaled = np.clip(audio_data, -1, 1)
    int16_audio = (scaled * 32767).astype(np.int16)
    
    # Save as WAV file
    output_path = Path(output_file)
    with wave.open(str(output_path), 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(int16_audio.tobytes())
    
    print(f"\n=== Audio saved ===")
    print(f"Output file: {output_path.absolute()}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {len(audio_data) / sample_rate:.2f} seconds")
    
    return str(output_path.absolute())

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_tts.py <text> [speaker_id] [language_id] [output_file]")
        print("Example: python3 test_tts.py 'Hello world' female hi output.wav")
        sys.exit(1)
    
    text = sys.argv[1]
    speaker_id = sys.argv[2] if len(sys.argv) > 2 else "female"
    language_id = sys.argv[3] if len(sys.argv) > 3 else "hi"
    output_file = sys.argv[4] if len(sys.argv) > 4 else "tts_output.wav"
    
    test_tts(text, speaker_id, language_id, output_file)
```

**Run the script:**
```bash
python3 test_tts.py "नमस्ते, यह एक परीक्षण है" female hi output.wav
```

**Expected Output:**
```
Generating speech:
  Text: नमस्ते, यह एक परीक्षण है
  Language: hi
  Speaker: female

Sending request to: http://localhost:9000/v2/models/tts/infer

=== Response received ===
Audio samples: 48510

=== Audio saved ===
Output file: /path/to/output.wav
Sample rate: 22050 Hz
Duration: 2.20 seconds
```

**Note**: The audio data in the response contains BASE64 content.

### Method 2: Using curl (Manual Testing)

```bash
curl -X POST http://localhost:9000/v2/models/tts/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "INPUT_TEXT",
        "shape": [1],
        "datatype": "BYTES",
        "data": ["Hello world"]
      },
      {
        "name": "INPUT_SPEAKER_ID",
        "shape": [1],
        "datatype": "BYTES",
        "data": ["female"]
      },
      {
        "name": "INPUT_LANGUAGE_ID",
        "shape": [1],
        "datatype": "BYTES",
        "data": ["hi"]
      }
    ]
  }' | python3 -m json.tool | head -50
```

**Note**: The response will contain a large array of audio data. You'll need a script to extract and save it as a WAV file.

### Method 3: Complete Python Example with Error Handling

```python
import requests
import numpy as np
import wave
from pathlib import Path

def generate_speech(text, speaker="female", language="hi", output="speech.wav"):
    """Generate speech and save to file"""
    try:
        url = "http://localhost:9000/v2/models/tts/infer"
        payload = {
            "inputs": [
                {"name": "INPUT_TEXT", "shape": [1], "datatype": "BYTES", "data": [text]},
                {"name": "INPUT_SPEAKER_ID", "shape": [1], "datatype": "BYTES", "data": [speaker]},
                {"name": "INPUT_LANGUAGE_ID", "shape": [1], "datatype": "BYTES", "data": [language]}
            ]
        }
        
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        # Extract audio
        for output in result.get("outputs", []):
            if output.get("name") == "OUTPUT_GENERATED_AUDIO":
                data = output["data"]
                audio = np.array(data[0] if isinstance(data[0], list) else data, dtype=np.float32)
                
                # Convert to int16 and save
                int16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
                with wave.open(output, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(22050)
                    wf.writeframes(int16.tobytes())
                
                return output
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

# Example usage
generate_speech("नमस्ते", "female", "hi", "hello_hindi.wav")
```

---

## 📊 Understanding the API

### Model Information

- **Model Name**: `tts`
- **Type**: Python backend model
- **Backend**: Triton Inference Server with Python backend
- **Max Batch Size**: 0 (batch size not used in current implementation)
- **GPU Required**: Yes (recommended for good performance)

### Input Format

**INPUT_TEXT** (Required)
- **Type**: BYTES (string)
- **Shape**: `[1]` (single text string)
- **Description**: The text to be converted to speech
- **Format**: UTF-8 encoded string in the target language's script
- **Length**: No strict limit, but longer texts take more time

**INPUT_SPEAKER_ID** (Required)
- **Type**: BYTES (string)
- **Shape**: `[1]`
- **Description**: Speaker voice identifier
- **Supported Values**: 
  - `"female"` - Female voice
  - `"male"` - Male voice (if supported for the language)
- **Note**: Available speakers may vary by language

**INPUT_LANGUAGE_ID** (Required)
- **Type**: BYTES (string)
- **Shape**: `[1]`
- **Description**: Language code for the text
- **Supported Languages**: See language list below

### Output Format

**OUTPUT_GENERATED_AUDIO**
- **Type**: FP32 (float32)
- **Shape**: `[-1]` (variable length audio array)
- **Description**: Generated audio waveform
- **Format**: Float32 array with values in range [-1.0, 1.0]
- **Sample Rate**: 22050 Hz
- **Channels**: Mono (single channel)
- **Bit Depth**: 16-bit equivalent (when converted to WAV)

### Supported Languages

The service supports **Indo-Aryan languages**:

- **Hindi** (hi)
- **Bengali** (bn)
- **Gujarati** (gu)
- **Marathi** (mr)
- **Punjabi** (pa)
- **And other Indo-Aryan languages!**

**Note**: Available languages and speakers may vary. Check the model documentation for the complete list.

---

## 🎵 Audio Output Specifications

### Generated Audio Format

- **Sample Rate**: 22050 Hz (standard for HiFiGAN vocoder)
- **Channels**: Mono (1 channel)
- **Format**: Float32 PCM (normalized to [-1, 1])
- **Bit Depth**: When saved as WAV, use 16-bit (standard)

### Converting to WAV File

When saving the output, convert as follows:

```python
# Audio is float32 in range [-1, 1]
audio_float = np.array(response_data, dtype=np.float32)

# Clip to valid range
audio_clipped = np.clip(audio_float, -1, 1)

# Convert to int16 for WAV format
audio_int16 = (audio_clipped * 32767).astype(np.int16)

# Save as WAV
import wave
with wave.open("output.wav", 'wb') as wf:
    wf.setnchannels(1)      # Mono
    wf.setsampwidth(2)      # 16-bit (2 bytes per sample)
    wf.setframerate(22050)  # Sample rate
    wf.writeframes(audio_int16.tobytes())
```

---

## 📝 Text Input Requirements

### Input Text Guidelines

- **Encoding**: UTF-8
- **Script**: Use native script for the language (e.g., Devanagari for Hindi)
- **Format**: Plain text (no special formatting required)
- **Length**: No strict limit, but:
  - Shorter sentences (10-50 words) work best
  - Very long texts may need to be split into sentences
  - Processing time increases with text length

### Tips for Best Results

1. **Use proper punctuation** for natural pauses
2. **Write in native script** for better pronunciation
3. **Use complete sentences** for natural intonation
4. **Avoid special characters** that might not be supported
5. **Check language support** before using
6. **Split long texts** into sentences for better quality

---

## ⚙️ Configuration Options

### Docker Run Options

You can customize the container with additional options:

```bash
docker run -d --gpus all \
  -p 9000:8000 \
  -p 9001:8001 \
  -p 9002:8002 \
  --shm-size=2g \
  --name indo-aryan-tts \
  --restart=always \
  -e NVIDIA_VISIBLE_DEVICES=all \
  ai4bharat/triton-indo-aryan-tts:latest \
  tritonserver \
  --model-repository=/home/triton_repo \
  --log-verbose=2 \
  --strict-model-config=false \
  --http-port=8000 \
  --grpc-port=8001 \
  --metrics-port=8002
```

**Options explained:**
- `--restart=always` = Automatically restart container if it crashes
- `-e NVIDIA_VISIBLE_DEVICES=all` = Use all GPUs (can specify specific GPU like "0" or "0,1")
- `--shm-size=2g` = Shared memory size (2GB required)
- `--model-repository=/home/triton_repo` = **Important**: TTS uses different model path than ASR/NMT

### Resource Allocation

- **GPU Memory**: Model requires several GB of GPU memory
- **System Memory**: At least 4GB RAM recommended
- **Shared Memory**: 2GB minimum (required for model loading)
- **Processing Time**: Typically 1-3 seconds per sentence (on GPU)

### Performance Optimization

- **Use GPU** for significantly faster generation
- **Batch processing** can be done by sending multiple requests
- **Cache frequently used phrases** if applicable to your use case
- **Shorter texts** process faster than very long texts

---

## 🛠️ Troubleshooting

### Problem: Service Won't Start

**Symptoms**: Container exits immediately or shows errors

**Solutions**:
1. **Check logs**: `docker logs indo-aryan-tts`
2. **Verify GPU**: `nvidia-smi` should show your GPU
3. **Check port conflicts**: `sudo lsof -i :9000` (should be empty)
4. **Verify Docker GPU access**: 
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```
5. **Check shared memory**: Ensure `--shm-size=2g` is set
6. **Verify model repository path**: TTS uses `/home/triton_repo` (different from others)

### Problem: "Out of Memory" Errors

**Symptoms**: CUDA out of memory errors in logs

**Solutions**:
1. **Check GPU memory**: `nvidia-smi`
2. **Process shorter texts** at a time
3. **Close other GPU applications**
4. **Restart the container** to clear GPU memory

### Problem: Slow Inference

**Symptoms**: Requests take a long time to process

**Solutions**:
1. **Verify GPU is being used**: Check logs for GPU usage
2. **Check GPU utilization**: `nvidia-smi -l 1` (should show activity)
3. **Use shorter texts** for faster processing
4. **Check if running on CPU** (should use GPU for good performance)

### Problem: Poor Audio Quality

**Symptoms**: Generated speech sounds unnatural or has artifacts

**Solutions**:
1. **Use proper text formatting** (punctuation, spacing)
2. **Check text encoding** (should be UTF-8)
3. **Use supported language** and verify language code
4. **Ensure text is in native script** for the language
5. **Check sample rate** when saving (should be 22050 Hz)

### Problem: No Audio Output

**Symptoms**: Request succeeds but no audio file is created

**Solutions**:
1. **Check response format** - audio is in `OUTPUT_GENERATED_AUDIO` field
2. **Verify data extraction** - audio data is a nested array
3. **Check conversion to WAV** - ensure proper format conversion
4. **Verify file permissions** - ensure you can write to output directory
5. **Check audio array length** - should be non-zero

### Problem: Language or Speaker Not Supported

**Symptoms**: Error about unsupported language or speaker

**Solutions**:
1. **Use correct language codes** from the supported list
2. **Verify speaker ID** - typically "female" or "male"
3. **Check language support** - not all Indo-Aryan languages may be supported
4. **Check model documentation** for exact supported values

### Problem: Cannot Connect to Service

**Symptoms**: Connection refused or timeout errors

**Solutions**:
1. **Verify service is running**: `docker ps`
2. **Check port mapping**: `docker ps` should show port mappings (9000-9002)
3. **Test from container**: 
   ```bash
   docker exec indo-aryan-tts curl http://localhost:8000/v2/health/ready
   ```
4. **Check firewall**: `sudo ufw status`
5. **Verify port 9000 is not in use** by another service

### Problem: Container Keeps Restarting

**Symptoms**: Container status shows "Restarting"

**Solutions**:
1. **Check logs**: `docker logs indo-aryan-tts`
2. **Verify GPU availability**: `nvidia-smi`
3. **Check system resources**: `free -h` and `df -h`
4. **Verify Docker runtime**: Ensure `runtime: nvidia` in docker-compose
5. **Check model repository path**: Ensure `/home/triton_repo` exists in container

---

## 📈 Monitoring and Metrics

### Health Check

```bash
curl http://localhost:9000/v2/health/ready
```

### Metrics Endpoint

```bash
curl http://localhost:9002/metrics
```

This provides Prometheus-compatible metrics including:
- Request count
- Inference latency
- GPU utilization
- Error rates

### View Real-Time Logs

```bash
docker logs -f indo-aryan-tts
```

Press `Ctrl+C` to stop viewing logs.

### Check Container Stats

```bash
docker stats indo-aryan-tts
```

Shows CPU, memory, GPU, and network usage in real-time.

---

## 🔄 Common Operations

### Stop the Service

```bash
docker stop indo-aryan-tts
```

### Start the Service

```bash
docker start indo-aryan-tts
```

### Restart the Service

```bash
docker restart indo-aryan-tts
```

### Remove the Service

```bash
docker stop indo-aryan-tts
docker rm indo-aryan-tts
```

### Update the Service

```bash
# Pull latest image
docker pull ai4bharat/triton-indo-aryan-tts:latest

# Stop and remove old container
docker stop indo-aryan-tts
docker rm indo-aryan-tts

# Start new container (use your preferred method)
docker-compose up -d indo-aryan-tts
```

---

## 📚 Additional Resources

### Service Documentation
- **Docker Hub**: https://hub.docker.com/r/ai4bharat/triton-indo-aryan-tts
- **AI4Bharat Indic-TTS**: https://github.com/AI4Bharat/Indic-TTS
- **AI4Bharat**: https://ai4bharat.iitm.ac.in/

### Technical References
- **Triton Server Docs**: https://docs.nvidia.com/deeplearning/triton-inference-server/
- **Docker Docs**: https://docs.docker.com/
- **HiFiGAN Vocoder**: https://github.com/jik876/hifi-gan

### Getting Help

If you encounter issues:
1. Check the logs: `docker logs indo-aryan-tts`
2. Review this guide's troubleshooting section
3. Check AI4Bharat GitHub repositories
4. Review Indic-TTS documentation
5. Review Triton Server documentation

---

## 📝 Quick Reference

### Essential Commands

```bash
# Pull image
docker pull ai4bharat/triton-indo-aryan-tts:latest

# Run with docker-compose (recommended)
docker-compose up -d indo-aryan-tts

# Or run directly
docker run -d --gpus all -p 9000:8000 -p 9001:8001 -p 9002:8002 \
  --shm-size=2g --name indo-aryan-tts \
  ai4bharat/triton-indo-aryan-tts:latest \
  tritonserver \
  --model-repository=/home/triton_repo \
  --log-verbose=2 \
  --strict-model-config=false \
  --http-port=8000 \
  --grpc-port=8001 \
  --metrics-port=8002

# Check status
docker ps
curl http://localhost:9000/v2/health/ready

# View logs
docker logs -f indo-aryan-tts

# Stop
docker stop indo-aryan-tts
```

### Port Information

- **HTTP API**: `http://localhost:9000`
- **gRPC API**: `localhost:9001`
- **Metrics**: `http://localhost:9002/metrics`

### Model Information

- **Model Name**: `tts`
- **Type**: Python backend
- **GPU Required**: Yes (recommended)
- **Shared Memory**: 2GB minimum
- **Model Repository**: `/home/triton_repo` (different from ASR/NMT)
- **Sample Rate**: 22050 Hz

### Quick Test

```bash
# Using Python script
python3 test_tts.py "नमस्ते" female hi output.wav

# Play the generated audio (if audio player is available)
aplay output.wav  # Linux
# or
afplay output.wav  # macOS
```

---

## ✅ Summary

You've learned how to:
1. ✅ Pull the TTS Docker image from Docker Hub
2. ✅ Run the TTS service container
3. ✅ Verify it's working correctly
4. ✅ Generate speech from text
5. ✅ Use the API for text-to-speech conversion
6. ✅ Save generated audio as WAV files
7. ✅ Troubleshoot common issues

The TTS-triton service is now ready to convert text to natural-sounding speech! For production use, consider setting up monitoring, load balancing, and proper security measures.

