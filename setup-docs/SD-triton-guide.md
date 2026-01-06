# SD-triton Service Guide

## 📖 What is SD-triton?

**SD** stands for **Speaker Diarization**. This service can analyze audio recordings and identify **"who spoke when"** - it segments the audio by different speakers and creates a timeline showing when each person was speaking.

### Real-World Use Cases
- **Meeting Transcription**: Identify who said what in meetings
- **Call Centers**: Track which agent and customer spoke when
- **Podcast Production**: Automatically segment by speakers
- **Interview Analysis**: Separate interviewer and interviewee speech
- **Court Recordings**: Identify different speakers in legal proceedings

---

## 🎯 What You Need Before Starting

### For Everyone (Non-Technical)

Before you can use this service, you need:
1. **A computer with Linux** (Ubuntu recommended)
2. **An NVIDIA graphics card** (GPU) - This makes the service run much faster
3. **Internet connection** - To download the necessary software and models
4. **HuggingFace account** - This service requires special access (see Authentication section)

### For Technical Users

**System Requirements:**
- **OS**: Linux (Ubuntu 20.04+)
- **GPU**: NVIDIA GPU with CUDA support
- **Docker**: Version 20.10+
- **Docker Compose**: Version 1.29+
- **NVIDIA Container Toolkit**: For GPU access in Docker
- **HuggingFace Account**: Required for model access
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

## 🔐 Authentication Setup (IMPORTANT!)

This service uses **gated models** on HuggingFace, which means you need special access.

### Step 1: Create HuggingFace Account

1. Go to: https://huggingface.co/join
2. Create a free account
3. Verify your email

### Step 2: Accept User Conditions

You need to accept conditions for TWO models:

1. **Speaker Diarization Model**:
   - Visit: https://huggingface.co/pyannote/speaker-diarization
   - Click **"Agree and Access Repository"**

2. **Segmentation Model**:
   - Visit: https://huggingface.co/pyannote/segmentation
   - Click **"Agree and Access Repository"**

### Step 3: Create Access Token

1. Go to: https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Name it (e.g., "SD-triton")
4. Select **"Read"** permissions
5. Click **"Generate token"**
6. **Copy the token** (starts with `hf_...`) - You won't see it again!

### Step 4: Set Environment Variable

```bash
export HUGGING_FACE_HUB_TOKEN="hf_your_token_here"
```

**Important**: You need to set this before building and running the service!

### Step 5: Verify Access

```bash
python3 -c "from huggingface_hub import login; login(token='hf_your_token_here'); from pyannote.audio import Pipeline; Pipeline.from_pretrained('pyannote/speaker-diarization@2.1', use_auth_token='hf_your_token_here')"
```

If successful, you should see "Login successful" and the pipeline will download.

---

## 🏗️ Understanding the Service Structure

The service uses pyannote.audio, a powerful speaker diarization toolkit.

```
SD-triton/
├── Dockerfile              # Recipe to build the container
├── README.md               # Documentation
├── test_client.py          # Test script
└── model_repository/       # Model storage
    └── speaker_diarization/
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

1. **Set your HuggingFace token** (if not already set):
   ```bash
   export HUGGING_FACE_HUB_TOKEN="hf_your_token_here"
   ```

2. **Open a terminal** (command line window)

3. **Navigate to the SD-triton folder:**
   ```bash
   cd SD-triton
   ```

4. **Build the image:**
   ```bash
   docker build --build-arg HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
     -t sd-triton:latest .
   ```
   
   **What this does:**
   - `docker build` = Start building
   - `--build-arg HUGGING_FACE_HUB_TOKEN=...` = Pass token to build process
   - `-t sd-triton:latest` = Name the image "sd-triton" with tag "latest"
   - `.` = Use the current directory (where Dockerfile is located)

5. **Wait for it to complete** (this may take 15-30 minutes the first time)
   - Downloads the base Triton server
   - Installs Python packages (PyTorch, pyannote.audio, etc.)
   - Downloads the speaker diarization models (requires authentication)
   - Sets everything up

#### Option B: Understanding What Happens During Build

The Dockerfile does these steps:
1. **Starts with Triton Server base image** - Pre-configured server
2. **Installs system dependencies** - Audio libraries (libsndfile1, ffmpeg)
3. **Installs Python packages** - PyTorch, torchaudio, pyannote.audio
4. **Copies the model code** - Your custom processing logic
5. **Pre-downloads the model** - Gets the pyannote speaker diarization model ready (requires token)
   - Model source: [https://huggingface.co/pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization)
   - Also requires: [https://huggingface.co/pyannote/segmentation](https://huggingface.co/pyannote/segmentation)

**Note**: If the token is not provided during build, the model will download on first run instead.

**Expected Output:**
```
Step 1/8 : FROM nvcr.io/nvidia/tritonserver:24.01-py3
...
Authenticated with HuggingFace
Downloading pyannote speaker diarization model...
Model downloaded successfully
...
Successfully built abc123def456
Successfully tagged sd-triton:latest
```

**Troubleshooting Build Issues:**
- **"Cannot connect to Docker daemon"**: Run `sudo systemctl start docker`
- **"No space left on device"**: Free up disk space (models are ~2GB)
- **"Authentication failed"**: Verify your HuggingFace token is correct
- **"Model access denied"**: Make sure you accepted user conditions for both models
- **"Network timeout"**: Check internet connection, the build downloads large files

---

## 📥 How the Speaker Diarization Model Was Obtained from HuggingFace

### Model Source

The SD-triton service uses the **pyannote.audio** speaker diarization models from HuggingFace:
- **Main Model**: [https://huggingface.co/pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization)
- **Segmentation Model**: [https://huggingface.co/pyannote/segmentation](https://huggingface.co/pyannote/segmentation)
- **Model Type**: End-to-End Speaker Diarization
- **Library**: pyannote.audio
- **Access**: Gated models (requires HuggingFace account and accepting user conditions)

### About the Model

The pyannote speaker diarization system:

- **Identifies "who spoke when"** in audio recordings:
  - Segments audio by different speakers
  - Creates a timeline showing when each person was speaking
  - Handles overlapping speech
  - Automatically determines number of speakers (or accepts manual input)

- **Uses end-to-end architecture**:
  - Voice Activity Detection (VAD)
  - Speaker Embedding Extraction
  - Clustering for speaker identification
  - Overlap-aware resegmentation

- **Performance**:
  - **Real-time Factor**: ~2.5% (processes 1 hour of audio in ~1.5 minutes)
  - **Accuracy**: DER (Diarization Error Rate) ranges from ~8% to ~32% depending on audio quality
  - **Features**:
    - End-to-end speaker segmentation
    - Overlap-aware resegmentation
    - Automatic voice activity detection
    - No manual number of speakers required (though it can be provided)

### How the Model is Downloaded

During the Docker build or first run, the models are downloaded from HuggingFace:

```python
from pyannote.audio import Pipeline

# Downloads pyannote speaker diarization pipeline
pipeline = Pipeline.from_pretrained(
    'pyannote/speaker-diarization@2.1',
    use_auth_token='hf_your_token_here'
)
```

**What happens:**
1. HuggingFace Hub authenticates using your token
2. Downloads two models:
   - `pyannote/speaker-diarization` - Main diarization pipeline
   - `pyannote/segmentation` - Segmentation model (used by the pipeline)
3. Models are cached locally
4. Models are ready to use for inference

**Note**: These are **gated models**, which means:
- You need a HuggingFace account
- You must accept user conditions for BOTH models:
  - [pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization)
  - [pyannote/segmentation](https://huggingface.co/pyannote/segmentation)
- You need a HuggingFace access token with "Read" permissions

### Model Architecture

The system uses:
- **Voice Activity Detection**: Identifies when speech is present
- **Speaker Embedding**: Extracts features representing each speaker's voice
- **Clustering**: Groups similar voice embeddings (each cluster = one speaker)
- **Segmentation**: Creates timeline of when each speaker was active
- **Overlap Handling**: Detects and handles overlapping speech

### Research Paper

- **Paper**: [End-to-end speaker segmentation for overlap-aware resegmentation](https://arxiv.org/abs/2106.04624)
- **Authors**: Hervé Bredin, Antoine Laurent

### References

- **Main Model**: [https://huggingface.co/pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization)
- **Segmentation Model**: [https://huggingface.co/pyannote/segmentation](https://huggingface.co/pyannote/segmentation)
- **pyannote.audio Library**: [https://github.com/pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio)
- **Research Paper**: [https://arxiv.org/abs/2106.04624](https://arxiv.org/abs/2106.04624)

---

## 🚀 Step 2: Running the Service

### What is Running?

Running the service means starting it up so it can accept requests. It's like opening a restaurant for business.

### Step-by-Step Run Instructions

#### Basic Run (For Testing)

**Important**: Make sure your HuggingFace token is set!

```bash
export HUGGING_FACE_HUB_TOKEN="hf_your_token_here"

docker run --gpus all \
  -p 8700:8000 \
  -p 8701:8001 \
  -p 8702:8002 \
  -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
  --name sd-server \
  sd-triton:latest
```

**What each part means:**
- `docker run` = Start a container
- `--gpus all` = Use all available GPUs (makes it faster)
- `-p 8700:8000` = Map port 8700 on your computer to port 8000 in container
- `-p 8701:8001` = Map gRPC port
- `-p 8702:8002` = Map metrics port
- `-e HUGGING_FACE_HUB_TOKEN=...` = Pass the token to the container
- `--name sd-server` = Name the container "sd-server"
- `sd-triton:latest` = Use the image we built

#### Run in Background (Recommended for Production)

```bash
export HUGGING_FACE_HUB_TOKEN="hf_your_token_here"

docker run -d --gpus all \
  -p 8700:8000 \
  -p 8701:8001 \
  -p 8702:8002 \
  -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
  --name sd-server \
  sd-triton:latest
```

The `-d` flag runs it in the background (detached mode).

#### Using Docker Compose (Easiest Method)

If you're using the main docker-compose.yml file:

```bash
export HUGGING_FACE_HUB_TOKEN="hf_your_token_here"
docker-compose up -d sd-server
```

This automatically handles all the configuration.

### Understanding Ports

- **Port 8700 (HTTP)**: Main entrance for web requests
- **Port 8701 (gRPC)**: Fast lane for program-to-program communication
- **Port 8702 (Metrics)**: Monitoring room for checking service health

---

## ✅ Step 3: Verifying the Service is Running

### Check if Container is Running

```bash
docker ps
```

You should see `sd-server` in the list with status "Up".

### Check Service Health

```bash
curl http://localhost:8700/v2/health/ready
```

**Expected Response:** `{"status":"ready"}`

If you see this, the service is ready to accept requests!

### List Available Models

```bash
curl http://localhost:8700/v2/models
```

**Expected Response:**
```json
{
  "models": [
    {
      "name": "speaker_diarization",
      "platform": "python",
      "versions": ["1"]
    }
  ]
}
```

### View Logs

```bash
docker logs sd-server
```

Look for:
- `"Server is ready to receive inference requests"` = Success!
- `"Authentication failed"` = Token issue, check your HuggingFace token
- Any error messages = Something went wrong

---

## 🧪 Step 4: Testing the Service

> **Note**: If you're accessing the service from a remote machine, replace `localhost` with your server's IP address. For example, if your server IP is `192.168.1.100`, use `http://192.168.1.100:8700` instead of `http://localhost:8700`.

### Method 1: Manual Testing with curl

#### Step 1: Prepare Your Audio File

Convert your audio to base64:
```bash
AUDIO_B64=$(base64 -w 0 your_audio.wav)
```

#### Step 2: Send Request (Auto-detect speakers)

```bash
curl -X POST http://localhost:8700/v2/models/speaker_diarization/infer \
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
        \"name\": \"NUM_SPEAKERS\",
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

#### Step 3: Send Request (Specify number of speakers)

```bash
curl -X POST http://localhost:8700/v2/models/speaker_diarization/infer \
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
        \"name\": \"NUM_SPEAKERS\",
        \"shape\": [1, 1],
        \"datatype\": \"BYTES\",
        \"data\": [[\"2\"]]
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
      "data": ["{\"total_segments\": 5, \"num_speakers\": 2, \"speakers\": [\"SPEAKER_00\", \"SPEAKER_01\"], \"segments\": [{\"start_time\": 0.5, \"end_time\": 2.1, \"duration\": 1.6, \"speaker\": \"SPEAKER_00\"}, ...]}"]
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
import sys

def infer_speaker_diarization(audio_file_path, num_speakers=None):
    # Read audio file
    with open(audio_file_path, "rb") as f:
        audio_bytes = f.read()
    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

    # Prepare request
    url = "http://localhost:8700/v2/models/speaker_diarization/infer"
    inputs = [
        {
            "name": "AUDIO_DATA",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [[audio_b64]]
        },
        {
            "name": "NUM_SPEAKERS",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [[str(num_speakers) if num_speakers else ""]]
        }
    ]

    payload = {
        "inputs": inputs,
        "outputs": [
            {"name": "DIARIZATION_RESULT"}
        ]
    }

    # Send request
    response = requests.post(url, json=payload)
    response.raise_for_status()
    result = response.json()
    
    # Parse and display results
    diarization_output = json.loads(result["outputs"][0]["data"][0])
    
    print(f"Total segments: {diarization_output['total_segments']}")
    print(f"Number of speakers: {diarization_output['num_speakers']}")
    print(f"Speakers: {diarization_output['speakers']}")
    print("\nSegments:")
    for segment in diarization_output['segments']:
        print(f"  [{segment['start_time']:.2f}s - {segment['end_time']:.2f}s]: "
              f"{segment['speaker']} (duration: {segment['duration']:.2f}s)")

if __name__ == "__main__":
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "audio.wav"
    num_speakers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    infer_speaker_diarization(audio_file, num_speakers)
```

Run it:
```bash
python3 test_my_audio.py your_audio.wav
# Or with specific number of speakers:
python3 test_my_audio.py your_audio.wav 2
```

**Expected Output:**
```
Total segments: 5
Number of speakers: 2
Speakers: ['SPEAKER_00', 'SPEAKER_01']

Segments:
  [0.50s - 2.10s]: SPEAKER_00 (duration: 1.60s)
  [2.50s - 4.00s]: SPEAKER_01 (duration: 1.50s)
  [4.20s - 6.50s]: SPEAKER_00 (duration: 2.30s)
  [6.80s - 8.20s]: SPEAKER_01 (duration: 1.40s)
  [8.50s - 10.00s]: SPEAKER_00 (duration: 1.50s)
```

---

## 📊 Understanding the API

### Input Format

1. **AUDIO_DATA** (Required)
   - **Type**: Base64-encoded string
   - **Format**: WAV, MP3, FLAC, or other audio formats
   - **What to send**: Your audio file converted to base64

2. **NUM_SPEAKERS** (Optional)
   - **Type**: String (number as string)
   - **Format**: Number of speakers (e.g., "2", "3")
   - **What to send**: 
     - Empty string `""` = Automatically detect number of speakers
     - Specific number like `"2"` = Expect exactly 2 speakers

### Output Format

The service returns a JSON string with:

```json
{
  "total_segments": 5,
  "num_speakers": 2,
  "speakers": ["SPEAKER_00", "SPEAKER_01"],
  "segments": [
    {
      "start_time": 0.5,
      "end_time": 2.1,
      "duration": 1.6,
      "speaker": "SPEAKER_00"
    },
    {
      "start_time": 2.5,
      "end_time": 4.0,
      "duration": 1.5,
      "speaker": "SPEAKER_01"
    }
  ]
}
```

**Fields:**
- **total_segments**: Number of speaker segments found
- **num_speakers**: Number of different speakers detected
- **speakers**: List of speaker labels (SPEAKER_00, SPEAKER_01, etc.)
- **segments**: Array of segments with:
  - **start_time**: When the segment starts (in seconds)
  - **end_time**: When the segment ends (in seconds)
  - **duration**: Length of the segment (in seconds)
  - **speaker**: Which speaker was speaking (SPEAKER_00, SPEAKER_01, etc.)

---

## 🧠 How It Works (Technical Details)

### Diarization Process

1. **Voice Activity Detection**:
   - Identifies when speech is present (filters out silence)
   - Removes non-speech segments

2. **Speaker Embedding**:
   - Extracts features that represent each speaker's voice
   - Creates embeddings for different voice characteristics

3. **Clustering**:
   - Groups similar voice embeddings together
   - Each cluster represents a different speaker
   - Automatically determines number of speakers (if not specified)

4. **Segmentation**:
   - Creates timeline of when each speaker was active
   - Handles overlapping speech
   - Produces continuous segments

### Performance Characteristics

- **Real-time Factor**: ~2.5% (processes 1 hour of audio in ~1.5 minutes)
- **Accuracy**: DER (Diarization Error Rate) ranges from ~8% to ~32% depending on audio quality
- **Features**:
  - End-to-end speaker segmentation
  - Overlap-aware resegmentation
  - Automatic voice activity detection
  - No manual number of speakers required (though it can be provided)

---

## ⚙️ Configuration Options

### Adjusting Performance

You can modify `model_repository/speaker_diarization/config.pbtxt` to change:

1. **Max Batch Size**: How many requests to process together
   - Current: 16
   - Increase for more throughput (needs more GPU memory)
   - Decrease if running out of memory

2. **Dynamic Batching**: Automatically groups requests
   - Current: Enabled with sizes [1, 2, 4, 8, 16]
   - Adjust based on your workload

3. **GPU Instances**: Number of model copies
   - Current: 1
   - Increase for higher throughput (uses more GPU memory)

**After changing config.pbtxt, rebuild the image:**
```bash
export HUGGING_FACE_HUB_TOKEN="hf_your_token_here"
docker build --build-arg HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
  -t sd-triton:latest .
docker stop sd-server
docker rm sd-server
docker run -d --gpus all -p 8700:8000 -p 8701:8001 -p 8702:8002 \
  -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
  --name sd-server sd-triton:latest
```

---

## 🛠️ Troubleshooting

### Problem: Service Won't Start

**Symptoms**: Container exits immediately or shows errors

**Solutions**:
1. **Check logs**: `docker logs sd-server`
2. **Verify GPU**: `nvidia-smi` should show your GPU
3. **Check port conflicts**: `sudo lsof -i :8700` (should be empty)
4. **Verify Docker GPU access**: 
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

### Problem: Authentication Errors

**Symptoms**: "Authentication failed" or "Model access denied"

**Solutions**:
1. **Verify token is set**: `echo $HUGGING_FACE_HUB_TOKEN`
2. **Check token is correct**: Should start with `hf_`
3. **Accept user conditions**: Visit both model pages and accept conditions
4. **Regenerate token** if expired
5. **Set token in container**: Make sure `-e HUGGING_FACE_HUB_TOKEN=...` is used

### Problem: Model Download Fails

**Symptoms**: Errors downloading model during build or runtime

**Solutions**:
1. **Check internet connection**
2. **Verify model access**: Make sure you accepted conditions for both models
3. **Check token**: Ensure token has "Read" permissions
4. **Retry**: Network issues can be temporary

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
3. **Note**: Processing time scales with audio duration (real-time factor ~2.5%)
4. **Use shorter audio files** for faster processing

### Problem: Wrong Number of Speakers

**Symptoms**: Service detects incorrect number of speakers

**Solutions**:
1. **Specify number of speakers** if you know it
2. **Use clear audio** (reduce background noise)
3. **Ensure distinct speakers** (similar voices may be confused)
4. **Review segments**: Check if segmentation makes sense

### Problem: Cannot Connect to Service

**Symptoms**: Connection refused or timeout errors

**Solutions**:
1. **Verify service is running**: `docker ps`
2. **Check port mapping**: `docker ps` should show port mappings
3. **Test from container**: 
   ```bash
   docker exec sd-server curl http://localhost:8000/v2/health/ready
   ```
4. **Check firewall**: `sudo ufw status`

---

## 📈 Monitoring and Metrics

### Health Check

```bash
curl http://localhost:8700/v2/health/ready
```

### Metrics Endpoint

```bash
curl http://localhost:8702/metrics
```

This provides Prometheus-compatible metrics including:
- Request count
- Inference latency
- GPU utilization
- Error rates

### View Real-Time Logs

```bash
docker logs -f sd-server
```

Press `Ctrl+C` to stop viewing logs.

---

## 🔄 Common Operations

### Stop the Service

```bash
docker stop sd-server
```

### Start the Service

```bash
docker start sd-server
```

### Restart the Service

```bash
docker restart sd-server
```

### Remove the Service

```bash
docker stop sd-server
docker rm sd-server
```

### Update the Service

```bash
# Rebuild with latest changes
export HUGGING_FACE_HUB_TOKEN="hf_your_token_here"
docker build --build-arg HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
  -t sd-triton:latest .

# Stop and remove old container
docker stop sd-server
docker rm sd-server

# Start new container
docker run -d --gpus all -p 8700:8000 -p 8701:8001 -p 8702:8002 \
  -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
  --name sd-server sd-triton:latest
```

---

## 📚 Additional Resources

### Service Documentation
- **Detailed README**: `SD-triton/README.md`
- **Model Source (HuggingFace)**: [https://huggingface.co/pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization)
  - This is where the speaker diarization model was obtained from
- **Segmentation Model**: [https://huggingface.co/pyannote/segmentation](https://huggingface.co/pyannote/segmentation)
- **pyannote.audio**: [https://github.com/pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio)
- **Paper**: [End-to-end speaker segmentation for overlap-aware resegmentation](https://arxiv.org/abs/2106.04624)

### Technical References
- **Triton Server Docs**: https://docs.nvidia.com/deeplearning/triton-inference-server/
- **Docker Docs**: https://docs.docker.com/

### Getting Help

If you encounter issues:
1. Check the logs: `docker logs sd-server`
2. Review this guide's troubleshooting section
3. Check the service README: `SD-triton/README.md`
4. Review pyannote.audio documentation

---

## 📝 Quick Reference

### Essential Commands

```bash
# Set token (required!)
export HUGGING_FACE_HUB_TOKEN="hf_your_token_here"

# Build
cd SD-triton
docker build --build-arg HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
  -t sd-triton:latest .

# Run
docker run -d --gpus all -p 8700:8000 -p 8701:8001 -p 8702:8002 \
  -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
  --name sd-server sd-triton:latest

# Check status
docker ps
curl http://localhost:8700/v2/health/ready

# Test
cd SD-triton
python3 test_client.py audio.wav

# View logs
docker logs -f sd-server

# Stop
docker stop sd-server
```

### Port Information

- **HTTP API**: `http://localhost:8700`
- **gRPC API**: `localhost:8701`
- **Metrics**: `http://localhost:8702/metrics`

### Model Information

- **Model Name**: `speaker_diarization`
- **Backend**: Python
- **Max Batch Size**: 16
- **GPU Required**: Yes
- **Authentication Required**: Yes (HuggingFace)
- **Real-time Factor**: ~2.5%

---

## ✅ Summary

You've learned how to:
1. ✅ Set up HuggingFace authentication
2. ✅ Build the SD-triton Docker image
3. ✅ Run the service with authentication
4. ✅ Verify it's working
5. ✅ Test speaker diarization
6. ✅ Use the API (auto-detect and specify speakers)
7. ✅ Understand how diarization works
8. ✅ Troubleshoot common issues

The SD-triton service is now ready to identify speakers in audio! For production use, consider setting up monitoring, load balancing, and proper security measures.



