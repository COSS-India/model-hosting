# Speaker Diarization Triton Deployment

This directory contains the Triton Inference Server deployment for **pyannote/speaker-diarization**, which performs automatic speaker diarization on audio.

## 📋 Overview

- **Model**: pyannote/speaker-diarization@2.1
- **Task**: Speaker Diarization (identifying "who spoke when")
- **Framework**: pyannote.audio
- **Backend**: Python (Triton)
- **Max Speakers**: Automatically detected, or can be specified (min/max bounds)
- **Port**: 8700 (HTTP), 8701 (gRPC), 8702 (Metrics)

## 🔐 Authentication

This model requires authentication with HuggingFace. You need to:

1. **Accept User Conditions**:
   - Visit: https://huggingface.co/pyannote/speaker-diarization
   - Visit: https://huggingface.co/pyannote/segmentation
   - Accept the user conditions on both pages.

2. **Create Access Token**:
   - Go to: https://huggingface.co/settings/tokens
   - Create a new token with "Read" permissions.
   - Copy the token (starts with `hf_...`).

3. **Set Token as Environment Variable**:
   ```bash
   export HUGGING_FACE_HUB_TOKEN="hf_your_token_here"
   ```
   This environment variable must be set when building and running the Docker container.

## 🏗️ Directory Structure

```
SD-triton/
├── Dockerfile                          # Docker build configuration
├── README.md                           # This file
├── test_client.py                      # Test client for inference
└── model_repository/
    └── speaker_diarization/            # Model name
        ├── config.pbtxt                # Triton model configuration
        └── 1/                          # Model version
            └── model.py                # Python backend implementation
```

## 🚀 Quick Start

### Build the Docker Image

```bash
cd /home/ubuntu/incubalm/SD-triton

# Set your HuggingFace token (required for model download during build)
export HUGGING_FACE_HUB_TOKEN="hf_your_token_here"

# Build the image
docker build -t sd-triton:latest .
```

### Run the Triton Server

```bash
# Run with GPU support and expose on port 8700
docker run --gpus all -p 8700:8000 -p 8701:8001 -p 8702:8002 \
  -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
  --name sd-triton-server \
  sd-triton:latest

# Or run in detached mode
docker run -d --gpus all -p 8700:8000 -p 8701:8001 -p 8702:8002 \
  -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
  --name sd-triton-server \
  sd-triton:latest
```

### Check Server Health

```bash
# Check if server is ready
curl http://localhost:8700/v2/health/ready

# List available models
curl http://localhost:8700/v2/models
```

## 📊 Input/Output Format

### Triton API Format

**Input Parameters:**

1. **AUDIO_DATA** (Required)
   - Type: `TYPE_STRING` (BYTES)
   - Shape: `[1, 1]`
   - Format: Base64 encoded audio data (e.g., WAV, MP3)

2. **NUM_SPEAKERS** (Optional)
   - Type: `TYPE_STRING` (BYTES)
   - Shape: `[1, 1]`
   - Format: Number of speakers as string (e.g., "2", "3")
   - If empty or not provided, the model automatically detects the number of speakers

**Output:**

- **DIARIZATION_RESULT**
  - Type: `TYPE_STRING` (BYTES)
  - Shape: `[1, 1]`
  - Format: JSON string containing diarization results.
  - Structure:
    ```json
    {
      "total_segments": 5,
      "num_speakers": 2,
      "speakers": ["SPEAKER_00", "SPEAKER_01"],
      "segments": [
        {"start_time": 0.5, "end_time": 2.1, "duration": 1.6, "speaker": "SPEAKER_00"},
        {"start_time": 2.5, "end_time": 4.0, "duration": 1.5, "speaker": "SPEAKER_01"},
        // ... more segments
      ]
    }
    ```

### Example Request (Python)

```python
import requests
import json
import base64

def infer_speaker_diarization(audio_file_path, num_speakers=None, server_url="http://localhost:8700"):
    with open(audio_file_path, "rb") as f:
        audio_bytes = f.read()
    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

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

    response = requests.post(f"{server_url}/v2/models/speaker_diarization/infer", json=payload)
    response.raise_for_status()
    result = response.json()
    diarization_output = json.loads(result["outputs"][0]["data"][0])
    return diarization_output

if __name__ == "__main__":
    results = infer_speaker_diarization("audio.wav")
    print(json.dumps(results, indent=2))
```

## 🧪 Testing

### Using the Test Client

```bash
# Basic usage (auto-detect speakers)
python3 test_client.py audio.wav

# With specific number of speakers
python3 test_client.py audio.wav --num-speakers 2

# Pretty print JSON output
python3 test_client.py audio.wav --pretty
```

## ⚙️ Configuration

### Model Configuration (config.pbtxt)

- **Backend**: Python
- **Max Batch Size**: 16
- **Instance Group**: 1 GPU instance
- **Dynamic Batching**: Enabled with preferred batch sizes [1, 2, 4, 8, 16]

### Performance

- **Real-time Factor**: Around 2.5% using one Nvidia Tesla V100 SXM2 GPU (for neural inference) and one Intel Cascade Lake 6248 CPU (for clustering). This means ~1.5 minutes to process a one-hour conversation.
- **Accuracy**: Benchmarked on various datasets (e.g., AISHELL-4, AMI, CALLHOME) with DERs ranging from ~8% to ~32%.
- **Features**:
  - End-to-end speaker segmentation
  - Overlap-aware resegmentation
  - Automatic voice activity detection
  - No manual number of speakers required (though it can be provided)

## 🎵 Audio Format Requirements

- **Format**: WAV, MP3, FLAC, or other formats supported by `pyannote.audio`.
- **Sample Rate**: Any (automatically handled by `pyannote.audio` which resamples to 16kHz internally).
- **Channels**: Mono or Stereo (automatically converted to mono).

## 📝 Notes

- The model automatically detects the number of speakers if not specified.
- Processing time scales with audio duration (real-time factor ~2.5%).
- The model handles overlapped speech and voice activity detection automatically.
- GPU is recommended for faster processing.

## 📚 References

- **Model**: https://huggingface.co/pyannote/speaker-diarization
- **Paper**: [End-to-end speaker segmentation for overlap-aware resegmentation](https://arxiv.org/abs/2106.04624)
- **pyannote.audio**: https://github.com/pyannote/pyannote-audio
- **Triton Documentation**: https://docs.nvidia.com/deeplearning/triton-inference-server/

## 📄 License

The pyannote/speaker-diarization model is released under the MIT License.












