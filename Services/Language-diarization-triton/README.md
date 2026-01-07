# Language Diarization Triton Deployment

This directory contains the Triton Inference Server deployment for **W2V-E2E-Language-Diarization**, which performs end-to-end spoken language diarization on audio.

## 📋 Overview

- **Model**: Language Diarization using Wav2Vec-based language identification
- **Task**: Spoken Language Diarization (segmenting audio by language)
- **Base Model**: speechbrain/lang-id-voxlingua107-ecapa
- **Languages**: 107 languages supported
- **Backend**: Python (Triton)
- **Framework**: PyTorch + SpeechBrain

## 🏗️ Directory Structure

```
Language-diarization-triton/
├── Dockerfile                          # Docker build configuration
├── README.md                           # This file
├── test_client.py                      # Test client script
└── model_repository/
    └── lang_diarization/              # Model name
        ├── config.pbtxt               # Triton model configuration
        └── 1/                         # Model version
            └── model.py               # Python backend implementation
```

## 🚀 Quick Start

### Build the Docker Image

```bash
cd /home/ubuntu/incubalm/Language-diarization-triton

# Build the image
docker build -t lang-diarization-triton:latest .
```

### Run the Triton Server

```bash
# Run with GPU support on port 8600
docker run --gpus all -p 8600:8000 -p 8601:8001 -p 8602:8002 \
  --name lang-diarization-triton \
  lang-diarization-triton:latest

# Or run in detached mode
docker run -d --gpus all -p 8600:8000 -p 8601:8001 -p 8602:8002 \
  --name lang-diarization-triton \
  lang-diarization-triton:latest
```

### Check Server Health

```bash
# Check if server is ready
curl http://localhost:8600/v2/health/ready

# List available models
curl http://localhost:8600/v2/models

# Get model metadata
curl http://localhost:8600/v2/models/lang_diarization
```

## 📊 Input/Output Format

### Triton API Format

**Input Parameters:**

1. **AUDIO_DATA** (Required)
   - Type: `TYPE_STRING` (BYTES)
   - Shape: `[1, 1]`
   - Format: Base64-encoded audio data (WAV format recommended)
   - Sample Rate: Automatically resampled to 16kHz

2. **LANGUAGE** (Optional)
   - Type: `TYPE_STRING` (BYTES)
   - Shape: `[1, 1]`
   - Format: Target language code (e.g., "ta", "gu", "te", "hi")
   - If empty or not provided, detects all languages in the audio

**Output:**

1. **DIARIZATION_RESULT**
   - Type: `TYPE_STRING` (BYTES)
   - Shape: `[1, 1]`
   - Format: JSON string containing diarization results
   - Structure:
     ```json
     {
       "total_segments": 5,
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

### Example Request (HTTP)

```bash
# First, encode your audio file to base64
AUDIO_B64=$(base64 -w 0 your_audio.wav)

# Diarization for all languages
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

# Diarization for specific language (e.g., Tamil)
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

## ⚙️ Configuration

### Model Configuration (config.pbtxt)

- **Backend**: Python
- **Max Batch Size**: 32
- **Instance Group**: 1 GPU instance
- **Dynamic Batching**: Enabled with preferred batch sizes [1, 2, 4, 8, 16, 32]

### Diarization Parameters

- **Segment Duration**: 2.0 seconds per segment
- **Overlap**: 0.5 seconds between segments
- **Minimum Segment Duration**: 1.0 second

These parameters can be adjusted in `model.py` if needed.

## 🎵 Audio Format Requirements

- **Format**: WAV, MP3, FLAC, or other formats supported by torchaudio
- **Sample Rate**: Any (automatically resampled to 16kHz by the model)
- **Channels**: Mono or Stereo (automatically converted to mono)
- **Duration**: No strict limit, but longer audio will produce more segments

## 🧪 Testing

### Python Test Client

```python
import requests
import json
import base64

# Read and encode audio file
with open("test_audio.wav", "rb") as f:
    audio_bytes = f.read()
    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

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

response = requests.post(url, json=payload)
result = response.json()

# Parse the output
diarization_result = json.loads(result["outputs"][0]["data"][0])

print(f"Total segments: {diarization_result['total_segments']}")
for segment in diarization_result['segments']:
    print(f"  [{segment['start_time']:.2f}s - {segment['end_time']:.2f}s]: {segment['language']} (confidence: {segment['confidence']:.4f})")
```

Run the test client:
```bash
python3 test_client.py
```

## 📝 How It Works

1. **Audio Segmentation**: The audio is divided into overlapping segments (2 seconds each with 0.5s overlap)
2. **Language Detection**: Each segment is analyzed using the VoxLingua107 language identification model
3. **Diarization**: Results are combined to create a timeline of language changes
4. **Filtering** (optional): If a target language is specified, only segments matching that language are returned

## 📚 References

- **Original Repository**: https://github.com/jagabandhumishra/W2V-E2E-Language-Diarization
- **Paper**: [End to End Spoken Language Diarization with Wav2vec Embeddings](https://www.isca-speech.org/archive/interspeech_2023/mishra23_interspeech.html)
- **Base Model**: https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa
- **SpeechBrain**: https://speechbrain.github.io/
- **Triton Documentation**: https://docs.nvidia.com/deeplearning/triton-inference-server/

## 📄 License

The W2V-E2E-Language-Diarization model is based on research code. Please refer to the original repository for licensing information.













