# Audio Language Detection (ALD) Triton Deployment

This directory contains the Triton Inference Server deployment for the **speechbrain/lang-id-voxlingua107-ecapa** model, which performs spoken language identification from audio.

## 📋 Overview

- **Model**: speechbrain/lang-id-voxlingua107-ecapa
- **Task**: Spoken Language Identification
- **Languages**: 107 languages (Abkhazian, Afrikaans, Amharic, Arabic, Assamese, Azerbaijani, Bashkir, Belarusian, Bulgarian, Bengali, Tibetan, Breton, Bosnian, Catalan, Cebuano, Czech, Welsh, Danish, German, Greek, English, Esperanto, Spanish, Estonian, Basque, Persian, Finnish, Faroese, French, Galician, Guarani, Gujarati, Manx, Hausa, Hawaiian, Hindi, Croatian, Haitian, Hungarian, Armenian, Interlingua, Indonesian, Icelandic, Italian, Hebrew, Japanese, Javanese, Georgian, Kazakh, Central Khmer, Kannada, Korean, Latin, Luxembourgish, Lingala, Lao, Lithuanian, Latvian, Malagasy, Maori, Macedonian, Malayalam, Mongolian, Marathi, Malay, Maltese, Burmese, Nepali, Dutch, Norwegian Nynorsk, Norwegian, Occitan, Panjabi, Polish, Pushto, Portuguese, Romanian, Russian, Sanskrit, Scots, Sindhi, Sinhala, Slovak, Slovenian, Shona, Somali, Albanian, Serbian, Sundanese, Swedish, Swahili, Tamil, Telugu, Tajik, Thai, Turkmen, Tagalog, Turkish, Tatar, Ukrainian, Urdu, Uzbek, Vietnamese, Waray, Yiddish, Yoruba, Mandarin Chinese)
- **Backend**: Python (Triton)
- **Framework**: PyTorch + SpeechBrain
- **Architecture**: ECAPA-TDNN

## 🏗️ Directory Structure

```
ALD-triton/
├── Dockerfile                          # Docker build configuration
├── README.md                           # This file
├── test_client.py                      # Test client script
└── model_repository/
    └── ald/                           # Model name
        ├── config.pbtxt               # Triton model configuration
        └── 1/                         # Model version
            └── model.py               # Python backend implementation
```

## 🚀 Quick Start

### Build the Docker Image

```bash
cd /home/ubuntu/incubalm/ALD-triton

# Build the image
docker build -t ald-triton:latest .
```

### Run the Triton Server

```bash
# Run with GPU support on port 8100
docker run --gpus all -p 8100:8000 -p 8101:8001 -p 8102:8002 \
  --name ald-triton \
  ald-triton:latest

# Or run in detached mode
docker run -d --gpus all -p 8100:8000 -p 8101:8001 -p 8102:8002 \
  --name ald-triton \
  ald-triton:latest
```

### Check Server Health

```bash
# Check if server is ready
curl http://localhost:8100/v2/health/ready

# List available models
curl http://localhost:8100/v2/models

# Get model metadata
curl http://localhost:8100/v2/models/ald
```

## 📊 Input/Output Format

### Triton API Format

**Input Parameters:**

1. **AUDIO_DATA** (Required)
   - Type: `TYPE_STRING` (BYTES)
   - Shape: `[1, 1]`
   - Format: Base64-encoded audio data (WAV format recommended)
   - Sample Rate: The model automatically resamples to 16kHz
   - Channels: The model automatically converts to mono

**Output:**

1. **LANGUAGE_CODE**
   - Type: `TYPE_STRING` (BYTES)
   - Shape: `[1, 1]`
   - Format: ISO language code (e.g., "en", "hi", "th", "es")
   - Example: `"en"` (English)

2. **CONFIDENCE**
   - Type: `TYPE_FP32`
   - Shape: `[1, 1]`
   - Format: Confidence score between 0.0 and 1.0
   - Example: `0.9850`

3. **ALL_SCORES**
   - Type: `TYPE_STRING` (BYTES)
   - Shape: `[1, 1]`
   - Format: JSON string containing prediction details
   - Structure:
     ```json
     {
       "predicted_language": "en",
       "confidence": 0.9850,
       "top_scores": [0.9850, 0.0100, 0.0030, 0.0015, 0.0005]
     }
     ```

### Example Request (HTTP)

```bash
# First, encode your audio file to base64
AUDIO_B64=$(base64 -w 0 your_audio.wav)

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
      {
        \"name\": \"LANGUAGE_CODE\"
      },
      {
        \"name\": \"CONFIDENCE\"
      },
      {
        \"name\": \"ALL_SCORES\"
      }
    ]
  }"
```

## ⚙️ Configuration

### Model Configuration (config.pbtxt)

- **Backend**: Python
- **Max Batch Size**: 64
- **Instance Group**: 1 GPU instance
- **Dynamic Batching**: Enabled with preferred batch sizes [1, 2, 4, 8, 16, 32, 64]

### Performance Tuning

To adjust performance:
1. Modify `max_batch_size` in `config.pbtxt`
2. Adjust `preferred_batch_size` for dynamic batching
3. Change `instance_group.count` for multiple model instances

## 🎵 Audio Format Requirements

- **Format**: WAV, MP3, FLAC, or other formats supported by torchaudio
- **Sample Rate**: Any (automatically resampled to 16kHz by the model)
- **Channels**: Mono or Stereo (automatically converted to mono by the model)
- **Duration**: No strict limit, but longer audio may take more processing time

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

response = requests.post(url, json=payload)
result = response.json()

# Parse the output
language_code = result["outputs"][0]["data"][0]
confidence = result["outputs"][1]["data"][0]
all_scores = json.loads(result["outputs"][2]["data"][0])

print(f"Detected Language: {language_code}")
print(f"Confidence: {confidence:.4f}")
print(f"Details: {all_scores}")
```

Run the test client:
```bash
python3 test_client.py
```

## 📝 Notes

- The model uses ECAPA-TDNN architecture optimized for speaker recognition, adapted for language identification
- Audio is automatically normalized (resampling to 16kHz and mono channel conversion)
- The model runs on GPU by default for better performance
- Error rate: 6.7% on the VoxLingua107 development dataset
- The model is trained on automatically collected YouTube data

## ⚠️ Limitations and Bias

Since the model is trained on VoxLingua107, it has some limitations:

- Accuracy on smaller languages may be limited
- May work worse on female speech than male speech (YouTube data bias)
- Doesn't work well on speech with foreign accents
- May not work well on children's speech and persons with speech disorders

## 📚 References

- **Model**: https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa
- **Paper**: [VoxLingua107: a Dataset for Spoken Language Recognition](https://arxiv.org/abs/2106.04624)
- **SpeechBrain**: https://speechbrain.github.io/
- **Triton Documentation**: https://docs.nvidia.com/deeplearning/triton-inference-server/

## 📄 License

The VoxLingua107 ECAPA-TDNN model is released under the Apache 2.0 License.













