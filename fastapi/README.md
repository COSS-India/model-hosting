# AI4Bharat Multilingual ASR FastAPI Server

FastAPI server hosting the AI4Bharat IndicConformer-600M-Multilingual ASR model from HuggingFace.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables:**
   ```bash
   export HF_TOKEN=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXxxx
   export DEVICE=cuda  # or "cpu" if no GPU available
   ```

3. **Run the server:**
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

   Or run directly:
   ```bash
   python app.py
   ```

## API Endpoints

### POST `/asr`
Transcribe audio. Accepts:
- **Multipart form data** with `audio` file field
- **Multipart form data** with `audio_b64` base64-encoded string
- **JSON** with `{"audio_b64": "base64_encoded_audio"}`

**Response:**
```json
{
  "text": "transcribed text",
  "latency_ms": 123.45
}
```

### POST `/asr/json`
Alternative endpoint that only accepts JSON format:
```json
{
  "audio_b64": "base64_encoded_audio_bytes"
}
```

### GET `/health`
Health check endpoint. Returns model status and configuration.

### GET `/`
API information and available endpoints.

### GET `/docs`
Interactive API documentation (Swagger UI).

## Model Information

- **Model:** `ai4bharat/indic-conformer-600m-multilingual`
- **Source:** HuggingFace (via Docker image: `ai4bharat/triton-multilingual-asr`)
- **Supported formats:** WAV, FLAC, and other formats supported by `soundfile`
- **Sample rate:** Auto-detected from model (typically 16kHz)

## Example Usage

### Using curl with file upload:
```bash
curl -X POST "http://localhost:8000/asr" \
  -F "audio=@audio_file.wav"
```

### Using curl with base64:
```bash
# Encode audio to base64
AUDIO_B64=$(base64 -w 0 audio_file.wav)

# Send request
curl -X POST "http://localhost:8000/asr/json" \
  -H "Content-Type: application/json" \
  -d "{\"audio_b64\": \"$AUDIO_B64\"}"
```

### Using Python:
```python
import requests
import base64

# Read and encode audio
with open("audio.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

# Send request
response = requests.post(
    "http://localhost:8000/asr/json",
    json={"audio_b64": audio_b64}
)

result = response.json()
print(f"Transcription: {result['text']}")
print(f"Latency: {result['latency_ms']} ms")
```

## Environment Variables

- `HF_TOKEN`: HuggingFace authentication token (required)
- `DEVICE`: Device to use - "cuda" or "cpu" (default: auto-detect)
- `MODEL_NAME`: HuggingFace model name (default: `ai4bharat/indic-conformer-600m-multilingual`)

## Notes

- The model is loaded on server startup, which may take a few minutes
- For GPU inference, ensure CUDA is available and PyTorch is installed with CUDA support
- The server uses a single worker by default for GPU scenarios
- Audio is automatically resampled to match the model's expected sample rate

