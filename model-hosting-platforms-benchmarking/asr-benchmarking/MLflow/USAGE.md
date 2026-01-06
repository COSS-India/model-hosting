# MLflow ASR Service - Usage Guide

This guide explains how to use the MLflow ASR service for transcribing audio files.

## Quick Start

1. **Start the server** (if not already running):
   ```bash
   source mlflow/bin/activate
   mlflow models serve -m "mlruns/0/models/m-<model_id>/artifacts" --no-conda --host 0.0.0.0 --port 5000
   ```

2. **Test with audio file**:
   ```bash
   ./test_curl.sh ta ctc
   ```

## API Usage

### Endpoint

**URL**: `http://localhost:5000/asr`  
**Method**: `POST`  
**Content-Type**: `application/json`

### Request Format

The API expects a JSON payload with the following structure:

```json
{
  "audio_base64": "<base64_encoded_audio>",
  "lang": "ta",
  "decoding": "ctc"
}
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `audio_base64` | string | Yes | - | Base64-encoded audio file (WAV format recommended) |
| `lang` | string | No | `"hi"` | Language code for transcription (e.g., `"hi"`, `"ta"`, `"te"`) |
| `decoding` | string | No | `"ctc"` | Decoding strategy: `"ctc"` or `"greedy"` |

### Supported Languages

The model supports multiple Indic languages. Common language codes:

- `hi` - Hindi
- `ta` - Tamil
- `te` - Telugu
- `kn` - Kannada
- `ml` - Malayalam
- `mr` - Marathi
- `gu` - Gujarati
- `bn` - Bengali
- `or` - Odia
- `pa` - Punjabi

### Response Format

```json
{
  "transcription": "transcribed text in the specified language",
  "lang": "ta",
  "decoding": "ctc"
}
```

## Usage Examples

### Example 1: Using the Test Script

```bash
# Tamil audio with CTC decoding
./test_curl.sh ta ctc

# Hindi audio with greedy decoding
./test_curl.sh hi greedy
```

### Example 2: Using curl (One-liner)

```bash
curl -X POST http://localhost:5000/asr \
  -H "Content-Type: application/json" \
  -d "$(python3 -c "
import base64, json
audio_b64 = base64.b64encode(open('/home/ubuntu/Benchmarking/ta2.wav', 'rb').read()).decode('utf-8')
print(json.dumps({'dataframe_records': [{'audio_base64': audio_b64, 'lang': 'ta', 'decoding': 'ctc'}]}))
")" | python3 -m json.tool
```

### Example 3: Using Python

```python
import requests
import base64
import json

# Read and encode audio file
with open('ta2.wav', 'rb') as f:
    audio_bytes = f.read()
    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

# Prepare payload
payload = {
    "dataframe_records": [{
        "audio_base64": audio_b64,
        "lang": "ta",
        "decoding": "ctc"
    }]
}

# Send request
response = requests.post(
    'http://localhost:5000/asr',
    json=payload,
    headers={'Content-Type': 'application/json'}
)

# Get result
result = response.json()
print(result['predictions'][0]['transcription'])
```

### Example 4: Batch Processing

```python
import requests
import base64
import json
import os

def transcribe_audio(audio_path, lang='ta', decoding='ctc', endpoint='http://localhost:5000/asr'):
    """Transcribe a single audio file."""
    with open(audio_path, 'rb') as f:
        audio_b64 = base64.b64encode(f.read()).decode('utf-8')
    
    payload = {
        "dataframe_records": [{
            "audio_base64": audio_b64,
            "lang": lang,
            "decoding": decoding
        }]
    }
    
    response = requests.post(endpoint, json=payload)
    return response.json()['transcription']

# Process multiple files
audio_files = ['audio1.wav', 'audio2.wav', 'audio3.wav']
for audio_file in audio_files:
    transcription = transcribe_audio(audio_file, lang='ta')
    print(f"{audio_file}: {transcription}")
```

### Example 5: Using JavaScript/Node.js

```javascript
const fs = require('fs');
const axios = require('axios');

async function transcribeAudio(audioPath, lang = 'ta', decoding = 'ctc') {
  // Read and encode audio file
  const audioBuffer = fs.readFileSync(audioPath);
  const audioBase64 = audioBuffer.toString('base64');
  
  // Prepare payload
  const payload = {
    audio_base64: audioBase64,
    lang: lang,
    decoding: decoding
  };
  
  // Send request
  const response = await axios.post(
    'http://localhost:5000/asr',
    payload,
    { headers: { 'Content-Type': 'application/json' } }
  );
  
  return response.data.transcription;
}

// Usage
transcribeAudio('ta2.wav', 'ta', 'ctc')
  .then(transcription => console.log(transcription))
  .catch(error => console.error(error));
```

## Audio File Requirements

### Supported Formats

- **WAV** (recommended)
- **MP3** (may require additional dependencies)
- **FLAC**
- Other formats supported by `torchaudio`

### Audio Specifications

- **Sample Rate**: Any (automatically resampled to 16kHz)
- **Channels**: Mono or stereo (stereo is converted to mono)
- **Bit Depth**: 16-bit or 32-bit
- **Duration**: No strict limit, but longer files take more time

### Preparing Audio Files

**Convert to WAV format** (if needed):

```bash
# Using ffmpeg
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav

# Using sox
sox input.mp3 -r 16000 -c 1 output.wav
```

**Normalize audio** (optional):

```bash
ffmpeg -i input.wav -af "volume=normalize" output.wav
```

## Performance Tips

1. **Use WAV format**: Faster processing, no decoding overhead
2. **Mono audio**: Reduces processing time
3. **16kHz sample rate**: Matches model's expected rate
4. **Batch requests**: Process multiple files in parallel (if server supports it)
5. **GPU usage**: Ensure GPU is available for faster inference

## Error Handling

### Common Errors

**1. Invalid Audio Format**
```json
{
  "error_code": "BAD_REQUEST",
  "message": "Failed to open the input"
}
```
**Solution**: Ensure audio file is valid and properly encoded.

**2. Missing Field**
```json
{
  "error_code": "BAD_REQUEST",
  "message": "Missing required field: audio_base64"
}
```
**Solution**: Include `audio_base64` field in request.

**3. Server Error**
```json
{
  "error_code": "INTERNAL_SERVER_ERROR",
  "message": "Model loading failed"
}
```
**Solution**: Check server logs, verify model is properly loaded.

### Error Handling in Code

```python
import requests

try:
    response = requests.post(
        'http://localhost:5000/asr',
        json=payload,
        timeout=60  # 60 second timeout
    )
    response.raise_for_status()  # Raise exception for HTTP errors
    result = response.json()
    transcription = result['predictions'][0]['transcription']
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
except KeyError as e:
    print(f"Unexpected response format: {e}")
except Exception as e:
    print(f"Error: {e}")
```

## Monitoring

### Check Server Status

```bash
# Health check
curl http://localhost:5000/health

# Check if server is running
ps aux | grep mlflow

# Check port
netstat -tlnp | grep 5000
```

### View Server Logs

If running in foreground, logs appear in terminal. If running in background:

```bash
# If using nohup
tail -f mlflow_server.log

# If using screen
screen -r mlflow
```

### Monitor GPU Usage

```bash
# Continuous monitoring
watch -n 1 nvidia-smi

# One-time check
nvidia-smi
```

## Advanced Usage

### Custom Model Path

If you've logged multiple models:

```bash
# List all models
ls mlruns/0/models/

# Use specific model
mlflow models serve -m "mlruns/0/models/m-<different_model_id>/artifacts" --no-conda
```

### Multiple Workers

For better performance with multiple concurrent requests:

```bash
mlflow models serve \
  -m "mlruns/0/models/m-<model_id>/artifacts" \
  --no-conda \
  --host 0.0.0.0 \
  --port 5000 \
  --workers 4
```

### Custom Port

```bash
mlflow models serve \
  -m "mlruns/0/models/m-<model_id>/artifacts" \
  --no-conda \
  --host 0.0.0.0 \
  --port 8080
```

Then update your requests to use port 8080.

## Integration Examples

### Flask Application

```python
from flask import Flask, request, jsonify
import requests
import base64

app = Flask(__name__)
MLFLOW_ENDPOINT = 'http://localhost:5000/asr'

@app.route('/transcribe', methods=['POST'])
def transcribe():
    audio_file = request.files['audio']
    lang = request.form.get('lang', 'ta')
    decoding = request.form.get('decoding', 'ctc')
    
    # Encode audio
    audio_b64 = base64.b64encode(audio_file.read()).decode('utf-8')
    
    # Call MLflow service
    payload = {
        "dataframe_records": [{
            "audio_base64": audio_b64,
            "lang": lang,
            "decoding": decoding
        }]
    }
    
    response = requests.post(MLFLOW_ENDPOINT, json=payload)
    result = response.json()
    
    return jsonify({
        "transcription": result['transcription']
    })

if __name__ == '__main__':
    app.run(port=3000)
```

## Best Practices

1. **Always handle errors**: Network issues, invalid audio, server errors
2. **Set timeouts**: Prevent hanging requests
3. **Validate audio**: Check file size, format before sending
4. **Use appropriate language code**: Match the language of your audio
5. **Monitor performance**: Track latency and success rates
6. **Cache results**: If processing the same audio multiple times

---

For more information, see [README.md](README.md) and [SETUP.md](SETUP.md).

