# API Reference

Complete API reference for the MLflow ASR service.

## Base URL

```
http://localhost:5000
```

For Docker deployments, replace `localhost` with your server IP or domain.

## Endpoints

### POST /asr

Transcribe audio to text.

#### Request

**Headers**:
```
Content-Type: application/json
```

**Body**:
```json
{
  "dataframe_records": [
    {
      "audio_base64": "string (required)",
      "lang": "string (optional, default: 'hi')",
      "decoding": "string (optional, default: 'ctc')"
    }
  ]
}
```

**Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `audio_base64` | string | Yes | - | Base64-encoded audio file (WAV format recommended) |
| `lang` | string | No | `"hi"` | Language code (see [Supported Languages](#supported-languages)) |
| `decoding` | string | No | `"ctc"` | Decoding strategy: `"ctc"` or `"greedy"` |

**Example Request**:
```bash
curl -X POST http://localhost:5000/asr \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_records": [{
      "audio_base64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=",
      "lang": "ta",
      "decoding": "ctc"
    }]
  }'
```

#### Response

**Success (200 OK)**:
```json
{
  "transcription": "transcribed text in the specified language",
  "lang": "ta",
  "decoding": "ctc"
}
```

**Error (400 Bad Request)**:
```json
{
  "error_code": "BAD_REQUEST",
  "message": "Error description",
  "stack_trace": "Detailed error traceback"
}
```

**Error (500 Internal Server Error)**:
```json
{
  "error_code": "INTERNAL_SERVER_ERROR",
  "message": "Model inference failed",
  "stack_trace": "..."
}
```

#### Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request (invalid input) |
| 500 | Internal Server Error (model error) |
| 503 | Service Unavailable (model loading) |

### GET /health

Check service health status.

#### Request

No parameters required.

#### Response

**Success (200 OK)**:
```
OK
```

**Service Unavailable (503)**:
```
Service Unavailable
```

### GET /docs

Interactive API documentation (Swagger UI).

### GET /redoc

Alternative API documentation (ReDoc).

## Supported Languages

| Code | Language |
|------|----------|
| `hi` | Hindi |
| `ta` | Tamil |
| `te` | Telugu |
| `kn` | Kannada |
| `ml` | Malayalam |
| `mr` | Marathi |
| `gu` | Gujarati |
| `bn` | Bengali |
| `or` | Odia |
| `pa` | Punjabi |

## Decoding Strategies

### CTC (Connectionist Temporal Classification)

- Default decoding method
- Good for general use cases
- Faster inference
- May produce more accurate results

### Greedy

- Alternative decoding method
- Simpler algorithm
- May be faster in some cases
- Results may vary

## Request Examples

### Python

```python
import requests
import base64

# Read and encode audio
with open('audio.wav', 'rb') as f:
    audio_b64 = base64.b64encode(f.read()).decode('utf-8')

# Prepare request
payload = {
    "audio_base64": audio_b64,
    "lang": "ta",
    "decoding": "ctc"
}

# Send request
response = requests.post(
    'http://localhost:5000/asr',
    json=payload
)

# Get result
result = response.json()
print(result['transcription'])
```

### JavaScript/Node.js

```javascript
const fs = require('fs');
const axios = require('axios');

// Read and encode audio
const audioBuffer = fs.readFileSync('audio.wav');
const audioBase64 = audioBuffer.toString('base64');

// Prepare request
const payload = {
  audio_base64: audioBase64,
  lang: 'ta',
  decoding: 'ctc'
};

// Send request
axios.post('http://localhost:5000/asr', payload)
  .then(response => {
    console.log(response.data.transcription);
  })
  .catch(error => {
    console.error('Error:', error.response.data);
  });
```

### cURL

```bash
# Create payload file
python3 -c "
import base64, json
audio_b64 = base64.b64encode(open('audio.wav', 'rb').read()).decode('utf-8')
payload = {'audio_base64': audio_b64, 'lang': 'ta', 'decoding': 'ctc'}
json.dump(payload, open('/tmp/payload.json', 'w'))
"

# Send request
curl -X POST http://localhost:5000/asr \
  -H "Content-Type: application/json" \
  -d @/tmp/payload.json
```

## Error Handling

### Common Errors

#### Invalid Audio Format

```json
{
  "error_code": "BAD_REQUEST",
  "message": "Failed to open the input (Invalid data found when processing input)"
}
```

**Solution**: Ensure audio is in WAV format and properly encoded.

#### Missing Field

```json
{
  "error_code": "BAD_REQUEST",
  "message": "Missing required field: audio_base64"
}
```

**Solution**: Include `audio_base64` in the request.

#### Model Loading Error

```json
{
  "error_code": "INTERNAL_SERVER_ERROR",
  "message": "Model loading failed"
}
```

**Solution**: Check server logs, verify model is accessible, check HF_TOKEN.

#### Service Unavailable

```json
{
  "error_code": "SERVICE_UNAVAILABLE",
  "message": "Model is still loading"
}
```

**Solution**: Wait for model to finish loading (first request takes longer).

## Rate Limiting

Currently, there is no rate limiting implemented. For production deployments, consider:
- Implementing rate limiting
- Using a reverse proxy (nginx, traefik)
- Load balancing for high traffic

## Timeout

Default timeout is typically 60 seconds. For long audio files:
- Consider chunking audio
- Increase timeout in client
- Use async requests

## Best Practices

1. **Audio Format**: Use WAV format, 16kHz, mono for best results
2. **Base64 Encoding**: Ensure proper base64 encoding (no line breaks)
3. **Error Handling**: Always check response status and handle errors
4. **Retries**: Implement retry logic for transient errors
5. **Timeouts**: Set appropriate timeouts for requests
6. **Language Code**: Use correct language code matching audio content
7. **Batch Processing**: For multiple files, send separate requests (or implement batching)

## Response Time

Typical response times:
- **First Request**: 30-60 seconds (model loading)
- **Subsequent Requests**: 1-5 seconds (depending on audio length)
- **GPU**: Faster inference with GPU
- **CPU**: Slower but functional

## Audio File Size Limits

- **Recommended**: < 10MB per file
- **Maximum**: Limited by server memory and timeout
- **Best Practice**: Keep files under 5MB for optimal performance

---

**Last Updated**: December 2024

