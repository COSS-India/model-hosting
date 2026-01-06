# Curl Commands for Triton IndicTrans v2

## Basic Health Checks

### Check if server is ready
```bash
curl -v http://localhost:8000/v2/health/ready
```

### Check if server is live
```bash
curl -v http://localhost:8000/v2/health/live
```

## Model Discovery

### List all available models
```bash
curl http://localhost:8000/v2/models
```

### Get server metadata
```bash
curl http://localhost:8000/v2
```

### Get repository index
```bash
curl http://localhost:8000/v2/repository/index
```

## Model Information

### Get model metadata
```bash
curl http://localhost:8000/v2/models/nmt | python3 -m json.tool
```

### Get model configuration
```bash
curl http://localhost:8000/v2/models/nmt/config | python3 -m json.tool
```

### Get model statistics
```bash
curl http://localhost:8000/v2/models/nmt/stats | python3 -m json.tool
```

## Inference Requests

**Model Name:** `nmt`  
**Language Codes:** `en` (English), `hi` (Hindi)  
**Important:** Shape must be `[1,1]` for inputs (not `[1]`)

### English to Hindi Translation
```bash
curl -X POST http://localhost:8000/v2/models/nmt/infer \
  -H 'Content-Type: application/json' \
  -d '{"inputs":[{"name":"INPUT_TEXT","datatype":"BYTES","shape":[1,1],"data":["Hello, how are you?"]},{"name":"INPUT_LANGUAGE_ID","datatype":"BYTES","shape":[1,1],"data":["en"]},{"name":"OUTPUT_LANGUAGE_ID","datatype":"BYTES","shape":[1,1],"data":["hi"]}]}' | python3 -m json.tool
```

### Hindi to English Translation
```bash
curl -X POST http://localhost:8000/v2/models/nmt/infer \
  -H 'Content-Type: application/json' \
  -d '{"inputs":[{"name":"INPUT_TEXT","datatype":"BYTES","shape":[1,1],"data":["नमस्ते, आप कैसे हैं?"]},{"name":"INPUT_LANGUAGE_ID","datatype":"BYTES","shape":[1,1],"data":["hi"]},{"name":"OUTPUT_LANGUAGE_ID","datatype":"BYTES","shape":[1,1],"data":["en"]}]}' | python3 -m json.tool
```

### Pretty Formatted Request (English to Hindi)
```bash
curl -X POST http://localhost:8000/v2/models/nmt/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": [
      {
        "name": "INPUT_TEXT",
        "datatype": "BYTES",
        "shape": [1, 1],
        "data": ["Hello, how are you?"]
      },
      {
        "name": "INPUT_LANGUAGE_ID",
        "datatype": "BYTES",
        "shape": [1, 1],
        "data": ["en"]
      },
      {
        "name": "OUTPUT_LANGUAGE_ID",
        "datatype": "BYTES",
        "shape": [1, 1],
        "data": ["hi"]
      }
    ]
  }' | python3 -m json.tool
```

## Pretty Print JSON Responses

Add `| python3 -m json.tool` or `| jq` to format JSON responses:

```bash
curl -s http://localhost:8000/v2/models | python3 -m json.tool
```

## Quick Test Script

Run the automated test script to check server status and discover models:

```bash
./test_curl.sh
```

## Notes

1. **Model Name**: Replace `MODEL_NAME` with the actual model name from the models list
2. **Input Format**: The exact input names, shapes, and data types depend on the model configuration
3. **Language Codes**: Common Indic language codes: `hin` (Hindi), `eng` (English), `tel` (Telugu), `tam` (Tamil), `kan` (Kannada), `mal` (Malayalam), `guj` (Gujarati), `mar` (Marathi), `pan` (Punjabi), `ben` (Bengali), `ori` (Odia), `asm` (Assamese)

## Getting Started

1. Start the server:
   ```bash
   docker-compose up -d
   ```

2. Wait for server to be ready:
   ```bash
   curl http://localhost:8000/v2/health/ready
   ```

3. List models:
   ```bash
   curl http://localhost:8000/v2/models
   ```

4. Get model metadata to understand input/output format:
   ```bash
   curl http://localhost:8000/v2/models/nmt | python3 -m json.tool
   ```

5. Make inference requests using the format shown above

## Working Examples

**Model Name:** `nmt`  
**Input Names:** `INPUT_TEXT`, `INPUT_LANGUAGE_ID`, `OUTPUT_LANGUAGE_ID`  
**Output Name:** `OUTPUT_TEXT`  
**Language Codes:** `en` (English), `hi` (Hindi)  
**Shape:** Must be `[1,1]` for all inputs

**Note:** The `/v2/models` endpoint (list all models) returns 400, but individual model endpoints work perfectly.

