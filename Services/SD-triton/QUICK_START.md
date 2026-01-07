# Quick Start Guide for SD-triton

## Prerequisites

**IMPORTANT**: This service requires a HuggingFace access token. 

1. Get your token from: https://huggingface.co/settings/tokens
2. Accept model conditions at:
   - https://huggingface.co/pyannote/speaker-diarization
   - https://huggingface.co/pyannote/segmentation
3. Set the token as an environment variable:
   ```bash
   export HUGGING_FACE_HUB_TOKEN="your_token_here"
   ```

## Step 1: Build Docker Image

```bash
cd /home/ubuntu/incubalm/SD-triton

# Make sure HUGGING_FACE_HUB_TOKEN is set
if [ -z "${HUGGING_FACE_HUB_TOKEN}" ]; then
    echo "ERROR: HUGGING_FACE_HUB_TOKEN is not set!"
    echo "Set it with: export HUGGING_FACE_HUB_TOKEN='your_token_here'"
    exit 1
fi

docker build --build-arg HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" -t sd-triton:latest .
```

This will take several minutes as it downloads dependencies and the model.

## Step 2: Start the Server

```bash
# Stop any existing container
docker stop sd-triton-server 2>/dev/null
docker rm sd-triton-server 2>/dev/null

# Start the server (make sure HUGGING_FACE_HUB_TOKEN is set)
docker run -d --gpus all \
  -p 8700:8000 -p 8701:8001 -p 8702:8002 \
  -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
  --name sd-triton-server \
  sd-triton:latest
```

Or use the script:
```bash
chmod +x start_server.sh
./start_server.sh
```

## Step 3: Wait for Server to be Ready

The server needs to:
1. Start Triton
2. Download the model (if not cached)
3. Load the model into memory

This can take 1-2 minutes on first run.

Check status:
```bash
# Check if server is ready
curl http://localhost:8700/v2/health/ready

# Check logs
docker logs sd-triton-server

# Watch logs in real-time
docker logs -f sd-triton-server
```

Look for these messages in logs:
- `[OK] HuggingFace authentication successful`
- `[OK] Model loaded successfully on device: cuda`
- `[OK] Speaker Diarization model ready for inference`

## Step 4: Test with Audio File

```bash
# Test with your audio file
cd /home/ubuntu/incubalm/SD-triton
python3 test_client.py ../ALD-triton/ta2.wav

# Or with specific number of speakers
python3 test_client.py ../ALD-triton/ta2.wav --num-speakers 2

# Pretty print JSON
python3 test_client.py ../ALD-triton/ta2.wav --pretty
```

## Troubleshooting

### Server not ready
- Wait longer (model download can take time)
- Check logs: `docker logs sd-triton-server`
- Verify model conditions are accepted on HuggingFace

### Model download fails
- Verify you accepted conditions at:
  - https://huggingface.co/pyannote/speaker-diarization
  - https://huggingface.co/pyannote/segmentation
- Check token is valid: `echo $HUGGING_FACE_HUB_TOKEN`

### Port already in use
```bash
# Find what's using port 8700
sudo lsof -i :8700

# Or use a different port
docker run -d --gpus all \
  -p 8701:8000 -p 8702:8001 -p 8703:8002 \
  -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
  --name sd-triton-server \
  sd-triton:latest
```
Then update test_client.py to use port 8701.












