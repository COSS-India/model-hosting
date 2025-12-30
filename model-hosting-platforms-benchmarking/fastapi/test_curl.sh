#!/bin/bash
# Example curl commands for testing the ASR endpoint

# Test 1: Upload audio file directly (multipart/form-data)
echo "Test 1: Upload audio file"
curl -X POST "http://localhost:8000/asr" \
  -F "audio=@audio_file.wav"

# Test 2: Using base64-encoded audio in form data
echo -e "\n\nTest 2: Base64 audio in form data"
AUDIO_B64=$(base64 -w 0 audio_file.wav)
curl -X POST "http://localhost:8000/asr" \
  -F "audio_b64=$AUDIO_B64"

# Test 3: Using JSON with base64-encoded audio
echo -e "\n\nTest 3: JSON with base64 audio"
AUDIO_B64=$(base64 -w 0 audio_file.wav)
curl -X POST "http://localhost:8000/asr/json" \
  -H "Content-Type: application/json" \
  -d "{\"audio_b64\": \"$AUDIO_B64\"}"

# Test 4: Health check
echo -e "\n\nTest 4: Health check"
curl -X GET "http://localhost:8000/health"

# Test 5: Root endpoint
echo -e "\n\nTest 5: Root endpoint"
curl -X GET "http://localhost:8000/"

