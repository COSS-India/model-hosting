#!/bin/bash
# Test MLflow ASR service with ta2.wav

AUDIO_FILE="/home/ubuntu/Benchmarking/ta2.wav"
ENDPOINT="http://localhost:5000/asr"
LANG="${1:-ta}"  # Default to 'ta' (Tamil), can override: ./test_curl.sh hi
DECODING="${2:-ctc}"  # Default to 'ctc', can override: ./test_curl.sh ta greedy

echo "Testing MLflow ASR service with $AUDIO_FILE"
echo "Language: $LANG, Decoding: $DECODING"
echo ""

# Create JSON payload with base64-encoded audio
python3 << EOF
import base64
import json
import sys

# Read and encode audio file
with open("$AUDIO_FILE", "rb") as f:
    audio_bytes = f.read()
    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

# Create payload (new /asr endpoint format)
payload = {
    "audio_base64": audio_b64,
    "lang": "$LANG",
    "decoding": "$DECODING"
}

# Write to temp file
with open("/tmp/mlflow_payload.json", "w") as f:
    json.dump(payload, f)

print(f"Payload created: {len(audio_b64)} characters (base64)")
EOF

# Send request
echo "Sending request to $ENDPOINT..."
echo ""

RESPONSE=$(curl -X POST "$ENDPOINT" \
  -H "Content-Type: application/json" \
  -d @/tmp/mlflow_payload.json \
  -w "\nHTTP_STATUS:%{http_code}" \
  -s)

# Extract HTTP status and body
HTTP_STATUS=$(echo "$RESPONSE" | grep -o "HTTP_STATUS:[0-9]*" | cut -d: -f2)
BODY=$(echo "$RESPONSE" | sed 's/HTTP_STATUS:[0-9]*$//')

echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
echo ""
echo "HTTP Status: $HTTP_STATUS"

echo ""
echo "---"
echo "Done!"

