#!/bin/bash
# Curl command to test BentoML ASR server with ta2.wav
# Works with both local server and containerized service

ENDPOINT="${1:-http://localhost:3000/asr}"
AUDIO_FILE="${2:-/home/ubuntu/Benchmarking/ta2.wav}"
LANG="${3:-ta}"
STRATEGY="${4:-ctc}"

echo "Testing BentoML ASR server..."
echo "Endpoint: $ENDPOINT"
echo "Audio file: $AUDIO_FILE"
echo "Language: $LANG"
echo "Strategy: $STRATEGY"
echo ""

# Check if audio file exists
if [ ! -f "$AUDIO_FILE" ]; then
    echo "Error: Audio file not found at $AUDIO_FILE"
    exit 1
fi

# Test with multipart form data
echo "Sending request..."
RESPONSE=$(curl -X POST "$ENDPOINT" \
  -F "file=@${AUDIO_FILE}" \
  -F "lang=${LANG}" \
  -F "strategy=${STRATEGY}" \
  -w "\nHTTP_STATUS:%{http_code}" \
  -s)

HTTP_STATUS=$(echo "$RESPONSE" | grep -o "HTTP_STATUS:[0-9]*" | cut -d: -f2)
BODY=$(echo "$RESPONSE" | sed 's/HTTP_STATUS:[0-9]*$//')

echo "Response:"
echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
echo ""
echo "HTTP Status: $HTTP_STATUS"

if [ "$HTTP_STATUS" = "200" ]; then
    echo "✅ Success!"
else
    echo "❌ Failed!"
fi

echo ""
echo "Usage: $0 [endpoint] [audio_file] [lang] [strategy]"
echo "Example: $0 http://localhost:3000/asr /path/to/audio.wav ta ctc"

