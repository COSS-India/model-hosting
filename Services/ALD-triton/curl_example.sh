#!/bin/bash
# Example curl command for ALD Triton Inference Server
# Usage: ./curl_example.sh <audio_file_path>

if [ $# -eq 0 ]; then
    echo "Usage: $0 <audio_file_path>"
    echo "Example: $0 test_audio.wav"
    exit 1
fi

AUDIO_FILE="$1"
SERVER_URL="http://localhost:8100"

if [ ! -f "$AUDIO_FILE" ]; then
    echo "Error: Audio file not found: $AUDIO_FILE"
    exit 1
fi

# Encode audio file to base64
echo "Encoding audio file to base64..."
AUDIO_B64=$(base64 -w 0 "$AUDIO_FILE")

# Make inference request
echo "Making inference request to $SERVER_URL/v2/models/ald/infer..."
echo ""

curl -X POST "$SERVER_URL/v2/models/ald/infer" \
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
  }" | python3 -m json.tool

echo ""













