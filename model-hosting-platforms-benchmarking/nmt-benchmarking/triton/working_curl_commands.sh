#!/bin/bash

# Working Curl Commands for Triton IndicTrans v2
# Model name: nmt
# Language codes: en (English), hi (Hindi)

SERVER="http://localhost:8000"
MODEL="nmt"

echo "=== Working Curl Commands ==="
echo ""

# 1. Health Check
echo "1. Health Check:"
echo "curl -v ${SERVER}/v2/health/ready"
echo ""

# 2. Get Model Metadata (WORKS)
echo "2. Get Model Metadata:"
echo "curl ${SERVER}/v2/models/${MODEL}"
echo ""

# 3. Get Model Configuration (WORKS)
echo "3. Get Model Configuration:"
echo "curl ${SERVER}/v2/models/${MODEL}/config | python3 -m json.tool"
echo ""

# 4. Inference Request - English to Hindi
echo "4. Example: English to Hindi (en -> hi):"
echo "curl -X POST ${SERVER}/v2/models/${MODEL}/infer \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"inputs\":[{\"name\":\"INPUT_TEXT\",\"datatype\":\"BYTES\",\"shape\":[1,1],\"data\":[\"Hello, how are you?\"]},{\"name\":\"INPUT_LANGUAGE_ID\",\"datatype\":\"BYTES\",\"shape\":[1,1],\"data\":[\"en\"]},{\"name\":\"OUTPUT_LANGUAGE_ID\",\"datatype\":\"BYTES\",\"shape\":[1,1],\"data\":[\"hi\"]}]}'"
echo ""

# 5. Inference Request - Hindi to English
echo "5. Example: Hindi to English (hi -> en):"
echo "curl -X POST ${SERVER}/v2/models/${MODEL}/infer \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"inputs\":[{\"name\":\"INPUT_TEXT\",\"datatype\":\"BYTES\",\"shape\":[1,1],\"data\":[\"नमस्ते, आप कैसे हैं?\"]},{\"name\":\"INPUT_LANGUAGE_ID\",\"datatype\":\"BYTES\",\"shape\":[1,1],\"data\":[\"hi\"]},{\"name\":\"OUTPUT_LANGUAGE_ID\",\"datatype\":\"BYTES\",\"shape\":[1,1],\"data\":[\"en\"]}]}'"
echo ""

# 6. Pretty formatted version
echo "6. Pretty formatted request (English to Hindi):"
echo "curl -X POST ${SERVER}/v2/models/${MODEL}/infer \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{"
echo "    \"inputs\": ["
echo "      {"
echo "        \"name\": \"INPUT_TEXT\","
echo "        \"datatype\": \"BYTES\","
echo "        \"shape\": [1, 1],"
echo "        \"data\": [\"Hello, how are you?\"]"
echo "      },"
echo "      {"
echo "        \"name\": \"INPUT_LANGUAGE_ID\","
echo "        \"datatype\": \"BYTES\","
echo "        \"shape\": [1, 1],"
echo "        \"data\": [\"en\"]"
echo "      },"
echo "      {"
echo "        \"name\": \"OUTPUT_LANGUAGE_ID\","
echo "        \"datatype\": \"BYTES\","
echo "        \"shape\": [1, 1],"
echo "        \"data\": [\"hi\"]"
echo "      }"
echo "    ]"
echo "  }' | python3 -m json.tool"
echo ""

echo "Note: Language codes - en (English), hi (Hindi)"
echo "Note: Shape must be [1,1] for single inputs (not [1])"
echo "Note: /v2/models (list endpoint) returns 400, but individual model endpoints work fine."

