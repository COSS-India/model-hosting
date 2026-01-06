#!/bin/bash

# Triton Inference Server - Curl Commands for IndicTrans v2
# Server running on http://localhost:8000

echo "=== Triton IndicTrans v2 - Curl Commands ==="
echo ""

# 1. Health Check - Check if server is ready
echo "1. Health Check (Server Ready):"
echo "curl -v http://localhost:8000/v2/health/ready"
echo ""

# 2. Server Health - Check if server is live
echo "2. Server Health (Live):"
echo "curl -v http://localhost:8000/v2/health/live"
echo ""

# 3. List all models
echo "3. List Available Models:"
echo "curl http://localhost:8000/v2/models"
echo ""

# 4. Get model metadata (replace MODEL_NAME with actual model name)
echo "4. Get Model Metadata:"
echo "curl http://localhost:8000/v2/models/MODEL_NAME"
echo ""

# 5. Get model configuration (replace MODEL_NAME with actual model name)
echo "5. Get Model Configuration:"
echo "curl http://localhost:8000/v2/models/MODEL_NAME/config"
echo ""

# 6. Get model statistics
echo "6. Get Model Statistics:"
echo "curl http://localhost:8000/v2/models/MODEL_NAME/stats"
echo ""

# 7. Inference Request - Text Translation (adjust based on actual model input format)
echo "7. Inference Request - Translation:"
echo "curl -X POST http://localhost:8000/v2/models/MODEL_NAME/infer \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{"
echo "    \"inputs\": ["
echo "      {"
echo "        \"name\": \"input_text\","
echo "        \"shape\": [1],"
echo "        \"datatype\": \"BYTES\","
echo "        \"data\": [\"Hello, how are you?\"]"
echo "      }"
echo "    ],"
echo "    \"outputs\": ["
echo "      {"
echo "        \"name\": \"output_text\""
echo "      }"
echo "    ]"
echo "  }'"
echo ""

# 8. Inference Request with source and target language
echo "8. Inference Request with Language Parameters:"
echo "curl -X POST http://localhost:8000/v2/models/MODEL_NAME/infer \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{"
echo "    \"inputs\": ["
echo "      {"
echo "        \"name\": \"input_text\","
echo "        \"shape\": [1],"
echo "        \"datatype\": \"BYTES\","
echo "        \"data\": [\"नमस्ते, आप कैसे हैं?\"]"
echo "      },"
echo "      {"
echo "        \"name\": \"src_lang\","
echo "        \"shape\": [1],"
echo "        \"datatype\": \"BYTES\","
echo "        \"data\": [\"hin\"]"
echo "      },"
echo "      {"
echo "        \"name\": \"tgt_lang\","
echo "        \"shape\": [1],"
echo "        \"datatype\": \"BYTES\","
echo "        \"data\": [\"eng\"]"
echo "      }"
echo "    ]"
echo "  }'"
echo ""

# 9. Server Metadata
echo "9. Server Metadata:"
echo "curl http://localhost:8000/v2"
echo ""

# 10. Repository Index
echo "10. Repository Index:"
echo "curl http://localhost:8000/v2/repository/index"
echo ""

echo "=== Note: Replace MODEL_NAME with the actual model name from the models list ==="

