#!/bin/bash

# Quick test script for Triton IndicTrans v2
# This script tests the server and attempts to get model information

set -e

SERVER_URL="http://localhost:8000"
MODEL_NAME=""  # Will be detected automatically

echo "Testing Triton IndicTrans v2 Server..."
echo "========================================"
echo ""

# Test 1: Health check
echo "1. Checking server health..."
if curl -s -f "${SERVER_URL}/v2/health/ready" > /dev/null; then
    echo "✓ Server is ready"
else
    echo "✗ Server is not ready"
    exit 1
fi
echo ""

# Test 2: Get available models
echo "2. Fetching available models..."
MODELS_RESPONSE=$(curl -s "${SERVER_URL}/v2/models")
echo "$MODELS_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$MODELS_RESPONSE"
echo ""

# Extract first model name if available
MODEL_NAME=$(echo "$MODELS_RESPONSE" | python3 -c "import sys, json; data = json.load(sys.stdin); print(data.get('models', [{}])[0].get('name', ''))" 2>/dev/null || echo "")

if [ -z "$MODEL_NAME" ]; then
    echo "⚠ Could not detect model name. Please check the models list above."
    echo "   Update MODEL_NAME in this script or use the curl commands directly."
    exit 0
fi

echo "Detected model: $MODEL_NAME"
echo ""

# Test 3: Get model metadata
echo "3. Fetching model metadata for '$MODEL_NAME'..."
curl -s "${SERVER_URL}/v2/models/${MODEL_NAME}" | python3 -m json.tool || curl -s "${SERVER_URL}/v2/models/${MODEL_NAME}"
echo ""

# Test 4: Get model configuration
echo "4. Fetching model configuration for '$MODEL_NAME'..."
curl -s "${SERVER_URL}/v2/models/${MODEL_NAME}/config" | python3 -m json.tool || curl -s "${SERVER_URL}/v2/models/${MODEL_NAME}/config"
echo ""

echo "========================================"
echo "Server is operational!"
echo ""
echo "To make an inference request, use:"
echo "  curl -X POST ${SERVER_URL}/v2/models/${MODEL_NAME}/infer \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"inputs\": [{\"name\": \"...\", \"shape\": [...], \"datatype\": \"...\", \"data\": [...]}]}'"

