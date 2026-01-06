#!/bin/bash

# Test script for FastAPI IndicTrans2 NMT service

BASE_URL="http://localhost:8000"

echo "=========================================="
echo "FastAPI IndicTrans2 NMT Service Tests"
echo "=========================================="
echo ""

# Test health endpoint
echo "1. Testing health endpoint..."
curl -X GET "$BASE_URL/health" | python3 -m json.tool
echo ""
echo ""

# Test English to Hindi
echo "2. Testing English to Hindi translation..."
curl -X POST "$BASE_URL/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?",
    "src_lang": "eng_Latn",
    "tgt_lang": "hin_Deva"
  }' | python3 -m json.tool
echo ""
echo ""

# Test Hindi to English
echo "3. Testing Hindi to English translation..."
curl -X POST "$BASE_URL/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "नमस्ते, आप कैसे हैं?",
    "src_lang": "hin_Deva",
    "tgt_lang": "eng_Latn"
  }' | python3 -m json.tool
echo ""
echo ""

# Test Hindi to Marathi (Indic-Indic)
echo "4. Testing Hindi to Marathi translation (Indic-Indic)..."
curl -X POST "$BASE_URL/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "नमस्ते, आप कैसे हैं?",
    "src_lang": "hin_Deva",
    "tgt_lang": "mar_Deva"
  }' | python3 -m json.tool
echo ""
echo ""

echo "=========================================="
echo "Tests completed!"
echo "=========================================="

