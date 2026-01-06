#!/bin/bash

# Curl test commands for BentoML IndicTrans2 NMT service
ENDPOINT="${1:-http://localhost:3000/translate}"

echo "=========================================="
echo "Testing BentoML IndicTrans2 NMT Service"
echo "Endpoint: $ENDPOINT"
echo "=========================================="
echo ""

# Test 1: English to Hindi
echo "Test 1: English to Hindi (eng_Latn -> hin_Deva)"
echo "----------------------------------------"
curl -X POST "$ENDPOINT" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?",
    "src_lang": "eng_Latn",
    "tgt_lang": "hin_Deva"
  }' | python3 -m json.tool
echo ""
echo ""

# Test 2: Hindi to English
echo "Test 2: Hindi to English (hin_Deva -> eng_Latn)"
echo "----------------------------------------"
curl -X POST "$ENDPOINT" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "नमस्ते, आप कैसे हैं?",
    "src_lang": "hin_Deva",
    "tgt_lang": "eng_Latn"
  }' | python3 -m json.tool
echo ""
echo ""

# Test 3: Hindi to Marathi (Indic-Indic)
echo "Test 3: Hindi to Marathi (hin_Deva -> mar_Deva)"
echo "----------------------------------------"
curl -X POST "$ENDPOINT" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "नमस्ते, आप कैसे हैं?",
    "src_lang": "hin_Deva",
    "tgt_lang": "mar_Deva"
  }' | python3 -m json.tool
echo ""
echo ""

echo "=========================================="
echo "Testing Complete"
echo "=========================================="

