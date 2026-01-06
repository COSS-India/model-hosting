#!/bin/bash
# Test curl commands for MLflow NMT service

# English to Hindi
curl -X POST http://localhost:5000/invocations \
  -H "Content-Type: application/json" \
  -d '{"inputs":{"text":"Hello, how are you?","src_lang":"eng_Latn","tgt_lang":"hin_Deva"}}' | python3 -m json.tool

# Hindi to English
curl -X POST http://localhost:5000/invocations \
  -H "Content-Type: application/json" \
  -d '{"inputs":{"text":"नमस्ते, आप कैसे हैं?","src_lang":"hin_Deva","tgt_lang":"eng_Latn"}}' | python3 -m json.tool

# Hindi to Marathi (Indic-Indic)
curl -X POST http://localhost:5000/invocations \
  -H "Content-Type: application/json" \
  -d '{"inputs":{"text":"नमस्ते","src_lang":"hin_Deva","tgt_lang":"mar_Deva"}}' | python3 -m json.tool

