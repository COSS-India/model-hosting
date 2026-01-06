# Curl Commands for BentoML IndicTrans2 Service

## Basic Translation Requests

### English to Hindi
```bash
curl -X POST http://localhost:3000/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?",
    "src_lang": "eng_Latn",
    "tgt_lang": "hin_Deva"
  }' | python3 -m json.tool
```

### Hindi to English
```bash
curl -X POST http://localhost:3000/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "नमस्ते, आप कैसे हैं?",
    "src_lang": "hin_Deva",
    "tgt_lang": "eng_Latn"
  }' | python3 -m json.tool
```

### Hindi to Marathi (Indic-Indic)
```bash
curl -X POST http://localhost:3000/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "नमस्ते, आप कैसे हैं?",
    "src_lang": "hin_Deva",
    "tgt_lang": "mar_Deva"
  }' | python3 -m json.tool
```

## One-Line Commands

### English to Hindi (compact)
```bash
curl -X POST http://localhost:3000/translate -H "Content-Type: application/json" -d '{"text":"Hello, how are you?","src_lang":"eng_Latn","tgt_lang":"hin_Deva"}' | python3 -m json.tool
```

### Hindi to English (compact)
```bash
curl -X POST http://localhost:3000/translate -H "Content-Type: application/json" -d '{"text":"नमस्ते, आप कैसे हैं?","src_lang":"hin_Deva","tgt_lang":"eng_Latn"}' | python3 -m json.tool
```

## Language Codes (FLORES Format)

The service uses FLORES language codes. Common examples:

- `eng_Latn` - English
- `hin_Deva` - Hindi (Devanagari script)
- `mar_Deva` - Marathi
- `guj_Gujr` - Gujarati
- `tel_Telu` - Telugu
- `tam_Taml` - Tamil
- `kan_Knda` - Kannada
- `mal_Mlym` - Malayalam
- `ben_Beng` - Bengali
- `pan_Guru` - Punjabi
- `ory_Orya` - Odia

For the complete list of 22 languages, see the [IndicTrans2 documentation](https://github.com/AI4Bharat/IndicTrans2).

## Expected Response Format

```json
{
  "translation": "नमस्ते, आप कैसे हैं?",
  "src_lang": "eng_Latn",
  "tgt_lang": "hin_Deva"
}
```

## Test Script

Use the provided test script for multiple test cases:

```bash
# Test with default endpoint (localhost:3000)
./test_curl.sh

# Test with custom endpoint
./test_curl.sh http://localhost:3001/translate
```

