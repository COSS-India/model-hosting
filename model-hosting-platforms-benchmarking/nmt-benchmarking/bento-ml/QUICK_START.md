# Quick Start Guide - BentoML IndicTrans2

## Prerequisites

1. **Request Access to Models** (Required):
   - Go to https://huggingface.co/ai4bharat/indictrans2-en-indic-1B and click "Request access"
   - Go to https://huggingface.co/ai4bharat/indictrans2-indic-en-1B and click "Request access"
   - Go to https://huggingface.co/ai4bharat/indictrans2-indic-indic-1B and click "Request access"
   - Wait for approval (usually quick)

2. **Get Your HuggingFace Token**:
   - Go to https://huggingface.co/settings/tokens
   - Create a token with "read" permissions
   - Copy the token

## Start the Service

```bash
cd /home/ubuntu/nmt-benchmarking/bento-ml

# Set your HuggingFace token (replace with your actual token)
export HF_TOKEN=YOUR_HF_TOKEN_HERE

# Start the service
./start_service.sh
```

## Test the Service

```bash
# English to Hindi
curl -X POST http://localhost:3000/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello, how are you?","src_lang":"eng_Latn","tgt_lang":"hin_Deva"}' | python3 -m json.tool

# Or use the test script
./test_curl.sh
```

## Troubleshooting

### 403 Forbidden Error

If you get a 403 error, it means:
1. Your token doesn't have access to the models yet
2. You need to request access on HuggingFace (see Prerequisites above)
3. Wait for access approval, then try again

### Check Service Status

```bash
# View logs
tail -f bentoml.log

# Check if service is running
ps aux | grep bentoml
```

### Stop the Service

```bash
pkill -f "bentoml serve"
```

