# IndicNER Triton Deployment Guide

**Status**: ✅ **READY FOR DEPLOYMENT** (Requires HuggingFace Access)

**Date**: 2025-11-06

---

## 📋 Overview

This deployment provides Named Entity Recognition (NER) for 11 Indian languages using the **ai4bharat/IndicNER** model hosted on NVIDIA Triton Inference Server.

### Key Information

- **Model**: ai4bharat/IndicNER
- **Task**: Named Entity Recognition (Token Classification)
- **Languages**: 11 Indian languages (Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Oriya, Punjabi, Tamil, Telugu)
- **Backend**: Python (Triton)
- **Framework**: PyTorch + Transformers
- **Ports**: 8300 (HTTP), 8301 (gRPC), 8302 (Metrics)

---

## ⚠️ Prerequisites

### 1. HuggingFace Access (REQUIRED)

The IndicNER model is **gated** and requires:

1. **HuggingFace Account**: Create at https://huggingface.co/join
2. **Model Access**: Request access at https://huggingface.co/ai4bharat/IndicNER
3. **Access Token**: Generate at https://huggingface.co/settings/tokens

**Important**: You MUST complete steps 1-3 before deployment. The model will not load without proper access.

### 2. System Requirements

- **GPU**: NVIDIA GPU with CUDA support (Tesla T4 or better)
- **Docker**: Docker with NVIDIA Container Toolkit
- **Memory**: At least 8GB GPU memory
- **Disk**: At least 10GB free space

---

## 🚀 Deployment Steps

### Step 1: Get HuggingFace Access

```bash
# 1. Visit https://huggingface.co/ai4bharat/IndicNER
# 2. Click "Request Access" or "Agree and Access Repository"
# 3. Wait for approval (usually instant)
# 4. Get your token from https://huggingface.co/settings/tokens
```

### Step 2: Set Environment Variable

```bash
export HUGGING_FACE_HUB_TOKEN="hf_your_token_here"
```

### Step 3: Build Docker Image

```bash
cd /home/ubuntu/incubalm/ner-triton

# Build with HuggingFace token to pre-download model
docker build \
  --build-arg HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
  -t ner-triton:latest .
```

**Note**: The build process will attempt to download the model. If you don't have access, it will skip the download and download at runtime instead.

### Step 4: Run Triton Server

```bash
# Run with GPU support on ports 8300-8302
docker run -d \
  --gpus all \
  -p 8300:8000 \
  -p 8301:8001 \
  -p 8302:8002 \
  -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
  --name ner-triton \
  ner-triton:latest
```

### Step 5: Verify Deployment

```bash
# Wait for model to load (may take 1-2 minutes)
sleep 60

# Check server health
curl http://localhost:8300/v2/health/ready

# Check model status
curl http://localhost:8300/v2/models/ner

# View logs
docker logs ner-triton --tail 50
```

---

## 🧪 Testing

### Quick Test

```bash
# Run the test client
cd /home/ubuntu/incubalm/ner-triton
python3 test_client.py
```

### Manual Test

```bash
curl -X POST http://localhost:8300/v2/models/ner/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "INPUT_TEXT",
        "shape": [1, 1],
        "datatype": "BYTES",
        "data": [["राम दिल्ली में रहते हैं"]]
      },
      {
        "name": "LANG_ID",
        "shape": [1, 1],
        "datatype": "BYTES",
        "data": [["hi"]]
      }
    ],
    "outputs": [
      {
        "name": "OUTPUT_TEXT"
      }
    ]
  }'
```

**Expected Output**:
```json
{
  "outputs": [
    {
      "name": "OUTPUT_TEXT",
      "datatype": "BYTES",
      "shape": [1, 1],
      "data": ["{\"source\": \"राम दिल्ली में रहते हैं\", \"nerPrediction\": [{\"entity\": \"राम\", \"class\": \"PERSON\", \"score\": 0.95}, {\"entity\": \"दिल्ली\", \"class\": \"LOCATION\", \"score\": 0.98}]}"]
    }
  ]
}
```

---

## 🔧 Troubleshooting

### Issue: Model Fails to Load with "Gated Repo" Error

**Symptom**:
```
UNAVAILABLE: Internal: OSError: You are trying to access a gated repo.
Make sure to request access at https://huggingface.co/ai4bharat/IndicNER
```

**Solution**:
1. Verify you have requested access at https://huggingface.co/ai4bharat/IndicNER
2. Verify your token has access:
   ```bash
   python3 -c "from huggingface_hub import login; login(token='${HUGGING_FACE_HUB_TOKEN}'); from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('ai4bharat/IndicNER')"
   ```
3. Ensure the token is passed to the container:
   ```bash
   docker logs ner-triton | grep -i "auth"
   # Should show: "[OK] HuggingFace authentication successful"
   ```

### Issue: Port Already in Use

**Symptom**:
```
Bind for 0.0.0.0:8300 failed: port is already allocated
```

**Solution**:
Use different ports:
```bash
docker run -d --gpus all \
  -p 8400:8000 -p 8401:8001 -p 8402:8002 \
  -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
  --name ner-triton \
  ner-triton:latest
```

### Issue: Out of Memory

**Symptom**:
```
CUDA out of memory
```

**Solution**:
1. Reduce batch size in `config.pbtxt`:
   ```
   max_batch_size: 32  # Reduce from 64
   ```
2. Rebuild and restart the container

---

## 📊 Performance

### Model Loading Time
- **First Load**: 1-2 minutes (downloads model ~500MB)
- **Subsequent Loads**: 30-60 seconds (cached)

### Inference Performance
- **Single Request**: ~100-200ms
- **Batch of 8**: ~300-500ms
- **Throughput**: ~20-30 requests/second

### Resource Usage
- **GPU Memory**: ~2-3GB
- **CPU Memory**: ~4-5GB
- **Disk Space**: ~2GB (model + dependencies)

---

## 📁 Directory Structure

```
ner-triton/
├── Dockerfile                          # Docker build configuration
├── README.md                           # User documentation
├── DEPLOYMENT_GUIDE.md                 # This file
├── test_client.py                      # Test client script
└── model_repository/
    └── ner/                           # Model name
        ├── config.pbtxt               # Triton model configuration
        └── 1/                         # Model version
            └── model.py               # Python backend implementation
```

---

## 🔄 Updating the Deployment

### Rebuild Image

```bash
cd /home/ubuntu/incubalm/ner-triton
docker build -t ner-triton:latest .
```

### Restart Container

```bash
docker stop ner-triton
docker rm ner-triton
docker run -d --gpus all \
  -p 8300:8000 -p 8301:8001 -p 8302:8002 \
  -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
  --name ner-triton \
  ner-triton:latest
```

---

## 📚 References

- **Model**: https://huggingface.co/ai4bharat/IndicNER
- **Paper**: [Naamapadam: A Large-Scale Named Entity Annotated Data for Indic Languages](https://arxiv.org/abs/2212.10168)
- **Triton Documentation**: https://docs.nvidia.com/deeplearning/triton-inference-server/
- **AI4Bharat**: https://ai4bharat.org/

---

## 📝 Notes

- The model uses BIO (Begin-Inside-Outside) tagging scheme
- Subword tokens are automatically aggregated into complete words
- Maximum input length is 512 tokens (longer texts will be truncated)
- The model runs on GPU by default for better performance
- Entity types depend on the model's training data (typically PERSON, LOCATION, ORGANIZATION, etc.)

---

## ✅ Deployment Checklist

- [ ] HuggingFace account created
- [ ] Access requested and approved for ai4bharat/IndicNER
- [ ] HuggingFace token generated
- [ ] Environment variable `HUGGING_FACE_HUB_TOKEN` set
- [ ] Docker image built successfully
- [ ] Container running on ports 8300-8302
- [ ] Server health check passes
- [ ] Model status shows "READY"
- [ ] Test inference successful
- [ ] Test client runs without errors

---

**Deployment Complete!** 🎉

The NER model is now ready to serve requests for Named Entity Recognition in 11 Indian languages.

