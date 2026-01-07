# IndicNER Triton Deployment

This directory contains the Triton Inference Server deployment for the **ai4bharat/IndicNER** model, which performs Named Entity Recognition (NER) for 11 Indian languages.

## 📋 Overview

- **Model**: ai4bharat/IndicNER
- **Task**: Named Entity Recognition (Token Classification)
- **Languages**: 11 Indian languages (Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Oriya, Punjabi, Tamil, Telugu)
- **Backend**: Python (Triton)
- **Framework**: PyTorch + Transformers

## 🏗️ Directory Structure

```
ner-triton/
├── Dockerfile                          # Docker build configuration
├── README.md                           # This file
└── model_repository/
    └── ner/                           # Model name
        ├── config.pbtxt               # Triton model configuration
        └── 1/                         # Model version
            └── model.py               # Python backend implementation
```

## 🚀 Quick Start

### Build the Docker Image

```bash
cd /home/ubuntu/incubalm/ner-triton

# If the model is gated, set your HuggingFace token
export HUGGING_FACE_HUB_TOKEN="your_token_here"

# Build the image
docker build -t ner-triton:latest .
```

### Run the Triton Server

```bash
# Run with GPU support
docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
  --name ner-triton \
  ner-triton:latest

# Or run in detached mode
docker run -d --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
  --name ner-triton \
  ner-triton:latest
```

### Check Server Health

```bash
# Check if server is ready
curl http://localhost:8000/v2/health/ready

# List available models
curl http://localhost:8000/v2/models

# Get model metadata
curl http://localhost:8000/v2/models/ner
```

## 📊 Input/Output Format

### Triton API Format

**Input Parameters:**

1. **INPUT_TEXT** (Required)
   - Type: `TYPE_STRING` (BYTES)
   - Shape: `[1, 1]`
   - Format: Text string in one of the supported Indian languages
   - Example: `"राम दिल्ली में रहते हैं"`

2. **LANG_ID** (Required)
   - Type: `TYPE_STRING` (BYTES)
   - Shape: `[1, 1]`
   - Supported Values: `["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]`
   - Example: `"hi"` (Hindi)

**Output:**

- **OUTPUT_TEXT**
  - Type: `TYPE_STRING` (BYTES)
  - Shape: `[1, 1]`
  - Format: JSON string containing NER predictions
  - Structure:
    ```json
    {
      "source": "राम दिल्ली में रहते हैं",
      "nerPrediction": [
        {"entity": "राम", "class": "PERSON", "score": 0.95},
        {"entity": "दिल्ली", "class": "LOCATION", "score": 0.98}
      ]
    }
    ```

### Example Request (HTTP)

```bash
curl -X POST http://localhost:8000/v2/models/ner/infer \
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

## 🌍 Supported Languages

| Language | ISO Code |
|----------|----------|
| Assamese | `as` |
| Bengali | `bn` |
| Gujarati | `gu` |
| Hindi | `hi` |
| Kannada | `kn` |
| Malayalam | `ml` |
| Marathi | `mr` |
| Oriya | `or` |
| Punjabi | `pa` |
| Tamil | `ta` |
| Telugu | `te` |

## 🏷️ Entity Types

The IndicNER model recognizes various entity types including:
- **PERSON**: Names of people
- **LOCATION**: Geographic locations
- **ORGANIZATION**: Companies, institutions, etc.
- And other entity types as defined by the model

The exact entity types are determined by the model's training data and configuration.

## ⚙️ Configuration

### Model Configuration (config.pbtxt)

- **Backend**: Python
- **Max Batch Size**: 64
- **Instance Group**: 1 GPU instance
- **Dynamic Batching**: Enabled with preferred batch sizes [1, 2, 4, 8, 16, 32, 64]

### Performance Tuning

To adjust performance:
1. Modify `max_batch_size` in `config.pbtxt`
2. Adjust `preferred_batch_size` for dynamic batching
3. Change `instance_group.count` for multiple model instances

## 🔐 Authentication

⚠️ **IMPORTANT**: The IndicNER model is **gated** on HuggingFace and requires authentication and access approval.

### Steps to Get Access:

1. **Create a HuggingFace Account** (if you don't have one):
   - Go to: https://huggingface.co/join

2. **Request Access to the Model**:
   - Visit: https://huggingface.co/ai4bharat/IndicNER
   - Click on "Request Access" or "Agree and Access Repository"
   - Wait for approval (usually instant for most users)

3. **Get Your Access Token**:
   - Go to: https://huggingface.co/settings/tokens
   - Create a new token with "Read" permissions
   - Copy the token (starts with `hf_...`)

4. **Set the Token as Environment Variable**:
   ```bash
   export HUGGING_FACE_HUB_TOKEN="hf_your_token_here"
   ```

### Verifying Access:

You can verify your token has access by running:
```bash
python3 -c "from huggingface_hub import login; login(token='hf_your_token_here'); from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('ai4bharat/IndicNER')"
```

If successful, you should see "Login successful" and the tokenizer will download without errors.

## 🧪 Testing

### Python Test Client

```python
import requests
import json

url = "http://localhost:8000/v2/models/ner/infer"

payload = {
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
}

response = requests.post(url, json=payload)
result = response.json()

# Parse the output
output_text = result["outputs"][0]["data"][0]
ner_result = json.loads(output_text)

print("Source:", ner_result["source"])
print("Entities:")
for entity in ner_result["nerPrediction"]:
    print(f"  - {entity['entity']}: {entity['class']} (score: {entity['score']:.2f})")
```

## 📝 Notes

- The model uses BIO (Begin-Inside-Outside) tagging scheme for entity recognition
- Subword tokens are automatically aggregated into complete words
- Maximum input length is 512 tokens (longer texts will be truncated)
- The model runs on GPU by default for better performance

## 📚 References

- **Model**: https://huggingface.co/ai4bharat/IndicNER
- **Paper**: [Naamapadam: A Large-Scale Named Entity Annotated Data for Indic Languages](https://arxiv.org/abs/2212.10168)
- **Triton Documentation**: https://docs.nvidia.com/deeplearning/triton-inference-server/

## 📄 License

The IndicNER model is released under the MIT License.

