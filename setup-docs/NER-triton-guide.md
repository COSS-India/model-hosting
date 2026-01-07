# ner-triton Service Guide

## 📖 What is ner-triton?

**NER** stands for **Named Entity Recognition**. This service can read text in Indian languages and automatically identify important entities like:
- **People's names** (e.g., "Rahul", "Priya")
- **Places** (e.g., "Delhi", "Mumbai", "Bangalore")
- **Organizations** (e.g., "Tata", "Infosys")
- **And more** depending on the language

### Real-World Use Cases
- **Information Extraction**: Extract key information from documents
- **Search Engines**: Improve search by understanding entities
- **Content Tagging**: Automatically tag content with relevant entities
- **Customer Support**: Extract names and locations from support tickets
- **Data Mining**: Extract structured data from unstructured text

---

## 🎯 What You Need Before Starting

### For Everyone (Non-Technical)

Before you can use this service, you need:
1. **A computer with Linux** (Ubuntu recommended)
2. **An NVIDIA graphics card** (GPU) - This makes the service run much faster
3. **Internet connection** - To download the necessary software and models
4. **HuggingFace account** - This service requires special access (see Authentication section)

### For Technical Users

**System Requirements:**
- **OS**: Linux (Ubuntu 20.04+)
- **GPU**: NVIDIA GPU with CUDA support
- **Docker**: Version 20.10+
- **Docker Compose**: Version 1.29+
- **NVIDIA Container Toolkit**: For GPU access in Docker
- **HuggingFace Account**: Required for model access
- **Hardware Specifications**: May vary depending on the scale of your application
- **Tested Machine**: g4dn.2xlarge (For detailed specifications and pricing, check [AWS EC2 g4dn.2xlarge](https://instances.vantage.sh/aws/ec2/g4dn.2xlarge?currency=USD))

> **Note**: The model used in this service is provided as a reference implementation. You can replace it with your own trained model or use different model variants based on your specific requirements, performance needs, and use case.

**Software Installation:**
```bash
# Install Docker
sudo apt-get update
sudo apt-get install -y docker.io docker-compose

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

---

## 🔐 Authentication Setup (IMPORTANT!)

This service uses a **gated model** on HuggingFace, which means you need special access.

### Step 1: Create HuggingFace Account

1. Go to: https://huggingface.co/join
2. Create a free account
3. Verify your email

### Step 2: Request Model Access

1. Visit: https://huggingface.co/ai4bharat/IndicNER
2. Click **"Request Access"** or **"Agree and Access Repository"**
3. Wait for approval (usually instant for most users)

### Step 3: Create Access Token

1. Go to: https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Name it (e.g., "NER-triton")
4. Select **"Read"** permissions
5. Click **"Generate token"**
6. **Copy the token** (starts with `hf_...`) - You won't see it again!

### Step 4: Set Environment Variable

```bash
export HUGGING_FACE_HUB_TOKEN="hf_your_token_here"
```

**Important**: You need to set this before building and running the service!

### Step 5: Verify Access

```bash
python3 -c "from huggingface_hub import login; login(token='hf_your_token_here'); from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('ai4bharat/IndicNER')"
```

If successful, you should see "Login successful" and the tokenizer will download.

---

## 🏗️ Understanding the Service Structure

The service uses a BERT-based model fine-tuned for Indian languages to identify entities in text.

```
ner-triton/
├── Dockerfile              # Recipe to build the container
├── README.md               # Documentation
├── test_client.py          # Test script
└── model_repository/       # Model storage
    └── ner/
        ├── config.pbtxt    # Service configuration
        └── 1/
            └── model.py    # Processing logic
```

---

## 🔨 Step 1: Building the Docker Image

### What is Building?

Building a Docker image packages everything needed to run the service: the code, the AI model, and all dependencies. Once built, you can run it anywhere.

### Step-by-Step Build Instructions

#### Option A: Simple Build (Recommended for Beginners)

1. **Set your HuggingFace token** (if not already set):
   ```bash
   export HUGGING_FACE_HUB_TOKEN="hf_your_token_here"
   ```

2. **Open a terminal** (command line window)

3. **Navigate to the ner-triton folder:**
   ```bash
   cd ner-triton
   ```

4. **Build the image:**
   ```bash
   docker build -t ner-triton:latest .
   ```
   
   **What this does:**
   - `docker build` = Start building
   - `-t ner-triton:latest` = Name the image "ner-triton" with tag "latest"
   - `.` = Use the current directory (where Dockerfile is located)

5. **Wait for it to complete** (this may take 10-20 minutes the first time)
   - Downloads the base Triton server
   - Installs Python packages (transformers, torch, etc.)
   - Downloads the IndicNER model (requires authentication)
   - Sets everything up

#### Option B: Understanding What Happens During Build

The Dockerfile does these steps:
1. **Starts with Triton Server base image** - Pre-configured server
2. **Installs Python packages** - Transformers, PyTorch, and other ML libraries
3. **Copies the model code** - Your custom processing logic
4. **Pre-downloads the model** - Gets the IndicNER model ready (requires token)
   - Model source: [https://huggingface.co/ai4bharat/IndicNER](https://huggingface.co/ai4bharat/IndicNER)

**Note**: If the token is not set during build, the model will download on first run instead.

**Expected Output:**
```
Step 1/6 : FROM nvcr.io/nvidia/tritonserver:24.01-py3
...
Downloading IndicNER model...
Model downloaded successfully
...
Successfully built abc123def456
Successfully tagged ner-triton:latest
```

**Troubleshooting Build Issues:**
- **"Cannot connect to Docker daemon"**: Run `sudo systemctl start docker`
- **"No space left on device"**: Free up disk space
- **"Authentication failed"**: Verify your HuggingFace token is correct
- **"Model access denied"**: Make sure you requested access to the model
- **"Network timeout"**: Check internet connection, the build downloads large files

---

## 📥 How the IndicNER Model Was Obtained from HuggingFace

### Model Source

The ner-triton service uses the **IndicNER** (Indic Named Entity Recognition) model from HuggingFace:
- **Model Repository**: [https://huggingface.co/ai4bharat/IndicNER](https://huggingface.co/ai4bharat/IndicNER)
- **Model Type**: Named Entity Recognition for Indian Languages
- **Organization**: AI4Bharat
- **Access**: Gated model (requires HuggingFace account and access request)

### About the Model

IndicNER is a BERT-based model fine-tuned for Named Entity Recognition in Indian languages that:

- **Supports 11 Indian languages**:
  - Assamese (as), Bengali (bn), Gujarati (gu), Hindi (hi), Kannada (kn)
  - Malayalam (ml), Marathi (mr), Oriya (or), Punjabi (pa), Tamil (ta), Telugu (te)

- **Recognizes entity types**:
  - **PERSON**: Names of people (e.g., "Rahul", "Priya")
  - **LOCATION**: Geographic locations (e.g., "Delhi", "Mumbai", "Bangalore")
  - **ORGANIZATION**: Companies, institutions (e.g., "Tata", "Infosys")
  - And more entity types depending on the training data

- **Based on Naamapadam dataset**:
  - Large-scale named entity annotated data for Indic languages
  - Paper: [Naamapadam: A Large-Scale Named Entity Annotated Data for Indic Languages](https://arxiv.org/abs/2212.10168)

### How the Model is Downloaded

During the Docker build or first run, the model is downloaded from HuggingFace:

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Downloads IndicNER model from HuggingFace
tokenizer = AutoTokenizer.from_pretrained('ai4bharat/IndicNER')
model = AutoModelForTokenClassification.from_pretrained('ai4bharat/IndicNER')
```

**What happens:**
1. HuggingFace Hub authenticates using your token
2. Downloads the IndicNER model files (weights, tokenizer, config)
3. Model is cached locally
4. Model is ready to use for inference

**Note**: This is a **gated model**, which means:
- You need a HuggingFace account
- You must request access to the model
- You need a HuggingFace access token with "Read" permissions

### Model Architecture

- **Base Model**: BERT-based architecture
- **Task**: Token Classification (Named Entity Recognition)
- **Input**: Text in one of 11 supported Indian languages
- **Output**: BIO-tagged tokens (B-ENTITY, I-ENTITY, O)
- **Max Input Length**: 512 tokens

### Performance Characteristics

- **Accuracy**: High for well-trained languages
- **Speed**: Fast on GPU, slower on CPU
- **Best For**: Clear text with proper language code
- **Limitations**: Longer texts are truncated to 512 tokens

### References

- **HuggingFace Model**: [https://huggingface.co/ai4bharat/IndicNER](https://huggingface.co/ai4bharat/IndicNER)
- **Research Paper**: [Naamapadam: A Large-Scale Named Entity Annotated Data for Indic Languages](https://arxiv.org/abs/2212.10168)
- **Organization**: AI4Bharat

---

## 🚀 Step 2: Running the Service

### What is Running?

Running the service means starting it up so it can accept requests. It's like opening a restaurant for business.

### Step-by-Step Run Instructions

#### Basic Run (For Testing)

**Important**: Make sure your HuggingFace token is set!

```bash
export HUGGING_FACE_HUB_TOKEN="hf_your_token_here"

docker run --gpus all \
  -p 8300:8000 \
  -p 8301:8001 \
  -p 8302:8002 \
  -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
  --name ner-triton \
  ner-triton:latest
```

**What each part means:**
- `docker run` = Start a container
- `--gpus all` = Use all available GPUs (makes it faster)
- `-p 8300:8000` = Map port 8300 on your computer to port 8000 in container
- `-p 8301:8001` = Map gRPC port
- `-p 8302:8002` = Map metrics port
- `-e HUGGING_FACE_HUB_TOKEN=...` = Pass the token to the container
- `--name ner-triton` = Name the container "ner-triton"
- `ner-triton:latest` = Use the image we built

#### Run in Background (Recommended for Production)

```bash
export HUGGING_FACE_HUB_TOKEN="hf_your_token_here"

docker run -d --gpus all \
  -p 8300:8000 \
  -p 8301:8001 \
  -p 8302:8002 \
  -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
  --name ner-triton \
  ner-triton:latest
```

The `-d` flag runs it in the background (detached mode).

#### Using Docker Compose (Easiest Method)

If you're using the main docker-compose.yml file:

```bash
export HUGGING_FACE_HUB_TOKEN="hf_your_token_here"
docker-compose up -d ner-triton
```

This automatically handles all the configuration.

### Understanding Ports

- **Port 8300 (HTTP)**: Main entrance for web requests
- **Port 8301 (gRPC)**: Fast lane for program-to-program communication
- **Port 8302 (Metrics)**: Monitoring room for checking service health

---

## ✅ Step 3: Verifying the Service is Running

### Check if Container is Running

```bash
docker ps
```

You should see `ner-triton` in the list with status "Up".

### Check Service Health

```bash
curl http://localhost:8300/v2/health/ready
```

**Expected Response:** `{"status":"ready"}`

If you see this, the service is ready to accept requests!

### List Available Models

```bash
curl http://localhost:8300/v2/models
```

**Expected Response:**
```json
{
  "models": [
    {
      "name": "ner",
      "platform": "python",
      "versions": ["1"]
    }
  ]
}
```

### View Logs

```bash
docker logs ner-triton
```

Look for:
- `"Server is ready to receive inference requests"` = Success!
- `"Authentication failed"` = Token issue, check your HuggingFace token
- Any error messages = Something went wrong

---

## 🧪 Step 4: Testing the Service

> **Note**: If you're accessing the service from a remote machine, replace `localhost` with your server's IP address. For example, if your server IP is `192.168.1.100`, use `http://192.168.1.100:8300` instead of `http://localhost:8300`.

### Method 1: Manual Testing with curl

#### Hindi Example

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

**Expected Response:**
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

#### Tamil Example

```bash
curl -X POST http://localhost:8300/v2/models/ner/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "INPUT_TEXT",
        "shape": [1, 1],
        "datatype": "BYTES",
        "data": [["ராமன் சென்னையில் வசிக்கிறார்"]]
      },
      {
        "name": "LANG_ID",
        "shape": [1, 1],
        "datatype": "BYTES",
        "data": [["ta"]]
      }
    ],
    "outputs": [
      {
        "name": "OUTPUT_TEXT"
      }
    ]
  }'
```

### Method 2: Python Test Script

Create a file `test_my_text.py`:

```python
import requests
import json

# Your text to test (Hindi example)
text = "राम दिल्ली में रहते हैं"
lang_id = "hi"  # Language code

# Prepare request
url = "http://localhost:8300/v2/models/ner/infer"
payload = {
    "inputs": [
        {
            "name": "INPUT_TEXT",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [[text]]
        },
        {
            "name": "LANG_ID",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [[lang_id]]
        }
    ],
    "outputs": [
        {"name": "OUTPUT_TEXT"}
    ]
}

# Send request
response = requests.post(url, json=payload)
result = response.json()

# Parse and display results
output_text = result["outputs"][0]["data"][0]
ner_result = json.loads(output_text)

print(f"Source: {ner_result['source']}")
print("\nEntities:")
for entity in ner_result['nerPrediction']:
    print(f"  - {entity['entity']}: {entity['class']} (score: {entity['score']:.2f})")
```

Run it:
```bash
python3 test_my_text.py
```

**Expected Output:**
```
Source: राम दिल्ली में रहते हैं
Entities:
  - राम: PERSON (score: 0.95)
  - दिल्ली: LOCATION (score: 0.98)
```

---

## 📊 Understanding the API

### Input Format

1. **INPUT_TEXT** (Required)
   - **Type**: String (text)
   - **What to send**: The text you want to analyze
   - **Supports**: 11 Indian languages

2. **LANG_ID** (Required)
   - **Type**: String (language code)
   - **What to send**: The language code of your text
   - **Supported Values**: `as`, `bn`, `gu`, `hi`, `kn`, `ml`, `mr`, `or`, `pa`, `ta`, `te`

### Output Format

The service returns a JSON string with:

```json
{
  "source": "original input text",
  "nerPrediction": [
    {
      "entity": "extracted entity text",
      "class": "entity type",
      "score": 0.95
    }
  ]
}
```

**Fields:**
- **source**: The text you sent
- **nerPrediction**: Array of detected entities with:
  - **entity**: The actual text that was identified
  - **class**: Type of entity (PERSON, LOCATION, ORGANIZATION, etc.)
  - **score**: Confidence score (0.0 to 1.0)

### Supported Languages

| Language | ISO Code | Script |
|----------|----------|--------|
| Assamese | `as` | Bengali |
| Bengali | `bn` | Bengali |
| Gujarati | `gu` | Gujarati |
| Hindi | `hi` | Devanagari |
| Kannada | `kn` | Kannada |
| Malayalam | `ml` | Malayalam |
| Marathi | `mr` | Devanagari |
| Oriya | `or` | Odia |
| Punjabi | `pa` | Gurmukhi |
| Tamil | `ta` | Tamil |
| Telugu | `te` | Telugu |

### Entity Types

The model recognizes various entity types including:
- **PERSON**: Names of people
- **LOCATION**: Geographic locations (cities, countries, etc.)
- **ORGANIZATION**: Companies, institutions, etc.
- **And more** depending on the training data

---

## 🧠 How It Works (Technical Details)

### NER Process

1. **Tokenization**:
   - Text is split into tokens (words/subwords)
   - Uses language-specific tokenizers

2. **Model Inference**:
   - BERT-based model processes the tokens
   - Each token gets a label (BIO scheme):
     - **B-ENTITY**: Beginning of an entity
     - **I-ENTITY**: Inside an entity
     - **O**: Outside (not an entity)

3. **Entity Extraction**:
   - Tokens are grouped into complete entities
   - Subword tokens are merged back into words
   - Confidence scores are calculated

4. **Output Formatting**:
   - Entities are formatted with their types
   - Results are returned as JSON

### Performance Characteristics

- **Max Input Length**: 512 tokens (longer texts are truncated)
- **Accuracy**: High for well-trained languages
- **Speed**: Fast on GPU, slower on CPU
- **Best For**: Clear text with proper language code

---

## ⚙️ Configuration Options

### Adjusting Performance

You can modify `model_repository/ner/config.pbtxt` to change:

1. **Max Batch Size**: How many requests to process together
   - Current: 64
   - Increase for more throughput (needs more GPU memory)
   - Decrease if running out of memory

2. **Dynamic Batching**: Automatically groups requests
   - Current: Enabled with sizes [1, 2, 4, 8, 16, 32, 64]
   - Adjust based on your workload

3. **GPU Instances**: Number of model copies
   - Current: 1
   - Increase for higher throughput (uses more GPU memory)

**After changing config.pbtxt, rebuild the image:**
```bash
docker build -t ner-triton:latest .
docker stop ner-triton
docker rm ner-triton
docker run -d --gpus all -p 8300:8000 -p 8301:8001 -p 8302:8002 \
  -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
  --name ner-triton ner-triton:latest
```

---

## 🛠️ Troubleshooting

### Problem: Service Won't Start

**Symptoms**: Container exits immediately or shows errors

**Solutions**:
1. **Check logs**: `docker logs ner-triton`
2. **Verify GPU**: `nvidia-smi` should show your GPU
3. **Check port conflicts**: `sudo lsof -i :8300` (should be empty)
4. **Verify Docker GPU access**: 
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

### Problem: Authentication Errors

**Symptoms**: "Authentication failed" or "Model access denied"

**Solutions**:
1. **Verify token is set**: `echo $HUGGING_FACE_HUB_TOKEN`
2. **Check token is correct**: Should start with `hf_`
3. **Request model access**: Visit https://huggingface.co/ai4bharat/IndicNER
4. **Regenerate token** if expired
5. **Set token in container**: Make sure `-e HUGGING_FACE_HUB_TOKEN=...` is used

### Problem: Model Download Fails

**Symptoms**: Errors downloading model during build or runtime

**Solutions**:
1. **Check internet connection**
2. **Verify model access**: Make sure you requested access
3. **Check token**: Ensure token has "Read" permissions
4. **Retry**: Network issues can be temporary

### Problem: "Out of Memory" Errors

**Symptoms**: CUDA out of memory errors in logs

**Solutions**:
1. **Reduce batch size** in config.pbtxt
2. **Use shorter text inputs** (max 512 tokens)
3. **Check GPU memory**: `nvidia-smi`
4. **Close other GPU applications**

### Problem: No Entities Detected

**Symptoms**: Service returns empty entity list

**Solutions**:
1. **Check language code**: Make sure it matches the text language
2. **Use longer text**: At least a few sentences
3. **Verify text contains entities**: Names, places, organizations
4. **Check text quality**: Clear, well-formatted text works better

### Problem: Wrong Entities Detected

**Symptoms**: Service identifies incorrect entities

**Solutions**:
1. **Verify language code**: Must match the text language
2. **Check text quality**: Clear, properly formatted text
3. **Review confidence scores**: Low scores may indicate uncertainty
4. **Try multiple examples**: Some texts may be ambiguous

### Problem: Cannot Connect to Service

**Symptoms**: Connection refused or timeout errors

**Solutions**:
1. **Verify service is running**: `docker ps`
2. **Check port mapping**: `docker ps` should show port mappings
3. **Test from container**: 
   ```bash
   docker exec ner-triton curl http://localhost:8000/v2/health/ready
   ```
4. **Check firewall**: `sudo ufw status`

---

## 📈 Monitoring and Metrics

### Health Check

```bash
curl http://localhost:8300/v2/health/ready
```

### Metrics Endpoint

```bash
curl http://localhost:8302/metrics
```

This provides Prometheus-compatible metrics including:
- Request count
- Inference latency
- GPU utilization
- Error rates

### View Real-Time Logs

```bash
docker logs -f ner-triton
```

Press `Ctrl+C` to stop viewing logs.

---

## 🔄 Common Operations

### Stop the Service

```bash
docker stop ner-triton
```

### Start the Service

```bash
docker start ner-triton
```

### Restart the Service

```bash
docker restart ner-triton
```

### Remove the Service

```bash
docker stop ner-triton
docker rm ner-triton
```

### Update the Service

```bash
# Rebuild with latest changes
export HUGGING_FACE_HUB_TOKEN="hf_your_token_here"
docker build -t ner-triton:latest .

# Stop and remove old container
docker stop ner-triton
docker rm ner-triton

# Start new container
docker run -d --gpus all -p 8300:8000 -p 8301:8001 -p 8302:8002 \
  -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
  --name ner-triton ner-triton:latest
```

---

## 📚 Additional Resources

### Service Documentation
- **Detailed README**: `ner-triton/README.md`
- **Model Source (HuggingFace)**: [https://huggingface.co/ai4bharat/IndicNER](https://huggingface.co/ai4bharat/IndicNER)
  - This is where the IndicNER model was obtained from
- **Paper**: [Naamapadam: A Large-Scale Named Entity Annotated Data for Indic Languages](https://arxiv.org/abs/2212.10168)

### Technical References
- **Triton Server Docs**: https://docs.nvidia.com/deeplearning/triton-inference-server/
- **Docker Docs**: https://docs.docker.com/
- **HuggingFace Docs**: https://huggingface.co/docs

### Getting Help

If you encounter issues:
1. Check the logs: `docker logs ner-triton`
2. Review this guide's troubleshooting section
3. Check the service README: `ner-triton/README.md`
4. Review HuggingFace model page for access issues

---

## 📝 Quick Reference

### Essential Commands

```bash
# Set token (required!)
export HUGGING_FACE_HUB_TOKEN="hf_your_token_here"

# Build
cd ner-triton
docker build -t ner-triton:latest .

# Run
docker run -d --gpus all -p 8300:8000 -p 8301:8001 -p 8302:8002 \
  -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
  --name ner-triton ner-triton:latest

# Check status
docker ps
curl http://localhost:8300/v2/health/ready

# Test
cd ner-triton
python3 test_client.py

# View logs
docker logs -f ner-triton

# Stop
docker stop ner-triton
```

### Port Information

- **HTTP API**: `http://localhost:8300`
- **gRPC API**: `localhost:8301`
- **Metrics**: `http://localhost:8302/metrics`

### Model Information

- **Model Name**: `ner`
- **Backend**: Python
- **Max Batch Size**: 64
- **GPU Required**: Yes
- **Authentication Required**: Yes (HuggingFace)
- **Supported Languages**: 11 Indian languages

---

## ✅ Summary

You've learned how to:
1. ✅ Set up HuggingFace authentication
2. ✅ Build the ner-triton Docker image
3. ✅ Run the service with authentication
4. ✅ Verify it's working
5. ✅ Test named entity recognition
6. ✅ Use the API with different languages
7. ✅ Troubleshoot common issues

The ner-triton service is now ready to extract entities from Indian language text! For production use, consider setting up monitoring, load balancing, and proper security measures.



