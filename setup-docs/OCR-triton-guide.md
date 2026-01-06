# Surya-ocr-triton Service Guide

## 📖 What is Surya-ocr-triton?

**Surya OCR** is an Optical Character Recognition service that can **read text from images**. It can extract text from photos, scanned documents, screenshots, and other images. It supports **90+ languages** including all major Indic languages (Hindi, Tamil, Telugu, Bengali, etc.) and many international languages.

### Real-World Use Cases
- **Document Digitization**: Convert scanned documents to editable text
- **Receipt Processing**: Extract information from receipts and invoices
- **Form Processing**: Automatically fill forms from scanned documents
- **Content Extraction**: Extract text from images for search and indexing
- **Accessibility**: Make images accessible by extracting text for screen readers

---

## 🎯 What You Need Before Starting

### For Everyone (Non-Technical)

Before you can use this service, you need:
1. **A computer with Linux** (Ubuntu recommended)
2. **An NVIDIA graphics card** (GPU) - This makes the service run much faster
3. **Internet connection** - To download the necessary software and models

### For Technical Users

**System Requirements:**
- **OS**: Linux (Ubuntu 20.04+)
- **GPU**: NVIDIA GPU with CUDA support (recommended for performance)
- **Docker**: Version 20.10+
- **Docker Compose**: Version 1.29+
- **NVIDIA Container Toolkit**: For GPU access in Docker
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

## 🏗️ Understanding the Service Structure

Surya OCR uses advanced deep learning models to detect and recognize text in images.

```
surya-ocr-triton/
├── Dockerfile              # Recipe to build the container
├── README.md               # Documentation
├── test_client.py          # Test script
├── sample_payload.json     # Example request payload
└── model_repository/       # Model storage
    └── surya_ocr/
        ├── config.pbtxt    # Service configuration
        └── 1/
            └── model.py    # Processing logic
```

---

## 🔨 Step 1: Building the Docker Image

### What is Building?

Building a Docker image packages everything needed to run the service: the code, the AI models, and all dependencies. Once built, you can run it anywhere.

### Step-by-Step Build Instructions

#### Option A: Simple Build (Recommended for Beginners)

1. **Open a terminal** (command line window)

2. **Navigate to the surya-ocr-triton folder:**
   ```bash
   cd surya-ocr-triton
   ```

3. **Build the image:**
   ```bash
   docker build -t surya-ocr-triton:latest .
   ```
   
   **What this does:**
   - `docker build` = Start building
   - `-t surya-ocr-triton:latest` = Name the image "surya-ocr-triton" with tag "latest"
   - `.` = Use the current directory (where Dockerfile is located)

4. **Wait for it to complete** (this may take 15-30 minutes the first time)
   - Downloads the base Triton server (24.08 version)
   - Installs Surya OCR package (which includes PyTorch and all dependencies)
   - Downloads OCR models (detection and recognition models)
   - Sets everything up

#### Option B: Understanding What Happens During Build

The Dockerfile does these steps:
1. **Starts with Triton Server base image** - Pre-configured server (version 24.08 for newer PyTorch support)
2. **Installs system dependencies** - Graphics libraries (libgl1-mesa-glx, libglib2.0-0)
3. **Installs Surya OCR** - The main OCR package (version 0.17.0)
   - Source: [https://github.com/datalab-to/surya](https://github.com/datalab-to/surya)
   - This automatically installs PyTorch, transformers, and other dependencies
4. **Copies the model code** - Your custom processing logic
5. **Models download on first use** - OCR models are downloaded when the service first runs
   - Detection and recognition models are downloaded from the Surya repository

**Expected Output:**
```
Step 1/5 : FROM nvcr.io/nvidia/tritonserver:24.08-py3
...
Installing surya-ocr==0.17.0
...
Successfully built abc123def456
Successfully tagged surya-ocr-triton:latest
```

**Troubleshooting Build Issues:**
- **"Cannot connect to Docker daemon"**: Run `sudo systemctl start docker`
- **"No space left on device"**: Free up disk space (models are ~2-3GB)
- **"Network timeout"**: Check internet connection, the build downloads large files
- **"Package installation failed"**: Retry the build, network issues can be temporary

---

## 📥 How the Surya OCR Model Was Obtained from GitHub

### Model Source

The Surya-ocr-triton service uses the **Surya OCR** package from GitHub:
- **Repository**: [https://github.com/datalab-to/surya](https://github.com/datalab-to/surya)
- **Model Type**: Optical Character Recognition (OCR)
- **Package**: `surya-ocr` (PyPI package, version 0.17.0)
- **License**: Open source

### About Surya OCR

Surya OCR is a state-of-the-art OCR system that:

- **Supports 90+ languages** including:
  - **Indic Languages**: Hindi, Bengali, Tamil, Telugu, Kannada, Malayalam, Gujarati, Marathi, Punjabi, Odia, Assamese, Urdu, and more
  - **International Languages**: English, Spanish, French, German, Italian, Portuguese, Chinese, Japanese, Korean, Arabic, and many more

- **Two-stage OCR process**:
  1. **Text Detection**: Detects where text is located in the image
     - Creates bounding boxes around text regions
     - Handles rotated and curved text
  2. **Text Recognition**: Recognizes the actual text in each detected region
     - Supports multiple languages automatically
     - Handles various fonts and styles

- **Performance**:
  - **Accuracy**: Benchmarks favorably vs Google Cloud Vision (0.97 vs 0.88 similarity)
  - **Speed**: 
    - Testing config (batch_size=64): 2-5 seconds per page
    - Production config (batch_size=512): 0.6-1 second per page
  - **GPU Accelerated**: Much faster on GPU than CPU
  - **Best For**: Printed text in documents (not handwriting)

### How the Model is Downloaded

During the Docker build process, the Surya OCR package is installed via pip:

```bash
pip install surya-ocr==0.17.0
```

**What happens:**
1. Pip installs the `surya-ocr` package from PyPI
2. This automatically installs dependencies:
   - PyTorch
   - Transformers
   - Other ML libraries
3. **Models download on first use**:
   - Detection model: Downloads when first image is processed
   - Recognition model: Downloads when first image is processed
   - Models are cached locally after first download
4. Models are ready to use for OCR inference

### Model Architecture

The system uses:
- **Detection Model**: Deep learning model for text detection in images
- **Recognition Model**: Deep learning model for text recognition (supports 90+ languages)
- **Post-Processing**: Combines text from all regions, orders logically, provides confidence scores

### Installation

The Surya OCR package can be installed directly:

```bash
pip install surya-ocr
```

Or use the specific version:
```bash
pip install surya-ocr==0.17.0
```

### Usage Example

You can use Surya OCR directly in Python:

```python
from surya.ocr import run_ocr
from PIL import Image

# Load image
image = Image.open("your_image.png")

# Run OCR
predictions = run_ocr([image])

# Get results
for prediction in predictions:
    for text_line in prediction.text_lines:
        print(f"Text: {text_line.text}")
        print(f"Confidence: {text_line.confidence}")
```

### References

- **GitHub Repository**: [https://github.com/datalab-to/surya](https://github.com/datalab-to/surya)
- **PyPI Package**: [https://pypi.org/project/surya-ocr/](https://pypi.org/project/surya-ocr/)
- **Documentation**: Available in the GitHub repository

---

## 🚀 Step 2: Running the Service

### What is Running?

Running the service means starting it up so it can accept requests. It's like opening a restaurant for business.

### Step-by-Step Run Instructions

#### Basic Run (For Testing)

```bash
docker run --gpus all \
  -p 8400:8000 \
  -p 8401:8001 \
  -p 8402:8002 \
  -e TORCH_DEVICE=cuda \
  -e RECOGNITION_BATCH_SIZE=64 \
  -e DETECTOR_BATCH_SIZE=8 \
  --name surya-ocr-server \
  surya-ocr-triton:latest
```

**What each part means:**
- `docker run` = Start a container
- `--gpus all` = Use all available GPUs (makes it faster)
- `-p 8400:8000` = Map port 8400 on your computer to port 8000 in container
- `-p 8401:8001` = Map gRPC port
- `-p 8402:8002` = Map metrics port
- `-e TORCH_DEVICE=cuda` = Use GPU (use "cpu" if no GPU available)
- `-e RECOGNITION_BATCH_SIZE=64` = Process 64 text lines at once (adjust based on VRAM)
- `-e DETECTOR_BATCH_SIZE=8` = Process 8 detection regions at once
- `--name surya-ocr-server` = Name the container "surya-ocr-server"
- `surya-ocr-triton:latest` = Use the image we built

#### Run in Background (Recommended for Production)

```bash
docker run -d --gpus all \
  -p 8400:8000 \
  -p 8401:8001 \
  -p 8402:8002 \
  -e TORCH_DEVICE=cuda \
  -e RECOGNITION_BATCH_SIZE=64 \
  -e DETECTOR_BATCH_SIZE=8 \
  --name surya-ocr-server \
  surya-ocr-triton:latest
```

The `-d` flag runs it in the background (detached mode).

#### Using Docker Compose (Easiest Method)

If you're using the main docker-compose.yml file:

```bash
docker-compose up -d surya-ocr-server
```

This automatically handles all the configuration.

### Understanding Ports

- **Port 8400 (HTTP)**: Main entrance for web requests
- **Port 8401 (gRPC)**: Fast lane for program-to-program communication
- **Port 8402 (Metrics)**: Monitoring room for checking service health

### Understanding Environment Variables

You can adjust these to optimize performance:

- **TORCH_DEVICE**: `cuda` (GPU) or `cpu` (CPU only)
- **RECOGNITION_BATCH_SIZE**: How many text lines to process together
  - Default: 64 (uses ~2-4GB VRAM)
  - Production: 512 (uses ~20GB VRAM, much faster)
  - Lower if running out of memory: 32, 16, or 8
- **DETECTOR_BATCH_SIZE**: How many text regions to detect together
  - Default: 8 (uses ~2-3GB VRAM)
  - Production: 36 (uses ~16GB VRAM)
  - Lower if running out of memory: 4 or 2

---

## ✅ Step 3: Verifying the Service is Running

### Check if Container is Running

```bash
docker ps
```

You should see `surya-ocr-server` in the list with status "Up".

### Check Service Health

```bash
curl http://localhost:8400/v2/health/ready
```

**Expected Response:** `{"status":"ready"}`

If you see this, the service is ready to accept requests!

**Note**: The first request may take longer as models download and load.

### List Available Models

```bash
curl http://localhost:8400/v2/models
```

**Expected Response:**
```json
{
  "models": [
    {
      "name": "surya_ocr",
      "platform": "python",
      "versions": ["1"]
    }
  ]
}
```

### View Logs

```bash
docker logs surya-ocr-server
```

Look for:
- `"Server is ready to receive inference requests"` = Success!
- `"Downloading models..."` = Models are downloading (first time only)
- `"Using device: cuda"` = GPU is being used
- Any error messages = Something went wrong

---

## 🧪 Step 4: Testing the Service

> **Note**: If you're accessing the service from a remote machine, replace `localhost` with your server's IP address. For example, if your server IP is `192.168.1.100`, use `http://192.168.1.100:8400` instead of `http://localhost:8400`.

### Method 1: Manual Testing with curl

#### Step 1: Prepare Your Image

Convert your image to base64:
```bash
IMAGE_B64=$(base64 -w 0 your_image.png)
```

#### Step 2: Send Request

```bash
curl -X POST http://localhost:8400/v2/models/surya_ocr/infer \
  -H "Content-Type: application/json" \
  -d "{
    \"inputs\": [
      {
        \"name\": \"IMAGE_DATA\",
        \"shape\": [1, 1],
        \"datatype\": \"BYTES\",
        \"data\": [[\"$IMAGE_B64\"]]
      }
    ],
    \"outputs\": [
      {
        \"name\": \"OUTPUT_TEXT\"
      }
    ]
  }"
```

**Expected Response:**
```json
{
  "outputs": [
    {
      "name": "OUTPUT_TEXT",
      "datatype": "BYTES",
      "shape": [1, 1],
      "data": ["{\"success\": true, \"full_text\": \"Extracted text here...\", \"text_lines\": [{\"text\": \"Line 1\", \"confidence\": 0.95, \"bbox\": [x1, y1, x2, y2], \"polygon\": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]}, ...], \"image_bbox\": [0, 0, width, height]}"]
    }
  ]
}
```

#### Step 3: Using Sample Payload

You can also use the provided sample payload:

```bash
# First, create the payload (if not already created)
cd surya-ocr-triton
python3 create_test_payload.py  # Creates sample_payload.json

# Then send the request
curl -X POST http://localhost:8400/v2/models/surya_ocr/infer \
  -H "Content-Type: application/json" \
  -d @sample_payload.json
```

### Method 2: Python Test Script

Create a file `test_my_image.py`:

```python
import requests
import json
import base64
from PIL import Image

# Read your image file
with open("your_image.png", "rb") as f:
    image_bytes = f.read()
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')

# Prepare request
url = "http://localhost:8400/v2/models/surya_ocr/infer"
payload = {
    "inputs": [
        {
            "name": "IMAGE_DATA",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [[image_b64]]
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
ocr_result = json.loads(output_text)

if ocr_result['success']:
    print(f"Full Text:\n{ocr_result['full_text']}")
    print(f"\nText Lines ({len(ocr_result['text_lines'])}):")
    for i, line in enumerate(ocr_result['text_lines'], 1):
        print(f"  {i}. {line['text']} (confidence: {line['confidence']:.2f})")
else:
    print(f"Error: {ocr_result.get('error', 'Unknown error')}")
```

Run it:
```bash
python3 test_my_image.py
```

**Expected Output:**
```
Full Text:
Hello World
This is a test image.

Text Lines (2):
  1. Hello World (confidence: 0.95)
  2. This is a test image. (confidence: 0.92)
```

---

## 📊 Understanding the API

### Input Format

**IMAGE_DATA** (Required)
- **Type**: Base64-encoded string
- **Format**: PNG, JPEG, BMP, TIFF, WebP
- **What to send**: Your image file converted to base64
- **Recommended**: PNG or JPEG for best results

### Output Format

The service returns a JSON string with:

```json
{
  "success": true,
  "full_text": "All extracted text combined with newlines...",
  "text_lines": [
    {
      "text": "Line of text",
      "confidence": 0.95,
      "bbox": [x1, y1, x2, y2],
      "polygon": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    }
  ],
  "image_bbox": [0, 0, width, height]
}
```

**Fields:**
- **success**: Boolean indicating if OCR was successful
- **full_text**: All detected text combined with newlines
- **text_lines**: Array of detected text lines with:
  - **text**: The text content
  - **confidence**: Confidence score (0.0 to 1.0)
  - **bbox**: Axis-aligned bounding box [x1, y1, x2, y2]
  - **polygon**: Polygon coordinates (clockwise from top-left)
- **image_bbox**: Bounding box of the entire image
- **error**: Error message (only present if success is false)

### Supported Languages

Surya OCR supports **90+ languages** including:

**Indic Languages:**
- Hindi, Bengali, Tamil, Telugu, Kannada, Malayalam, Gujarati, Marathi, Punjabi, Odia, Assamese, Urdu, and more

**International Languages:**
- English, Spanish, French, German, Italian, Portuguese, Chinese, Japanese, Korean, Arabic, and many more

See the [Surya documentation](https://github.com/datalab-to/surya) for the complete list.

---

## 🧠 How It Works (Technical Details)

### OCR Process

1. **Text Detection**:
   - Detects where text is located in the image
   - Creates bounding boxes around text regions
   - Handles rotated and curved text

2. **Text Recognition**:
   - Recognizes the actual text in each detected region
   - Supports multiple languages automatically
   - Handles various fonts and styles

3. **Post-Processing**:
   - Combines text from all regions
   - Orders text logically (top-to-bottom, left-to-right)
   - Provides confidence scores

### Performance Characteristics

- **Accuracy**: Benchmarks favorably vs Google Cloud Vision (0.97 vs 0.88 similarity)
- **Speed**: 
  - Testing config (batch_size=64): 2-5 seconds per page
  - Production config (batch_size=512): 0.6-1 second per page
- **GPU Accelerated**: Much faster on GPU than CPU
- **Best For**: Printed text in documents (not handwriting)

---

## ⚙️ Configuration Options

### Adjusting Performance

You can modify `model_repository/surya_ocr/config.pbtxt` to change:

1. **Max Batch Size**: How many requests to process together
   - Current: 8
   - Increase for more throughput (needs more GPU memory)
   - Decrease if running out of memory

2. **Dynamic Batching**: Automatically groups requests
   - Current: Enabled with sizes [1, 2, 4, 8]
   - Adjust based on your workload

3. **GPU Instances**: Number of model copies
   - Current: 1
   - Increase for higher throughput (uses more GPU memory)

### Adjusting Batch Sizes

You can change batch sizes when running the container:

**For Testing (Lower VRAM usage):**
```bash
docker run -d --gpus all -p 8400:8000 -p 8401:8001 -p 8402:8002 \
  -e RECOGNITION_BATCH_SIZE=32 \
  -e DETECTOR_BATCH_SIZE=4 \
  --name surya-ocr-server surya-ocr-triton:latest
```

**For Production (Higher throughput):**
```bash
docker run -d --gpus all -p 8400:8000 -p 8401:8001 -p 8402:8002 \
  -e RECOGNITION_BATCH_SIZE=512 \
  -e DETECTOR_BATCH_SIZE=36 \
  --name surya-ocr-server surya-ocr-triton:latest
```

**After changing config.pbtxt, rebuild the image:**
```bash
docker build -t surya-ocr-triton:latest .
docker stop surya-ocr-server
docker rm surya-ocr-server
docker run -d --gpus all -p 8400:8000 -p 8401:8001 -p 8402:8002 \
  -e TORCH_DEVICE=cuda \
  -e RECOGNITION_BATCH_SIZE=64 \
  -e DETECTOR_BATCH_SIZE=8 \
  --name surya-ocr-server surya-ocr-triton:latest
```

---

## 🛠️ Troubleshooting

### Problem: Service Won't Start

**Symptoms**: Container exits immediately or shows errors

**Solutions**:
1. **Check logs**: `docker logs surya-ocr-server`
2. **Verify GPU**: `nvidia-smi` should show your GPU
3. **Check port conflicts**: `sudo lsof -i :8400` (should be empty)
4. **Verify Docker GPU access**: 
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

### Problem: "Out of Memory" Errors

**Symptoms**: CUDA out of memory errors in logs

**Solutions**:
1. **Reduce batch sizes**:
   ```bash
   -e RECOGNITION_BATCH_SIZE=16
   -e DETECTOR_BATCH_SIZE=2
   ```
2. **Use smaller images** (resize before sending)
3. **Check GPU memory**: `nvidia-smi`
4. **Close other GPU applications**
5. **Use CPU mode** (slower but works):
   ```bash
   -e TORCH_DEVICE=cpu
   ```

### Problem: Slow Processing

**Symptoms**: Requests take a very long time to process

**Solutions**:
1. **Verify GPU is being used**: Check logs for "Using device: cuda"
2. **Check GPU utilization**: `nvidia-smi -l 1` (should show activity)
3. **Increase batch sizes** (if memory allows)
4. **Use smaller images** for faster processing
5. **Note**: First request is slower (models load into memory)

### Problem: Poor OCR Quality

**Symptoms**: Service extracts incorrect or missing text

**Solutions**:
1. **Increase image resolution** (text should be clearly readable)
2. **Ensure image is not blurry** or low quality
3. **Preprocess image** if needed:
   - Binarize (convert to black and white)
   - Deskew (straighten rotated text)
   - Enhance contrast
4. **Check that language is supported** (90+ languages supported)
5. **Use clear, printed text** (handwriting not well supported)

### Problem: Models Not Downloading

**Symptoms**: Service fails to load models

**Solutions**:
1. **Check internet connection**
2. **Check disk space**: `df -h` (models need ~2-3GB)
3. **Review logs**: `docker logs surya-ocr-server` for specific errors
4. **Retry**: Network issues can be temporary
5. **Note**: Models download on first use, first request may take longer

### Problem: Cannot Connect to Service

**Symptoms**: Connection refused or timeout errors

**Solutions**:
1. **Verify service is running**: `docker ps`
2. **Check port mapping**: `docker ps` should show port mappings
3. **Test from container**: 
   ```bash
   docker exec surya-ocr-server curl http://localhost:8000/v2/health/ready
   ```
4. **Check firewall**: `sudo ufw status`
5. **Wait for models to load**: First request may take 1-2 minutes

### Problem: GPU Not Available

**Symptoms**: Service runs on CPU instead of GPU

**Solutions**:
1. **Verify NVIDIA drivers**: `nvidia-smi`
2. **Install nvidia-container-toolkit** (see Prerequisites)
3. **Restart Docker**: `sudo systemctl restart docker`
4. **Verify GPU in container**: 
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```
5. **Check logs**: Should show "Using device: cuda" not "cpu"

---

## 📈 Monitoring and Metrics

### Health Check

```bash
curl http://localhost:8400/v2/health/ready
```

### Metrics Endpoint

```bash
curl http://localhost:8402/metrics
```

This provides Prometheus-compatible metrics including:
- Request count
- Inference latency
- GPU utilization
- Error rates

### View Real-Time Logs

```bash
docker logs -f surya-ocr-server
```

Press `Ctrl+C` to stop viewing logs.

### Resource Usage

**Testing Configuration** (batch_size=64):
- **VRAM**: ~3-5 GB
- **RAM**: ~3 GB
- **Inference Time**: 2-5 seconds per page

**Production Configuration** (batch_size=512):
- **VRAM**: ~20 GB
- **RAM**: ~5 GB
- **Inference Time**: 0.6-1 second per page

---

## 🔄 Common Operations

### Stop the Service

```bash
docker stop surya-ocr-server
```

### Start the Service

```bash
docker start surya-ocr-server
```

### Restart the Service

```bash
docker restart surya-ocr-server
```

### Remove the Service

```bash
docker stop surya-ocr-server
docker rm surya-ocr-server
```

### Update the Service

```bash
# Rebuild with latest changes
docker build -t surya-ocr-triton:latest .

# Stop and remove old container
docker stop surya-ocr-server
docker rm surya-ocr-server

# Start new container
docker run -d --gpus all -p 8400:8000 -p 8401:8001 -p 8402:8002 \
  -e TORCH_DEVICE=cuda \
  -e RECOGNITION_BATCH_SIZE=64 \
  -e DETECTOR_BATCH_SIZE=8 \
  --name surya-ocr-server surya-ocr-triton:latest
```

---

## 📚 Additional Resources

### Service Documentation
- **Detailed README**: `surya-ocr-triton/README.md`
- **Model Source (GitHub)**: [https://github.com/datalab-to/surya](https://github.com/datalab-to/surya)
  - This is where the Surya OCR package and models were obtained from
- **Surya Documentation**: Available in the GitHub repository

### Technical References
- **Triton Server Docs**: https://docs.nvidia.com/deeplearning/triton-inference-server/
- **Docker Docs**: https://docs.docker.com/

### Getting Help

If you encounter issues:
1. Check the logs: `docker logs surya-ocr-server`
2. Review this guide's troubleshooting section
3. Check the service README: `surya-ocr-triton/README.md`
4. Review Surya OCR documentation on GitHub

---

## 📝 Quick Reference

### Essential Commands

```bash
# Build
cd surya-ocr-triton
docker build -t surya-ocr-triton:latest .

# Run
docker run -d --gpus all -p 8400:8000 -p 8401:8001 -p 8402:8002 \
  -e TORCH_DEVICE=cuda \
  -e RECOGNITION_BATCH_SIZE=64 \
  -e DETECTOR_BATCH_SIZE=8 \
  --name surya-ocr-server surya-ocr-triton:latest

# Check status
docker ps
curl http://localhost:8400/v2/health/ready

# Test
cd surya-ocr-triton
python3 test_client.py

# View logs
docker logs -f surya-ocr-server

# Stop
docker stop surya-ocr-server
```

### Port Information

- **HTTP API**: `http://localhost:8400`
- **gRPC API**: `localhost:8401`
- **Metrics**: `http://localhost:8402/metrics`

### Model Information

- **Model Name**: `surya_ocr`
- **Backend**: Python
- **Max Batch Size**: 8
- **GPU Required**: Recommended (works on CPU but slower)
- **Supported Languages**: 90+ languages

### Batch Size Recommendations

| VRAM Available | RECOGNITION_BATCH_SIZE | DETECTOR_BATCH_SIZE |
|---------------|----------------------|-------------------|
| 4-8 GB | 32-64 | 4-8 |
| 8-16 GB | 64-128 | 8-16 |
| 16+ GB | 256-512 | 24-36 |

---

## ✅ Summary

You've learned how to:
1. ✅ Build the Surya-ocr-triton Docker image
2. ✅ Run the service with optimal batch sizes
3. ✅ Verify it's working
4. ✅ Test OCR extraction
5. ✅ Use the API with images
6. ✅ Adjust performance settings
7. ✅ Troubleshoot common issues

The Surya-ocr-triton service is now ready to extract text from images! For production use, consider setting up monitoring, load balancing, and proper security measures.



