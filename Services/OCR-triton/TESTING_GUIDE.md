# Surya OCR Triton - Testing Guide

This guide provides comprehensive instructions for testing the Surya OCR Triton deployment.

## Prerequisites

- Docker with GPU support (nvidia-docker)
- NVIDIA GPU with CUDA support
- Python 3.x (for test clients)
- curl (for command-line testing)

## Quick Test

The fastest way to verify the deployment:

```bash
# 1. Start the server
docker run --gpus all --rm -d \
  -p 8400:8000 -p 8401:8001 -p 8402:8002 \
  --name surya-ocr-triton \
  -e RECOGNITION_BATCH_SIZE=64 \
  -e DETECTOR_BATCH_SIZE=8 \
  surya-ocr-triton:latest

# 2. Wait for server to be ready (about 30 seconds for model loading)
sleep 30

# 3. Run the test client
python3 test_client.py
```

Expected output:
```
================================================================================
Surya OCR Triton Server Test Client
================================================================================

Checking server health...
✅ Server is ready!

Running OCR tests...
...
🎉 All tests passed!
```

## Testing Methods

### 1. Python Test Client (Recommended)

The `test_client.py` script provides comprehensive testing with synthetic images.

**Features**:
- Automatic server health check
- Multiple test cases (simple text, multi-line, numbers/symbols)
- Detailed results with confidence scores
- Success/failure summary

**Usage**:
```bash
python3 test_client.py
```

**Requirements**:
```bash
# Ubuntu/Debian
sudo apt-get install -y python3-pil python3-requests

# Or with pip (in virtual environment)
pip install pillow requests
```

### 2. cURL Testing

For command-line testing or integration with shell scripts.

#### Step 1: Generate Test Payload

```bash
python3 create_test_payload.py
```

This creates:
- `sample_image.png`: A test image with sample text
- `sample_payload.json`: JSON payload for Triton inference

#### Step 2: Send Request

```bash
curl -X POST http://localhost:8400/v2/models/surya_ocr/infer \
  -H "Content-Type: application/json" \
  -d @sample_payload.json
```

#### Step 3: Parse Response

The response is JSON with this structure:
```json
{
  "model_name": "surya_ocr",
  "model_version": "1",
  "outputs": [
    {
      "name": "OUTPUT_TEXT",
      "datatype": "BYTES",
      "shape": [1, 1],
      "data": ["{\"success\": true, \"text_lines\": [...], ...}"]
    }
  ]
}
```

The actual OCR results are in `outputs[0].data[0]` as a JSON string.

#### Pretty Print Response

```bash
curl -X POST http://localhost:8400/v2/models/surya_ocr/infer \
  -H "Content-Type: application/json" \
  -d @sample_payload.json | jq '.'
```

### 3. Custom Image Testing

#### Using Python

```python
import base64
import json
import requests
from PIL import Image

# Load your image
image = Image.open("your_image.png")

# Convert to base64
from io import BytesIO
buffer = BytesIO()
image.save(buffer, format="PNG")
image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

# Create payload
payload = {
    "inputs": [
        {
            "name": "IMAGE_DATA",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [image_base64]
        }
    ]
}

# Send request
response = requests.post(
    "http://localhost:8400/v2/models/surya_ocr/infer",
    json=payload
)

# Parse results
result = json.loads(response.json()["outputs"][0]["data"][0])
print("Full text:", result["full_text"])
for line in result["text_lines"]:
    print(f"- {line['text']} (confidence: {line['confidence']:.2f})")
```

#### Using create_test_payload.py

```bash
# Create payload from your image
python3 create_test_payload.py your_image.png

# Test with cURL
curl -X POST http://localhost:8400/v2/models/surya_ocr/infer \
  -H "Content-Type: application/json" \
  -d @sample_payload.json
```

## Health Checks

### Server Ready

```bash
curl http://localhost:8400/v2/health/ready
```

- Returns empty response with HTTP 200 if ready
- Returns HTTP 503 if not ready

### Model Ready

```bash
curl http://localhost:8400/v2/models/surya_ocr/ready
```

### Model Metadata

```bash
curl http://localhost:8400/v2/models/surya_ocr
```

Returns model information:
```json
{
  "name": "surya_ocr",
  "versions": ["1"],
  "platform": "python",
  "inputs": [...],
  "outputs": [...]
}
```

## Performance Testing

### Single Request Latency

```bash
time curl -X POST http://localhost:8400/v2/models/surya_ocr/infer \
  -H "Content-Type: application/json" \
  -d @sample_payload.json > /dev/null
```

### Concurrent Requests

```bash
# Install apache bench
sudo apt-get install apache2-utils

# Run 100 requests with 10 concurrent connections
ab -n 100 -c 10 -p sample_payload.json -T application/json \
  http://localhost:8400/v2/models/surya_ocr/infer
```

### Metrics

View Prometheus metrics:
```bash
curl http://localhost:8402/metrics
```

Key metrics to monitor:
- `nv_inference_request_success`: Successful requests
- `nv_inference_request_failure`: Failed requests
- `nv_inference_queue_duration_us`: Queue time
- `nv_inference_compute_infer_duration_us`: Inference time
- `nv_gpu_utilization`: GPU utilization
- `nv_gpu_memory_used_bytes`: GPU memory usage

## Troubleshooting

### Server Not Starting

**Check logs**:
```bash
docker logs surya-ocr-triton
```

**Common issues**:
- GPU not available: Ensure `--gpus all` flag is used
- Port conflicts: Check if ports 8400-8402 are available
- Insufficient memory: Reduce batch sizes

### Model Not Loading

**Check model status**:
```bash
curl http://localhost:8400/v2/models/surya_ocr
```

**Common issues**:
- Model files missing: Rebuild Docker image
- CUDA errors: Check GPU compatibility (Compute Capability 6.0+)
- Memory errors: Reduce `RECOGNITION_BATCH_SIZE` and `DETECTOR_BATCH_SIZE`

### Low Accuracy

**Possible causes**:
- Image quality: Ensure images are clear and high-resolution
- Image format: Use PNG or JPEG for best results
- Text size: Very small text may have lower accuracy
- Language: Ensure the language is supported by Surya OCR

**Tips for better accuracy**:
- Use images with at least 300 DPI
- Ensure good contrast between text and background
- Avoid skewed or rotated images
- Use appropriate image preprocessing

### Slow Performance

**Optimization tips**:
1. Increase batch sizes (if GPU memory allows):
   ```bash
   -e RECOGNITION_BATCH_SIZE=128 -e DETECTOR_BATCH_SIZE=16
   ```

2. Use multiple GPU instances (in config.pbtxt):
   ```
   instance_group [
     {
       count: 2
       kind: KIND_GPU
     }
   ]
   ```

3. Adjust dynamic batching parameters (in config.pbtxt):
   ```
   dynamic_batching {
     preferred_batch_size: [1, 2, 4, 8, 16]
     max_queue_delay_microseconds: 50000
   }
   ```

## Test Cases

### Recommended Test Scenarios

1. **Simple Text**: Single line, clear font
2. **Multi-line Document**: Paragraphs with multiple lines
3. **Mixed Content**: Text with numbers and symbols
4. **Multi-language**: Text in different languages
5. **Complex Layout**: Tables, columns, mixed orientations
6. **Low Quality**: Scanned documents, photos of text
7. **Edge Cases**: Very small text, handwriting, artistic fonts

### Sample Test Images

The repository includes `sample_image.png` with:
- Document title
- Multiple paragraphs
- Bullet points
- Language examples
- Mixed content

## Integration Testing

### Example: Batch Processing

```python
import os
import base64
import json
import requests
from PIL import Image
from io import BytesIO

def process_image(image_path):
    # Load and encode image
    image = Image.open(image_path)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Create payload
    payload = {
        "inputs": [{
            "name": "IMAGE_DATA",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [image_base64]
        }]
    }
    
    # Send request
    response = requests.post(
        "http://localhost:8400/v2/models/surya_ocr/infer",
        json=payload
    )
    
    # Parse results
    result = json.loads(response.json()["outputs"][0]["data"][0])
    return result

# Process multiple images
image_dir = "path/to/images"
for filename in os.listdir(image_dir):
    if filename.endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(image_dir, filename)
        result = process_image(image_path)
        print(f"{filename}: {result['full_text'][:50]}...")
```

## Continuous Testing

### Automated Testing Script

```bash
#!/bin/bash
# test_deployment.sh

echo "Testing Surya OCR Triton Deployment..."

# 1. Health check
echo "1. Checking server health..."
if curl -f http://localhost:8400/v2/health/ready > /dev/null 2>&1; then
    echo "✅ Server is ready"
else
    echo "❌ Server is not ready"
    exit 1
fi

# 2. Model check
echo "2. Checking model status..."
if curl -f http://localhost:8400/v2/models/surya_ocr/ready > /dev/null 2>&1; then
    echo "✅ Model is ready"
else
    echo "❌ Model is not ready"
    exit 1
fi

# 3. Inference test
echo "3. Running inference test..."
if python3 test_client.py > /dev/null 2>&1; then
    echo "✅ Inference test passed"
else
    echo "❌ Inference test failed"
    exit 1
fi

echo "✅ All tests passed!"
```

Make it executable and run:
```bash
chmod +x test_deployment.sh
./test_deployment.sh
```

## Conclusion

This testing guide covers all aspects of testing the Surya OCR Triton deployment. For production deployments, consider:

1. Setting up automated testing in CI/CD pipelines
2. Implementing monitoring and alerting
3. Load testing with realistic workloads
4. Regular accuracy validation with ground truth data
5. Performance benchmarking and optimization

For issues or questions, refer to:
- README.md for general documentation
- DEPLOYMENT_SUMMARY.md for deployment details
- Triton documentation: https://docs.nvidia.com/deeplearning/triton-inference-server/
- Surya OCR documentation: https://github.com/datalab-to/surya

