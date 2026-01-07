# Surya OCR Triton - Usage Examples

This guide provides practical examples for using the Surya OCR Triton server with your own images.

## 🎯 Quick Start

### Method 1: Simple Python Script (Recommended)

Create a file `ocr_image.py`:

```python
#!/usr/bin/env python3
import requests
import base64
import json
import sys

def ocr_image(image_path):
    """Perform OCR on an image file"""
    
    # Read and encode image
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    # Create payload
    payload = {
        "inputs": [{
            "name": "IMAGE_DATA",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [image_data]
        }]
    }
    
    # Send request
    response = requests.post(
        "http://localhost:8400/v2/models/surya_ocr/infer",
        json=payload,
        timeout=60
    )
    
    if response.status_code == 200:
        result = json.loads(response.json()['outputs'][0]['data'][0])
        return result
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 ocr_image.py <image_path>")
        sys.exit(1)
    
    result = ocr_image(sys.argv[1])
    
    if result:
        print("\n" + "="*70)
        print("OCR RESULTS")
        print("="*70 + "\n")
        print(result['full_text'])
        print("\n" + "="*70)
        print(f"Detected {len(result['text_lines'])} text lines")
        print(f"Average confidence: {sum(l['confidence'] for l in result['text_lines'])/len(result['text_lines']):.2%}")
        print("="*70)
```

**Usage:**
```bash
chmod +x ocr_image.py
python3 ocr_image.py your_document.png
```

---

### Method 2: Using cURL

**Step 1:** Create a helper script `create_payload.sh`:

```bash
#!/bin/bash
IMAGE_PATH=$1

if [ -z "$IMAGE_PATH" ]; then
    echo "Usage: ./create_payload.sh <image_path>"
    exit 1
fi

python3 -c "
import base64
import json

with open('$IMAGE_PATH', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

payload = {
    'inputs': [{
        'name': 'IMAGE_DATA',
        'shape': [1, 1],
        'datatype': 'BYTES',
        'data': [image_data]
    }]
}

with open('payload.json', 'w') as f:
    json.dump(payload, f)

print('Payload created: payload.json')
"
```

**Step 2:** Use it:

```bash
chmod +x create_payload.sh
./create_payload.sh my_image.jpg
curl -X POST http://localhost:8400/v2/models/surya_ocr/infer \
  -H "Content-Type: application/json" \
  -d @payload.json | jq -r '.outputs[0].data[0] | fromjson | .full_text'
```

---

### Method 3: One-Liner Bash Script

Create `ocr.sh`:

```bash
#!/bin/bash
IMAGE_PATH=$1

if [ -z "$IMAGE_PATH" ]; then
    echo "Usage: ./ocr.sh <image_path>"
    exit 1
fi

python3 -c "
import base64, json, requests

with open('$IMAGE_PATH', 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode('utf-8')

payload = {
    'inputs': [{
        'name': 'IMAGE_DATA',
        'shape': [1, 1],
        'datatype': 'BYTES',
        'data': [img_b64]
    }]
}

r = requests.post('http://localhost:8400/v2/models/surya_ocr/infer', json=payload)
result = json.loads(r.json()['outputs'][0]['data'][0])

print('\n' + '='*70)
print('OCR RESULTS')
print('='*70 + '\n')
print(result['full_text'])
print('\n' + '='*70)
print(f\"Lines: {len(result['text_lines'])} | Avg Confidence: {sum(l['confidence'] for l in result['text_lines'])/len(result['text_lines']):.2%}\")
print('='*70)
"
```

**Usage:**
```bash
chmod +x ocr.sh
./ocr.sh receipt.jpg
```

---

## 📊 Understanding the Output

The OCR returns a JSON structure with the following fields:

```json
{
  "success": true,
  "full_text": "Complete text from the image",
  "text_lines": [
    {
      "text": "Individual line of text",
      "confidence": 0.9876,
      "bbox": [x1, y1, x2, y2],
      "polygon": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    }
  ],
  "image_bbox": [0, 0, width, height]
}
```

**Fields:**
- `success`: Boolean indicating if OCR was successful
- `full_text`: All detected text combined with newlines
- `text_lines`: Array of individual text lines with details
  - `text`: The recognized text
  - `confidence`: Confidence score (0-1)
  - `bbox`: Bounding box [x1, y1, x2, y2]
  - `polygon`: Four corner points of the text region
- `image_bbox`: Original image dimensions

---

## 🌍 Multilingual Support

Surya OCR supports **90+ languages** including:

- **Indian Languages**: Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Urdu
- **European Languages**: English, Spanish, French, German, Italian, Portuguese, Russian
- **Asian Languages**: Chinese, Japanese, Korean, Thai, Vietnamese, Indonesian
- **Middle Eastern**: Arabic, Persian, Hebrew
- And many more!

### Example: Hindi Text OCR

```bash
# Test with Hindi text
python3 test_hindi_ocr.py
```

**Sample Output:**
```
OCR RESULTS
======================================================================
1. नमस्ते, यह एक परीक्षण है।
2. सूर्य ओसीआर हिंदी पाठ को पहचान सकता है।
3. भारत एक विशाल देश है।

Average Confidence: 98.95%
```

---

## 🖼️ Supported Image Formats

- PNG
- JPEG/JPG
- BMP
- TIFF
- WebP

---

## 💡 Advanced Usage

### Batch Processing Multiple Images

```python
import os
import glob

def batch_ocr(image_folder):
    """Process all images in a folder"""
    
    image_files = glob.glob(os.path.join(image_folder, "*.png")) + \
                  glob.glob(os.path.join(image_folder, "*.jpg"))
    
    results = {}
    
    for image_path in image_files:
        print(f"Processing: {image_path}")
        result = ocr_image(image_path)
        
        if result:
            results[image_path] = result['full_text']
    
    return results

# Usage
results = batch_ocr("./documents")
for path, text in results.items():
    print(f"\n{path}:\n{text}\n")
```

### Extract Text with Coordinates

```python
def ocr_with_coordinates(image_path):
    """Get text with bounding box coordinates"""
    
    result = ocr_image(image_path)
    
    if result:
        for i, line in enumerate(result['text_lines'], 1):
            x1, y1, x2, y2 = line['bbox']
            print(f"Line {i}: '{line['text']}'")
            print(f"  Position: ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})")
            print(f"  Confidence: {line['confidence']:.2%}")
            print()

# Usage
ocr_with_coordinates("document.png")
```

### Save Results to File

```python
def ocr_to_file(image_path, output_path):
    """Perform OCR and save results to text file"""
    
    result = ocr_image(image_path)
    
    if result:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result['full_text'])
        print(f"Results saved to: {output_path}")

# Usage
ocr_to_file("receipt.jpg", "receipt.txt")
```

---

## 🔍 Testing Examples

### Test with Sample Images

```bash
# Test with English text
python3 test_client.py

# Test with Hindi text
python3 test_hindi_ocr.py

# Test with your own image
python3 ocr_image.py my_document.png
```

### Health Check

```bash
# Check if server is ready
curl http://localhost:8400/v2/health/ready

# Check model status
curl http://localhost:8400/v2/models/surya_ocr
```

---

## ⚙️ Performance Tips

1. **Image Quality**: Higher resolution images generally produce better results
2. **Contrast**: Ensure good contrast between text and background
3. **Orientation**: Images should be properly oriented (not rotated)
4. **Batch Size**: Adjust `RECOGNITION_BATCH_SIZE` and `DETECTOR_BATCH_SIZE` environment variables for performance tuning
5. **Timeout**: For large images, increase the timeout value in requests

---

## 🐛 Troubleshooting

### Server Not Responding
```bash
# Check if container is running
docker ps | grep surya-ocr-triton

# Check server logs
docker logs surya-ocr-triton

# Restart server
docker stop surya-ocr-triton
docker run --gpus all --rm -d -p 8400:8000 -p 8401:8001 -p 8402:8002 \
  --name surya-ocr-triton surya-ocr-triton:latest
```

### Low Confidence Scores
- Check image quality and resolution
- Ensure proper lighting and contrast
- Verify text is not too small or blurry
- Try preprocessing the image (enhance contrast, denoise)

### Timeout Errors
- Increase timeout in your request
- Reduce image size if very large
- Check server resources (GPU memory, CPU)

---

## 📚 Additional Resources

- **Full Documentation**: See `README.md`
- **Testing Guide**: See `TESTING_GUIDE.md`
- **Deployment Details**: See `DEPLOYMENT_SUMMARY.md`
- **API Reference**: See `README.md` - API Endpoints section

---

## 🎓 Example Workflow

```bash
# 1. Start the server (if not running)
docker run --gpus all --rm -d -p 8400:8000 -p 8401:8001 -p 8402:8002 \
  --name surya-ocr-triton surya-ocr-triton:latest

# 2. Wait for models to load (~60-90 seconds)
sleep 90

# 3. Check server is ready
curl http://localhost:8400/v2/health/ready

# 4. Process your image
python3 ocr_image.py my_document.png

# 5. View results
cat output.txt
```

---

**Happy OCR-ing! 🚀**

