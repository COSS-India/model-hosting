#!/usr/bin/env python3
"""
Test client for Surya OCR Triton deployment.
Tests OCR functionality with synthetic and real images.
"""

import requests
import json
import sys
import base64
import io
from PIL import Image, ImageDraw, ImageFont


def create_test_image(text="Hello World\nThis is a test", width=800, height=400):
    """
    Create a simple test image with text.
    
    Args:
        text: Text to render on the image
        width: Image width
        height: Image height
    
    Returns:
        PIL.Image: Generated test image
    """
    # Create white background
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Try to use a larger font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    # Draw text
    draw.text((50, 50), text, fill='black', font=font)
    
    return image


def image_to_base64(image):
    """
    Convert PIL Image to base64 string.
    
    Args:
        image: PIL Image
    
    Returns:
        str: Base64 encoded image
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_base64


def test_ocr(image_base64, server_url="http://localhost:8400"):
    """
    Test OCR for a given image.
    
    Args:
        image_base64: Base64 encoded image
        server_url: Triton server URL
    
    Returns:
        dict: OCR result
    """
    url = f"{server_url}/v2/models/surya_ocr/infer"
    
    payload = {
        "inputs": [
            {
                "name": "IMAGE_DATA",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [[image_base64]]
            }
        ],
        "outputs": [
            {
                "name": "OUTPUT_TEXT"
            }
        ]
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        output_text = result["outputs"][0]["data"][0]
        ocr_result = json.loads(output_text)
        
        return ocr_result
    
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Error parsing response: {e}")
        print(f"Response: {response.text if 'response' in locals() else 'N/A'}")
        return None


def check_server_health(server_url="http://localhost:8400"):
    """Check if Triton server is ready."""
    try:
        response = requests.get(f"{server_url}/v2/health/ready", timeout=5)
        return response.status_code == 200
    except:
        return False


def main():
    server_url = "http://localhost:8400"
    
    # Check server health
    print("=" * 80)
    print("Surya OCR Triton Server Test Client")
    print("=" * 80)
    print()
    
    print("Checking server health...")
    if not check_server_health(server_url):
        print("❌ Server is not ready. Please start the Triton server first.")
        print("   Run: docker run --gpus all -p 8400:8000 -p 8401:8001 -p 8402:8002 surya-ocr-triton:latest")
        sys.exit(1)
    
    print("✅ Server is ready!")
    print()
    
    # Test cases
    test_cases = [
        {
            "name": "Simple English Text",
            "text": "Hello World\nThis is a test of Surya OCR",
            "description": "Basic English text recognition"
        },
        {
            "name": "Multi-line Document",
            "text": "Document Title\n\nParagraph 1: This is the first paragraph.\nParagraph 2: This is the second paragraph.\n\nConclusion",
            "description": "Multi-paragraph document"
        },
        {
            "name": "Numbers and Symbols",
            "text": "Invoice #12345\nTotal: $1,234.56\nDate: 2024-01-15",
            "description": "Text with numbers and symbols"
        }
    ]
    
    # Run tests
    print("Running OCR tests...")
    print("=" * 80)
    print()
    
    success_count = 0
    total_count = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}/{total_count}: {test_case['name']}")
        print(f"Description: {test_case['description']}")
        print(f"Input text: {test_case['text'][:50]}...")
        
        # Create test image
        image = create_test_image(test_case['text'])
        image_base64 = image_to_base64(image)
        
        # Run OCR
        result = test_ocr(image_base64, server_url)
        
        if result and result.get('success'):
            print(f"✅ OCR successful!")
            print(f"   Lines detected: {len(result.get('text_lines', []))}")
            print(f"   Full text preview: {result.get('full_text', '')[:100]}...")
            
            # Show first few lines with confidence
            for idx, line in enumerate(result.get('text_lines', [])[:3]):
                print(f"   Line {idx + 1}: '{line['text']}' (confidence: {line['confidence']:.4f})")
            
            success_count += 1
        else:
            error_msg = result.get('error', 'Unknown error') if result else 'No response'
            print(f"❌ OCR failed: {error_msg}")
        
        print()
        print("-" * 80)
        print()
    
    # Summary
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Total tests: {total_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {total_count - success_count}")
    print(f"Success rate: {success_count/total_count*100:.1f}%")
    print()
    
    if success_count == total_count:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the logs.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

