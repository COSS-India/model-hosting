#!/usr/bin/env python3
"""
Helper script to create a test payload for Surya OCR Triton server.
This script creates a sample image with text and generates a JSON payload file.
"""

import base64
import json
import sys
from PIL import Image, ImageDraw, ImageFont


def create_sample_image(output_path="sample_image.png"):
    """
    Create a sample document image with text.
    
    Args:
        output_path: Path to save the image
    
    Returns:
        PIL.Image: Created image
    """
    # Create a document-like image
    width, height = 1200, 800
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Try to use a nice font
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
        body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
    except:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
    
    # Draw document content
    y_position = 50
    
    # Title
    draw.text((50, y_position), "Sample Document", fill='black', font=title_font)
    y_position += 80
    
    # Body paragraphs
    paragraphs = [
        "This is a test document for Surya OCR.",
        "It contains multiple lines of text.",
        "",
        "Surya OCR supports 90+ languages including:",
        "- English, Hindi, Bengali, Tamil, Telugu",
        "- Chinese, Japanese, Korean",
        "- Arabic, French, German, Spanish",
        "",
        "The system can detect and recognize text",
        "with high accuracy and speed.",
    ]
    
    for para in paragraphs:
        draw.text((50, y_position), para, fill='black', font=body_font)
        y_position += 45
    
    # Save image
    image.save(output_path)
    print(f"✅ Sample image created: {output_path}")
    
    return image


def image_to_base64(image_path):
    """
    Convert image file to base64 string.
    
    Args:
        image_path: Path to image file
    
    Returns:
        str: Base64 encoded image
    """
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    return image_base64


def create_payload_file(image_base64, output_path="sample_payload.json"):
    """
    Create a JSON payload file for Triton inference.
    
    Args:
        image_base64: Base64 encoded image
        output_path: Path to save the payload file
    """
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
    
    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)
    
    print(f"✅ Payload file created: {output_path}")
    print(f"   Payload size: {len(json.dumps(payload)) / 1024:.2f} KB")


def main():
    print("=" * 80)
    print("Surya OCR Test Payload Generator")
    print("=" * 80)
    print()
    
    # Check if user provided an image path
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"Using provided image: {image_path}")
        
        try:
            # Verify image exists and is valid
            Image.open(image_path)
        except Exception as e:
            print(f"❌ Error: Could not open image: {e}")
            sys.exit(1)
    else:
        # Create a sample image
        image_path = "sample_image.png"
        print("No image provided. Creating a sample image...")
        create_sample_image(image_path)
    
    print()
    print("Converting image to base64...")
    image_base64 = image_to_base64(image_path)
    print(f"✅ Image encoded (size: {len(image_base64) / 1024:.2f} KB)")
    
    print()
    print("Creating payload file...")
    create_payload_file(image_base64, "sample_payload.json")
    
    print()
    print("=" * 80)
    print("Done! You can now test the server with:")
    print()
    print("  curl -X POST http://localhost:8400/v2/models/surya_ocr/infer \\")
    print("    -H \"Content-Type: application/json\" \\")
    print("    -d @sample_payload.json")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()

