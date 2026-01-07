#!/usr/bin/env python3
"""
Test Surya OCR with Hindi text to verify multilingual support
Can also test with custom images provided by the user
"""

import requests
import base64
import json
from PIL import Image, ImageDraw, ImageFont
import io
import sys
import argparse
import os

def create_hindi_test_image():
    """
    Create a test image with Hindi text
    """
    # Create a white background image
    width, height = 800, 600
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Try to use a font that supports Hindi (Devanagari script)
    # Common fonts that support Hindi on Ubuntu
    font_paths = [
        '/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf',
        '/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf',
        '/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ]
    
    font = None
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, 40)
            print(f"Using font: {font_path}")
            break
        except:
            continue
    
    if font is None:
        print("Warning: Could not load Hindi font, using default font")
        font = ImageFont.load_default()
    
    # Hindi text samples
    hindi_texts = [
        "नमस्ते, यह एक परीक्षण है।",  # Hello, this is a test.
        "सूर्य ओसीआर हिंदी पाठ को पहचान सकता है।",  # Surya OCR can recognize Hindi text.
        "भारत एक विशाल देश है।",  # India is a vast country.
        "मुझे हिंदी पढ़ना अच्छा लगता है।",  # I like reading Hindi.
        "आज का दिन बहुत अच्छा है।",  # Today is a very good day.
    ]
    
    # Draw title
    draw.text((50, 30), "Hindi Text OCR Test", fill='black', font=font)
    draw.line([(50, 80), (750, 80)], fill='black', width=2)
    
    # Draw Hindi text lines
    y_position = 120
    for i, text in enumerate(hindi_texts, 1):
        draw.text((50, y_position), f"{i}. {text}", fill='black', font=font)
        y_position += 80
    
    return image

def test_hindi_ocr(image, server_url="http://65.1.35.3:8400"):
    """
    Send image to Surya OCR and get results

    Args:
        image: PIL Image object
        server_url: Triton server URL
    """
    # Convert image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # Create payload
    payload = {
        "inputs": [
            {
                "name": "IMAGE_DATA",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [image_data]
            }
        ]
    }
    
    # Send request
    url = f"{server_url}/v2/models/surya_ocr/infer"
    print(f"\nSending request to {url}...")
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            output_data = json.loads(result['outputs'][0]['data'][0])
            
            print("\n" + "="*70)
            print("HINDI OCR TEST RESULTS")
            print("="*70)
            
            if output_data['success']:
                print(f"\n✅ OCR Success!")
                print(f"Total text lines detected: {len(output_data['text_lines'])}")
                
                print("\n" + "-"*70)
                print("FULL TEXT OUTPUT:")
                print("-"*70)
                print(output_data['full_text'])
                
                print("\n" + "-"*70)
                print("DETAILED LINE-BY-LINE RESULTS:")
                print("-"*70)
                
                for i, line in enumerate(output_data['text_lines'], 1):
                    print(f"\nLine {i}:")
                    print(f"  Text: {line['text']}")
                    print(f"  Confidence: {line['confidence']:.4f} ({line['confidence']*100:.2f}%)")
                    print(f"  BBox: {line['bbox']}")
                
                # Calculate average confidence
                avg_confidence = sum(line['confidence'] for line in output_data['text_lines']) / len(output_data['text_lines'])
                print("\n" + "="*70)
                print(f"Average Confidence: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)")
                print("="*70)
                
                # Check if Hindi characters are present
                hindi_chars_found = any('\u0900' <= char <= '\u097F' for line in output_data['text_lines'] for char in line['text'])
                if hindi_chars_found:
                    print("\n✅ Hindi (Devanagari) characters detected in output!")
                else:
                    print("\n⚠️  No Hindi (Devanagari) characters found in output")
                
                return output_data
            else:
                print("\n❌ OCR Failed")
                print(output_data)
                return None
        else:
            print(f"\n❌ HTTP Error: {response.status_code}")
            print(response.text)
            return None
            
    except requests.exceptions.Timeout:
        print("\n❌ Request timed out (60 seconds)")
        return None
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return None

def load_custom_image(image_path):
    """
    Load an image from file path

    Args:
        image_path: Path to the image file

    Returns:
        PIL Image object or None if failed
    """
    try:
        image = Image.open(image_path)
        print(f"✅ Image loaded: {image_path}")
        print(f"   Size: {image.size[0]}x{image.size[1]} pixels")
        print(f"   Format: {image.format}")
        return image
    except FileNotFoundError:
        print(f"❌ Error: File not found: {image_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading image: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description='Test Surya OCR with Hindi text or custom images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with auto-generated Hindi text image
  python3 test_hindi_ocr.py

  # Test with your own image
  python3 test_hindi_ocr.py --image my_document.png
  python3 test_hindi_ocr.py -i receipt.jpg

  # Use custom server URL
  python3 test_hindi_ocr.py --image doc.png --server http://localhost:8400
        """
    )

    parser.add_argument('--image', '-i', metavar='PATH',
                        help='Path to your image file (PNG, JPG, etc.)')
    parser.add_argument('--server', '-s', default='http://65.1.35.3:8400',
                        help='Triton server URL (default: http://65.1.35.3:8400)')
    parser.add_argument('--save-generated', action='store_true',
                        help='Save the auto-generated test image (only when not using --image)')

    args = parser.parse_args()

    print("="*70)
    print("SURYA OCR - TEXT RECOGNITION TEST")
    print("="*70)

    # Determine which image to use
    if args.image:
        # User provided an image
        print(f"\n1. Loading custom image: {args.image}")
        image = load_custom_image(args.image)
        if image is None:
            sys.exit(1)
        image_path = args.image
    else:
        # Generate Hindi test image
        print("\n1. Creating test image with Hindi text...")
        image = create_hindi_test_image()

        # Save the test image if requested or by default
        image_path = "hindi_test_image.png"
        image.save(image_path)
        print(f"✅ Test image saved: {image_path}")

    # Test OCR
    print(f"\n2. Testing OCR (Server: {args.server})...")
    result = test_hindi_ocr(image, server_url=args.server)

    if result:
        print("\n" + "="*70)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("="*70)
        if not args.image:
            print(f"\nYou can view the generated test image: {image_path}")
    else:
        print("\n" + "="*70)
        print("TEST FAILED")
        print("="*70)
        sys.exit(1)

if __name__ == "__main__":
    main()

