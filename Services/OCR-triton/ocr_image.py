#!/usr/bin/env python3
"""
Simple script to perform OCR on any image using Surya OCR Triton server

Usage:
    python3 ocr_image.py <image_path>
    python3 ocr_image.py <image_path> --detailed
    python3 ocr_image.py <image_path> --json
    python3 ocr_image.py <image_path> --output result.txt
"""

import requests
import base64
import json
import sys
import argparse

def ocr_image(image_path, server_url="http://65.1.35.3:8400"):
    """
    Perform OCR on an image file
    
    Args:
        image_path: Path to the image file
        server_url: Triton server URL (default: http://localhost:8400)
    
    Returns:
        dict: OCR results or None if failed
    """
    try:
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
        url = f"{server_url}/v2/models/surya_ocr/infer"
        response = requests.post(url, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = json.loads(response.json()['outputs'][0]['data'][0])
            return result
        else:
            print(f"❌ HTTP Error: {response.status_code}", file=sys.stderr)
            print(response.text, file=sys.stderr)
            return None
            
    except FileNotFoundError:
        print(f"❌ Error: File not found: {image_path}", file=sys.stderr)
        return None
    except requests.exceptions.Timeout:
        print("❌ Error: Request timed out (60 seconds)", file=sys.stderr)
        return None
    except Exception as e:
        print(f"❌ Error: {str(e)}", file=sys.stderr)
        return None

def print_simple_results(result, image_path):
    """Print simple text output"""
    print("\n" + "="*70)
    print(f"OCR RESULTS: {image_path}")
    print("="*70 + "\n")
    print(result['full_text'])
    print("\n" + "="*70)
    print(f"Detected {len(result['text_lines'])} text lines")
    if result['text_lines']:
        avg_conf = sum(l['confidence'] for l in result['text_lines']) / len(result['text_lines'])
        print(f"Average confidence: {avg_conf:.2%}")
    print("="*70)

def print_detailed_results(result, image_path):
    """Print detailed line-by-line results"""
    print("\n" + "="*70)
    print(f"DETAILED OCR RESULTS: {image_path}")
    print("="*70 + "\n")
    
    for i, line in enumerate(result['text_lines'], 1):
        print(f"Line {i}:")
        print(f"  Text: {line['text']}")
        print(f"  Confidence: {line['confidence']:.4f} ({line['confidence']*100:.2f}%)")
        print(f"  BBox: {line['bbox']}")
        print(f"  Polygon: {line['polygon']}")
        print()
    
    print("="*70)
    print(f"Total lines: {len(result['text_lines'])}")
    if result['text_lines']:
        avg_conf = sum(l['confidence'] for l in result['text_lines']) / len(result['text_lines'])
        print(f"Average confidence: {avg_conf:.4f} ({avg_conf*100:.2f}%)")
    print("="*70)

def print_json_results(result):
    """Print results as JSON"""
    print(json.dumps(result, indent=2, ensure_ascii=False))

def save_to_file(result, output_path):
    """Save OCR results to a text file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result['full_text'])
        print(f"✅ Results saved to: {output_path}")
    except Exception as e:
        print(f"❌ Error saving file: {str(e)}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(
        description='Perform OCR on images using Surya OCR Triton server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 ocr_image.py document.png
  python3 ocr_image.py receipt.jpg --detailed
  python3 ocr_image.py invoice.png --output invoice.txt
  python3 ocr_image.py form.jpg --json > result.json
        """
    )
    
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--detailed', '-d', action='store_true',
                        help='Show detailed line-by-line results')
    parser.add_argument('--json', '-j', action='store_true',
                        help='Output results as JSON')
    parser.add_argument('--output', '-o', metavar='FILE',
                        help='Save text results to file')
    parser.add_argument('--server', '-s', default='http://65.1.35.3:8400',
                        help='Triton server URL (default: http://65.1.35.3:8400)')
    
    args = parser.parse_args()
    
    # Perform OCR
    result = ocr_image(args.image_path, args.server)
    
    if result is None:
        sys.exit(1)
    
    if not result.get('success', False):
        print("❌ OCR failed", file=sys.stderr)
        sys.exit(1)
    
    # Output results based on flags
    if args.json:
        print_json_results(result)
    elif args.detailed:
        print_detailed_results(result, args.image_path)
    else:
        print_simple_results(result, args.image_path)
    
    # Save to file if requested
    if args.output:
        save_to_file(result, args.output)

if __name__ == "__main__":
    main()

