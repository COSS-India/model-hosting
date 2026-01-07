#!/usr/bin/env python3
"""
Python script to test ALD Triton Inference Server using curl-like requests.
Usage: python3 curl_example.py <audio_file_path>
"""

import sys
import base64
import json
import requests
import argparse


def encode_audio_file(file_path):
    """
    Read and encode audio file to base64.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        str: Base64 encoded audio data
    """
    try:
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        return audio_b64
    except Exception as e:
        print(f"Error reading audio file: {e}")
        return None


def make_inference_request(audio_file_path, server_url="http://localhost:8100"):
    """
    Make inference request to ALD Triton server.
    
    Args:
        audio_file_path: Path to audio file
        server_url: Triton server URL
        
    Returns:
        dict: Response from server
    """
    # Encode audio file
    print(f"Encoding audio file: {audio_file_path}")
    audio_b64 = encode_audio_file(audio_file_path)
    if audio_b64 is None:
        return None
    
    # Prepare request payload
    url = f"{server_url}/v2/models/ald/infer"
    
    payload = {
        "inputs": [
            {
                "name": "AUDIO_DATA",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [[audio_b64]]
            }
        ],
        "outputs": [
            {
                "name": "LANGUAGE_CODE"
            },
            {
                "name": "CONFIDENCE"
            },
            {
                "name": "ALL_SCORES"
            }
        ]
    }
    
    # Make request
    print(f"Making inference request to {url}...")
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Test ALD Triton Inference Server with audio file"
    )
    parser.add_argument(
        "audio_file",
        help="Path to audio file (WAV, MP3, FLAC, etc.)"
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:8100",
        help="Triton server URL (default: http://localhost:8100)"
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty print JSON output"
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    import os
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        sys.exit(1)
    
    # Make inference request
    result = make_inference_request(args.audio_file, args.server_url)
    
    if result is None:
        print("❌ Inference failed!")
        sys.exit(1)
    
    # Print results
    print("\n" + "=" * 80)
    print("Inference Results")
    print("=" * 80)
    
    if args.pretty:
        print(json.dumps(result, indent=2))
    else:
        # Parse and display results in a readable format
        try:
            language_code = result["outputs"][0]["data"][0]
            confidence = result["outputs"][1]["data"][0]
            all_scores = json.loads(result["outputs"][2]["data"][0])
            
            print(f"✅ Detected Language: {language_code}")
            print(f"✅ Confidence: {confidence:.4f}")
            print(f"\nDetailed Scores:")
            print(json.dumps(all_scores, indent=2))
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print("Raw response:")
            print(json.dumps(result, indent=2))
    
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()













