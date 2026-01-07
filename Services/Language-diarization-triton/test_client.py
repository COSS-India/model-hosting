#!/usr/bin/env python3
"""
Test client for Language Diarization Triton deployment.
Tests language diarization on audio files.
"""

import requests
import json
import sys
import base64
import os
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


def test_diarization(audio_file_path, target_language="", server_url="http://localhost:8600"):
    """
    Test language diarization for a given audio file.
    
    Args:
        audio_file_path: Path to audio file
        target_language: Optional target language code (e.g., "ta", "gu", "te")
        server_url: Triton server URL
    
    Returns:
        dict: Diarization result
    """
    url = f"{server_url}/v2/models/lang_diarization/infer"
    
    # Encode audio file
    audio_b64 = encode_audio_file(audio_file_path)
    if audio_b64 is None:
        return None
    
    payload = {
        "inputs": [
            {
                "name": "AUDIO_DATA",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [[audio_b64]]
            },
            {
                "name": "LANGUAGE",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [[target_language]]
            }
        ],
        "outputs": [
            {
                "name": "DIARIZATION_RESULT"
            }
        ]
    }
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        diarization_result = json.loads(result["outputs"][0]["data"][0])
        
        return diarization_result
    
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Error parsing response: {e}")
        print(f"Response: {response.text if 'response' in locals() else 'N/A'}")
        return None


def check_server_health(server_url="http://localhost:8600"):
    """Check if Triton server is ready."""
    try:
        response = requests.get(f"{server_url}/v2/health/ready", timeout=5)
        return response.status_code == 200
    except:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test Language Diarization Triton Server"
    )
    parser.add_argument(
        "audio_file",
        help="Path to audio file (WAV, MP3, FLAC, etc.)"
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:8600",
        help="Triton server URL (default: http://localhost:8600)"
    )
    parser.add_argument(
        "--language",
        default="",
        help="Target language code (e.g., 'ta', 'gu', 'te'). Leave empty for all languages."
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty print JSON output"
    )
    
    args = parser.parse_args()
    
    # Check server health
    print("=" * 80)
    print("Language Diarization Triton Server Test Client")
    print("=" * 80)
    print()
    
    print("Checking server health...")
    if not check_server_health(args.server_url):
        print("❌ Server is not ready. Please start the Triton server first.")
        print(f"   Run: docker run --gpus all -p 8600:8000 -p 8601:8001 -p 8602:8002 lang-diarization-triton:latest")
        sys.exit(1)
    
    print("✅ Server is ready!")
    print()
    
    # Check if audio file exists
    if not os.path.exists(args.audio_file):
        print(f"❌ Audio file not found: {args.audio_file}")
        sys.exit(1)
    
    print(f"Testing with audio file: {args.audio_file}")
    if args.language:
        print(f"Target language: {args.language}")
    else:
        print("Target language: All languages")
    print("-" * 80)
    print()
    
    result = test_diarization(args.audio_file, args.language, args.server_url)
    
    if result:
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return 1
        
        print(f"✅ Success!")
        print(f"Total segments found: {result['total_segments']}")
        print(f"Target language filter: {result.get('target_language', 'all')}")
        print()
        
        if result['total_segments'] > 0:
            print("Diarization Results:")
            print("-" * 80)
            for i, segment in enumerate(result['segments'], 1):
                print(f"Segment {i}:")
                print(f"  Time: {segment['start_time']:.2f}s - {segment['end_time']:.2f}s (duration: {segment['duration']:.2f}s)")
                print(f"  Language: {segment['language']}")
                print(f"  Confidence: {segment['confidence']:.4f}")
                print()
        else:
            print("No segments found matching the criteria.")
        
        if args.pretty:
            print("Full JSON Output:")
            print(json.dumps(result, indent=2))
        
        return 0
    else:
        print(f"❌ Failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())













