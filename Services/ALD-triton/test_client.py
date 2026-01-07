#!/usr/bin/env python3
"""
Test client for Audio Language Detection (ALD) Triton deployment.
Tests language identification from audio files.
"""

import requests
import json
import sys
import base64
import os


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


def test_ald_inference(audio_file_path, server_url="http://localhost:8100"):
    """
    Test ALD inference for a given audio file.
    
    Args:
        audio_file_path: Path to audio file
        server_url: Triton server URL
    
    Returns:
        dict: Language detection result
    """
    url = f"{server_url}/v2/models/ald/infer"
    
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
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        
        language_code = result["outputs"][0]["data"][0]
        confidence = result["outputs"][1]["data"][0]
        all_scores = json.loads(result["outputs"][2]["data"][0])
        
        return {
            "language_code": language_code,
            "confidence": confidence,
            "all_scores": all_scores
        }
    
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Error parsing response: {e}")
        print(f"Response: {response.text if 'response' in locals() else 'N/A'}")
        return None


def check_server_health(server_url="http://localhost:8100"):
    """Check if Triton server is ready."""
    try:
        response = requests.get(f"{server_url}/v2/health/ready", timeout=5)
        return response.status_code == 200
    except:
        return False


def main():
    server_url = "http://localhost:8100"
    
    # Check server health
    print("=" * 80)
    print("Audio Language Detection (ALD) Triton Server Test Client")
    print("=" * 80)
    print()
    
    print("Checking server health...")
    if not check_server_health(server_url):
        print("❌ Server is not ready. Please start the Triton server first.")
        print("   Run: docker run --gpus all -p 8100:8000 -p 8101:8001 -p 8102:8002 ald-triton:latest")
        sys.exit(1)
    
    print("✅ Server is ready!")
    print()
    
    # Check if audio file is provided as argument
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        if not os.path.exists(audio_file):
            print(f"❌ Audio file not found: {audio_file}")
            sys.exit(1)
        
        print(f"Testing with audio file: {audio_file}")
        print("-" * 80)
        
        result = test_ald_inference(audio_file, server_url)
        
        if result:
            print(f"✅ Success!")
            print(f"Detected Language: {result['language_code']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Details: {json.dumps(result['all_scores'], indent=2)}")
            return 0
        else:
            print(f"❌ Failed!")
            return 1
    else:
        print("Usage: python3 test_client.py <audio_file_path>")
        print()
        print("Example:")
        print("  python3 test_client.py test_audio.wav")
        print()
        print("Note: The audio file should be in a format supported by torchaudio")
        print("      (WAV, MP3, FLAC, etc.). The model will automatically resample")
        print("      to 16kHz and convert to mono channel.")
        return 1


if __name__ == "__main__":
    sys.exit(main())













