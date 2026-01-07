#!/usr/bin/env python3
"""
Test client for Speaker Diarization Triton deployment - Local Machine Version
This script can be run from your local machine to test the remote Triton server.
Tests speaker diarization on audio files.
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


def test_diarization(audio_file_path, num_speakers=None, server_url="http://localhost:8700"):
    """
    Test speaker diarization for a given audio file.
    
    Args:
        audio_file_path: Path to audio file
        num_speakers: Optional number of speakers (None for auto-detection)
        server_url: Triton server URL (can be remote IP/domain)
    
    Returns:
        dict: Diarization result
    """
    url = f"{server_url}/v2/models/speaker_diarization/infer"
    
    # Encode audio file
    print(f"Reading and encoding audio file: {audio_file_path}")
    audio_b64 = encode_audio_file(audio_file_path)
    if audio_b64 is None:
        return None
    
    # Prepare num_speakers parameter
    num_speakers_str = str(num_speakers) if num_speakers is not None else ""
    
    payload = {
        "inputs": [
            {
                "name": "AUDIO_DATA",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [[audio_b64]]
            },
            {
                "name": "NUM_SPEAKERS",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [[num_speakers_str]]
            }
        ],
        "outputs": [
            {
                "name": "DIARIZATION_RESULT"
            }
        ]
    }
    
    try:
        print(f"Sending request to {url}...")
        if num_speakers:
            print(f"Number of speakers: {num_speakers}")
        else:
            print("Auto-detecting number of speakers...")
        
        response = requests.post(url, json=payload, timeout=300)  # 5 minute timeout for long audio
        response.raise_for_status()
        
        result = response.json()
        diarization_result = json.loads(result["outputs"][0]["data"][0])
        
        return diarization_result
    
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection Error: Could not connect to server at {server_url}")
        print(f"   Make sure the server is running and accessible.")
        print(f"   Error: {e}")
        return None
    except requests.exceptions.Timeout as e:
        print(f"❌ Timeout Error: Request took too long (>5 minutes)")
        print(f"   The audio file might be too large or server is slow.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Request Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response status: {e.response.status_code}")
            print(f"   Response: {e.response.text}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"❌ Error parsing response: {e}")
        print(f"   Response: {response.text if 'response' in locals() else 'N/A'}")
        return None


def check_server_health(server_url="http://localhost:8700"):
    """Check if Triton server is ready."""
    try:
        response = requests.get(f"{server_url}/v2/health/ready", timeout=5)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False
    except:
        return False


def print_results(diarization_result, pretty=False):
    """Print diarization results in a readable format."""
    if "error" in diarization_result:
        print(f"❌ Error: {diarization_result['error']}")
        return
    
    print("\n" + "=" * 80)
    print("Speaker Diarization Results")
    print("=" * 80)
    print(f"Number of speakers detected: {diarization_result['num_speakers']}")
    print(f"Total segments: {diarization_result['total_segments']}")
    print(f"Speakers: {', '.join(diarization_result['speakers'])}")
    print()
    
    if diarization_result['total_segments'] > 0:
        print("Segments:")
        print("-" * 80)
        for i, segment in enumerate(diarization_result['segments'], 1):
            print(f"Segment {i}:")
            print(f"  Time: {segment['start_time']:.2f}s - {segment['end_time']:.2f}s "
                  f"(duration: {segment['duration']:.2f}s)")
            print(f"  Speaker: {segment['speaker']}")
            print()
    else:
        print("No segments found.")
    
    if pretty:
        print("\nFull JSON Output:")
        print(json.dumps(diarization_result, indent=2))
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Test Speaker Diarization Triton deployment from local machine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Connect to remote server (replace with your server IP)
  python3 test_client_local.py audio.wav --server-url http://your-server-ip:8700
  
  # Auto-detect speakers
  python3 test_client_local.py audio.wav --server-url http://192.168.1.100:8700
  
  # Specify 2 speakers
  python3 test_client_local.py audio.wav --server-url http://your-server-ip:8700 --num-speakers 2
  
  # Use localhost (if running on same machine)
  python3 test_client_local.py audio.wav --server-url http://localhost:8700
  
  # Pretty print JSON output
  python3 test_client_local.py audio.wav --server-url http://your-server-ip:8700 --pretty
        """
    )
    
    parser.add_argument(
        "audio_file",
        help="Path to audio file on your local machine (WAV, MP3, FLAC, etc.)"
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:8700",
        help="Triton server URL (default: http://localhost:8700). Use your server IP for remote access, e.g., http://192.168.1.100:8700"
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Number of speakers (optional, auto-detected if not specified)"
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty print full JSON output"
    )
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not os.path.exists(args.audio_file):
        print(f"❌ Error: Audio file not found: {args.audio_file}")
        print(f"   Please provide a valid path to an audio file on your local machine.")
        sys.exit(1)
    
    # Check server health
    print("=" * 80)
    print("SD-triton Test Client (Local Machine)")
    print("=" * 80)
    print()
    print(f"Server URL: {args.server_url}")
    print(f"Audio file: {args.audio_file}")
    print()
    print("Checking server health...")
    
    if not check_server_health(args.server_url):
        print("❌ Server is not ready or not accessible.")
        print()
        print("Troubleshooting:")
        print(f"  1. Verify the server is running: curl {args.server_url}/v2/health/ready")
        print(f"  2. Check if the server IP/port is correct: {args.server_url}")
        print("  3. Ensure firewall allows connections on port 8700")
        print("  4. If using remote server, make sure it's accessible from your network")
        print()
        print("To start the server on the remote machine:")
        print("  docker run -d --gpus all \\")
        print("    -p 8700:8000 -p 8701:8001 -p 8702:8002 \\")
        print("    -e HUGGING_FACE_HUB_TOKEN='your_token' \\")
        print("    --name sd-triton-server sd-triton:latest")
        sys.exit(1)
    
    print("✅ Server is ready!")
    print()
    
    # Perform inference
    print(f"Processing audio file: {args.audio_file}")
    if args.num_speakers:
        print(f"Number of speakers: {args.num_speakers}")
    else:
        print("Auto-detecting number of speakers...")
    print()
    
    result = test_diarization(
        args.audio_file,
        args.num_speakers,
        args.server_url
    )
    
    if result:
        print_results(result, args.pretty)
        sys.exit(0)
    else:
        print("❌ Inference failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()












