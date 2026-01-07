#!/usr/bin/env python3
"""
Direct test runner - checks server and runs test
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_client import test_diarization, check_server_health, print_results

def main():
    server_url = "http://localhost:8700"
    audio_file = "../ALD-triton/ta2.wav"
    
    print("=" * 80)
    print("SD-triton Test Runner")
    print("=" * 80)
    print()
    
    # Check server health
    print("Checking server health...")
    if not check_server_health(server_url):
        print("❌ Server is not ready. Please start the Triton server first.")
        print()
        print("Run:")
        print("  export HUGGING_FACE_HUB_TOKEN='your_token_here'")
        print("  docker run -d --gpus all -p 8700:8000 -p 8701:8001 -p 8702:8002 \\")
        print("    -e HUGGING_FACE_HUB_TOKEN=\"${HUGGING_FACE_HUB_TOKEN}\" \\")
        print("    --name sd-triton-server sd-triton:latest")
        print()
        print("Then wait 1-2 minutes and run this script again.")
        sys.exit(1)
    
    print("✅ Server is ready!")
    print()
    
    # Check if audio file exists
    if not os.path.exists(audio_file):
        print(f"❌ Error: Audio file not found: {audio_file}")
        sys.exit(1)
    
    # Run test
    print(f"Processing audio file: {audio_file}")
    print("Auto-detecting number of speakers...")
    print()
    
    result = test_diarization(audio_file, num_speakers=None, server_url=server_url)
    
    if result:
        print_results(result, pretty=True)
        sys.exit(0)
    else:
        print("❌ Inference failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()












