#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Configuration
IMAGE_NAME="ai4bharat/triton-multilingual-asr:latest"
CONTAINER_NAME="asr"
HTTP_PORT=5000
GRPC_PORT=5001
METRICS_PORT=5002
MODEL_NAME="asr_am_ensemble"

# Function to print colored messages
print_message() {
    echo -e "${2}${1}${NC}"
}

# Function to show spinner animation
show_spinner() {
    local pid=$1
    local message=$2
    local spinstr='|/-\'
    local delay=0.1
    
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf "\r${CYAN}[${spinstr:0:1}]${NC} ${message}..."
        spinstr=$temp${spinstr%"$temp"}
        sleep $delay
    done
    printf "\r${GREEN}[✓]${NC} ${message} - Done!\n"
}

# Function to wait for service to be ready
wait_for_service() {
    local max_attempts=120
    local attempt=1
    
    print_message "Waiting for service to be ready..." "$CYAN"
    print_message "This may take 1-2 minutes while models are being loaded..." "$YELLOW"
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f http://localhost:${HTTP_PORT}/v2/health/ready > /dev/null 2>&1; then
            print_message "Service is ready!" "$GREEN"
            return 0
        fi
        
        printf "\r${YELLOW}[${attempt}/${max_attempts}]${NC} Waiting for service to start... (checking every 2s)"
        sleep 2
        attempt=$((attempt + 1))
    done
    
    print_message "Service failed to start within expected time!" "$RED"
    print_message "Checking container logs..." "$YELLOW"
    docker logs --tail 30 ${CONTAINER_NAME} 2>&1 | tail -15
    return 1
}

# Function to check if file exists
check_audio_file() {
    local file=$1
    if [ ! -f "$file" ]; then
        print_message "Warning: Audio file not found: $file" "$YELLOW"
        print_message "Skipping audio test. You can test later with your own audio file." "$YELLOW"
        return 1
    fi
    return 0
}

# Clear screen
clear

# Banner
echo -e "${BOLD}${CYAN}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║      Multilingual ASR Service Setup & Test Script         ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}\n"

# Step 1: Check if Docker is running
print_message "Step 1: Checking Docker..." "$BLUE"
if ! docker info > /dev/null 2>&1; then
    print_message "ERROR: Docker is not running!" "$RED"
    exit 1
fi
print_message "Docker is running ✓" "$GREEN"
echo ""

# Step 2: Stop and remove existing container if exists
print_message "Step 2: Cleaning up existing container..." "$BLUE"
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    print_message "Found existing container: ${CONTAINER_NAME}" "$YELLOW"
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_message "Stopping container..." "$YELLOW"
        docker stop ${CONTAINER_NAME} > /dev/null 2>&1
    fi
    print_message "Removing container..." "$YELLOW"
    docker rm ${CONTAINER_NAME} > /dev/null 2>&1
    print_message "Container removed ✓" "$GREEN"
else
    print_message "No existing container found ✓" "$GREEN"
fi
echo ""

# Step 3: Check if image exists, if not pull it
print_message "Step 3: Checking Docker image..." "$BLUE"
if docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "^${IMAGE_NAME}$"; then
    print_message "Image already exists: ${IMAGE_NAME}" "$GREEN"
    image_size=$(docker images --format "{{.Size}}" ${IMAGE_NAME} 2>/dev/null | head -1)
    if [ ! -z "$image_size" ]; then
        print_message "Image size: $image_size" "$CYAN"
    fi
else
    print_message "Image not found. Pulling ${IMAGE_NAME}..." "$YELLOW"
    print_message "This may take several minutes (image size: ~several GB)..." "$YELLOW"
    echo ""
    
    # Pull image with progress
    docker pull ${IMAGE_NAME} &
    pull_pid=$!
    show_spinner $pull_pid "Pulling Docker image"
    
    # Wait for pull to complete
    wait $pull_pid
    
    if [ $? -eq 0 ]; then
        print_message "Image pulled successfully!" "$GREEN"
    else
        print_message "Failed to pull image!" "$RED"
        exit 1
    fi
fi
echo ""

# Step 4: Run the container
print_message "Step 4: Starting Docker container..." "$BLUE"
print_message "Container name: ${CONTAINER_NAME}" "$CYAN"
print_message "HTTP port: ${HTTP_PORT} -> 8000" "$CYAN"
print_message "gRPC port: ${GRPC_PORT} -> 8001" "$CYAN"
print_message "Metrics port: ${METRICS_PORT} -> 8002" "$CYAN"
echo ""

docker run -d --gpus all \
    -p ${HTTP_PORT}:8000 \
    -p ${GRPC_PORT}:8001 \
    -p ${METRICS_PORT}:8002 \
    --shm-size=2g \
    --name ${CONTAINER_NAME} \
    ${IMAGE_NAME} \
    tritonserver --model-repository=/models > /dev/null 2>&1

if [ $? -eq 0 ]; then
    print_message "Container started successfully!" "$GREEN"
    echo ""
    
    # Show container logs in background for monitoring
    print_message "Container logs (initial):" "$CYAN"
    docker logs --tail 5 ${CONTAINER_NAME} 2>&1 | head -10
    echo ""
else
    print_message "Failed to start container!" "$RED"
    exit 1
fi

# Step 5: Wait for service to be ready
echo ""
if ! wait_for_service; then
    print_message "Please check the logs manually: docker logs -f ${CONTAINER_NAME}" "$YELLOW"
    exit 1
fi
echo ""

# Step 6: Test model endpoint
print_message "Step 6: Testing model information endpoint..." "$BLUE"
echo ""
print_message "Testing: GET http://localhost:${HTTP_PORT}/v2/models/${MODEL_NAME}" "$CYAN"
echo ""

response=$(curl -s http://localhost:${HTTP_PORT}/v2/models/${MODEL_NAME} 2>&1)

if echo "$response" | grep -q "name"; then
    print_message "Model endpoint test: SUCCESS ✓" "$GREEN"
    echo ""
    print_message "Model Information:" "$BOLD"
    echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
else
    print_message "Model endpoint test: FAILED ✗" "$RED"
    echo "Response: $response"
    print_message "Checking if models are loaded..." "$YELLOW"
    docker logs --tail 20 ${CONTAINER_NAME} 2>&1 | grep -i "model\|loaded\|ready" | tail -10
fi
echo ""

# Step 7: Test inference endpoint (if audio file available)
print_message "Step 7: Testing ASR inference endpoint..." "$BLUE"
echo ""

# Check for common test audio files
TEST_AUDIO=""
if [ -f "ta2.wav" ]; then
    TEST_AUDIO="ta2.wav"
    TEST_LANG="ta"
elif [ -f "test.wav" ]; then
    TEST_AUDIO="test.wav"
    TEST_LANG="hi"
elif [ -f "/home/ubuntu/Benchmarking/test.wav" ]; then
    TEST_AUDIO="/home/ubuntu/Benchmarking/test.wav"
    TEST_LANG="hi"
elif [ -f "/home/ubuntu/Benchmarking/ta2.wav" ]; then
    TEST_AUDIO="/home/ubuntu/Benchmarking/ta2.wav"
    TEST_LANG="ta"
fi

if [ ! -z "$TEST_AUDIO" ] && check_audio_file "$TEST_AUDIO"; then
    print_message "Found test audio file: ${TEST_AUDIO}" "$GREEN"
    print_message "Language: ${TEST_LANG}" "$CYAN"
    echo ""
    print_message "Note: Testing ASR requires encoding audio to base64." "$YELLOW"
    print_message "Creating Python test script..." "$CYAN"
    
    # Create a temporary test script
    TEST_SCRIPT="/tmp/test_asr_$$.py"
    cat > ${TEST_SCRIPT} << 'PYTHON_EOF'
#!/usr/bin/env python3
import sys
import json
import base64
import requests

# Try to import required libraries
try:
    import wave
    import numpy as np
    HAS_WAVE_NUMPY = True
except ImportError:
    HAS_WAVE_NUMPY = False

audio_file = sys.argv[1]
lang_id = sys.argv[2]
endpoint = sys.argv[3]

if HAS_WAVE_NUMPY:
    try:
        # Read WAV file and extract audio signal
        with wave.open(audio_file, 'rb') as wav_file:
            # Get audio parameters
            sample_rate = wav_file.getframerate()
            num_channels = wav_file.getnchannels()
            num_frames = wav_file.getnframes()
            sample_width = wav_file.getsampwidth()
            
            # Read audio data
            audio_bytes = wav_file.readframes(num_frames)
            
            # Convert to numpy array based on sample width
            if sample_width == 1:
                audio_array = np.frombuffer(audio_bytes, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
            elif sample_width == 2:
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 4:
                audio_array = np.frombuffer(audio_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            
            # Convert to mono if stereo
            if num_channels == 2:
                audio_array = audio_array.reshape(-1, 2).mean(axis=1)
            
            # Normalize audio signal
            audio_array = audio_array.flatten().tolist()
            num_samples = len(audio_array)
        
        # Create request payload matching model interface
        payload = {
            "inputs": [
                {
                    "name": "AUDIO_SIGNAL",
                    "shape": [1, num_samples],
                    "datatype": "FP32",
                    "data": [audio_array]
                },
                {
                    "name": "NUM_SAMPLES",
                    "shape": [1, 1],
                    "datatype": "INT32",
                    "data": [[num_samples]]
                },
                {
                    "name": "LANG_ID",
                    "shape": [1, 1],
                    "datatype": "BYTES",
                    "data": [[lang_id]]
                }
            ],
            "outputs": [
                {
                    "name": "TRANSCRIPTS"
                }
            ]
        }
        
        response = requests.post(endpoint, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Extract transcript
        if "outputs" in result:
            for output in result["outputs"]:
                if output.get("name") == "TRANSCRIPTS" and "data" in output:
                    transcript = output["data"][0]
                    print(f"\n=== Transcript ===")
                    print(transcript)
                    
    except Exception as e:
        print(f"Error processing WAV file: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        HAS_WAVE_NUMPY = False

if not HAS_WAVE_NUMPY:
    # Fallback: try base64 encoded WAV data (simpler approach)
    try:
        print("Note: Using base64 encoded WAV format (wave/numpy not available)", file=sys.stderr)
        with open(audio_file, 'rb') as f:
            audio_data = f.read()
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        payload = {
            "inputs": [
                {
                    "name": "WAV_DATA",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": [audio_base64]
                },
                {
                    "name": "LANGUAGE_ID",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": [lang_id]
                }
            ],
            "outputs": [
                {
                    "name": "TRANSCRIPTS"
                }
            ]
        }
        
        response = requests.post(endpoint, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Extract transcript
        if "outputs" in result:
            for output in result["outputs"]:
                if output.get("name") == "TRANSCRIPTS" and "data" in output:
                    transcript = output["data"][0]
                    print(f"\n=== Transcript ===")
                    print(transcript)
    except Exception as e2:
        print(f"Error: {e2}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
PYTHON_EOF

    chmod +x ${TEST_SCRIPT}
    
    print_message "Running ASR inference test..." "$CYAN"
    echo ""
    
    if python3 ${TEST_SCRIPT} "${TEST_AUDIO}" "${TEST_LANG}" "http://localhost:${HTTP_PORT}/v2/models/${MODEL_NAME}/infer" 2>&1; then
        print_message "ASR inference test: SUCCESS ✓" "$GREEN"
    else
        print_message "ASR inference test: FAILED ✗" "$RED"
        print_message "Note: This might be due to audio file format or network issues." "$YELLOW"
    fi
    
    # Cleanup
    rm -f ${TEST_SCRIPT}
else
    print_message "No test audio file found. Skipping inference test." "$YELLOW"
    print_message "To test ASR, you can use:" "$CYAN"
    echo "  python3 -c \"
import base64, json, requests, sys
with open('your_audio.wav', 'rb') as f:
    audio_b64 = base64.b64encode(f.read()).decode('utf-8')
payload = {
    'inputs': [
        {'name': 'WAV_DATA', 'shape': [1], 'datatype': 'BYTES', 'data': [audio_b64]},
        {'name': 'LANGUAGE_ID', 'shape': [1], 'datatype': 'BYTES', 'data': ['hi']}
    ],
    'outputs': [{'name': 'TRANSCRIPTS'}]
}
r = requests.post('http://localhost:${HTTP_PORT}/v2/models/${MODEL_NAME}/infer', json=payload)
print(json.dumps(r.json(), indent=2, ensure_ascii=False))
\""
fi
echo ""

# Summary
echo -e "${BOLD}${GREEN}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                    Setup Complete!                        ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

print_message "Service Details:" "$BOLD"
echo "  - Container: ${CONTAINER_NAME}"
echo "  - HTTP Endpoint: http://localhost:${HTTP_PORT}"
echo "  - gRPC Endpoint: localhost:${GRPC_PORT}"
echo "  - Metrics Endpoint: http://localhost:${METRICS_PORT}/metrics"
echo "  - Model Name: ${MODEL_NAME}"
echo ""
print_message "Supported Languages:" "$CYAN"
echo "  - Multiple Indic languages (hi, ta, te, mr, gu, kn, ml, pa, bn, or, as, etc.)"
echo ""
print_message "Useful Commands:" "$CYAN"
echo "  - View logs: docker logs -f ${CONTAINER_NAME}"
echo "  - Stop service: docker stop ${CONTAINER_NAME}"
echo "  - Start service: docker start ${CONTAINER_NAME}"
echo "  - Remove service: docker stop ${CONTAINER_NAME} && docker rm ${CONTAINER_NAME}"
echo ""
print_message "Example Python test script:" "$CYAN"
echo "  # Create test_asr.py (requires wave and numpy):"
echo "  import wave, numpy as np, json, requests"
echo "  with wave.open('audio.wav', 'rb') as wav:"
echo "      sample_rate = wav.getframerate()"
echo "      num_frames = wav.getnframes()"
echo "      audio_bytes = wav.readframes(num_frames)"
echo "      audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0"
echo "      audio_array = audio_array.flatten().tolist()"
echo "      num_samples = len(audio_array)"
echo "  payload = {"
echo "      'inputs': ["
echo "          {'name': 'AUDIO_SIGNAL', 'shape': [1, num_samples], 'datatype': 'FP32', 'data': [audio_array]},"
echo "          {'name': 'NUM_SAMPLES', 'shape': [1, 1], 'datatype': 'INT32', 'data': [[num_samples]]},"
echo "          {'name': 'LANG_ID', 'shape': [1, 1], 'datatype': 'BYTES', 'data': [['hi']]}"
echo "      ],"
echo "      'outputs': [{'name': 'TRANSCRIPTS'}]"
echo "  }"
echo "  r = requests.post('http://localhost:${HTTP_PORT}/v2/models/${MODEL_NAME}/infer', json=payload)"
echo "  print(json.dumps(r.json(), indent=2, ensure_ascii=False))"
echo ""
print_message "Important Notes:" "$YELLOW"
echo "  - Audio files must be in WAV format"
echo "  - Audio must be base64 encoded for API requests"
echo "  - Supported language codes: hi, ta, te, mr, gu, kn, ml, pa, bn, or, as, etc."
echo ""

