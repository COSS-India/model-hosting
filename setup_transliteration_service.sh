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
IMAGE_NAME="ai4bharat/triton-indic-xlit:latest"
CONTAINER_NAME="indic-xlit-server"
HTTP_PORT=8200
GRPC_PORT=8201
METRICS_PORT=8202
MODEL_NAME="transliteration"

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
    local max_attempts=180
    local attempt=1
    
    print_message "Waiting for service to be ready..." "$CYAN"
    print_message "This may take 3-5 minutes on first run while models are being downloaded and loaded..." "$YELLOW"
    
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

# Clear screen
clear

# Banner
echo -e "${BOLD}${CYAN}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║   Indic Transliteration Service Setup & Test Script      ║"
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
else
    print_message "Image not found. Pulling ${IMAGE_NAME}..." "$YELLOW"
    print_message "This may take several minutes (image size: ~13.1GB)..." "$YELLOW"
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
    --shm-size=2gb \
    --name ${CONTAINER_NAME} \
    ${IMAGE_NAME} \
    tritonserver --model-repository=/models --log-verbose=1 --strict-readiness=false > /dev/null 2>&1

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
    print_message "Checking container logs..." "$YELLOW"
    docker logs --tail 20 ${CONTAINER_NAME} 2>&1 | tail -10
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
fi
echo ""

# Step 7: Test inference endpoint
print_message "Step 7: Testing inference endpoint..." "$BLUE"
echo ""
print_message "Test Input:" "$CYAN"
echo "  - Text: 'namaste'"
echo "  - Input Language: 'en'"
echo "  - Output Language: 'hi'"
echo "  - Word Level: true"
echo "  - Top K: 5"
echo ""

inference_response=$(curl -s -X POST http://localhost:${HTTP_PORT}/v2/models/${MODEL_NAME}/infer \
    -H "Content-Type: application/json" \
    -d '{
    "inputs": [
      {
        "name": "INPUT_TEXT",
        "shape": [1],
        "datatype": "BYTES",
        "data": ["namaste"]
      },
      {
        "name": "INPUT_LANGUAGE_ID",
        "shape": [1],
        "datatype": "BYTES",
        "data": ["en"]
      },
      {
        "name": "OUTPUT_LANGUAGE_ID",
        "shape": [1],
        "datatype": "BYTES",
        "data": ["hi"]
      },
      {
        "name": "IS_WORD_LEVEL",
        "shape": [1],
        "datatype": "BOOL",
        "data": [true]
      },
      {
        "name": "TOP_K",
        "shape": [1],
        "datatype": "UINT8",
        "data": [5]
      }
    ],
    "outputs": [
      {
        "name": "OUTPUT_TEXT"
      }
    ]
  }' 2>&1)

if echo "$inference_response" | grep -q "OUTPUT_TEXT"; then
    print_message "Inference endpoint test: SUCCESS ✓" "$GREEN"
    echo ""
    print_message "Transliteration Result:" "$BOLD"
    echo "$inference_response" | python3 -m json.tool 2>/dev/null || echo "$inference_response"
else
    print_message "Inference endpoint test: FAILED ✗" "$RED"
    echo "Response: $inference_response"
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
print_message "Useful Commands:" "$CYAN"
echo "  - View logs: docker logs -f ${CONTAINER_NAME}"
echo "  - Stop service: docker stop ${CONTAINER_NAME}"
echo "  - Start service: docker start ${CONTAINER_NAME}"
echo "  - Remove service: docker stop ${CONTAINER_NAME} && docker rm ${CONTAINER_NAME}"
echo ""

