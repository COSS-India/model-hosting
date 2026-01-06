#!/bin/bash

# Start FastAPI IndicTrans2 service with HuggingFace token

# Set your HuggingFace token here or export it before running
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN environment variable is not set."
    echo "Please set it with: export HF_TOKEN=your_token_here"
    exit 1
fi

cd /home/ubuntu/nmt-benchmarking/fastapi
source fastapi/bin/activate

# Login to HuggingFace (optional, but helps with authentication)
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true

# Start the service
PORT=${1:-8000}  # Default to port 8000 if not provided

echo "Starting FastAPI service with HF_TOKEN..."
echo "Service will be available at http://localhost:${PORT}"
echo "API Documentation: http://localhost:${PORT}/docs"
echo ""

# Run in background with nohup
nohup env HF_TOKEN="$HF_TOKEN" uvicorn app:app --host 0.0.0.0 --port "$PORT" > fastapi.log 2>&1 &

echo "Service started in background. PID: $!"
echo "View logs with: tail -f fastapi.log"
echo ""
echo "Note: If you get 403 errors, you may need to request access to the models at:"
echo "  - https://huggingface.co/ai4bharat/indictrans2-en-indic-1B"
echo "  - https://huggingface.co/ai4bharat/indictrans2-indic-en-1B"
echo "  - https://huggingface.co/ai4bharat/indictrans2-indic-indic-1B"

