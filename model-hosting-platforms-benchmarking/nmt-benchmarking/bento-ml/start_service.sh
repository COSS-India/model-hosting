#!/bin/bash

# Start BentoML IndicTrans2 service with HuggingFace token

# Set your HuggingFace token here or export it before running
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN environment variable is not set."
    echo "Please set it with: export HF_TOKEN=your_token_here"
    echo "Or edit this script to set it."
    exit 1
fi

cd /home/ubuntu/nmt-benchmarking/bento-ml
source bento/bin/activate

# Login to HuggingFace (optional, but helps with authentication)
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true

# Start the service
echo "Starting BentoML service with HF_TOKEN..."
echo "Service will be available at http://localhost:3000"
echo ""

nohup env HF_TOKEN="$HF_TOKEN" bentoml serve indictrans_nmt:latest --port 3000 --host 0.0.0.0 > bentoml.log 2>&1 &

echo "Service started in background. PID: $!"
echo "View logs with: tail -f bentoml.log"
echo ""
echo "Note: If you get 403 errors, you may need to request access to the models at:"
echo "  - https://huggingface.co/ai4bharat/indictrans2-en-indic-1B"
echo "  - https://huggingface.co/ai4bharat/indictrans2-indic-en-1B"
echo "  - https://huggingface.co/ai4bharat/indictrans2-indic-indic-1B"

