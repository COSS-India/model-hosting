#!/bin/bash

# Real-time GPU monitoring script for Triton inference

echo "Monitoring GPU utilization during Triton inference..."
echo "Press Ctrl+C to stop"
echo ""

# Function to run inference
run_inference() {
    curl -s -X POST http://localhost:8000/v2/models/nmt/infer \
      -H 'Content-Type: application/json' \
      -d '{"inputs":[{"name":"INPUT_TEXT","datatype":"BYTES","shape":[1,1],"data":["Hello, how are you? This is a longer text to test GPU utilization during neural machine translation between English and Hindi."]},{"name":"INPUT_LANGUAGE_ID","datatype":"BYTES","shape":[1,1],"data":["en"]},{"name":"OUTPUT_LANGUAGE_ID","datatype":"BYTES","shape":[1,1],"data":["hi"]}]}' > /dev/null
}

# Monitor GPU
monitor_gpu() {
    while true; do
        clear
        echo "=== GPU Utilization Monitor ==="
        echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
        nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu --format=csv
        echo ""
        echo "Press Ctrl+C to stop monitoring"
        sleep 1
    done
}

# Run inference in background and monitor
if [ "$1" = "--inference" ]; then
    echo "Running continuous inference requests..."
    (
        while true; do
            run_inference
            sleep 2
        done
    ) &
    INFERENCE_PID=$!
    echo "Inference PID: $INFERENCE_PID"
    echo ""
    monitor_gpu
    kill $INFERENCE_PID 2>/dev/null
else
    monitor_gpu
fi

