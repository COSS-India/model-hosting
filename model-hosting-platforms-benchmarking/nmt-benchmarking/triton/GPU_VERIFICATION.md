# GPU Verification and Monitoring Guide

This guide helps you verify that the Triton server is running on GPU and monitor GPU utilization.

## Quick Verification

### 1. Run Comprehensive Status Check
```bash
cd /home/ubuntu/nmt-benchmarking/triton
./check_triton_status.sh
```

This script checks:
- Container status
- Server health
- Model configuration (GPU vs CPU)
- GPU availability
- Port status
- Metrics

### 2. Verify GPU Usage
```bash
./verify_gpu.sh
```

This script:
- Verifies container can see GPU
- Checks model instance configuration
- Monitors GPU utilization during inference
- Shows Triton metrics

## Manual Verification Steps

### Check Container is Using GPU
```bash
# Check if GPU is visible in container
docker exec triton-indictrans-v2 nvidia-smi

# Check container GPU access
docker inspect triton-indictrans-v2 | grep -i gpu
```

### Verify Model Instance is on GPU
```bash
curl http://localhost:8000/v2/models/nmt/config | python3 -m json.tool | grep -A 5 instance_group
```

Look for:
```json
"instance_group": [
    {
        "kind": "KIND_GPU",  // ✓ This confirms GPU usage
        "gpus": [0],
        ...
    }
]
```

If you see `"kind": "KIND_CPU"`, the model is running on CPU.

### Monitor GPU Utilization in Real-Time

**Option 1: Continuous monitoring**
```bash
watch -n 1 nvidia-smi
```

**Option 2: Interactive monitoring with inference**
```bash
./monitor_gpu.sh --inference
```

**Option 3: Simple nvidia-smi check**
```bash
nvidia-smi
```

### Monitor GPU During Inference

**Step 1: Open a terminal and start monitoring**
```bash
watch -n 0.5 nvidia-smi
```

**Step 2: In another terminal, run inference**
```bash
curl -X POST http://localhost:8000/v2/models/nmt/infer \
  -H 'Content-Type: application/json' \
  -d '{"inputs":[{"name":"INPUT_TEXT","datatype":"BYTES","shape":[1,1],"data":["Hello, how are you? This is a longer text to test GPU utilization during neural machine translation."]},{"name":"INPUT_LANGUAGE_ID","datatype":"BYTES","shape":[1,1],"data":["en"]},{"name":"OUTPUT_LANGUAGE_ID","datatype":"BYTES","shape":[1,1],"data":["hi"]}]}'
```

**What to look for:**
- GPU utilization should increase (often 30-100%)
- GPU memory usage should increase
- Temperature may rise slightly

## Triton Metrics

### Check Triton Metrics Endpoint
```bash
curl http://localhost:8002/metrics | grep -i gpu
```

### Key Metrics to Monitor
```bash
# Inference requests
curl http://localhost:8002/metrics | grep nv_inference_request

# Execution count
curl http://localhost:8002/metrics | grep nv_inference_exec

# GPU utilization (if available)
curl http://localhost:8002/metrics | grep nv_gpu_utilization
```

## Verify Docker Compose GPU Configuration

Check `docker-compose.yml`:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

## Expected GPU Behavior

### When Model is Running on GPU:
1. ✅ `nvidia-smi` shows process using GPU
2. ✅ GPU utilization increases during inference
3. ✅ Model config shows `"kind": "KIND_GPU"`
4. ✅ Container has access to GPU devices
5. ✅ Triton logs show CUDA initialization

### Signs of CPU Fallback:
1. ❌ GPU utilization stays at 0%
2. ❌ Model config shows `"kind": "KIND_CPU"`
3. ❌ No GPU processes in `nvidia-smi`
4. ❌ Container cannot access GPU

## Troubleshooting

### GPU Not Detected in Container
```bash
# Check NVIDIA Docker runtime
docker info | grep -i runtime

# Restart container with explicit GPU access
docker-compose down
docker-compose up -d
```

### Model Still Running on CPU
1. Check docker-compose.yml has GPU configuration
2. Verify NVIDIA Docker runtime is installed
3. Check container logs: `docker logs triton-indictrans-v2`
4. Verify GPU is available: `nvidia-smi`

### Low GPU Utilization
- Batch size might be too small
- Model might be very fast (small models)
- Check if multiple requests trigger higher utilization
- Run multiple concurrent requests

## Benchmark GPU vs CPU

To verify GPU is actually faster:

```bash
# Time inference on GPU (should be configured)
time curl -s -X POST http://localhost:8000/v2/models/nmt/infer \
  -H 'Content-Type: application/json' \
  -d '{"inputs":[{"name":"INPUT_TEXT","datatype":"BYTES","shape":[1,1],"data":["Hello"]},{"name":"INPUT_LANGUAGE_ID","datatype":"BYTES","shape":[1,1],"data":["en"]},{"name":"OUTPUT_LANGUAGE_ID","datatype":"BYTES","shape":[1,1],"data":["hi"]}]}'
```

GPU inference should be significantly faster than CPU.

## Useful Commands Summary

```bash
# Quick status check
./check_triton_status.sh

# Verify GPU usage
./verify_gpu.sh

# Monitor GPU in real-time
./monitor_gpu.sh

# Manual checks
nvidia-smi
docker exec triton-indictrans-v2 nvidia-smi
curl http://localhost:8000/v2/models/nmt/config | python3 -m json.tool
curl http://localhost:8002/metrics | grep gpu
```

