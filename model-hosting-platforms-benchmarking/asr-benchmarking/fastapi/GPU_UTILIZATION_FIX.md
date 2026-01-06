# Why GPU Utilization is Still 0% - Root Cause Analysis

## Critical Finding

The model **requires `onnxruntime`**, which suggests it may be using ONNX for inference. However, the more likely issue is that **the model's `transcribe()` method is a high-level wrapper that performs all inference on CPU**, even if the model weights are loaded on GPU.

## Root Causes

### 1. Model's `transcribe()` Method May Be CPU-Only
- The `transcribe()` method is likely a convenience wrapper
- It may do preprocessing on CPU and call lower-level methods
- Even if model weights are on GPU, the wrapper might not use them

### 2. Server Not Restarted
- The server is running with **old code** (started before fixes)
- **You must restart the server** for changes to take effect

### 3. Input Data Not on GPU
- Even with our fixes, if `transcribe()` doesn't accept GPU tensors, it falls back to CPU
- The method signature might not support GPU tensors

## Immediate Actions Required

### Step 1: Restart the Server
```bash
# Find the server process
ps aux | grep uvicorn

# Kill it (replace PID with actual process ID)
kill <PID>

# Restart with new code
cd /home/ubuntu/Benchmarking/benchmark
source fastapi/bin/activate  # if using virtualenv
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Step 2: Check Server Logs on Startup
Look for diagnostic messages:
```
[DIAGNOSTIC] Model parameters - GPU: X, CPU: Y
[DIAGNOSTIC] Model type: <class '...'>
[DIAGNOSTIC] Has transcribe: True/False
```

### Step 3: Monitor GPU During Inference
```bash
# In one terminal
watch -n 0.5 nvidia-smi

# In another terminal, send a test request
curl -X POST "http://localhost:8000/asr" \
  -F "audio=@/path/to/audio.wav" \
  -F "lang=en"
```

## If GPU Utilization is Still 0% After Restart

### Option 1: Check Model's Actual Implementation
The model's `transcribe()` method might be hardcoded to use CPU. Check:
```python
# In Python, after loading model
import inspect
print(inspect.getsource(model.transcribe))
```

### Option 2: Use Model's Lower-Level Methods
Instead of `transcribe()`, try using the model's forward pass directly:
```python
# Instead of model.transcribe(data)
# Try:
with torch.no_grad():
    output = model.forward(data_tensor)
```

### Option 3: Check if Model is ONNX-Based
If the model is ONNX, it needs special configuration:
```python
import onnxruntime as ort
session = ort.InferenceSession(
    model_path,
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
```

### Option 4: Verify Model Actually Uses GPU
Add this check in the inference code:
```python
# Before inference
if DEVICE == "cuda":
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    
# ... inference ...

# After inference
if DEVICE == "cuda":
    end_event.record()
    torch.cuda.synchronize()
    print(f"GPU time: {start_event.elapsed_time(end_event)} ms")
```

## Expected Diagnostic Output

After restart, you should see in server logs:
```
Loading model: ai4bharat/indic-conformer-600m-multilingual
Using device: cuda
[DIAGNOSTIC] Model parameters - GPU: <large number>, CPU: 0
[DIAGNOSTIC] Model type: <class '...'>
[DIAGNOSTIC] Has transcribe: True
Model loaded successfully. Sample rate: 16000 Hz
```

During inference:
```
[INFERENCE] Model device: cuda:0, Input type: <class 'numpy.ndarray'>
```

## Most Likely Solution

If the model's `transcribe()` method is CPU-only, you may need to:

1. **Use the model's forward pass directly** instead of `transcribe()`
2. **Check the model's documentation** for GPU inference examples
3. **Contact the model maintainers** or check GitHub issues for GPU usage

## Quick Test

After restarting the server, run this test:
```python
import requests
import base64

# Read audio file
with open("/path/to/audio.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

# Send request
response = requests.post(
    "http://localhost:8000/asr/json",
    json={"audio_b64": audio_b64, "lang": "en"}
)
print(response.json())
```

While this runs, monitor GPU with `nvidia-smi`. If GPU utilization is still 0%, the model's `transcribe()` method is likely CPU-only.

