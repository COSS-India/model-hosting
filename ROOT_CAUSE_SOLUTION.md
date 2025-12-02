# Root Cause Found and Fixed! 🎯

## The Problem

**GPU utilization was 0% because the model uses ONNX Runtime, but only the CPU version was installed.**

## Root Cause

1. The model `ai4bharat/indic-conformer-600m-multilingual` uses **ONNX Runtime** for inference
2. Only `onnxruntime` (CPU version) was installed
3. ONNX Runtime tried to use `CUDAExecutionProvider` but it wasn't available
4. Model fell back to `CPUExecutionProvider`, running entirely on CPU
5. Result: GPU utilization = 0%

## Evidence from Logs

```
Specified provider 'CUDAExecutionProvider' is not in available provider names.
Available providers: 'AzureExecutionProvider, CPUExecutionProvider'
```

## Solution Applied

1. ✅ Uninstalled `onnxruntime` (CPU version)
2. ✅ Installed `onnxruntime-gpu` (GPU version)
3. ✅ Verified CUDAExecutionProvider is now available:
   ```
   Available providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
   ```
4. ✅ Restarted server with GPU-enabled ONNX Runtime

## Expected Behavior Now

- ✅ ONNX Runtime will use `CUDAExecutionProvider`
- ✅ Model inference will run on GPU
- ✅ GPU utilization should increase during inference (typically 30-90%)
- ✅ GPU memory usage will increase
- ✅ Inference latency should be lower

## Verification

After the server restarts, you should see:
- No warnings about CUDAExecutionProvider not being available
- GPU utilization > 0% during inference
- Diagnostic logs showing model using GPU

## Test It

Run a benchmark and monitor GPU:
```bash
# Terminal 1: Monitor GPU
watch -n 0.5 nvidia-smi

# Terminal 2: Run benchmark
cd /home/ubuntu/Benchmarking/benchmark
python3 benchmark_asr.py \
  --url http://127.0.0.1:8000/asr \
  --audio /path/to/audio.wav \
  --rate 10 \
  --duration 30 \
  --lang ta
```

You should now see GPU utilization increase during inference!

## Key Lesson

**When using ONNX models, you MUST install `onnxruntime-gpu` instead of `onnxruntime` to enable GPU acceleration.**

The CPU version (`onnxruntime`) will work but will always use CPU, regardless of CUDA availability.

