# GPU Utilization Analysis - Why Server Not Utilizing GPU

## Executive Summary

The server is not utilizing GPU because **all requests are failing with HTTP 400 status**, preventing the model from executing inference. Additionally, there are potential issues with how GPU tensors are handled during inference.

## Issues Identified

### 1. **CRITICAL: Missing `lang` Parameter in Benchmark Script** ✅ FIXED

**Location:** `/home/ubuntu/Benchmarking/benchmark/benchmark_asr.py` line 128-132

**Problem:**
- The benchmark script sends requests without the required `lang` parameter
- The API endpoint requires `lang` parameter (see `app.py` line 256-268)
- All 602 requests returned HTTP 400 status code
- Since requests fail, the model inference never runs, so GPU utilization stays at 0%

**Evidence:**
- `requests.csv` shows all requests with `status=400`
- `gpu_samples.csv` shows `gpu_util_percent=0.0` throughout
- `benchmark_asr.py` only sends `audio` field, missing `lang`

**Fix Applied:**
- Added `data.add_field('lang', 'en')` to the benchmark script (defaults to English)

### 2. **Potential Issue: Input Data Not Moved to GPU**

**Location:** `/home/ubuntu/Benchmarking/benchmark/app.py` lines 294-330

**Problem:**
- When using `model.transcribe()` or `model.generate()`, the input data (numpy arrays or file paths) may not be explicitly moved to GPU
- The model is loaded with `device_map="auto"` which places it on GPU, but input preprocessing might happen on CPU
- Only the `callable(model)` branch (line 337-338) explicitly moves tensors to GPU

**Impact:**
- Even if requests succeed, inference might run on CPU if input tensors aren't on GPU
- This would explain why GPU utilization remains low even after fixing the request issue

**Recommendation:**
- Ensure input tensors are moved to GPU before calling model methods
- Verify that `transcribe()` and `generate()` methods handle GPU tensors correctly
- Consider using `torch.cuda.current_device()` to verify tensor device placement

### 3. **Model Loading Configuration**

**Location:** `/home/ubuntu/Benchmarking/benchmark/app.py` lines 69-75

**Current Configuration:**
```python
model = AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None,
    token=HF_TOKEN,
    trust_remote_code=True
)
```

**Analysis:**
- `device_map="auto"` should automatically place model on GPU
- However, this might not work correctly for all model architectures
- The fallback `model.to(DEVICE)` on line 78 only runs if `DEVICE == "cpu"`

**Recommendation:**
- Add explicit device placement verification after model loading
- Log the actual device where model layers are placed
- Consider using `device_map="cuda:0"` instead of `"auto"` for more explicit control

## Root Cause Analysis

### Primary Cause: Request Failures
1. Benchmark script doesn't send required `lang` parameter
2. API rejects all requests with 400 status
3. Model inference never executes
4. GPU utilization remains at 0%

### Secondary Cause: Input Tensor Device Placement
1. Model may be on GPU via `device_map="auto"`
2. Input data (numpy arrays) are created on CPU
3. When calling `transcribe()` or `generate()`, input might not be moved to GPU
4. Inference runs on CPU even though model is on GPU

## Verification Steps

After fixing the benchmark script, verify:

1. **Request Status:**
   ```bash
   # Check if requests now return 200 status
   tail -20 /home/ubuntu/bench_run1/requests.csv
   ```

2. **GPU Utilization:**
   ```bash
   # Monitor GPU during benchmark
   watch -n 0.5 nvidia-smi
   ```

3. **Model Device Placement:**
   ```python
   # Add to app.py startup to verify model is on GPU
   if DEVICE == "cuda":
       for name, param in model.named_parameters():
           print(f"{name}: {param.device}")
   ```

4. **Input Tensor Device:**
   ```python
   # Add logging to verify input tensors are on GPU
   print(f"Input tensor device: {data_tensor.device}")
   ```

## Recommendations

### Immediate Actions:
1. ✅ **FIXED:** Add `lang` parameter to benchmark script
2. Re-run benchmark to verify requests succeed
3. Monitor GPU utilization during successful requests

### Code Improvements:
1. Add explicit GPU tensor placement for all inference paths
2. Add device verification logging in model loading
3. Add input tensor device logging in inference code
4. Consider using `torch.cuda.synchronize()` before/after inference for accurate timing

### Testing:
1. Run benchmark with fixed script
2. Verify requests return 200 status
3. Check GPU utilization increases during inference
4. Compare CPU vs GPU inference latency

## Additional Issue Found: Input Data Not on GPU

**Location:** `/home/ubuntu/Benchmarking/benchmark/app.py` lines 294-318, 415-440

**Problem:**
- Even after requests succeed (status 200), GPU utilization remains at 0%
- The `model.transcribe()` method is being called with CPU data:
  - File paths (line 300, 422) - model reads file and processes on CPU
  - Numpy arrays (line 310, 432) - CPU numpy arrays, not GPU tensors
- The model's `transcribe()` method might be doing all preprocessing and inference on CPU, even if model weights are on GPU

**Fix Applied:**
- Modified inference code to convert numpy arrays to GPU tensors before calling `transcribe()`
- Added fallback logic to try GPU tensor first, then numpy array if tensor doesn't work
- Updated both `/asr` and `/asr/json` endpoints

**Note:** The model's `transcribe()` method might still not use GPU if it's a high-level wrapper that does CPU preprocessing. In that case, we may need to:
1. Use the model's lower-level methods directly
2. Or ensure the model's transcribe method is configured to use GPU

## Expected Behavior After Fix

- Requests should return HTTP 200 status ✅ (Already working)
- GPU utilization should increase during inference (typically 30-90% depending on model)
- GPU memory usage should increase (model weights + activations)
- Latency should be lower than CPU inference

**If GPU utilization is still 0% after this fix:**
- The model's `transcribe()` method might be doing CPU-only inference
- Check if the model has lower-level methods that can be called with GPU tensors
- Consider using the model's forward pass directly instead of `transcribe()`

## Files Modified

1. `/home/ubuntu/Benchmarking/benchmark/benchmark_asr.py`
   - Added `lang` parameter to requests (line 133)
   - Added `--lang` command-line argument

2. `/home/ubuntu/Benchmarking/benchmark/app.py`
   - Updated `/asr` endpoint to use GPU tensors when calling `transcribe()` (lines 294-318)
   - Updated `/asr/json` endpoint to use GPU tensors when calling `transcribe()` (lines 415-440)
   - Added GPU tensor conversion before calling model methods
   - Added fallback logic for models that don't accept GPU tensors directly

## Next Steps

1. Re-run the benchmark with the fixed script
2. Monitor GPU utilization during the run
3. If GPU utilization is still low, investigate input tensor device placement
4. Consider adding explicit GPU tensor operations in the inference code

