# Diagnostic Steps for GPU Utilization Issues

## Quick Checks

### 1. Verify Requests Are Succeeding
```bash
# Check request status codes
tail -20 /path/to/bench_run/requests.csv | awk -F',' '{print $3}' | sort | uniq -c
# Should show mostly 200 status codes
```

### 2. Check GPU Availability
```bash
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
nvidia-smi
```

### 3. Verify Model Device Placement
Add this to `app.py` in the `load_model()` function after model loading:
```python
if DEVICE == "cuda":
    print("\n=== Model Device Check ===")
    gpu_params = sum(1 for p in model.parameters() if p.device.type == 'cuda')
    cpu_params = sum(1 for p in model.parameters() if p.device.type == 'cpu')
    print(f"Parameters on GPU: {gpu_params}")
    print(f"Parameters on CPU: {cpu_params}")
```

### 4. Monitor GPU During Inference
```bash
# In one terminal, run benchmark
python3 benchmark_asr.py --url http://127.0.0.1:8000/asr --audio test.wav --rate 10 --duration 30

# In another terminal, monitor GPU
watch -n 0.5 nvidia-smi
```

### 5. Check What Inference Method Is Being Used
Add logging to `app.py` in the inference code:
```python
print(f"Using inference method: transcribe={hasattr(model, 'transcribe')}, generate={hasattr(model, 'generate')}, callable={callable(model)}")
if hasattr(model, 'transcribe'):
    print(f"Input data type: {type(data)}, device: {data.device if hasattr(data, 'device') else 'numpy array'}")
```

### 6. Test Direct GPU Tensor Inference
If `transcribe()` still doesn't use GPU, try calling the model's forward pass directly:
```python
# After model loading, check if model has a forward method
if hasattr(model, 'forward'):
    # Try calling forward directly with GPU tensors
    # This bypasses the transcribe() wrapper
```

## Common Issues and Solutions

### Issue: Model's `transcribe()` method does CPU preprocessing
**Solution:** Use model's lower-level methods or forward pass directly

### Issue: Input tensors not on GPU
**Solution:** Explicitly move tensors to GPU: `tensor.cuda()` or `tensor.to('cuda')`

### Issue: Model loaded with `device_map="auto"` but not using GPU
**Solution:** 
- Check actual device placement with diagnostic code above
- Try explicit device placement: `model.to('cuda')` instead of `device_map="auto"`

### Issue: Batch size too small
**Solution:** Process multiple requests in a batch to better utilize GPU

### Issue: Model is ONNX or quantized
**Solution:** Some ONNX models run on CPU by default. Check model type and configuration.


