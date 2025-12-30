#!/usr/bin/env python3
"""
Quick script to verify GPU monitoring is working correctly.
Run this to test if pynvml can access and monitor your GPU.
"""

import sys

try:
    import pynvml
    print("[✓] pynvml is installed")
except ImportError:
    print("[✗] pynvml is NOT installed")
    print("    Install with: pip install pynvml")
    sys.exit(1)

try:
    pynvml.nvmlInit()
    print("[✓] NVML initialized successfully")
    
    device_count = pynvml.nvmlDeviceGetCount()
    print(f"[✓] Found {device_count} GPU device(s)")
    
    if device_count == 0:
        print("[✗] No GPU devices found")
        sys.exit(1)
    
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        name_str = name.decode('utf-8') if isinstance(name, bytes) else name
        
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        print(f"\nGPU {i}: {name_str}")
        print(f"  Utilization: {util.gpu}% (GPU), {util.memory}% (Memory)")
        print(f"  Memory: {mem_info.used / (1024*1024):.0f}MB / {mem_info.total / (1024*1024):.0f}MB")
        print(f"  Memory Used: {(mem_info.used / mem_info.total) * 100:.1f}%")
    
    # Test continuous sampling
    print("\n[TEST] Sampling GPU for 5 seconds...")
    import time
    samples = []
    for i in range(10):
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        samples.append((util.gpu, mem_info.used / (1024*1024)))
        time.sleep(0.5)
    
    utils = [s[0] for s in samples]
    mems = [s[1] for s in samples]
    print(f"[✓] Collected {len(samples)} samples")
    print(f"  GPU Utilization: min={min(utils)}%, max={max(utils)}%, avg={sum(utils)/len(utils):.1f}%")
    print(f"  GPU Memory: min={min(mems):.0f}MB, max={max(mems):.0f}MB, avg={sum(mems)/len(mems):.0f}MB")
    
    pynvml.nvmlShutdown()
    print("\n[✓] GPU monitoring test PASSED")
    
except pynvml.NVMLError as e:
    print(f"[✗] NVML Error: {e}")
    print("    This might indicate:")
    print("    - NVIDIA drivers not installed")
    print("    - Insufficient permissions")
    print("    - GPU not accessible")
    sys.exit(1)
except Exception as e:
    print(f"[✗] Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

