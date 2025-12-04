#!/usr/bin/env python3
"""
benchmark_asr.py - Comprehensive ASR Benchmark Tool

Usage:
  python benchmark_asr.py \
    --endpoint http://127.0.0.1:8000/asr \
    --audio /path/to/audio.wav \
    --lang_id ta \
    --outputdir /home/ubuntu/bench_results

Output:
  - benchmark_results.xlsx: Excel file with all metrics in separate columns
  - Individual CSV files for detailed data (requests.csv, gpu_samples.csv, sys_samples.csv)
"""

import argparse
import asyncio
import aiohttp
import time
import csv
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import psutil
import threading
from collections import defaultdict

# Try pynvml import robustly
try:
    import pynvml
    PYNVML_PRESENT = True
except Exception:
    PYNVML_PRESENT = False

# Try pandas and openpyxl for Excel export
try:
    import pandas as pd
    PANDAS_PRESENT = True
except ImportError:
    PANDAS_PRESENT = False
    print("[WARNING] pandas not installed. Install with: pip install pandas openpyxl")

# ------------------------------
# Sampling helpers (GPU & Sys)
# ------------------------------
def gpu_sampler(stop_event, sample_interval, out_list):
    """Samples GPU utilization and memory every sample_interval seconds until stop_event is set.
    Appends tuples (ts, util_percent, mem_used_mb) to out_list."""
    if not PYNVML_PRESENT:
        print("[gpu_sampler] pynvml not available, skipping GPU sampling.")
        return
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # single GPU index 0
    except Exception as e:
        print("[gpu_sampler] nvml init error:", e)
        return

    while not stop_event.is_set():
        ts = time.time()
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu  # percent
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used_mb = int(mem_info.used / (1024*1024))
        except Exception as e:
            util = -1
            mem_used_mb = -1
        out_list.append((ts, float(util), int(mem_used_mb)))
        stop_event.wait(sample_interval)
    # final sample
    try:
        ts = time.time()
        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_used_mb = int(mem_info.used / (1024*1024))
        out_list.append((ts, float(util), int(mem_used_mb)))
    except Exception:
        pass

def sys_sampler(stop_event, sample_interval, out_list):
    """Samples CPU percent and memory usage. Appends tuples (ts, cpu_percent, mem_used_mb, mem_total_mb)."""
    # warmup call for psutil.cpu_percent
    psutil.cpu_percent(interval=None)
    while not stop_event.is_set():
        ts = time.time()
        cpu = psutil.cpu_percent(interval=None)  # non-blocking percent since last call
        vm = psutil.virtual_memory()
        mem_used_mb = int((vm.total - vm.available) / (1024*1024))
        mem_total_mb = int(vm.total / (1024*1024))
        out_list.append((ts, float(cpu), int(mem_used_mb), int(mem_total_mb)))
        stop_event.wait(sample_interval)
    # final sample
    ts = time.time()
    cpu = psutil.cpu_percent(interval=None)
    vm = psutil.virtual_memory()
    mem_used_mb = int((vm.total - vm.available) / (1024*1024))
    mem_total_mb = int(vm.total / (1024*1024))
    out_list.append((ts, float(cpu), int(mem_used_mb), int(mem_total_mb)))

# ------------------------------
# Load generator (rate-controlled)
# ------------------------------
async def run_rate_test(endpoint, audio_path, rate, duration_s, lang_id):
    """
    Sends requests at approx 'rate' req/sec for 'duration_s' seconds.
    Returns list of (ts, latency_ms, status).
    """
    # Preload audio bytes into memory
    with open(audio_path, "rb") as fh:
        audio_bytes = fh.read()
    filename = Path(audio_path).name

    results = []  # (ts, latency_ms, status)
    stop_time = time.time() + duration_s

    # Use aiohttp session
    timeout = aiohttp.ClientTimeout(total=300)  # Increased timeout for long requests
    conn = aiohttp.TCPConnector(limit=0)  # unlimited concurrent connections
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        # schedule inter-arrival
        if rate <= 0:
            return results
        interval = 1.0 / float(rate)
        next_send = time.time()
        tasks = []
        pbar = tqdm(total=duration_s, desc=f"RPS {rate}", unit="s")
        # Use limited concurrency semaphore to avoid too many tasks: allow up to 4*rate or 1000
        max_concurrency = min(1000, max(50, int(rate*4)))
        sem = asyncio.Semaphore(max_concurrency)

        async def do_request(i):
            nonlocal session, filename, audio_bytes, results, lang_id
            async with sem:
                start = time.time()
                try:
                    data = aiohttp.FormData()
                    data.add_field('file',
                                   audio_bytes,
                                   filename=filename,
                                   content_type='audio/wav')
                    data.add_field('lang', lang_id)
                    data.add_field('strategy', 'ctc')  # Add strategy field for BentoML service
                    async with session.post(endpoint, data=data) as resp:
                        await resp.read()  # ensure body read
                        latency = (time.time() - start) * 1000.0
                        results.append((start, latency, resp.status))
                except Exception as e:
                    latency = (time.time() - start) * 1000.0
                    # use status 0 for failures
                    results.append((start, latency, 0))

        # Loop and schedule requests
        while time.time() < stop_time:
            now = time.time()
            if now >= next_send:
                # schedule
                tasks.append(asyncio.create_task(do_request(len(tasks))))
                next_send += interval
            else:
                # sleep a tiny bit
                await asyncio.sleep(min(0.001, next_send - now))
            # update progress
            if int(time.time()) % 1 == 0:
                pbar.update(0)
        # wait for outstanding tasks to finish
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        pbar.close()
    return results

# ------------------------------
# Metrics Calculation
# ------------------------------
def compute_latency_percentiles(latencies_ms):
    """Calculate p50, p95, p99 latency percentiles."""
    arr = np.array(latencies_ms, dtype=float)
    if arr.size == 0:
        return {"p50": None, "p95": None, "p99": None, "mean": None, "min": None, "max": None}
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr))
    }

def compute_qps(results, start_time, end_time):
    """Calculate Queries Per Second."""
    if len(results) < 2:
        return 0.0
    total_requests = len(results)
    duration = max(1e-6, end_time - start_time)
    return float(total_requests) / duration

def compute_gpu_metrics(gpu_samples):
    """Calculate GPU utilization metrics."""
    if not gpu_samples or len(gpu_samples) == 0:
        return {
            "avg_util_percent": 0.0,
            "max_util_percent": 0.0,
            "min_util_percent": 0.0,
            "avg_mem_mb": 0.0,
            "max_mem_mb": 0.0
        }
    
    utils = [s[1] for s in gpu_samples if s[1] >= 0]
    mems = [s[2] for s in gpu_samples if s[2] >= 0]
    
    if not utils:
        return {
            "avg_util_percent": 0.0,
            "max_util_percent": 0.0,
            "min_util_percent": 0.0,
            "avg_mem_mb": float(np.mean(mems)) if mems else 0.0,
            "max_mem_mb": float(np.max(mems)) if mems else 0.0
        }
    
    return {
        "avg_util_percent": float(np.mean(utils)),
        "max_util_percent": float(np.max(utils)),
        "min_util_percent": float(np.min(utils)),
        "avg_mem_mb": float(np.mean(mems)) if mems else 0.0,
        "max_mem_mb": float(np.max(mems)) if mems else 0.0
    }

def compute_cpu_memory_metrics(sys_samples):
    """Calculate CPU and memory usage metrics."""
    if not sys_samples or len(sys_samples) == 0:
        return {
            "avg_cpu_percent": 0.0,
            "max_cpu_percent": 0.0,
            "avg_mem_used_mb": 0.0,
            "max_mem_used_mb": 0.0,
            "mem_total_mb": 0.0
        }
    
    cpus = [s[1] for s in sys_samples]
    mems = [s[2] for s in sys_samples]
    mem_total = sys_samples[0][3] if sys_samples else 0
    
    return {
        "avg_cpu_percent": float(np.mean(cpus)),
        "max_cpu_percent": float(np.max(cpus)),
        "avg_mem_used_mb": float(np.mean(mems)),
        "max_mem_used_mb": float(np.max(mems)),
        "mem_total_mb": float(mem_total)
    }

def compute_throughput_latency_relationship(results, audio_size_bytes):
    """Calculate throughput and its relationship with latency."""
    if len(results) < 2:
        return {
            "throughput_bytes_per_sec": 0.0,
            "throughput_mb_per_sec": 0.0,
            "latency_at_peak_throughput": 0.0
        }
    
    start_ts = min(r[0] for r in results)
    end_ts = max(r[0] for r in results)
    duration = max(1e-6, end_ts - start_ts)
    total_requests = len(results)
    
    throughput_bytes_per_sec = (total_requests * audio_size_bytes) / duration
    throughput_mb_per_sec = throughput_bytes_per_sec / (1024 * 1024)
    
    # Find latency at peak throughput (use p50 as representative)
    latencies = [r[1] for r in results if r[1] is not None and r[1] > 0]
    latency_at_peak = float(np.percentile(latencies, 50)) if latencies else 0.0
    
    return {
        "throughput_bytes_per_sec": throughput_bytes_per_sec,
        "throughput_mb_per_sec": throughput_mb_per_sec,
        "latency_at_peak_throughput": latency_at_peak
    }

# ------------------------------
# CSV Writing (for detailed data)
# ------------------------------
def write_requests_csv(out_dir, rows):
    out_path = Path(out_dir) / "requests.csv"
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ts_unix", "latency_ms", "status"])
        for r in rows:
            w.writerow([r[0], r[1], r[2]])
    return out_path

def write_gpu_csv(out_dir, rows):
    out_path = Path(out_dir) / "gpu_samples.csv"
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ts_unix", "gpu_util_percent", "gpu_mem_mb"])
        for r in rows:
            w.writerow([r[0], r[1], r[2]])
    return out_path

def write_sys_csv(out_dir, rows):
    out_path = Path(out_dir) / "sys_samples.csv"
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ts_unix", "cpu_percent", "mem_used_mb", "mem_total_mb"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], r[3]])
    return out_path

# ------------------------------
# Excel Export
# ------------------------------
def write_excel_summary(out_dir, metrics):
    """Write all metrics to Excel file."""
    excel_path = Path(out_dir) / "benchmark_results.xlsx"
    
    # Prepare data for Excel
    data = {
        "Timestamp": [time.strftime("%Y-%m-%d %H:%M:%S")],
        "Endpoint": [metrics["config"]["endpoint"]],
        
        # Latency Metrics
        "Latency_p50_ms": [metrics["latency"]["p50"]],
        "Latency_p95_ms": [metrics["latency"]["p95"]],
        "Latency_p99_ms": [metrics["latency"]["p99"]],
        "Latency_mean_ms": [metrics["latency"]["mean"]],
        "Latency_min_ms": [metrics["latency"]["min"]],
        "Latency_max_ms": [metrics["latency"]["max"]],
        
        # QPS Metrics
        "QPS": [metrics["qps"]],
        "Total_Requests": [metrics["total_requests"]],
        "Successful_Requests": [metrics["successful_requests"]],
        "Failed_Requests": [metrics["failed_requests"]],
        
        # GPU Metrics
        "GPU_Avg_Util_Percent": [metrics["gpu"]["avg_util_percent"]],
        "GPU_Max_Util_Percent": [metrics["gpu"]["max_util_percent"]],
        "GPU_Min_Util_Percent": [metrics["gpu"]["min_util_percent"]],
        "GPU_Avg_Mem_MB": [metrics["gpu"]["avg_mem_mb"]],
        "GPU_Max_Mem_MB": [metrics["gpu"]["max_mem_mb"]],
        
        # CPU & Memory Metrics
        "CPU_Avg_Percent": [metrics["cpu_memory"]["avg_cpu_percent"]],
        "CPU_Max_Percent": [metrics["cpu_memory"]["max_cpu_percent"]],
        "Memory_Avg_Used_MB": [metrics["cpu_memory"]["avg_mem_used_mb"]],
        "Memory_Max_Used_MB": [metrics["cpu_memory"]["max_mem_used_mb"]],
        "Memory_Total_MB": [metrics["cpu_memory"]["mem_total_mb"]],
        
        # Throughput Metrics
        "Throughput_Bytes_Per_Sec": [metrics["throughput"]["throughput_bytes_per_sec"]],
        "Throughput_MB_Per_Sec": [metrics["throughput"]["throughput_mb_per_sec"]],
        "Latency_at_Peak_Throughput_ms": [metrics["throughput"]["latency_at_peak_throughput"]],
        
        # Test Configuration
        "Test_Duration_Seconds": [metrics["config"]["duration_s"]],
        "Target_Rate_RPS": [metrics["config"]["rate"]],
        "Audio_Size_Bytes": [metrics["config"]["audio_size_bytes"]],
        "Language_ID": [metrics["config"]["lang_id"]],
    }
    
    df = pd.DataFrame(data)
    
    # If Excel file exists, append to it (for comparing multiple runs/endpoints)
    if excel_path.exists():
        try:
            existing_df = pd.read_excel(excel_path, engine='openpyxl')
            df = pd.concat([existing_df, df], ignore_index=True)
        except Exception as e:
            print(f"[WARNING] Could not read existing Excel file: {e}. Creating new file.")
    
    # Write to Excel
    df.to_excel(excel_path, index=False, engine='openpyxl')
    print(f"\n[SUCCESS] Benchmark results written to: {excel_path}")
    return excel_path

# ------------------------------
# Main runner
# ------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive ASR Benchmark Tool - Generates Excel report with all metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--endpoint", required=True, 
                       help="ASR endpoint URL (e.g. http://127.0.0.1:8000/asr)")
    parser.add_argument("--audio", required=True, 
                       help="Path to WAV file used for requests")
    parser.add_argument("--lang_id", required=True, 
                       help="Language code for ASR (e.g., 'en', 'hi', 'ta', 'te', 'mr')")
    parser.add_argument("--outputdir", required=True, 
                       help="Output directory for results")
    parser.add_argument("--rate", type=float, default=10.0, 
                       help="Target requests per second (default: 10.0)")
    parser.add_argument("--duration", type=int, default=30, 
                       help="Duration in seconds (default: 30)")
    parser.add_argument("--sample_interval", type=float, default=0.5, 
                       help="Sampling interval for GPU/CPU in seconds (default: 0.5)")
    
    args = parser.parse_args()
    
    if not PANDAS_PRESENT:
        print("[ERROR] pandas and openpyxl are required for Excel export.")
        print("Install with: pip install pandas openpyxl")
        return 1
    
    outdir = Path(args.outputdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Get audio file size
    audio_size_bytes = Path(args.audio).stat().st_size
    
    # Prepare sampler threads
    gpu_samples = []
    sys_samples = []
    stop_event = threading.Event()
    
    gpu_thread = None
    if PYNVML_PRESENT:
        gpu_thread = threading.Thread(
            target=gpu_sampler, 
            args=(stop_event, args.sample_interval, gpu_samples), 
            daemon=True
        )
    else:
        print("[WARNING] pynvml not installed — GPU sampling disabled. Install pynvml or nvidia-ml-py3 to enable.")
    
    sys_thread = threading.Thread(
        target=sys_sampler, 
        args=(stop_event, args.sample_interval, sys_samples), 
        daemon=True
    )
    
    # Start samplers
    if gpu_thread:
        gpu_thread.start()
    sys_thread.start()
    
    print(f"\n{'='*60}")
    print(f"ASR Benchmark Tool")
    print(f"{'='*60}")
    print(f"Endpoint: {args.endpoint}")
    print(f"Audio: {args.audio}")
    print(f"Language: {args.lang_id}")
    print(f"Rate: {args.rate} req/s")
    print(f"Duration: {args.duration}s")
    print(f"Output: {outdir}")
    print(f"{'='*60}\n")
    
    # Run load generator (async)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    start_time = time.time()
    try:
        results = loop.run_until_complete(
            run_rate_test(args.endpoint, args.audio, args.rate, args.duration, args.lang_id)
        )
    finally:
        end_time = time.time()
        # stop samplers
        stop_event.set()
        if gpu_thread:
            gpu_thread.join(timeout=2.0)
        sys_thread.join(timeout=2.0)
        loop.close()
    
    # Write detailed CSV files
    req_csv = write_requests_csv(outdir, results)
    sys_csv = write_sys_csv(outdir, sys_samples)
    if PYNVML_PRESENT:
        gpu_csv = write_gpu_csv(outdir, gpu_samples)
    else:
        gpu_csv = None
    
    # Calculate all metrics
    latencies = [r[1] for r in results if r[1] is not None and r[1] > 0]
    successful_requests = len([r for r in results if r[2] == 200])
    failed_requests = len(results) - successful_requests
    
    latency_metrics = compute_latency_percentiles(latencies)
    qps = compute_qps(results, start_time, end_time)
    gpu_metrics = compute_gpu_metrics(gpu_samples)
    cpu_memory_metrics = compute_cpu_memory_metrics(sys_samples)
    throughput_metrics = compute_throughput_latency_relationship(results, audio_size_bytes)
    
    # Compile all metrics
    all_metrics = {
        "latency": latency_metrics,
        "qps": qps,
        "total_requests": len(results),
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "gpu": gpu_metrics,
        "cpu_memory": cpu_memory_metrics,
        "throughput": throughput_metrics,
        "config": {
            "endpoint": args.endpoint,
            "duration_s": args.duration,
            "rate": args.rate,
            "audio_size_bytes": audio_size_bytes,
            "lang_id": args.lang_id
        }
    }
    
    # Write Excel summary
    excel_path = write_excel_summary(outdir, all_metrics)
    
    # Print summary to console
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"Endpoint: {args.endpoint}")
    print(f"\n--- Latency Metrics ---")
    print(f"  p50: {latency_metrics['p50']:.2f} ms")
    print(f"  p95: {latency_metrics['p95']:.2f} ms")
    print(f"  p99: {latency_metrics['p99']:.2f} ms")
    print(f"  Mean: {latency_metrics['mean']:.2f} ms")
    print(f"  Min: {latency_metrics['min']:.2f} ms")
    print(f"  Max: {latency_metrics['max']:.2f} ms")
    
    print(f"\n--- QPS Metrics ---")
    print(f"  QPS: {qps:.2f} req/s")
    print(f"  Total Requests: {len(results)}")
    print(f"  Successful: {successful_requests}")
    print(f"  Failed: {failed_requests}")
    
    print(f"\n--- GPU Metrics ---")
    print(f"  Avg Utilization: {gpu_metrics['avg_util_percent']:.2f}%")
    print(f"  Max Utilization: {gpu_metrics['max_util_percent']:.2f}%")
    print(f"  Avg Memory: {gpu_metrics['avg_mem_mb']:.2f} MB")
    print(f"  Max Memory: {gpu_metrics['max_mem_mb']:.2f} MB")
    
    print(f"\n--- CPU & Memory Metrics ---")
    print(f"  Avg CPU: {cpu_memory_metrics['avg_cpu_percent']:.2f}%")
    print(f"  Max CPU: {cpu_memory_metrics['max_cpu_percent']:.2f}%")
    print(f"  Avg Memory Used: {cpu_memory_metrics['avg_mem_used_mb']:.2f} MB")
    print(f"  Max Memory Used: {cpu_memory_metrics['max_mem_used_mb']:.2f} MB")
    print(f"  Total Memory: {cpu_memory_metrics['mem_total_mb']:.2f} MB")
    
    print(f"\n--- Throughput Metrics ---")
    print(f"  Throughput: {throughput_metrics['throughput_mb_per_sec']:.2f} MB/s")
    print(f"  Latency at Peak: {throughput_metrics['latency_at_peak_throughput']:.2f} ms")
    
    print(f"\n--- Output Files ---")
    print(f"  Excel Report: {excel_path}")
    print(f"  Requests CSV: {req_csv}")
    print(f"  System CSV: {sys_csv}")
    if gpu_csv:
        print(f"  GPU CSV: {gpu_csv}")
    print(f"{'='*60}\n")
    
    return 0

if __name__ == "__main__":
    exit(main())
