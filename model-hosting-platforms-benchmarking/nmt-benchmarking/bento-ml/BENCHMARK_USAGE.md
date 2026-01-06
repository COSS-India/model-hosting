# BentoML NMT Benchmark Usage Guide

## Prerequisites

Install required dependencies:

```bash
cd /home/ubuntu/nmt-benchmarking/bento-ml
source bento/bin/activate
pip install pandas openpyxl pynvml aiohttp tqdm numpy psutil
```

## Basic Usage

### English to Hindi Translation

```bash
python3 benchmark_nmt.py \
  --endpoint http://localhost:3000/translate \
  --input_text "Hello, how are you?" \
  --src_lang eng_Latn \
  --tgt_lang hin_Deva \
  --outputdir /home/ubuntu/nmt-benchmarking/bento-ml/bench_results \
  --rate 5.0 \
  --duration 30
```

### Hindi to English Translation

```bash
python3 benchmark_nmt.py \
  --endpoint http://localhost:3000/translate \
  --input_text "नमस्ते, आप कैसे हैं?" \
  --src_lang hin_Deva \
  --tgt_lang eng_Latn \
  --outputdir /home/ubuntu/nmt-benchmarking/bento-ml/bench_results \
  --rate 5.0 \
  --duration 30
```

### Indic to Indic Translation (Hindi to Marathi)

```bash
python3 benchmark_nmt.py \
  --endpoint http://localhost:3000/translate \
  --input_text "नमस्ते, आप कैसे हैं?" \
  --src_lang hin_Deva \
  --tgt_lang mar_Deva \
  --outputdir /home/ubuntu/nmt-benchmarking/bento-ml/bench_results \
  --rate 5.0 \
  --duration 30
```

## Advanced Usage

### High Load Test (10 RPS for 60 seconds)

```bash
python3 benchmark_nmt.py \
  --endpoint http://localhost:3000/translate \
  --input_text "Hello, how are you?" \
  --src_lang eng_Latn \
  --tgt_lang hin_Deva \
  --outputdir /home/ubuntu/nmt-benchmarking/bento-ml/bench_results \
  --rate 10.0 \
  --duration 60 \
  --sample_interval 0.5
```

### Quick Test (2 RPS for 10 seconds)

```bash
python3 benchmark_nmt.py \
  --endpoint http://localhost:3000/translate \
  --input_text "Hello" \
  --src_lang eng_Latn \
  --tgt_lang hin_Deva \
  --outputdir /home/ubuntu/nmt-benchmarking/bento-ml/bench_results \
  --rate 2.0 \
  --duration 10
```

## Output Files

The benchmark generates the following files in the output directory:

1. **benchmark_results.xlsx** - Excel file with all metrics in separate columns
2. **requests.csv** - Detailed request data (timestamp, latency, status)
3. **gpu_samples.csv** - GPU utilization and memory samples
4. **sys_samples.csv** - CPU and system memory samples

## Metrics Collected

### Latency Metrics
- p50, p95, p99 percentiles
- Mean, Min, Max latency

### QPS Metrics
- Queries Per Second (QPS)
- Total requests, successful/failed requests
- Success rate percentage

### GPU Metrics
- Average, Max, Min GPU utilization percentage
- Average and Max GPU memory usage (MB)

### CPU & Memory Metrics
- Average and Max CPU utilization percentage
- Average and Max system memory usage (MB)
- Total system memory (MB)

### Throughput Metrics
- Throughput in bytes/second and MB/second
- Latency at peak throughput

## Parameters

- `--endpoint`: BentoML service endpoint URL (required)
- `--input_text`: Text to translate (required)
- `--src_lang`: Source language code (required, e.g., `eng_Latn`, `hin_Deva`)
- `--tgt_lang`: Target language code (required, e.g., `eng_Latn`, `hin_Deva`)
- `--outputdir`: Output directory for results (required)
- `--rate`: Target requests per second (default: 10.0)
- `--duration`: Test duration in seconds (default: 30)
- `--sample_interval`: Sampling interval for GPU/CPU in seconds (default: 0.5)

## Example Output

The script will display a progress bar during execution and print a summary at the end:

```
============================================================
NMT Benchmark Tool - BentoML
============================================================
Endpoint: http://localhost:3000/translate
Input Text: Hello, how are you?
Source Language: eng_Latn
Target Language: hin_Deva
Rate: 5.0 req/s
Duration: 30s
Output: /home/ubuntu/nmt-benchmarking/bento-ml/bench_results
============================================================

RPS 5.0: 100%|████████████████| 30/30 [00:30<00:00,  1.00s/it]

[SUCCESS] Benchmark results written to: .../benchmark_results.xlsx

============================================================
BENCHMARK SUMMARY
============================================================
Endpoint: http://localhost:3000/translate

--- Latency Metrics ---
  p50: 1250.45 ms
  p95: 1800.23 ms
  p99: 2100.56 ms
  Mean: 1320.78 ms
  Min: 980.12 ms
  Max: 2200.45 ms

--- QPS Metrics ---
  QPS: 4.98 req/s
  Total Requests: 149
  Successful: 149
  Failed: 0
  Success Rate: 100.00%

...
```

