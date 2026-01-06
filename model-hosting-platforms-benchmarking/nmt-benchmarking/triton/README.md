# Triton IndicTrans v2 NMT Model Hosting

This directory contains the complete setup for hosting the AI4Bharat IndicTrans v2 Neural Machine Translation model using NVIDIA Triton Inference Server in a Docker container.

## 📋 Overview

This setup deploys the IndicTrans v2 NMT model as a Triton Inference Server service, providing:
- HTTP and gRPC endpoints for inference
- GPU acceleration support
- Health monitoring and metrics
- Easy deployment via Docker Compose

## 🐳 Docker Image

**Image Name:** `ai4bharat/triton-indictrans-v2:latest`  
**Source:** [Docker Hub](https://hub.docker.com/r/ai4bharat/triton-indictrans-v2)  
**Description:** Pre-configured Triton Inference Server with IndicTrans v2 NMT model

### Pulling the Docker Image

```bash
# Pull the latest image from Docker Hub
docker pull ai4bharat/triton-indictrans-v2:latest

# Verify the image was pulled successfully
docker images | grep triton-indictrans-v2
```

The image includes:
- NVIDIA Triton Inference Server 2.29.0
- Pre-loaded IndicTrans v2 NMT model
- Python backend for model execution
- All necessary dependencies

## 🚀 Hosting in Triton

### Method 1: Using Docker Compose (Recommended)

1. **Navigate to the triton directory:**
   ```bash
   cd /home/ubuntu/nmt-benchmarking/triton
   ```

2. **Start the Triton server:**
   ```bash
   docker-compose up -d
   ```

3. **Verify the container is running:**
   ```bash
   docker ps | grep triton-indictrans-v2
   ```

4. **Check server logs:**
   ```bash
   docker-compose logs -f
   ```

5. **Stop the server:**
   ```bash
   docker-compose down
   ```

### Method 2: Using Docker Directly

```bash
docker run -d \
  --name triton-indictrans-v2 \
  --gpus all \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  --shm-size=1g \
  --restart unless-stopped \
  ai4bharat/triton-indictrans-v2:latest
```

### Method 3: Using the Run Script

```bash
cd /home/ubuntu/nmt-benchmarking/triton
./run.sh
```

## 🔌 Endpoints

Once the container is running, the Triton Inference Server exposes:

| Endpoint | Port | Protocol | Description |
|----------|------|----------|-------------|
| HTTP API | 8000 | HTTP/REST | Main inference endpoint |
| gRPC API | 8001 | gRPC | gRPC inference endpoint |
| Metrics | 8002 | HTTP | Prometheus metrics endpoint |

### Base URLs
- **HTTP**: `http://localhost:8000`
- **gRPC**: `localhost:8001`
- **Metrics**: `http://localhost:8002/metrics`

## ✅ Verification

### Quick Status Check

```bash
cd /home/ubuntu/nmt-benchmarking/triton
./check_triton_status.sh
```

This script checks:
- Container status
- Server health
- Model configuration (GPU/CPU)
- GPU availability
- Port accessibility

### Verify Server Health

```bash
# Check if server is live
curl http://localhost:8000/v2/health/live

# Check if server is ready
curl http://localhost:8000/v2/health/ready
```

### Verify GPU Usage

```bash
# Run GPU verification script
./verify_gpu.sh

# Check model instance configuration
curl http://localhost:8000/v2/models/nmt/config | python3 -m json.tool | grep -A 3 instance_group
```

Expected output should show:
```json
"instance_group": [
    {
        "kind": "KIND_GPU",
        "gpus": [0]
    }
]
```

## 🔍 How Can You Say This is Running in Triton?

There are several definitive ways to verify that the model is actually running on **NVIDIA Triton Inference Server** and not another inference framework:

### 1. Check Triton Server Metadata

The most direct way is to query the Triton server's metadata endpoint, which only Triton provides:

```bash
curl http://localhost:8000/v2 | python3 -m json.tool
```

**Expected Response:**
```json
{
  "name": "triton",
  "version": "2.29.0",
  "extensions": [
    "classification",
    "sequence",
    "model_repository",
    "model_repository(unload_dependents)",
    "schedule_policy",
    "model_configuration",
    "system_shared_memory",
    "cuda_shared_memory",
    "binary_tensor_data",
    "statistics",
    "trace",
    "logging"
  ]
}
```

**Key Indicators:**
- ✅ `"name": "triton"` - Confirms it's Triton server
- ✅ `"version": "2.29.0"` - Shows Triton version
- ✅ Triton-specific extensions like `cuda_shared_memory`, `model_repository`, `statistics`

### 2. Verify Triton-Specific API Endpoints

Triton has specific API endpoints that other inference servers don't have:

```bash
# Triton health endpoints (v2 API)
curl http://localhost:8000/v2/health/live
curl http://localhost:8000/v2/health/ready

# Triton model repository endpoint
curl http://localhost:8000/v2/repository/index

# Triton metrics endpoint (Prometheus format)
curl http://localhost:8002/metrics | grep -i triton
```

**Triton-Specific Endpoints:**
- `/v2/health/live` - Triton health check
- `/v2/health/ready` - Triton readiness check
- `/v2/repository/index` - Model repository index
- `/v2/models/<model>/config` - Model configuration in Triton format
- `/v2/models/<model>/stats` - Model statistics

### 3. Check Container Logs for Triton Signatures

The container logs contain clear Triton server startup messages:

```bash
docker logs triton-indictrans-v2 2>&1 | grep -i triton
```

**Expected Triton Log Messages:**
```
I0102 05:55:18.878816 78 pinned_memory_manager.cc:240] Pinned memory pool is created
I0102 05:55:18.881262 78 cuda_memory_manager.cc:105] CUDA memory pool is created on device 0
I0102 05:55:28.518415 78 server.cc:633] 
+-------+---------+--------+
| Model | Version | Status |
+-------+---------+--------+
| nmt   | 1       | READY  |
+-------+---------+--------+
I0102 05:55:28.564996 78 grpc_server.cc:4819] Started GRPCInferenceService at 0.0.0.0:8001
I0102 05:55:28.565258 78 http_server.cc:3477] Started HTTPService at 0.0.0.0:8000
I0102 05:55:28.620332 78 http_server.cc:184] Started Metrics Service at 0.0.0.0:8002
```

**Key Triton Identifiers in Logs:**
- `pinned_memory_manager.cc` - Triton memory management
- `cuda_memory_manager.cc` - Triton CUDA memory management
- `server.cc` - Triton server initialization
- `grpc_server.cc` - Triton gRPC server
- `http_server.cc` - Triton HTTP server
- Model table format (ASCII table with Model/Version/Status)

### 4. Verify Triton Model Configuration Format

Triton uses a specific configuration format (config.pbtxt or JSON):

```bash
curl http://localhost:8000/v2/models/nmt/config | python3 -m json.tool
```

**Triton-Specific Configuration Elements:**
- `instance_group` with `kind: KIND_GPU` or `KIND_CPU`
- `dynamic_batching` configuration
- `optimization` settings with `priority`, `input_pinned_memory`
- `version_policy` settings
- Triton backend identifiers (`backend: "python"`)

### 5. Check Metrics Endpoint for Triton Metrics

Triton exposes Prometheus metrics with Triton-specific prefixes:

```bash
curl http://localhost:8002/metrics | grep nv_
```

**Expected Triton Metrics:**
```
nv_inference_request_success{model="nmt",version="1"}
nv_inference_exec_count{model="nmt",version="1"}
nv_inference_request_duration_us{model="nmt",version="1"}
nv_gpu_utilization{gpu_uuid="..."}
nv_energy_consumption{gpu_uuid="..."}
```

**Key Indicator:** Metrics prefixed with `nv_` are Triton-specific NVIDIA metrics.

### 6. Verify Triton Inference Request Format

Triton uses a specific inference request format with inputs/outputs structure:

```bash
curl -X POST http://localhost:8000/v2/models/nmt/infer \
  -H 'Content-Type: application/json' \
  -d '{"inputs":[...],"outputs":[...]}'
```

**Triton Request Format Characteristics:**
- Uses `/v2/models/<model>/infer` endpoint
- Request body has `inputs` and optionally `outputs` arrays
- Each input has `name`, `datatype`, `shape`, `data`
- Response includes `model_name`, `model_version`, `outputs`

### 7. Check Docker Image Details

Verify the container is using the Triton image:

```bash
docker inspect triton-indictrans-v2 | grep Image
```

**Expected Output:**
```json
"Image": "ai4bharat/triton-indictrans-v2:latest"
```

You can also check what's inside the container:

```bash
docker exec triton-indictrans-v2 ls /opt/tritonserver/
```

Should show Triton server binaries and directories.

### 8. Quick Verification Script

Run the comprehensive verification script:

```bash
./check_triton_status.sh
```

This script checks multiple Triton-specific indicators and confirms:
- Server name is "triton"
- Triton API endpoints are responding
- Model configuration is in Triton format
- Metrics endpoint exposes Triton metrics

### Summary: How to Know It's Triton

**Definitive Proof:**
1. ✅ Server metadata returns `"name": "triton"` and version number
2. ✅ Container logs show Triton server initialization messages
3. ✅ `/v2/health/ready` and `/v2/health/live` endpoints work (Triton v2 API)
4. ✅ Model config shows Triton-specific fields (`instance_group`, `dynamic_batching`)
5. ✅ Metrics endpoint has `nv_` prefixed metrics
6. ✅ Inference uses `/v2/models/<model>/infer` endpoint format
7. ✅ Docker image is `ai4bharat/triton-indictrans-v2:latest`

**If it were NOT Triton:**
- ❌ Server metadata would show a different name (e.g., "FastAPI", "BentoML", "TorchServe")
- ❌ Wouldn't have `/v2/health/ready` endpoint
- ❌ Wouldn't have Triton-specific configuration format
- ❌ Wouldn't expose `nv_` prefixed metrics
- ❌ Logs would show different server initialization messages

## 🤖 Model Information

### Model Details
- **Model Name:** `nmt`
- **Version:** `1`
- **Platform:** `python`
- **Backend:** Python backend with PyTorch

### Inputs
1. **INPUT_TEXT** (BYTES): Text to translate
2. **INPUT_LANGUAGE_ID** (BYTES): Source language code (e.g., "en", "hi")
3. **OUTPUT_LANGUAGE_ID** (BYTES): Target language code (e.g., "en", "hi")

### Outputs
1. **OUTPUT_TEXT** (BYTES): Translated text

### Supported Language Codes
- `en` - English
- `hi` - Hindi

## 📡 API Usage

### Get Server Metadata

```bash
curl http://localhost:8000/v2 | python3 -m json.tool
```

### Get Model Metadata

```bash
curl http://localhost:8000/v2/models/nmt | python3 -m json.tool
```

### Get Model Configuration

```bash
curl http://localhost:8000/v2/models/nmt/config | python3 -m json.tool
```

### Translation: English to Hindi

```bash
curl -X POST http://localhost:8000/v2/models/nmt/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": [
      {
        "name": "INPUT_TEXT",
        "datatype": "BYTES",
        "shape": [1, 1],
        "data": ["Hello, how are you?"]
      },
      {
        "name": "INPUT_LANGUAGE_ID",
        "datatype": "BYTES",
        "shape": [1, 1],
        "data": ["en"]
      },
      {
        "name": "OUTPUT_LANGUAGE_ID",
        "datatype": "BYTES",
        "shape": [1, 1],
        "data": ["hi"]
      }
    ]
  }' | python3 -m json.tool
```

**Response:**
```json
{
  "model_name": "nmt",
  "model_version": "1",
  "outputs": [
    {
      "name": "OUTPUT_TEXT",
      "datatype": "BYTES",
      "shape": [1, 1],
      "data": ["नमस्कार, आप कैसे हैं?"]
    }
  ]
}
```

### Translation: Hindi to English

```bash
curl -X POST http://localhost:8000/v2/models/nmt/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": [
      {
        "name": "INPUT_TEXT",
        "datatype": "BYTES",
        "shape": [1, 1],
        "data": ["नमस्ते, आप कैसे हैं?"]
      },
      {
        "name": "INPUT_LANGUAGE_ID",
        "datatype": "BYTES",
        "shape": [1, 1],
        "data": ["hi"]
      },
      {
        "name": "OUTPUT_LANGUAGE_ID",
        "datatype": "BYTES",
        "shape": [1, 1],
        "data": ["en"]
      }
    ]
  }' | python3 -m json.tool
```

## 📊 Monitoring

### Real-Time GPU Monitoring

```bash
# Monitor GPU during inference
./monitor_gpu.sh

# Monitor with continuous inference
./monitor_gpu.sh --inference
```

### Check Metrics

```bash
# View all metrics
curl http://localhost:8002/metrics

# Filter GPU metrics
curl http://localhost:8002/metrics | grep gpu

# Filter inference metrics
curl http://localhost:8002/metrics | grep inference
```

### Manual GPU Check

```bash
# Quick GPU status
nvidia-smi

# Continuous monitoring
watch -n 1 nvidia-smi
```

## 📁 Project Structure

```
triton/
├── README.md                      # This file
├── docker-compose.yml             # Docker Compose configuration
├── run.sh                         # Quick start script
├── check_triton_status.sh         # Status verification script
├── verify_gpu.sh                  # GPU verification script
├── monitor_gpu.sh                 # GPU monitoring script
├── working_curl_commands.sh       # Working curl command examples
├── curl_examples.txt              # Quick copy-paste curl commands
├── CURL_EXAMPLES.md               # Detailed curl documentation
└── GPU_VERIFICATION.md            # GPU verification guide
```

## 🔧 Configuration

### Docker Compose Configuration

The `docker-compose.yml` file configures:

- **GPU Access**: All GPUs available to container
- **Ports**: 8000 (HTTP), 8001 (gRPC), 8002 (Metrics)
- **Shared Memory**: 1GB for model loading
- **Restart Policy**: Automatically restart unless stopped
- **Health Check**: Automatic health monitoring

### Environment Variables

No additional environment variables required. The image comes pre-configured.

## 🐛 Troubleshooting

### Container Won't Start

1. **Check GPU availability:**
   ```bash
   nvidia-smi
   docker info | grep -i runtime
   ```

2. **Verify NVIDIA Docker runtime:**
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

3. **Check logs:**
   ```bash
   docker logs triton-indictrans-v2
   ```

### Model Not Found

If you see model errors:
1. Check container logs: `docker logs triton-indictrans-v2`
2. Verify model is loaded: `curl http://localhost:8000/v2/models/nmt`
3. Restart container: `docker-compose restart`

### GPU Not Being Used

1. **Verify GPU configuration:**
   ```bash
   ./verify_gpu.sh
   ```

2. **Check model instance:**
   ```bash
   curl http://localhost:8000/v2/models/nmt/config | python3 -m json.tool | grep kind
   ```
   Should show `"kind": "KIND_GPU"`

3. **Check container GPU access:**
   ```bash
   docker exec triton-indictrans-v2 nvidia-smi
   ```

### Port Already in Use

If ports are already in use:
1. Stop existing containers: `docker-compose down`
2. Or modify ports in `docker-compose.yml`
3. Check what's using the ports: `sudo netstat -tulpn | grep 8000`

## 📊 Benchmarking

A comprehensive benchmarking tool is available to measure NMT model performance with all the same metrics as the ASR benchmarking suite.

### Benchmark Script

The `benchmark_nmt.py` script measures:
- **Latency**: p50, p95, p99 percentiles (ms)
- **QPS**: Queries Per Second
- **GPU Utilization**: Average, max, and min (%)
- **GPU Memory**: Average and peak usage (MB)
- **CPU Usage**: Average and peak (%)
- **Memory Usage**: Average and peak (MB)
- **Throughput**: Bytes/MB per second
- **Success Rate**: Percentage of successful requests

### Basic Usage

```bash
cd /home/ubuntu/nmt-benchmarking/triton

python benchmark_nmt.py \
  --endpoint http://localhost:8000/v2/models/nmt/infer \
  --input_text "Hello, how are you?" \
  --src_lang en \
  --tgt_lang hi \
  --outputdir ./bench_results
```

### Advanced Usage

```bash
python benchmark_nmt.py \
  --endpoint http://localhost:8000/v2/models/nmt/infer \
  --input_text "Hello, how are you? This is a longer text for testing translation performance." \
  --src_lang en \
  --tgt_lang hi \
  --outputdir ./bench_results \
  --rate 20.0 \
  --duration 60 \
  --sample_interval 0.5
```

### Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--endpoint` | Yes | - | Triton NMT endpoint URL |
| `--input_text` | Yes | - | Text to translate |
| `--src_lang` | Yes | - | Source language code (e.g., "en", "hi") |
| `--tgt_lang` | Yes | - | Target language code (e.g., "en", "hi") |
| `--outputdir` | Yes | - | Output directory for results |
| `--rate` | No | 10.0 | Target requests per second |
| `--duration` | No | 30 | Benchmark duration in seconds |
| `--sample_interval` | No | 0.5 | Sampling interval for GPU/CPU (seconds) |

### Output Files

The benchmark generates:
- **`benchmark_results.xlsx`**: Excel file with all metrics in separate columns
- **`requests.csv`**: Detailed request-level data (timestamp, latency, status)
- **`gpu_samples.csv`**: GPU utilization samples over time
- **`sys_samples.csv`**: System resource samples (CPU, memory) over time

### Excel Report Columns

The Excel report includes:
- Timestamp and endpoint information
- **Latency Metrics**: p50, p95, p99, mean, min, max (ms)
- **QPS Metrics**: QPS, total requests, successful/failed requests, success rate
- **GPU Metrics**: Average, max, min utilization (%); average and max memory (MB)
- **CPU & Memory Metrics**: Average/max CPU (%); average/max memory (MB); total memory
- **Throughput Metrics**: Bytes/sec, MB/sec, latency at peak throughput
- **Test Configuration**: Duration, rate, text size, source/target languages

### Example Output

```
============================================================
NMT Benchmark Tool - Triton
============================================================
Endpoint: http://localhost:8000/v2/models/nmt/infer
Input Text: Hello, how are you?
Source Language: en
Target Language: hi
Rate: 10.0 req/s
Duration: 30s
Output: ./bench_results
============================================================

RPS 10.0: 100%|████████████████| 30/30 [00:30<00:00,  1.00s/it]

============================================================
BENCHMARK SUMMARY
============================================================
Endpoint: http://localhost:8000/v2/models/nmt/infer

--- Latency Metrics ---
  p50: 125.34 ms
  p95: 245.67 ms
  p99: 312.45 ms
  Mean: 135.23 ms
  Min: 98.12 ms
  Max: 456.78 ms

--- QPS Metrics ---
  QPS: 10.02 req/s
  Total Requests: 301
  Successful: 301
  Failed: 0
  Success Rate: 100.00%

--- GPU Metrics ---
  Avg Utilization: 45.23%
  Max Utilization: 78.56%
  Min Utilization: 12.34%
  Avg Memory: 5234.56 MB
  Max Memory: 5678.90 MB

--- CPU & Memory Metrics ---
  Avg CPU: 25.45%
  Max CPU: 45.67%
  Avg Memory Used: 8123.45 MB
  Max Memory Used: 9234.56 MB
  Total Memory: 30720.00 MB

--- Throughput Metrics ---
  Throughput: 0.0023 MB/s
  Latency at Peak: 125.34 ms

--- Output Files ---
  Excel Report: ./bench_results/benchmark_results.xlsx
  Requests CSV: ./bench_results/requests.csv
  System CSV: ./bench_results/sys_samples.csv
  GPU CSV: ./bench_results/gpu_samples.csv
============================================================
```

### Prerequisites

Install required dependencies:

```bash
pip install pandas openpyxl tqdm numpy psutil aiohttp

# For GPU monitoring (optional but recommended)
pip install pynvml  # or nvidia-ml-py3
```

### Notes

- The script supports concurrent requests with rate limiting
- GPU monitoring requires `pynvml` or `nvidia-ml-py3`
- Multiple benchmark runs append results to the same Excel file for comparison
- The script handles timeouts and errors gracefully
- Progress is shown with a progress bar during execution

## 📚 Additional Resources

- **Docker Hub**: https://hub.docker.com/r/ai4bharat/triton-indictrans-v2
- **AI4Bharat**: https://ai4bharat.iitm.ac.in/
- **Triton Documentation**: https://docs.nvidia.com/deeplearning/triton-inference-server/
- **Detailed Curl Examples**: See `CURL_EXAMPLES.md`
- **GPU Verification Guide**: See `GPU_VERIFICATION.md`

## 📝 Quick Reference

### Start Server
```bash
docker-compose up -d
```

### Stop Server
```bash
docker-compose down
```

### Check Status
```bash
./check_triton_status.sh
```

### Test Translation
```bash
curl -X POST http://localhost:8000/v2/models/nmt/infer \
  -H 'Content-Type: application/json' \
  -d '{"inputs":[{"name":"INPUT_TEXT","datatype":"BYTES","shape":[1,1],"data":["Hello"]},{"name":"INPUT_LANGUAGE_ID","datatype":"BYTES","shape":[1,1],"data":["en"]},{"name":"OUTPUT_LANGUAGE_ID","datatype":"BYTES","shape":[1,1],"data":["hi"]}]}' | python3 -m json.tool
```

### Monitor GPU
```bash
./monitor_gpu.sh
```

### Run Benchmark
```bash
python benchmark_nmt.py \
  --endpoint http://localhost:8000/v2/models/nmt/infer \
  --input_text "Hello, how are you?" \
  --src_lang en \
  --tgt_lang hi \
  --outputdir ./bench_results
```

## ✅ Verification Checklist

- [ ] Docker image pulled successfully
- [ ] Container is running
- [ ] Server health endpoints respond (200 OK)
- [ ] Model `nmt` is available
- [ ] Model instance is configured for GPU (KIND_GPU)
- [ ] GPU is accessible from container
- [ ] Translation requests work (en ↔ hi)
- [ ] GPU utilization increases during inference

---

**Last Updated**: January 2025  
**Triton Version**: 2.29.0  
**Model**: IndicTrans v2 NMT
