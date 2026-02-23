# GPT-OSS LLM Hosting — Setup & Architecture Guide

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Infrastructure Details](#2-infrastructure-details)
3. [Component Stack](#3-component-stack)
4. [Step-by-Step Setup](#4-step-by-step-setup)
5. [Service Configuration](#5-service-configuration)
6. [API Reference](#6-api-reference)
7. [How a Request Flows](#7-how-a-request-flows)
8. [Monitoring & Management](#8-monitoring--management)
9. [Security Considerations](#9-security-considerations)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        AWS EC2 Instance                        │
│                     (ap-south-1 / Mumbai)                      │
│                                                                │
│  ┌──────────────┐       ┌──────────────┐       ┌────────────┐  │
│  │  Your        │ HTTP  │  FastAPI      │ HTTP  │  Ollama    │  │
│  │  Backend     │──────▶│  Gateway      │──────▶│  Server    │  │
│  │  System      │ :8000 │  (app.py)     │:11434 │            │  │
│  └──────────────┘       └──────────────┘       └─────┬──────┘  │
│                                                      │         │
│                                                      ▼         │
│                                                ┌───────────┐   │
│                                                │ GPT-OSS   │   │
│                                                │ 20B Model │   │
│                                                │ (MXFP4)   │   │
│                                                └─────┬─────┘   │
│                                                      │         │
│                                                      ▼         │
│                                                ┌───────────┐   │
│                                                │ NVIDIA    │   │
│                                                │ Tesla T4  │   │
│                                                │ 15GB VRAM │   │
│                                                └───────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### How the LLM is hosted

- **GPT-OSS** is an open-weight model released by OpenAI under the Apache 2.0 license.
- The model runs **100% offline** on the server — no data leaves the machine.
- **Ollama** serves as the local LLM runtime. It loads the model into GPU memory and exposes a local HTTP API on port `11434`.
- **FastAPI** acts as the API gateway. It receives requests from external backends on port `8000`, constructs optimized prompts (with context, tone, etc.), forwards them to Ollama, and returns structured JSON responses.
- The model runs on the **NVIDIA Tesla T4 GPU** for hardware-accelerated inference.

---

## 2. Infrastructure Details

| Component              | Details                                    |
|------------------------|--------------------------------------------|
| **Cloud Provider**     | AWS (ap-south-1 / Mumbai)                  |
| **Instance Type**      | EC2 with NVIDIA Tesla T4 GPU               |
| **Instance ID**        | `i-0a1251ef5fab6ebec`                      |
| **Public IP**          | `13.201.75.118`                            |
| **OS**                 | Ubuntu 22.04.5 LTS                         |
| **Kernel**             | 6.8.0-1040-aws                             |
| **GPU**                | NVIDIA Tesla T4 — 15,360 MiB (15 GB) VRAM |
| **GPU Driver**         | NVIDIA 550.163.01                          |
| **CUDA Version**       | 13.1                                       |
| **System RAM**         | 62 GB                                      |
| **Disk**               | 194 GB (SSD)                               |
| **Security Group**     | `launch-wizard-21` (port 8000 open)        |

---

## 3. Component Stack

```
Layer 4 — Your Backend System (external consumer)
    │
    ▼
Layer 3 — FastAPI API Gateway       (port 8000, systemd: gpt-oss-api)
    │       ├── /api/translate
    │       ├── /api/summarize
    │       ├── /api/query
    │       └── /health
    ▼
Layer 2 — Ollama LLM Runtime        (port 11434, systemd: ollama)
    │       └── Loads & serves GPT-OSS model
    ▼
Layer 1 — NVIDIA GPU + CUDA         (Tesla T4, driver 550.163.01)
            └── Hardware acceleration for inference
```

| Layer | Software        | Version   | Role                              |
|-------|-----------------|-----------|-----------------------------------|
| 3     | FastAPI + Uvicorn | 0.115.6 / 0.34.0 | API gateway, prompt engineering |
| 2     | Ollama          | 0.15.6    | LLM runtime & model management   |
| 1     | NVIDIA Driver   | 550.163.01| GPU kernel driver                 |
| 1     | CUDA Toolkit    | 13.1      | GPU compute libraries             |

### LLM Model Details

| Property              | Value                         |
|-----------------------|-------------------------------|
| **Model Name**        | `gpt-oss:20b`                 |
| **Provider**          | OpenAI (open-weight release)  |
| **Parameters**        | 20.9 Billion                  |
| **Active Parameters** | 3.6 Billion (Mixture of Experts) |
| **Quantization**      | MXFP4                         |
| **Model Size on Disk**| ~13 GB                         |
| **GPU Memory Usage**  | ~12.5 GB                       |
| **Context Window**    | 131,072 tokens (128K)          |
| **License**           | Apache 2.0                     |
| **Capabilities**      | Completion, Tools, Thinking    |

---

## 4. Step-by-Step Setup

### Step 1: Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

This installs the Ollama binary to `/usr/local/bin/ollama` and creates a systemd service.

### Step 2: Install NVIDIA GPU Drivers

```bash
# Install kernel headers
sudo apt-get install -y linux-headers-$(uname -r)

# Install GCC 12 (required for kernel 6.8+)
sudo apt-get install -y gcc-12 g++-12
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100

# Install CUDA keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install NVIDIA driver
sudo apt-get install -y nvidia-driver-550

# Build the kernel module
sudo dkms install nvidia/550.163.01 -k $(uname -r)

# Load the driver
sudo modprobe nvidia

# Verify
nvidia-smi
```

### Step 3: Install CUDA Toolkit

```bash
sudo apt-get install -y cuda-toolkit-13-1

# Set environment variables
echo 'export PATH=/usr/local/cuda-13.1/bin:$PATH' | sudo tee -a /etc/profile.d/cuda.sh
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:$LD_LIBRARY_PATH' | sudo tee -a /etc/profile.d/cuda.sh
```

### Step 4: Configure Ollama for GPU

```bash
# Create override to pass CUDA environment to Ollama
sudo mkdir -p /etc/systemd/system/ollama.service.d

cat << 'EOF' | sudo tee /etc/systemd/system/ollama.service.d/cuda.conf
[Service]
Environment="PATH=/usr/local/cuda-13.1/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:/usr/lib/x86_64-linux-gnu"
Environment="CUDA_HOME=/usr/local/cuda-13.1"
Environment="CUDA_PATH=/usr/local/cuda-13.1"
EOF

sudo systemctl daemon-reload
sudo systemctl restart ollama
```

### Step 5: Pull the GPT-OSS Model

```bash
ollama pull gpt-oss:20b
```

This downloads the 13 GB model. Verify with:

```bash
ollama list
```

### Step 6: Set Up the FastAPI API Gateway

```bash
# Install Python dependencies
sudo apt-get install -y python3-pip
mkdir -p /home/ubuntu/gpt-oss-api
cd /home/ubuntu/gpt-oss-api

# Create requirements.txt
cat << 'EOF' > requirements.txt
fastapi==0.115.6
uvicorn[standard]==0.34.0
httpx==0.28.1
pydantic==2.10.4
EOF

pip3 install -r requirements.txt
```

Place the `app.py` file in `/home/ubuntu/gpt-oss-api/app.py` (see source code in repository).

### Step 7: Create systemd Service for the API

```bash
cat << 'EOF' | sudo tee /etc/systemd/system/gpt-oss-api.service
[Unit]
Description=GPT-OSS NLP API Gateway
After=network.target ollama.service
Wants=ollama.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/gpt-oss-api
Environment="PATH=/home/ubuntu/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=/home/ubuntu/.local/bin/uvicorn app:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable gpt-oss-api
sudo systemctl start gpt-oss-api
```

### Step 8: Open Port in AWS Security Group

In the AWS Console → EC2 → Security Groups → `launch-wizard-21`:

| Type       | Protocol | Port  | Source      |
|------------|----------|-------|-------------|
| Custom TCP | TCP      | 8000  | 0.0.0.0/0   |

---

## 5. Service Configuration

### Systemd Services

| Service          | Config File                                      | Starts At Boot | Port  |
|------------------|--------------------------------------------------|----------------|-------|
| `ollama`         | `/etc/systemd/system/ollama.service`             | Yes            | 11434 |
| `gpt-oss-api`    | `/etc/systemd/system/gpt-oss-api.service`        | Yes            | 8000  |

### File Locations

| File                                                  | Purpose                          |
|-------------------------------------------------------|----------------------------------|
| `/home/ubuntu/gpt-oss-api/app.py`                    | FastAPI application code         |
| `/home/ubuntu/gpt-oss-api/requirements.txt`          | Python dependencies              |
| `/etc/systemd/system/gpt-oss-api.service`            | API systemd service              |
| `/etc/systemd/system/ollama.service`                  | Ollama systemd service           |
| `/etc/systemd/system/ollama.service.d/cuda.conf`     | Ollama CUDA environment override |
| `/etc/profile.d/cuda.sh`                              | CUDA PATH for all users          |
| `/etc/modules-load.d/nvidia.conf`                     | Auto-load NVIDIA module at boot  |

### Model Storage

Ollama stores downloaded models at: `~/.ollama/models/` (for the ollama user)

---

## 6. API Reference

**Base URL:** `http://13.201.75.118:8000`
**Swagger UI:** `http://13.201.75.118:8000/docs`

### GET /health

Check service health.

```bash
curl http://13.201.75.118:8000/health
```

**Response:**
```json
{
    "status": "healthy",
    "ollama": "connected",
    "available_models": ["gpt-oss:20b"],
    "active_model": "gpt-oss:20b"
}
```

---

### POST /api/translate

Translate text between languages with optional context.

| Field             | Type   | Required | Description                                    |
|-------------------|--------|----------|------------------------------------------------|
| `text`            | string | Yes      | Text to translate                              |
| `source_language` | string | Yes      | Source language name                            |
| `target_language` | string | Yes      | Target language name                           |
| `tone`            | string | No       | `formal`, `informal`, `technical`              |
| `context`         | string | No       | Situational context (gender, audience, domain) |

**Context examples:**
- `"Speaker is a female"` — uses feminine verb forms (critical for Hindi, Arabic, etc.)
- `"Speaker is a male"` — uses masculine verb forms
- `"Sentence is addressed to an aged person"` — uses respectful honorifics
- `"Keep the language simple and rural"` — avoids complex vocabulary
- `"Sentence is in legal context"` — uses legal terminology

```bash
curl -X POST http://13.201.75.118:8000/api/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I am going to the market today",
    "source_language": "English",
    "target_language": "Hindi",
    "context": "Speaker is a female"
  }'
```

---

### POST /api/summarize

Summarize text with optional style and context.

| Field           | Type    | Required | Description                                         |
|-----------------|---------|----------|-----------------------------------------------------|
| `text`          | string  | Yes      | Text to summarize                                   |
| `max_sentences` | integer | No       | Maximum sentences in summary                        |
| `style`         | string  | No       | `brief`, `detailed`, `bullet_points`, `executive`   |
| `context`       | string  | No       | Audience or focus context                           |

```bash
curl -X POST http://13.201.75.118:8000/api/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Long text to summarize...",
    "max_sentences": 3,
    "style": "bullet_points",
    "context": "Target audience is non-technical managers"
  }'
```

---

### POST /api/query

General-purpose natural language query.

| Field           | Type   | Required | Description                      |
|-----------------|--------|----------|----------------------------------|
| `query`         | string | Yes      | Natural language query           |
| `system_prompt` | string | No       | System instruction for the model |
| `context`       | string | No       | Additional context               |

```bash
curl -X POST http://13.201.75.118:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain microservices architecture",
    "system_prompt": "You are a senior software architect.",
    "context": "Audience is junior developers"
  }'
```

---

### Unified Response Format

All endpoints return:

```json
{
    "success": true,
    "result": "...generated text...",
    "model": "gpt-oss:20b",
    "processing_time_seconds": 2.83
}
```

---

## 7. How a Request Flows

```
1. Your backend sends POST to http://13.201.75.118:8000/api/translate
       │
       ▼
2. FastAPI gateway receives the request
       │
       ▼
3. app.py constructs an optimized prompt:
       - Injects system prompt (translator persona)
       - Adds tone instruction (if provided)
       - Adds context instruction (gender, domain, audience)
       - Appends the source text
       │
       ▼
4. FastAPI sends the prompt to Ollama at http://127.0.0.1:11434/api/generate
       │
       ▼
5. Ollama loads GPT-OSS 20B into Tesla T4 GPU memory (~12.5 GB)
       │
       ▼
6. GPU performs inference (token generation)
       │
       ▼
7. Ollama returns the generated text to FastAPI
       │
       ▼
8. FastAPI wraps the result in a structured JSON response
       │
       ▼
9. Response sent back to your backend
```

**Typical response times:** 2–15 seconds depending on input/output length.

---

## 8. Monitoring & Management

### Check service status

```bash
sudo systemctl status ollama
sudo systemctl status gpt-oss-api
```

### View logs

```bash
# API gateway logs
sudo journalctl -u gpt-oss-api -f

# Ollama logs
sudo journalctl -u ollama -f
```

### Monitor GPU usage

```bash
# One-time check
nvidia-smi

# Live monitoring (updates every 1 second)
watch -n 1 nvidia-smi
```

### Restart services

```bash
sudo systemctl restart ollama
sudo systemctl restart gpt-oss-api
```

### Check model info

```bash
ollama list
ollama show gpt-oss:20b
```

---

## 9. Security Considerations

| Aspect                | Current State                          | Recommendation for Production         |
|-----------------------|----------------------------------------|---------------------------------------|
| **Data Privacy**      | All data stays on the server (offline) | No change needed                      |
| **Network Access**    | Port 8000 open to 0.0.0.0/0           | Restrict to your backend IP only      |
| **Authentication**    | None (open API)                        | Add API key / JWT authentication      |
| **HTTPS**             | HTTP only                              | Add TLS via Nginx reverse proxy       |
| **Ollama Port**       | Localhost only (127.0.0.1:11434)       | No change needed (not exposed)        |
| **Rate Limiting**     | None                                   | Add rate limiting in FastAPI/Nginx    |
| **Model License**     | Apache 2.0 — free for commercial use   | No restrictions                       |

---

*Last updated: February 2026*
*Server: 13.201.75.118 | Region: ap-south-1 (Mumbai)*
