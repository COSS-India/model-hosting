# vLLM + Gemma User Guide (Tesla T4, Ubuntu)

This guide documents the exact path used to get `google/gemma-2-2b-it` running with vLLM on a Tesla T4.

## 1) Create virtual environment

```bash
python3 -m venv /home/ubuntu/vllm-env
source /home/ubuntu/vllm-env/bin/activate
python -m pip install --upgrade pip
```

## 2) Install Hugging Face auth (gated model)

Gemma 2 is gated on Hugging Face. Make sure your account has access.

```bash
export HF_TOKEN="hf_xxx_your_token"
```

Optional persistent setup:

```bash
echo 'export HF_TOKEN="hf_xxx_your_token"' >> ~/.bashrc
source ~/.bashrc
```

Optional auth verification:

```bash
python -c "from huggingface_hub import whoami; print(whoami())"
```

## 3) Install/upgrade NVIDIA driver

On this machine, old drivers caused CUDA runtime failures.

Check recommended drivers:

```bash
sudo apt update
sudo apt install -y ubuntu-drivers-common
ubuntu-drivers devices
```

Install recommended driver (example from this setup):

```bash
sudo apt install -y nvidia-driver-595-open
sudo reboot
```

Verify after reboot:

```bash
nvidia-smi
```

## 4) Install stable vLLM stack

Nightly cu129 builds caused multiple runtime/kernel issues on Tesla T4.  
This stable stack is what worked best in this runbook.

```bash
source /home/ubuntu/vllm-env/bin/activate
pip uninstall -y vllm torch torchaudio torchvision flashinfer-python flashinfer-cubin triton
pip install --upgrade pip
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.6.6.post1
pip install -U "transformers==4.46.3" "tokenizers==0.20.3" "sentencepiece>=0.1.99"
```

## 5) Start Gemma server

Important for Tesla T4:
- Do not force `VLLM_ATTENTION_BACKEND=XFORMERS`.
- Use `--dtype=half` (T4 does not support bf16).
- Keep conservative memory/length settings.

```bash
source /home/ubuntu/vllm-env/bin/activate
unset VLLM_ATTENTION_BACKEND

python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-2-2b-it \
  --host 0.0.0.0 \
  --port 8001 \
  --dtype=half \
  --gpu-memory-utilization 0.70 \
  --max-model-len 1024 \
  --enforce-eager
```

## 6) Test that server is running

```bash
curl http://127.0.0.1:8001/v1/models
```

## 7) Test chat completion

Use only `user` role for this model template in this setup (no `system` role).

```bash
curl http://127.0.0.1:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-2-2b-it",
    "messages": [
      {"role": "user", "content": "Reply exactly with GEMMA_OK"}
    ],
    "temperature": 0,
    "max_tokens": 16
  }'
```

Alternative (no chat template):

```bash
curl http://127.0.0.1:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-2-2b-it",
    "prompt": "Reply exactly with GEMMA_OK",
    "temperature": 0,
    "max_tokens": 16
  }'
```

## 8) Docker path (background service)

If you prefer running in Docker, this is the known-good approach from this setup.

Prerequisites:
- Host GPU must work: `nvidia-smi`
- Docker GPU runtime must be configured (NVIDIA Container Toolkit + CDI)

Install/configure NVIDIA Container Toolkit:

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
&& curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
| sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
| sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
| sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null \
&& sudo apt update \
&& sudo apt install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

Optional non-root Docker usage (run as two separate commands):

```bash
sudo usermod -aG docker $USER
newgrp docker
```

Sanity-check GPU in Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Run vLLM in background:

```bash
docker rm -f gemma-vllm 2>/dev/null || true

docker run --gpus all -d \
  --name gemma-vllm \
  --restart unless-stopped \
  -p 8001:8000 \
  -e HF_TOKEN=hf_xxx_your_token \
  -v /home/ubuntu/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model google/gemma-2-2b-it \
  --dtype=float \
  --gpu-memory-utilization 0.95 \
  --max-model-len 256 \
  --max-num-seqs 1 \
  --enforce-eager
```

Useful Docker checks:

```bash
docker logs -f gemma-vllm
curl http://127.0.0.1:8001/v1/models
```

## Troubleshooting (from actual run history)

- `401 Unauthorized` / `GatedRepoError`:
  - Ensure HF access is granted and `HF_TOKEN` is set.
- `Address already in use`:
  - Use different port or free existing process.
- `failed to discover GPU vendor from CDI: no known GPU vendor found`:
  - Install/configure NVIDIA Container Toolkit, then generate CDI spec:
    `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker && sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`
- `cudaErrorInsufficientDriver` / old driver:
  - Upgrade NVIDIA driver and reboot.
- `The model type 'gemma2' does not support float16` (Docker `vllm/vllm-openai:latest` / vLLM 0.20.x):
  - Use `--dtype=float` for Gemma 2 on Tesla T4.
- `Bfloat16 ... compute capability >= 8.0` (older local vLLM path):
  - Use `--dtype=half` on Tesla T4 (compute capability 7.5).
- `System role not supported`:
  - Remove `system` message from chat payload.
- `No available memory for the cache blocks` / negative KV cache memory:
  - For Docker + float32 on T4, increase `--gpu-memory-utilization` (for example `0.95`) and reduce `--max-model-len` (for example `256` or `128`).
