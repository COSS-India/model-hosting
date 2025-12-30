# GPU Setup for Docker

## Error: "could not select device driver with capabilities: [[gpu]]"

This error occurs when NVIDIA Container Toolkit is not installed or configured.

## Quick Fix

### Option 1: Install NVIDIA Container Toolkit (Recommended)

Run the installation script:
```bash
cd /home/ubuntu/Benchmarking/fastapi
chmod +x install_nvidia_docker.sh
sudo ./install_nvidia_docker.sh
```

Or install manually:
```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker
sudo systemctl restart docker
```

### Option 2: Run Without GPU (Temporary Workaround)

If you need to test without GPU first:

1. **Use CPU version:**
   ```bash
   docker build -f Dockerfile.cpu -t asr-server:cpu .
   docker run -d --name asr-server -p 8000:8000 \
     -e HF_TOKEN=your_token_here \
     -e DEVICE=cpu \
     asr-server:cpu
   ```

2. **Or modify docker-compose.yml:**
   Remove the `deploy.resources.reservations.devices` section and set `DEVICE=cpu`

### Option 3: Use Docker with nvidia-docker2 (Legacy)

If you're using older Docker setup:
```bash
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

Then use:
```bash
docker run --runtime=nvidia ...
```

## Verify Installation

After installing, verify GPU access:
```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Should show GPU information
```

## Troubleshooting

1. **Check Docker runtime:**
   ```bash
   docker info | grep -i runtime
   # Should show: nvidia, nvidia-container-runtime, or nvidia-container-runtime-experimental
   ```

2. **Check NVIDIA driver:**
   ```bash
   nvidia-smi
   # Should show GPU information
   ```

3. **Restart Docker if needed:**
   ```bash
   sudo systemctl restart docker
   ```

4. **Check container logs:**
   ```bash
   docker logs asr-server
   ```

## After Installation

Once NVIDIA Container Toolkit is installed, you can run:

```bash
# With docker run
docker run -d --name asr-server --gpus all -p 8000:8000 \
  -e HF_TOKEN=your_token_here \
  -e DEVICE=cuda \
  asr-server:latest

# Or with docker-compose
docker-compose up -d
```



