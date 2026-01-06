# Docker Quick Start

Get the MLflow ASR service running in Docker in 3 steps!

## Prerequisites

- Docker installed
- Model already logged (run `python log_model.py` first)
- HuggingFace token (if required): `export HF_TOKEN="your_token"`

## Steps

### 1. Build the Image

```bash
./build_docker.sh
```

### 2. Run the Container

```bash
./run_docker.sh
```

### 3. Test It

```bash
./test_curl.sh ta ctc
```

## That's It! 🎉

Your service is running at `http://localhost:5000`

## Useful Commands

```bash
# View logs
docker logs -f mlflow-asr-service

# Stop container
docker stop mlflow-asr-service

# Remove container
docker rm mlflow-asr-service

# Restart
docker restart mlflow-asr-service
```

## Using Docker Compose

```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# View logs
docker-compose logs -f
```

## Troubleshooting

**Container won't start?**
```bash
docker logs mlflow-asr-service
```

**Port already in use?**
```bash
# Use different port
./run_docker.sh latest 8080
```

**GPU not working?**
```bash
# Check GPU access
docker exec mlflow-asr-service python -c "import torch; print(torch.cuda.is_available())"
```

For more details, see [DOCKER.md](DOCKER.md)






