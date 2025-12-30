# Quick Start Guide

Get the MLflow ASR service up and running in 5 minutes!

## Prerequisites

- Python 3.10+
- Virtual environment activated
- HuggingFace token (if required)

## Steps

### 1. Activate Environment

```bash
cd /home/ubuntu/Benchmarking/Frameworks/MLflow
source mlflow/bin/activate
```

### 2. Install Dependencies (if not already done)

```bash
pip install -r requirements.txt
```

### 3. Log the Model

```bash
python log_model.py
```

**Note the model ID** from the output or:
```bash
ls mlruns/0/models/
```

### 4. Start Server

```bash
python server.py
```

### 5. Test It

```bash
./test_curl.sh ta ctc
```

## That's It! 🎉

Your server is running at `http://localhost:5000`

- **API Docs**: http://localhost:5000/docs
- **Health Check**: `curl http://localhost:5000/health`

## Next Steps

- Read [README.md](README.md) for detailed documentation
- Check [USAGE.md](USAGE.md) for API usage examples
- See [SETUP.md](SETUP.md) for troubleshooting

## Common Commands

```bash
# Check if server is running
ps aux | grep mlflow

# Stop server
pkill -f "mlflow models serve"

# View logs (if running in background)
tail -f mlflow_server.log
```

---

**Need Help?** Check the [Troubleshooting](README.md#troubleshooting) section in README.md

