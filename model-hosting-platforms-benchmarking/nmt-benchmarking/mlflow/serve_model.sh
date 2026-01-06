#!/bin/bash

# Serve MLflow IndicTrans2 NMT Model

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN environment variable is not set."
    echo "Please set it with: export HF_TOKEN=your_token_here"
    exit 1
fi

cd /home/ubuntu/nmt-benchmarking/mlflow
source mlflow/bin/activate

# Set MLflow tracking URI
export MLFLOW_TRACKING_URI="file:///home/ubuntu/nmt-benchmarking/mlflow/mlruns"

# Get the latest model version (or specify a specific run_id)
# For now, we'll use the registered model
MODEL_NAME="IndicTransNMT"
PORT=${1:-5000}

echo "Starting MLflow model server..."
echo "Model: $MODEL_NAME"
echo "Port: $PORT"
echo ""

# Try to serve registered model, fallback to specific run_id if needed
export MLFLOW_TRACKING_URI="file:///home/ubuntu/nmt-benchmarking/mlflow/mlruns"

# Get the latest version of the registered model
LATEST_VERSION=$(ls -t mlruns/0/ 2>/dev/null | head -1)

if [ -n "$LATEST_VERSION" ]; then
    echo "Using run ID: $LATEST_VERSION"
    mlflow models serve -m "runs:/$LATEST_VERSION/indictrans_nmt_model" --port $PORT --host 0.0.0.0 --no-conda
elif mlflow models serve -m "models:/$MODEL_NAME/Staging" --port $PORT --host 0.0.0.0 --no-conda 2>/dev/null; then
    echo "Model served successfully on port $PORT from Staging"
elif mlflow models serve -m "models:/$MODEL_NAME/Production" --port $PORT --host 0.0.0.0 --no-conda 2>/dev/null; then
    echo "Model served successfully on port $PORT from Production"
else
    echo "ERROR: No model found. Please register the model first with:"
    echo "  python3 register_model.py"
    exit 1
fi

