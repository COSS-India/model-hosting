# MLflow ASR Service Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY mlflow_asr.py .
COPY log_model.py .
COPY server.py .

# Copy MLflow model artifacts
COPY mlruns/ ./mlruns/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=file:///app/mlruns

# Expose port
EXPOSE 5000

# Default model path (can be overridden)
ENV MODEL_PATH="mlruns/0/models/m-8f33614a5aeb46f6a4f4c8b0c64b9cf7/artifacts"
ENV PORT=5000
ENV HOST=0.0.0.0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run custom FastAPI server with /asr endpoint
CMD ["python", "server.py"]

