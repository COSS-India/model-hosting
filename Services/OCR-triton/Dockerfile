# Use NVIDIA Triton Inference Server as base image
# Using 24.08 which supports newer PyTorch versions required by surya-ocr
FROM nvcr.io/nvidia/tritonserver:24.08-py3

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Install surya-ocr which will bring in its required dependencies including PyTorch
RUN pip install --no-cache-dir \
    surya-ocr==0.17.0

# Copy model repository structure
COPY model_repository /models

# Note: Surya models will be downloaded on first use
# They will be cached in /root/.cache/huggingface/hub/

# Expose Triton ports
# 8000: HTTP
# 8001: gRPC
# 8002: Metrics
EXPOSE 8000 8001 8002

# Set environment variables
ENV MODEL_REPOSITORY=/models
ENV TORCH_DEVICE=cuda
# Reduced batch sizes for testing (saves VRAM)
ENV RECOGNITION_BATCH_SIZE=64
ENV DETECTOR_BATCH_SIZE=8

# Start Triton server
CMD ["tritonserver", "--model-repository=/models", "--log-verbose=1"]

