# Use NVIDIA Triton Inference Server as base image
FROM nvcr.io/nvidia/tritonserver:24.01-py3

# Set working directory
WORKDIR /workspace

# Install system dependencies for audio backends
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Use compatible versions: speechbrain 1.0.3 works with huggingface_hub < 0.20.0
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchaudio==2.1.0 \
    speechbrain==1.0.3 \
    "huggingface_hub<0.20.0" \
    protobuf==3.20.3 \
    requests \
    soundfile

# Copy model repository
COPY model_repository /models

# Pre-download the VoxLingua107 model to cache it in the image
RUN python3 -c "from speechbrain.inference.classifiers import EncoderClassifier; \
    print('Downloading VoxLingua107 ECAPA-TDNN model...'); \
    model = EncoderClassifier.from_hparams(source='speechbrain/lang-id-voxlingua107-ecapa', savedir='tmp_ald_model'); \
    print('Model downloaded successfully')" || echo "Model download skipped (will download at runtime)"

# Expose Triton ports
# 8000: HTTP
# 8001: gRPC
# 8002: Metrics
EXPOSE 8000 8001 8002

# Set environment variables
ENV MODEL_REPOSITORY=/models

# Start Triton server
CMD ["tritonserver", "--model-repository=/models", "--log-verbose=1"]

