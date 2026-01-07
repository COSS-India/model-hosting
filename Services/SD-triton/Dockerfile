# Use NVIDIA Triton Inference Server as base image
FROM nvcr.io/nvidia/tritonserver:24.01-py3

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Note: pyannote.audio 2.1.1 requires torchaudio<1.0, so we use compatible versions
RUN pip install --no-cache-dir \
    torch==1.13.1 \
    torchaudio==0.13.1 \
    "huggingface_hub<0.20.0" \
    protobuf==3.20.3 \
    requests \
    soundfile

# Install pyannote.audio separately to handle dependencies
RUN pip install --no-cache-dir \
    pyannote.audio==2.1.1

# Copy model repository
COPY model_repository /models

# Pre-download the pyannote speaker diarization model to cache it in the image
# Note: This model is gated and requires authentication
# Pass HUGGING_FACE_HUB_TOKEN as build arg: --build-arg HUGGING_FACE_HUB_TOKEN=your_token
ARG HUGGING_FACE_HUB_TOKEN
RUN if [ -n "$HUGGING_FACE_HUB_TOKEN" ]; then \
        python3 -c "from pyannote.audio import Pipeline; \
        import os; \
        from huggingface_hub import login; \
        hf_token = os.environ.get('HUGGING_FACE_HUB_TOKEN'); \
        login(token=hf_token); \
        print('Authenticated with HuggingFace'); \
        model_name='pyannote/speaker-diarization@2.1'; \
        print('Downloading pyannote speaker diarization model...'); \
        pipeline = Pipeline.from_pretrained(model_name, use_auth_token=hf_token); \
        print('Model downloaded successfully')"; \
    else \
        echo "Model download skipped (will download at runtime)"; \
    fi

# Expose Triton ports
# 8000: HTTP
# 8001: gRPC
# 8002: Metrics
EXPOSE 8000 8001 8002

# Set environment variables
ENV MODEL_REPOSITORY=/models

# Start Triton server
CMD ["tritonserver", "--model-repository=/models", "--log-verbose=1"]

