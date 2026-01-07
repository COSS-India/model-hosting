# Use NVIDIA Triton Inference Server as base image
FROM nvcr.io/nvidia/tritonserver:24.01-py3

# Set working directory
WORKDIR /workspace

# Install Python dependencies
RUN pip install --no-cache-dir \
    transformers==4.36.0 \
    torch==2.1.0 \
    sentencepiece==0.1.99 \
    protobuf==3.20.3 \
    accelerate==0.25.0 \
    huggingface_hub

# Copy model repository
COPY model_repository /models

# Pre-download the IndicNER model to cache it in the image
# Note: This model is gated and requires authentication
# Set HUGGING_FACE_HUB_TOKEN environment variable when building if needed
RUN python3 -c "from transformers import AutoTokenizer, AutoModelForTokenClassification; \
    import os; \
    from huggingface_hub import login; \
    hf_token = os.environ.get('HUGGING_FACE_HUB_TOKEN'); \
    if hf_token: \
        login(token=hf_token); \
        print('Authenticated with HuggingFace'); \
    model_name='ai4bharat/IndicNER'; \
    print('Downloading IndicNER model...'); \
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token); \
    model = AutoModelForTokenClassification.from_pretrained(model_name, token=hf_token); \
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

