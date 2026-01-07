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
    fasttext==0.9.2 \
    huggingface_hub

# Copy model repository structure
COPY model_repository /models

# Download IndicBERT tokenizer and model components
RUN python3 -c "from transformers import AutoTokenizer, AutoModel; \
    print('Downloading IndicBERT tokenizer and model...'); \
    tokenizer = AutoTokenizer.from_pretrained('ai4bharat/IndicBERTv2-MLM-only'); \
    model = AutoModel.from_pretrained('ai4bharat/IndicBERTv2-MLM-only'); \
    print('IndicBERT components downloaded successfully')"

# Download IndicLID models from GitHub releases
RUN cd /models/indiclid/1 && \
    echo "Downloading IndicLID models..." && \
    wget -q https://github.com/AI4Bharat/IndicLID/releases/download/v1.0/indiclid-ftn.zip && \
    wget -q https://github.com/AI4Bharat/IndicLID/releases/download/v1.0/indiclid-ftr.zip && \
    wget -q https://github.com/AI4Bharat/IndicLID/releases/download/v1.0/indiclid-bert.zip && \
    echo "Extracting models..." && \
    unzip -q indiclid-ftn.zip && \
    unzip -q indiclid-ftr.zip && \
    unzip -q indiclid-bert.zip && \
    echo "Renaming model files..." && \
    mv indiclid-ftn/model_baseline_roman.bin indiclid-ftn.bin 2>/dev/null || \
    mv indiclid-ftn/model*.bin indiclid-ftn.bin 2>/dev/null || \
    find indiclid-ftn -name "*.bin" -exec mv {} indiclid-ftn.bin \; && \
    mv indiclid-ftr/model_baseline_roman.bin indiclid-ftr.bin 2>/dev/null || \
    mv indiclid-ftr/model*.bin indiclid-ftr.bin 2>/dev/null || \
    find indiclid-ftr -name "*.bin" -exec mv {} indiclid-ftr.bin \; && \
    mv indiclid-bert/basline_nn_simple.pt indiclid-bert.pt 2>/dev/null || \
    mv indiclid-bert/*.pt indiclid-bert.pt 2>/dev/null || \
    find indiclid-bert -name "*.pt" -exec mv {} indiclid-bert.pt \; && \
    echo "Cleaning up..." && \
    rm -rf indiclid-ftn.zip indiclid-ftr.zip indiclid-bert.zip && \
    rm -rf indiclid-ftn indiclid-ftr indiclid-bert && \
    echo "Model files prepared:" && \
    ls -lh *.bin *.pt 2>/dev/null || echo "Warning: Model files may need manual verification"

# Expose Triton ports
# 8000: HTTP
# 8001: gRPC
# 8002: Metrics
EXPOSE 8000 8001 8002

# Set environment variables
ENV MODEL_REPOSITORY=/models

# Start Triton server
CMD ["tritonserver", "--model-repository=/models", "--log-verbose=1"]

