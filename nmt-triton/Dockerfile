# Standard Triton base (CPU-capable). Pick a newer tag if this one is gone.
FROM nvcr.io/nvidia/tritonserver:24.08-py3

# IMPORTANT: this base image does NOT ship the Python `torch` module for the
# Python backend, so we install it explicitly (CPU build).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
        transformers==4.53.2 sentencepiece sacremoses nltk regex mosestokenizer \
    && git clone https://github.com/VarunGumma/IndicTransToolkit /opt/IndicTransToolkit \
    && pip install --no-cache-dir --editable /opt/IndicTransToolkit \
    && python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Cache downloaded models here (mount a volume to persist across restarts)
ENV HF_HOME=/cache

COPY models /models

CMD ["tritonserver", "--model-repository=/models"]
