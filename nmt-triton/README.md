# nmt-triton — Local CPU NMT (IndicTrans2)

Docker image and Triton model repository for **Neural Machine Translation** using AI4Bharat **IndicTrans2** on **CPU** (no GPU required).

## Contents

```
nmt-triton/
├── Dockerfile
├── README.md
└── models/
    └── nmt/
        ├── config.pbtxt      # Triton model config
        └── 1/
            └── model.py      # IndicTrans2 Python backend
```

## Prerequisites

- Linux with Docker
- HuggingFace account with access to gated IndicTrans2 models
- HuggingFace **Read** token (`hf_...`)

Accept model access on HuggingFace for:

- [indictrans2-en-indic-dist-200M](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M)
- [indictrans2-indic-en-dist-200M](https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M)
- [indictrans2-indic-indic-dist-320M](https://huggingface.co/ai4bharat/indictrans2-indic-indic-dist-320M)

## Quick start

```bash
cd nmt-triton
docker build -t nmt-triton-cpu .

docker run -d \
  -p 8000:8000 -p 8002:8002 \
  -v hf-cache:/cache \
  -e HF_TOKEN=hf_your_token_here \
  --name indictrans \
  nmt-triton-cpu
```

Health check:

```bash
curl -s -o /dev/null -w "HTTP %{http_code}\n" http://localhost:8000/v2/health/ready
```

Test translation:

```bash
curl -X POST http://localhost:8000/v2/models/nmt/infer \
  -H "Content-Type: application/json" \
  -d '{"inputs":[
    {"name":"INPUT_TEXT","shape":[1,1],"datatype":"BYTES","data":["Hello, how are you?"]},
    {"name":"INPUT_LANGUAGE_ID","shape":[1,1],"datatype":"BYTES","data":["en"]},
    {"name":"OUTPUT_LANGUAGE_ID","shape":[1,1],"datatype":"BYTES","data":["hi"]}]}'
```

The **first request** may take 1–3 minutes while models download into the `hf-cache` volume.

## GPU deployment

For GPU-based NMT on Triton, see `setup-docs/NMT_guide.md` in this repository.

## Related

| Document | Location |
|----------|----------|
| GPU NMT Triton guide | `setup-docs/NMT_guide.md` |
