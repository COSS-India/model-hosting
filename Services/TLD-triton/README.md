# IndicLID Triton Inference Server Deployment

This directory contains a production-ready deployment of the IndicLID (Language Identification for Indian languages) model using NVIDIA Triton Inference Server.

## Overview

IndicLID is a language identifier for all 22 Indian languages listed in the Indian constitution in both native-script and romanized text. This deployment uses a two-stage classifier that is an ensemble of:

1. **IndicLID-FTN**: FastText model for native script text (24 classes)
2. **IndicLID-FTR**: FastText model for roman script text (21 classes)
3. **IndicLID-BERT**: Fine-tuned IndicBERT model for low-confidence roman script cases

The system can predict **47 classes** total:
- 24 native-script classes
- 21 roman-script classes
- English (eng_Latn)
- Other (other)

## Supported Languages

The model supports the following languages in both native and roman scripts:

| Language | Native Script Code | Roman Script Code |
|----------|-------------------|-------------------|
| Assamese | asm_Beng | asm_Latn |
| Bengali | ben_Beng | ben_Latn |
| Bodo | brx_Deva | brx_Latn |
| Dogri | doi_Deva | doi_Latn |
| Gujarati | guj_Gujr | guj_Latn |
| Hindi | hin_Deva | hin_Latn |
| Kannada | kan_Knda | kan_Latn |
| Kashmiri | kas_Arab, kas_Deva | kas_Latn |
| Konkani | kok_Deva | kok_Latn |
| Maithili | mai_Deva | mai_Latn |
| Malayalam | mal_Mlym | mal_Latn |
| Manipuri | mni_Beng, mni_Meti | mni_Latn |
| Marathi | mar_Deva | mar_Latn |
| Nepali | nep_Deva | nep_Latn |
| Odia | ori_Orya | ori_Latn |
| Punjabi | pan_Guru | pan_Latn |
| Sanskrit | san_Deva | san_Latn |
| Santali | sat_Olch | - |
| Sindhi | snd_Arab | snd_Latn |
| Tamil | tam_Tamil | tam_Latn |
| Telugu | tel_Telu | tel_Latn |
| Urdu | urd_Arab | urd_Latn |
| English | - | eng_Latn |

## Directory Structure

```
indiclid-triton/
├── Dockerfile                          # Docker image definition
├── README.md                           # This file
├── test_client.py                      # Test client for inference
└── model_repository/                   # Triton model repository
    └── indiclid/                       # IndicLID model
        ├── config.pbtxt                # Triton model configuration
        └── 1/                          # Version 1
            ├── model.py                # Python backend implementation
            ├── indiclid-ftn.bin        # FastText native model (downloaded)
            ├── indiclid-ftr.bin        # FastText roman model (downloaded)
            └── indiclid-bert.pt        # BERT model (downloaded)
```

## Building the Docker Image

```bash
cd indiclid-triton
docker build -t indiclid-triton:latest .
```

**Note**: The build process will automatically download the IndicLID models from the official GitHub releases.

## Running the Server

### Using Docker

```bash
docker run --gpus all \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  --name indiclid-server \
  indiclid-triton:latest
```

### Port Mapping

- **8000**: HTTP inference endpoint
- **8001**: gRPC inference endpoint
- **8002**: Metrics endpoint

## Testing the Deployment

### Using the Test Client

```bash
python3 test_client.py
```

This will run comprehensive tests for all supported languages in both native and roman scripts.

### Using curl

#### Native Script Example (Hindi)

```bash
curl -X POST http://localhost:8000/v2/models/indiclid/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "INPUT_TEXT",
        "shape": [1, 1],
        "datatype": "BYTES",
        "data": [["मैं भारत से हूं"]]
      }
    ],
    "outputs": [
      {
        "name": "OUTPUT_TEXT"
      }
    ]
  }'
```

Expected response:
```json
{
  "outputs": [
    {
      "name": "OUTPUT_TEXT",
      "datatype": "BYTES",
      "shape": [1, 1],
      "data": ["{\"input\": \"मैं भारत से हूं\", \"langCode\": \"hin_Deva\", \"confidence\": 0.9999, \"model\": \"IndicLID-FTN\"}"]
    }
  ]
}
```

#### Roman Script Example (Hindi)

```bash
curl -X POST http://localhost:8000/v2/models/indiclid/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "INPUT_TEXT",
        "shape": [1, 1],
        "datatype": "BYTES",
        "data": [["main bharat se hoon"]]
      }
    ],
    "outputs": [
      {
        "name": "OUTPUT_TEXT"
      }
    ]
  }'
```

#### Tamil Example

```bash
curl -X POST http://localhost:8000/v2/models/indiclid/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "INPUT_TEXT",
        "shape": [1, 1],
        "datatype": "BYTES",
        "data": [["நான் இந்தியாவிலிருந்து வந்தேன்"]]
      }
    ],
    "outputs": [
      {
        "name": "OUTPUT_TEXT"
      }
    ]
  }'
```

#### Bengali Example

```bash
curl -X POST http://localhost:8000/v2/models/indiclid/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "INPUT_TEXT",
        "shape": [1, 1],
        "datatype": "BYTES",
        "data": [["আমি ভারত থেকে এসেছি"]]
      }
    ],
    "outputs": [
      {
        "name": "OUTPUT_TEXT"
      }
    ]
  }'
```

## API Reference

### Input Format

- **INPUT_TEXT**: String containing the text to detect language for

### Output Format

The output is a JSON string with the following fields:

```json
{
  "input": "original input text",
  "langCode": "detected language code (e.g., hin_Deva, tam_Latn)",
  "confidence": 0.9999,
  "model": "model used for detection (IndicLID-FTN, IndicLID-FTR, or IndicLID-BERT)"
}
```

## Model Details

### Two-Stage Detection Pipeline

1. **Script Detection**: The system first determines if the text is primarily roman script or native script based on character analysis (threshold: 50%)

2. **Native Script Path** (if < 50% roman characters):
   - Uses **IndicLID-FTN** (FastText Native) model
   - Fast and accurate for native scripts
   - Returns result directly

3. **Roman Script Path** (if ≥ 50% roman characters):
   - **Stage 1**: Uses **IndicLID-FTR** (FastText Roman) model
   - If confidence > 0.6: Returns FTR result
   - If confidence ≤ 0.6: Proceeds to Stage 2
   - **Stage 2**: Uses **IndicLID-BERT** model for more accurate detection

### Performance

- **Throughput**: 
  - IndicLID-FTN: ~30,000 sentences/second
  - IndicLID-FTR: ~37,000 sentences/second
  - IndicLID-BERT: ~3 sentences/second (used only for low-confidence cases)

- **Accuracy**:
  - Native script: 98% F1-score
  - Roman script: 80% accuracy (75% F1-score)

## Monitoring

### Health Check

```bash
curl http://localhost:8000/v2/health/ready
```

### Metrics

```bash
curl http://localhost:8002/metrics
```

## Troubleshooting

### Model Loading Issues

If models fail to load, check the Docker logs:

```bash
docker logs indiclid-server
```

### GPU Not Available

If GPU is not available, the model will fall back to CPU. Check logs for:
```
[OK] Using device: cpu
```

For GPU support, ensure:
1. NVIDIA drivers are installed
2. nvidia-docker2 is installed
3. `--gpus all` flag is used when running the container

## References

- **IndicLID Repository**: https://github.com/AI4Bharat/IndicLID
- **Paper**: IndicLID: Language Identification for Indian Languages
- **Model Downloads**: https://github.com/AI4Bharat/IndicLID/releases/tag/v1.0
- **Triton Inference Server**: https://github.com/triton-inference-server/server

## License

The IndicLID models are released under the MIT License by AI4Bharat.

## Acknowledgements

This deployment is based on the IndicLID models developed by AI4Bharat (IIT Madras). We thank the original authors for making these models available.

