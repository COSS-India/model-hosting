# Architecture Documentation

This document describes the architecture and design of the MLflow ASR service.

## System Overview

The MLflow ASR service is a REST API that provides automatic speech recognition (ASR) capabilities for multiple Indic languages. It uses MLflow's PyFunc model flavor to serve a HuggingFace-based ASR model.

```
┌─────────────────┐
│   Client App    │
│  (curl/Python/  │
│   JavaScript)   │
└────────┬────────┘
         │ HTTP POST
         │ /asr
         ▼
┌─────────────────┐
│  MLflow Server  │
│  (Uvicorn)      │
│  Port: 5000     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PyFunc Model   │
│  Wrapper        │
│  (mlflow_asr.py)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Indic Conformer│
│  Model          │
│  (HuggingFace)  │
└─────────────────┘
```

## Components

### 1. MLflow Server

- **Framework**: Uvicorn (ASGI server)
- **Port**: 5000 (configurable)
- **Protocol**: HTTP/REST
- **Endpoint**: `/asr`
- **Health Check**: `/health`

The MLflow server handles:
- HTTP request/response handling
- Request validation
- Model invocation
- Error handling

### 2. PyFunc Model Wrapper

**File**: `mlflow_asr.py`

The `IndicConformerWrapper` class implements MLflow's `PythonModel` interface:

```python
class IndicConformerWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Model initialization
    
    def predict(self, context, model_input):
        # Model inference
```

**Responsibilities**:
- Model loading and initialization
- Audio preprocessing (base64 decoding, resampling)
- Model inference
- Response formatting

### 3. ASR Model

**Model**: `ai4bharat/indic-conformer-600m-multilingual`

- **Architecture**: Conformer-based transformer
- **Parameters**: 600M
- **Input**: Audio waveform (16kHz, mono)
- **Output**: Transcribed text
- **Supported Languages**: Multiple Indic languages
- **Decoding**: CTC or greedy

## Data Flow

### Request Flow

1. **Client** sends HTTP POST request with base64-encoded audio
2. **MLflow Server** receives and validates request
3. **PyFunc Wrapper** processes the request:
   - Decodes base64 audio
   - Loads audio using Python's `wave` module
   - Resamples to 16kHz if needed
   - Converts to mono if multi-channel
   - Converts to PyTorch tensor
4. **ASR Model** performs inference:
   - Processes audio through Conformer layers
   - Generates transcription using CTC/greedy decoding
5. **Response** is formatted and returned to client

### Audio Processing Pipeline

```
Base64 String
    ↓
Base64 Decode
    ↓
WAV Bytes
    ↓
Wave Module Parse
    ↓
NumPy Array
    ↓
PyTorch Tensor
    ↓
Resample (if needed)
    ↓
Mono Conversion (if needed)
    ↓
GPU Transfer (if available)
    ↓
Model Input
```

## Model Loading Strategy

### Lazy Loading

The model is loaded on the first request, not at server startup. This allows:
- Faster server startup
- Memory efficiency (model only loaded when needed)
- Easier debugging (errors appear on first request)

### Model Caching

Once loaded, the model remains in memory for subsequent requests, providing:
- Fast inference for multiple requests
- Reduced memory churn
- Better performance

## Error Handling

### Request Validation

- Base64 decoding errors
- Audio format validation
- Required field checks

### Model Errors

- Model loading failures
- Inference errors
- GPU/CPU fallback

### Response Format

All errors follow MLflow's standard format:

```json
{
  "error_code": "BAD_REQUEST",
  "message": "Error description",
  "stack_trace": "..."
}
```

## Deployment Architectures

### Local Development

```
┌──────────────┐
│   Python     │
│  Virtual Env │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ MLflow Serve │
│  (Direct)    │
└──────────────┘
```

### Docker Deployment

```
┌──────────────┐
│   Docker     │
│   Container  │
│              │
│ ┌──────────┐│
│ │ MLflow   ││
│ │ Server   ││
│ └──────────┘│
│ ┌──────────┐│
│ │ Model    ││
│ │ Artifacts││
│ └──────────┘│
└──────────────┘
```

### Production Deployment

```
┌──────────────┐
│ Load Balancer│
└──────┬───────┘
       │
   ┌───┴───┐
   │       │
┌──▼──┐ ┌──▼──┐
│App 1│ │App 2│
└─────┘ └─────┘
```

## Performance Considerations

### GPU Acceleration

- Automatic GPU detection
- CUDA support via PyTorch
- Falls back to CPU if GPU unavailable

### Audio Processing

- Efficient WAV parsing using Python's `wave` module
- NumPy for array operations
- PyTorch for tensor operations
- Resampling only when necessary

### Model Optimization

- Model quantization (future)
- Batch processing support (future)
- ONNX Runtime integration (model uses ONNX)

## Security Considerations

### Authentication

- HuggingFace token for model access
- Environment variable storage
- No token in code or logs

### Input Validation

- Base64 validation
- Audio format checks
- Size limits (implicit)

### Container Security

- Non-root user (recommended)
- Minimal base image
- Regular security updates

## Scalability

### Horizontal Scaling

- Stateless service design
- Multiple container instances
- Load balancer distribution

### Vertical Scaling

- GPU memory limits
- CPU core utilization
- Model size constraints

## Monitoring and Observability

### Health Checks

- `/health` endpoint
- Container health checks
- Model availability status

### Logging

- Request/response logging
- Error tracking
- Performance metrics

### Metrics (Future)

- Request latency
- Throughput (QPS)
- Error rates
- GPU utilization

## Dependencies

### Core Dependencies

- `mlflow>=3.7.0` - Model serving framework
- `torch>=2.0.0` - Deep learning framework
- `torchaudio>=2.0.0` - Audio processing
- `transformers>=4.30.0` - HuggingFace models
- `onnxruntime>=1.15.0` - ONNX inference
- `pandas>=1.5.0` - Data handling

### System Dependencies

- Python 3.10+
- CUDA (optional, for GPU)
- FFmpeg (not required, using wave module)

## Future Enhancements

1. **Batch Processing**: Support multiple audio files in one request
2. **Streaming**: Real-time audio transcription
3. **Model Versioning**: Multiple model versions
4. **Caching**: Response caching for identical inputs
5. **Metrics**: Prometheus/Grafana integration
6. **Authentication**: API key authentication
7. **Rate Limiting**: Request throttling

---

**Last Updated**: December 2024

