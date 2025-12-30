"""
FastAPI ASR server for AI4Bharat Multilingual ASR model.

This server hosts the AI4Bharat IndicConformer-600M-Multilingual ASR model
from HuggingFace, accessible via the /asr endpoint.

Endpoints:
  POST /asr
    - Accepts multipart/form-data with field "audio" (file: wav/flac/etc), OR
    - Accepts JSON {"audio_b64": "<base64 wav bytes>"}
    - Returns JSON {"text": "<transcription>", "latency_ms": <float>}

Environment Variables:
  HF_TOKEN - HuggingFace token for authentication (required)
  DEVICE - Device to use: "cuda" or "cpu" (default: auto-detect)
  MODEL_NAME - HuggingFace model name (default: ai4bharat/indic-conformer-600m-multilingual)

Run:
  pip install -r requirements.txt
  export HF_TOKEN=your_token_here
  uvicorn app:app --host 0.0.0.0 --port 8000
"""
import os
import io
import time
import base64
import tempfile
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import soundfile as sf
import torch
import numpy as np
from transformers import AutoModel, AutoProcessor, pipeline

# Configuration
HF_TOKEN = os.environ.get("HF_TOKEN", "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")
MODEL_NAME = os.environ.get("MODEL_NAME", "ai4bharat/indic-conformer-600m-multilingual")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(
    title="AI4Bharat Multilingual ASR API",
    description="FastAPI server hosting AI4Bharat IndicConformer-600M-Multilingual ASR model",
    version="1.0.0"
)

class ASRResponse(BaseModel):
    text: str
    latency_ms: float

# Global model and pipeline
asr_pipeline = None
sample_rate = 16000

@app.on_event("startup")
def load_model():
    """Load the ASR model on startup."""
    global asr_pipeline, sample_rate
    
    print(f"Loading model: {MODEL_NAME}")
    print(f"Using device: {DEVICE}")
    print(f"Using HuggingFace token: {HF_TOKEN[:10]}...")
    
    try:
        # Load model and processor from HuggingFace
        # This model uses a custom configuration, so we use AutoModel instead
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else None,
            token=HF_TOKEN,
            trust_remote_code=True
        )
        
        if DEVICE == "cpu":
            model = model.to(DEVICE)
        
        # Try to load processor, but it's optional for ONNX models
        processor = None
        try:
            processor = AutoProcessor.from_pretrained(MODEL_NAME, token=HF_TOKEN, trust_remote_code=True)
        except Exception as e:
            print(f"Warning: Could not load processor: {e}")
            print("Will use model's native inference methods")
        
        # Check if model has native inference methods (like transcribe)
        if hasattr(model, 'transcribe') or hasattr(model, 'generate'):
            # Use model's native methods
            asr_pipeline = model
            # Try to get sample rate from model config
            if hasattr(model, 'config') and hasattr(model.config, 'sample_rate'):
                sample_rate = model.config.sample_rate
            elif hasattr(model, 'sample_rate'):
                sample_rate = model.sample_rate
            else:
                sample_rate = 16000  # Default
        else:
            # Try to create pipeline if processor is available
            if processor is not None:
                pipeline_kwargs = {
                    "model": model,
                    "device": 0 if DEVICE == "cuda" else -1,
                    "torch_dtype": torch.float16 if DEVICE == "cuda" else torch.float32,
                }
                
                if hasattr(processor, 'tokenizer'):
                    pipeline_kwargs["tokenizer"] = processor.tokenizer
                
                if hasattr(processor, 'feature_extractor'):
                    pipeline_kwargs["feature_extractor"] = processor.feature_extractor
                    if hasattr(processor.feature_extractor, 'sampling_rate'):
                        sample_rate = processor.feature_extractor.sampling_rate
                elif hasattr(processor, 'image_processor'):
                    pipeline_kwargs["feature_extractor"] = processor.image_processor
                    if hasattr(processor.image_processor, 'sampling_rate'):
                        sample_rate = processor.image_processor.sampling_rate
                
                asr_pipeline = pipeline("automatic-speech-recognition", **pipeline_kwargs)
            else:
                # Fallback: use model directly
                asr_pipeline = model
                sample_rate = 16000  # Default
        
        print(f"Model loaded successfully. Sample rate: {sample_rate} Hz")
        
        # Diagnostic: Check device placement
        if DEVICE == "cuda" and torch.cuda.is_available():
            if hasattr(asr_pipeline, 'parameters'):
                gpu_params = sum(1 for p in asr_pipeline.parameters() if p.device.type == 'cuda')
                cpu_params = sum(1 for p in asr_pipeline.parameters() if p.device.type == 'cpu')
                print(f"[DIAGNOSTIC] Model parameters - GPU: {gpu_params}, CPU: {cpu_params}")
            print(f"[DIAGNOSTIC] Model type: {type(asr_pipeline)}")
            print(f"[DIAGNOSTIC] Has transcribe: {hasattr(asr_pipeline, 'transcribe')}")
            print(f"[DIAGNOSTIC] Has forward: {hasattr(asr_pipeline, 'forward')}")
        
        app.state.asr_pipeline = asr_pipeline
        app.state.processor = processor
        app.state.sample_rate = sample_rate
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Failed to load model {MODEL_NAME}: {e}")

def _read_audio_from_bytes(wav_bytes: bytes):
    """
    Read audio bytes (wav/flac/...) and return numpy array and sample rate.
    Raises HTTPException on failure.
    """
    try:
        data, sr = sf.read(io.BytesIO(wav_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read audio: {e}")
    return data, int(sr)

def _ensure_mono(data):
    """Convert to mono if necessary."""
    if getattr(data, "ndim", 1) > 1:
        return data.mean(axis=1)
    return data

def _resample_audio(data, orig_sr, target_sr):
    """Resample audio if needed."""
    if orig_sr == target_sr:
        return data
    
    try:
        import librosa
        return librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        raise HTTPException(
            status_code=400,
            detail=f"Sample rate mismatch ({orig_sr} vs {target_sr}) and librosa not available for resampling"
        )

@app.post("/asr", response_model=ASRResponse)
async def asr_endpoint(
    request: Request,
    audio: Optional[UploadFile] = File(None),
    audio_b64: Optional[str] = Form(None),
    lang: Optional[str] = Form(None)
):
    """
    ASR endpoint for speech-to-text transcription.
    
    Accepts either:
      - multipart/form-data with field "audio" (UploadFile) and "lang" (required), OR
      - multipart/form-data with field "audio_b64" (base64-encoded string) and "lang" (required), OR
      - JSON body: {"audio_b64": "...", "lang": "hi"} (lang is required)
    
    Language codes: hi (Hindi), en (English), ta (Tamil), te (Telugu), mr (Marathi), etc.
    
    Returns transcription and latency in milliseconds.
    """
    start_time = time.time()
    
    # Check if model is loaded
    if asr_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    # Determine input format
    content_type = request.headers.get("content-type", "").lower()
    wav_bytes = None
    
    if "application/json" in content_type:
        # Handle JSON request
        try:
            body = await request.json()
            if isinstance(body, dict) and "audio_b64" in body:
                audio_b64 = body["audio_b64"]
            else:
                raise ValueError("JSON must be object with key 'audio_b64'")
            
            if not isinstance(audio_b64, str):
                raise ValueError("'audio_b64' must be a base64-encoded string")
            
            try:
                wav_bytes = base64.b64decode(audio_b64)
            except Exception as e:
                raise ValueError(f"base64 decode error: {e}")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")
    else:
        # Handle multipart/form-data request
        try:
            form = await request.form()
            
            # Try to get audio file first
            if audio is not None:
                wav_bytes = await audio.read()
            elif "audio" in form:
                audio_file = form["audio"]
                if hasattr(audio_file, 'read'):
                    wav_bytes = await audio_file.read()
                else:
                    raise HTTPException(status_code=400, detail="'audio' field must be a file upload")
            elif audio_b64 is not None:
                try:
                    wav_bytes = base64.b64decode(audio_b64)
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"base64 decode error: {e}")
            elif "audio_b64" in form:
                audio_b64 = form["audio_b64"]
                try:
                    wav_bytes = base64.b64decode(audio_b64)
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"base64 decode error: {e}")
            else:
                raise HTTPException(
                    status_code=400,
                    detail="No audio provided. Send 'audio' file or 'audio_b64' in form data, or JSON with 'audio_b64'"
                )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read audio: {e}")
    
    if wav_bytes is None:
        raise HTTPException(status_code=400, detail="No audio data received")
    
    # Get language parameter - required
    # Get language parameter - required, no default
    language = lang
    if not language or language.strip() == "":
        # Try to get from JSON if JSON request
        if "application/json" in content_type:
            try:
                body = await request.json()
                language = body.get("lang")
            except:
                pass
        
        # Language is required - throw error if not provided
        if not language or language.strip() == "":
            raise HTTPException(status_code=400, detail="Language not passed. Please provide 'lang' parameter (e.g., 'hi', 'en', 'ta', etc.)")
    
    # Decode audio bytes to numpy array
    data, sr = _read_audio_from_bytes(wav_bytes)
    data = _ensure_mono(data)
    
    # Resample if needed
    target_sr = app.state.sample_rate
    if sr != target_sr:
        data = _resample_audio(data, sr, target_sr)
        sr = target_sr
    
    # Perform transcription
    try:
        # Convert to float32 numpy array
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        
        # Normalize if needed
        if np.abs(data).max() > 1.0:
            data = data / np.abs(data).max()
        
        # Run inference - check if model has native methods
        model = app.state.asr_pipeline
        
        # Diagnostic: Log device info
        if DEVICE == "cuda" and torch.cuda.is_available():
            if hasattr(model, 'parameters'):
                sample_param = next(iter(model.parameters()), None)
                if sample_param is not None:
                    print(f"[INFERENCE] Model device: {sample_param.device}, Input type: {type(data)}")
        
        # First, try transcribe method (most common for ASR models)
        if hasattr(model, 'transcribe'):
            # Use model's transcribe method - ensure GPU usage if available
            try:
                # Convert numpy array to GPU tensor if CUDA is available
                if DEVICE == "cuda" and torch.cuda.is_available() and isinstance(data, np.ndarray):
                    # Try passing as GPU tensor first
                    try:
                        data_tensor = torch.from_numpy(data).unsqueeze(0)  # Add batch dimension
                        data_tensor = data_tensor.cuda()
                        # Some models accept tensors directly
                        result = model.transcribe(data_tensor, lang=language)
                        if isinstance(result, list):
                            text = result[0] if result else ""
                        elif isinstance(result, str):
                            text = result
                        elif isinstance(result, dict):
                            text = result.get("text", result.get("transcription", ""))
                        else:
                            text = str(result)
                    except (TypeError, AttributeError) as tensor_error:
                        # If tensor doesn't work, try with numpy array (model might handle GPU internally)
                        result = model.transcribe(data, lang=language)
                        if isinstance(result, list):
                            text = result[0] if result else ""
                        elif isinstance(result, str):
                            text = result
                        elif isinstance(result, dict):
                            text = result.get("text", result.get("transcription", ""))
                        else:
                            text = str(result)
                else:
                    # CPU path or numpy array
                    result = model.transcribe(data, lang=language)
                    if isinstance(result, list):
                        text = result[0] if result else ""
                    elif isinstance(result, str):
                        text = result
                    elif isinstance(result, dict):
                        text = result.get("text", result.get("transcription", ""))
                    else:
                        text = str(result)
            except Exception as e:
                # Fallback: try with file path
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                        sf.write(tmp.name, data, sr)
                        result = model.transcribe(tmp.name, lang=language)
                        if isinstance(result, list):
                            text = result[0] if result else ""
                        elif isinstance(result, str):
                            text = result
                        elif isinstance(result, dict):
                            text = result.get("text", result.get("transcription", ""))
                        else:
                            text = str(result)
                except Exception as e2:
                    raise HTTPException(status_code=500, detail=f"Transcribe failed: {e}, {e2}")
        elif hasattr(model, 'generate'):
            # Use model's generate method - ensure GPU usage if available
            try:
                # Convert numpy array to GPU tensor if CUDA is available
                if DEVICE == "cuda" and torch.cuda.is_available() and isinstance(data, np.ndarray):
                    data_tensor = torch.from_numpy(data).unsqueeze(0)  # Add batch dimension
                    data_tensor = data_tensor.cuda()
                    result = model.generate(data_tensor, lang=language)
                else:
                    result = model.generate(data, lang=language)
                
                if isinstance(result, dict):
                    text = result.get("text", "")
                elif isinstance(result, str):
                    text = result
                else:
                    text = str(result)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Generate failed: {e}")
        elif callable(model):
            # Try to call model directly without unsupported parameters
            try:
                # Convert numpy array to tensor if needed
                if isinstance(data, np.ndarray):
                    data_tensor = torch.from_numpy(data).unsqueeze(0)  # Add batch dimension
                    if DEVICE == "cuda" and torch.cuda.is_available():
                        data_tensor = data_tensor.cuda()
                    result = model(data_tensor, lang=language)
                else:
                    result = model(data, lang=language)
                
                # Handle different result formats
                if isinstance(result, dict):
                    text = result.get("text", result.get("transcription", ""))
                elif isinstance(result, (list, tuple)) and len(result) > 0:
                    text = str(result[0])
                elif isinstance(result, str):
                    text = result
                else:
                    text = str(result)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")
        else:
            raise HTTPException(status_code=500, detail="Model does not support inference")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    
    latency_ms = (time.time() - start_time) * 1000.0
    
    return ASRResponse(text=text, latency_ms=latency_ms)

class AudioBase64Request(BaseModel):
    audio_b64: str
    lang: str

@app.post("/asr/json", response_model=ASRResponse)
async def asr_json_endpoint(request: AudioBase64Request):
    """
    Alternative ASR endpoint that accepts JSON with base64-encoded audio.
    
    Request body: {"audio_b64": "base64_encoded_audio_bytes", "lang": "hi"} (lang is required)
    
    Language codes: hi (Hindi), en (English), ta (Tamil), te (Telugu), mr (Marathi), etc.
    """
    start_time = time.time()
    
    if asr_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        wav_bytes = base64.b64decode(request.audio_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"base64 decode error: {e}")
    
    # Get language parameter - required, no default
    language = request.lang
    if not language or language.strip() == "":
        raise HTTPException(status_code=400, detail="Language not passed. Please provide 'lang' parameter (e.g., 'hi', 'en', 'ta', etc.)")
    
    # Decode audio bytes to numpy array
    data, sr = _read_audio_from_bytes(wav_bytes)
    data = _ensure_mono(data)
    
    # Resample if needed
    target_sr = app.state.sample_rate
    if sr != target_sr:
        data = _resample_audio(data, sr, target_sr)
        sr = target_sr
    
    # Perform transcription
    try:
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        
        if np.abs(data).max() > 1.0:
            data = data / np.abs(data).max()
        
        # Run inference - check if model has native methods
        model = app.state.asr_pipeline
        
        # First, try transcribe method (most common for ASR models)
        if hasattr(model, 'transcribe'):
            # Use model's transcribe method - ensure GPU usage if available
            try:
                # Convert numpy array to GPU tensor if CUDA is available
                if DEVICE == "cuda" and torch.cuda.is_available() and isinstance(data, np.ndarray):
                    # Try passing as GPU tensor first
                    try:
                        data_tensor = torch.from_numpy(data).unsqueeze(0)  # Add batch dimension
                        data_tensor = data_tensor.cuda()
                        # Some models accept tensors directly
                        result = model.transcribe(data_tensor, lang=language)
                        if isinstance(result, list):
                            text = result[0] if result else ""
                        elif isinstance(result, str):
                            text = result
                        elif isinstance(result, dict):
                            text = result.get("text", result.get("transcription", ""))
                        else:
                            text = str(result)
                    except (TypeError, AttributeError) as tensor_error:
                        # If tensor doesn't work, try with numpy array (model might handle GPU internally)
                        result = model.transcribe(data, lang=language)
                        if isinstance(result, list):
                            text = result[0] if result else ""
                        elif isinstance(result, str):
                            text = result
                        elif isinstance(result, dict):
                            text = result.get("text", result.get("transcription", ""))
                        else:
                            text = str(result)
                else:
                    # CPU path or numpy array
                    result = model.transcribe(data, lang=language)
                    if isinstance(result, list):
                        text = result[0] if result else ""
                    elif isinstance(result, str):
                        text = result
                    elif isinstance(result, dict):
                        text = result.get("text", result.get("transcription", ""))
                    else:
                        text = str(result)
            except Exception as e:
                # Fallback: try with file path
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                        sf.write(tmp.name, data, sr)
                        result = model.transcribe(tmp.name, lang=language)
                        if isinstance(result, list):
                            text = result[0] if result else ""
                        elif isinstance(result, str):
                            text = result
                        elif isinstance(result, dict):
                            text = result.get("text", result.get("transcription", ""))
                        else:
                            text = str(result)
                except Exception as e2:
                    raise HTTPException(status_code=500, detail=f"Transcribe failed: {e}, {e2}")
        elif hasattr(model, 'generate'):
            # Use model's generate method - ensure GPU usage if available
            try:
                # Convert numpy array to GPU tensor if CUDA is available
                if DEVICE == "cuda" and torch.cuda.is_available() and isinstance(data, np.ndarray):
                    data_tensor = torch.from_numpy(data).unsqueeze(0)  # Add batch dimension
                    data_tensor = data_tensor.cuda()
                    result = model.generate(data_tensor, lang=language)
                else:
                    result = model.generate(data, lang=language)
                
                if isinstance(result, dict):
                    text = result.get("text", "")
                elif isinstance(result, str):
                    text = result
                else:
                    text = str(result)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Generate failed: {e}")
        elif callable(model):
            # Try to call model directly without unsupported parameters
            try:
                # Convert numpy array to tensor if needed
                if isinstance(data, np.ndarray):
                    data_tensor = torch.from_numpy(data).unsqueeze(0)  # Add batch dimension
                    if DEVICE == "cuda" and torch.cuda.is_available():
                        data_tensor = data_tensor.cuda()
                    result = model(data_tensor, lang=language)
                else:
                    result = model(data, lang=language)
                
                # Handle different result formats
                if isinstance(result, dict):
                    text = result.get("text", result.get("transcription", ""))
                elif isinstance(result, (list, tuple)) and len(result) > 0:
                    text = str(result[0])
                elif isinstance(result, str):
                    text = result
                else:
                    text = str(result)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")
        else:
            raise HTTPException(status_code=500, detail="Model does not support inference")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    
    latency_ms = (time.time() - start_time) * 1000.0
    
    return ASRResponse(text=text, latency_ms=latency_ms)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": asr_pipeline is not None,
        "device": DEVICE,
        "sample_rate": sample_rate,
        "model_name": MODEL_NAME
    }

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "AI4Bharat Multilingual ASR API",
        "version": "1.0.0",
        "model": MODEL_NAME,
        "endpoints": {
            "/asr": "POST - Transcribe audio (multipart or JSON)",
            "/asr/json": "POST - Transcribe audio (JSON only)",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, workers=1)

