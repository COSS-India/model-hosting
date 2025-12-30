# service.py
import io
import os
import sys
import json

# Add BentoML's package path to Python path (for Docker containers)
bento_pkg_path = "/home/bentoml/bento/src/bento/lib/python3.10/site-packages"
if os.path.exists(bento_pkg_path) and bento_pkg_path not in sys.path:
    sys.path.insert(0, bento_pkg_path)

import torch
import torchaudio
import numpy as np
import bentoml
from bentoml import Service
from transformers import AutoModel
from starlette.requests import Request
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from typing import Optional, Any

# Load Bento model entry by tag name we saved earlier
# NOTE: This model does NOT have a tokenizer (it's an ASR model)
# Use model name without version to get the latest version
MODEL_NAME = "indic_conformer_600m_model"

# Lazy load model config to avoid errors at import time
def get_model_config():
    """Load model config from BentoML artifact"""
    try:
        import bentoml.pytorch
        # Get model by name (will get the latest version)
        # BentoML automatically uses the latest version when no tag is specified
        model_artifact = bentoml.pytorch.get(MODEL_NAME)
        
        # Try to load config from wrapper or JSON file
        config_path = os.path.join(model_artifact.path, "model_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                model_config = json.load(f)
        else:
            # Fallback: extract config from wrapper object
            try:
                wrapper = model_artifact.load_model()
                model_config = {
                    "model_id": wrapper.model_id,
                    "hf_token": wrapper.hf_token,
                    "torch_dtype": wrapper.torch_dtype,
                    "device": wrapper.device,
                }
            except Exception as e:
                print(f"Warning: Could not load config from wrapper: {e}")
                # Last resort: use labels and environment
                model_config = {
                    "model_id": model_artifact.labels.get("hf_id", "ai4bharat/indic-conformer-600m-multilingual"),
                    "hf_token": os.environ.get("HF_TOKEN", ""),
                    "torch_dtype": "float16" if torch.cuda.is_available() else "float32",
                    "device": model_artifact.labels.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
                }
        
        # Ensure HF_TOKEN is set
        if not model_config.get("hf_token") and os.environ.get("HF_TOKEN"):
            model_config["hf_token"] = os.environ.get("HF_TOKEN")
        
        return model_config
    except Exception as e:
        print(f"Error loading model config: {e}")
        print("Falling back to environment variables and defaults (this is normal in Docker)")
        # Fallback to environment variables only - this is expected in Docker if model isn't in bundle
        # The model will be loaded directly from HuggingFace at runtime
        model_config = {
            "model_id": os.environ.get("MODEL_ID", "ai4bharat/indic-conformer-600m-multilingual"),
            "hf_token": os.environ.get("HF_TOKEN", ""),
            "torch_dtype": "float16" if torch.cuda.is_available() else "float32",
            "device": os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
        }
        # Warn if HF_TOKEN is not set
        if not model_config["hf_token"]:
            print("WARNING: HF_TOKEN not set. Model loading may fail if the model requires authentication.")
        return model_config

# Create a custom runner that loads the model dynamically
# This avoids serialization issues with NeMo components
class ASRModelRunner:
    def __init__(self, model_config):
        self.model_config = model_config
        self._model = None
    
    def load_model(self):
        """Lazy load the model when needed"""
        if self._model is None:
            print(f"Loading model: {self.model_config['model_id']}")
            print(f"Using device: {self.model_config['device']}")
            
            # Determine torch dtype
            torch_dtype = torch.float16 if self.model_config["torch_dtype"] == "float16" else torch.float32
            
            # Load model
            self._model = AutoModel.from_pretrained(
                self.model_config["model_id"],
                token=self.model_config["hf_token"],
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map="auto" if self.model_config["device"] == "cuda" else None
            )
            
            # Explicitly move model to GPU if using CUDA (device_map="auto" might not work correctly)
            if self.model_config["device"] == "cuda" and torch.cuda.is_available():
                # Check if model is already on GPU
                first_param = next(self._model.parameters(), None)
                if first_param is not None and first_param.device.type != "cuda":
                    print(f"Moving model to GPU (currently on {first_param.device})")
                    self._model = self._model.cuda()
                else:
                    print(f"Model is on device: {first_param.device if first_param else 'unknown'}")
            
            # Set model to eval mode
            self._model.eval()
            print("Model loaded successfully!")
        return self._model
    
    def run(self, wav_tensor, lang, strategy):
        """Run inference"""
        model = self.load_model()
        
        # Ensure tensor is on the same device as model
        if self.model_config["device"] == "cuda" and torch.cuda.is_available():
            # Get model device
            first_param = next(model.parameters(), None)
            if first_param is not None:
                model_device = first_param.device
                if wav_tensor.device != model_device:
                    wav_tensor = wav_tensor.to(model_device)
        
        # Run inference with torch.no_grad() for efficiency
        with torch.no_grad():
            result = model(wav_tensor, lang, strategy)
        return result
    
    async def async_run(self, wav_tensor, lang, strategy):
        """Async run inference"""
        # For async, we need to ensure CUDA context is available in the thread
        # Use a thread pool executor that preserves CUDA context
        import asyncio
        import concurrent.futures
        
        # Create a thread pool that can handle CUDA
        loop = asyncio.get_event_loop()
        # Use a custom executor that maintains CUDA context
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        
        # Ensure CUDA context is set in the current thread before running
        if self.model_config["device"] == "cuda" and torch.cuda.is_available():
            # Initialize CUDA context in current thread
            _ = torch.cuda.current_device()
        
        return await loop.run_in_executor(executor, self.run, wav_tensor, lang, strategy)

# Global storage for multipart data (thread-safe)
import threading
_multipart_storage = threading.local()

# Middleware to handle multipart form data before BentoML parses it
class MultipartMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "POST" and request.url.path == "/asr":
            content_type = request.headers.get("content-type", "")
            if "multipart/form-data" in content_type:
                # Parse multipart form data
                form = await request.form()
                file_field = form.get("file")
                if file_field:
                    file_bytes = await file_field.read()
                    # Store in request state
                    request.state._multipart_data = {
                        "file": file_bytes,
                        "lang": form.get("lang", "hi"),
                        "strategy": form.get("strategy", "ctc")
                    }
                    # Also store in thread-local storage as backup
                    _multipart_storage.data = request.state._multipart_data
        return await call_next(request)

# Create service class for BentoML 1.4+
class IndicConformerASR:
    def __init__(self):
        self.model_config = None
        self.model_runner = None
    
    def _get_model_runner(self):
        """Get or create model runner (lazy initialization)"""
        if self.model_runner is None:
            self.model_config = get_model_config()
            self.model_runner = ASRModelRunner(self.model_config)
        return self.model_runner
    
    @bentoml.api(route="/asr")
    async def predict(self, request: Any = None) -> dict:
        """
        Accepts an audio file (wav/flac) and returns transcription.
        Multipart data is parsed by middleware and stored in request state.
        
        Args:
            request: HTTP request (will be injected by BentoML or accessed from middleware)
        
        Returns:
            dict with transcription result
        """
        # Get request - try multiple methods
        if request is None:
            # Try to get from BentoML context
            try:
                from bentoml._internal import context as ctx
                request = ctx.request.get()
            except Exception as e:
                # Fallback: use thread-local storage from middleware
                request = None
        
        # Get multipart data from request state (set by middleware)
        # The middleware stores data in a global or we access it differently
        if request is not None and hasattr(request.state, '_multipart_data'):
            multipart_data = request.state._multipart_data
            file_bytes = multipart_data["file"]
            lang = multipart_data["lang"]
            strategy = multipart_data["strategy"]
        elif request is not None:
            # Fallback: try to parse form data directly
            try:
                form = await request.form()
                file_field = form.get("file")
                if not file_field:
                    return {"error": "No file provided"}
                file_bytes = await file_field.read()
                lang = form.get("lang", "hi")
                strategy = form.get("strategy", "ctc")
            except Exception as e:
                return {"error": f"Failed to parse request: {str(e)}"}
        else:
            # Try to get from thread-local storage set by middleware
            if hasattr(_multipart_storage, 'data'):
                multipart_data = _multipart_storage.data
                file_bytes = multipart_data["file"]
                lang = multipart_data["lang"]
                strategy = multipart_data["strategy"]
            else:
                return {"error": "Could not access request data. Make sure middleware is properly configured."}
        
        # Get model runner (lazy initialization)
        runner = self._get_model_runner()
        config = self.model_config
        
        # Create buffer from file bytes
        buffer = io.BytesIO(file_bytes)

        # Load audio using Python's built-in wave module (no FFmpeg required)
        # This works for standard WAV files without external dependencies
        import wave
        import numpy as np
        
        buffer.seek(0)
        with wave.open(buffer, 'rb') as wav_file:
            # Get audio parameters
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sr = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            
            # Read audio data
            audio_bytes = wav_file.readframes(n_frames)
            
            # Convert bytes to numpy array based on sample width
            if sample_width == 1:
                # 8-bit unsigned
                waveform_np = np.frombuffer(audio_bytes, dtype=np.uint8).astype(np.float32)
                waveform_np = (waveform_np - 128) / 128.0
            elif sample_width == 2:
                # 16-bit signed
                waveform_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                waveform_np = waveform_np / 32768.0
            elif sample_width == 4:
                # 32-bit signed
                waveform_np = np.frombuffer(audio_bytes, dtype=np.int32).astype(np.float32)
                waveform_np = waveform_np / 2147483648.0
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            
            # Reshape for multi-channel audio
            if n_channels > 1:
                waveform_np = waveform_np.reshape(n_frames, n_channels).T
            else:
                waveform_np = waveform_np.reshape(1, -1)
            
            # Convert to torch tensor
            waveform = torch.from_numpy(waveform_np)
            
            # Convert to mono if stereo
            if waveform.dim() == 2 and waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            elif waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
        
        # Convert to mono and expected shape
        if waveform.dim() == 2:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed to 16k (model expects 16kHz)
        target_sr = 16000
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        
        # Move to GPU if available and configured
        if config["device"] == "cuda" and torch.cuda.is_available():
            waveform = waveform.cuda()

        # Call the model runner
        # Model interface: model(wav_tensor, lang, strategy)
        # Returns transcription text
        transcription = await runner.async_run(waveform, lang, strategy)

        return {"text": transcription, "lang": lang, "strategy": strategy}

# Create BentoML service with middleware
svc = Service("indic_conformer_asr", inner=IndicConformerASR)
svc.add_asgi_middleware(MultipartMiddleware)
