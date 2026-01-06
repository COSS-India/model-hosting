import mlflow.pyfunc
import base64, io
import os
import wave
import numpy as np
import torch
import torchaudio
import pandas as pd
from transformers import AutoModel

SAMPLE_RATE = 16000

class IndicConformerWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Get HF token from environment if available
        hf_token = os.environ.get("HF_TOKEN")
        
        model_kwargs = {
            "trust_remote_code": True
        }
        
        if hf_token:
            model_kwargs["token"] = hf_token
        
        self.model = AutoModel.from_pretrained(
            "ai4bharat/indic-conformer-600m-multilingual",
            **model_kwargs
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def _load_audio(self, b64):
        audio_bytes = base64.b64decode(b64)
        
        # Use Python's wave module instead of torchaudio to avoid torchcodec dependency
        try:
            with io.BytesIO(audio_bytes) as buffer:
                with wave.open(buffer, 'rb') as wf:
                    n_channels = wf.getnchannels()
                    sample_width = wf.getsampwidth()
                    sr = wf.getframerate()
                    n_frames = wf.getnframes()
                    audio_bytes_raw = wf.readframes(n_frames)

                    # Convert bytes to numpy array
                    if sample_width == 2:  # 16-bit signed
                        waveform_np = np.frombuffer(audio_bytes_raw, dtype=np.int16).astype(np.float32) / 32768.0
                    elif sample_width == 4:  # 32-bit signed
                        waveform_np = np.frombuffer(audio_bytes_raw, dtype=np.int32).astype(np.float32) / 2147483648.0
                    else:
                        raise ValueError(f"Unsupported sample width: {sample_width}")

                    # Handle multi-channel audio
                    if n_channels > 1:
                        waveform_np = waveform_np.reshape(n_frames, n_channels).T
                    else:
                        waveform_np = waveform_np.reshape(1, -1)
                    
                    # Convert to torch tensor
                    wav = torch.from_numpy(waveform_np)
        except Exception as e:
            # Fallback to torchaudio if wave module fails
            try:
                wav, sr = torchaudio.load(io.BytesIO(audio_bytes))
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
            except Exception:
                raise ValueError(f"Failed to load audio: {e}")

        # Resample if needed
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            wav = resampler(wav)

        return wav.to(self.device)

    def predict(self, context, model_input: pd.DataFrame):
        outputs = []

        for _, row in model_input.iterrows():
            wav = self._load_audio(row["audio_base64"])
            lang = row.get("lang", "hi")
            decoding = row.get("decoding", "ctc")

            with torch.no_grad():
                out = self.model(wav, lang, decoding)
                outputs.append({"transcription": out[0] if isinstance(out, list) else out})

        return pd.DataFrame(outputs)
