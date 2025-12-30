# download_model.py
import os
import argparse
import json
import bentoml
from transformers import AutoModel
import torch
import torch.nn as nn

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
# make sure you have agreed to the model terms on HF if the repo is gated

def main(hf_token=None):
    # Get HF token from parameter or environment variable
    if hf_token:
        HF_TOKEN = hf_token
    else:
        HF_TOKEN = os.environ.get("HF_TOKEN")
    
    if not HF_TOKEN:
        raise ValueError(
            "HF_TOKEN is required. Provide it as parameter: --token your_token_here\n"
            "Or set environment variable: export HF_TOKEN=your_token_here"
        )
    
    # Load model to verify it works (trust_remote_code must be True for this repo)
    # NOTE: This model is ONNX-based and contains NeMo components that can't be serialized
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {MODEL_ID}")
    print(f"Using device: {device}")
    
    model = AutoModel.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    print("Model loaded successfully!")
    
    # The model interface is: model(wav_tensor, lang, strategy)
    # where:
    #   - wav_tensor: audio waveform tensor
    #   - lang: language code (e.g., "hi", "en", "ta")
    #   - strategy: decoding strategy ("ctc" or "rnnt")
    
    # Since the model contains NeMo components that can't be serialized,
    # we'll create a minimal PyTorch Module wrapper that stores the config
    # This wrapper can be serialized by BentoML
    class ModelConfigWrapper(nn.Module):
        """A minimal PyTorch Module that stores model configuration"""
        def __init__(self, model_id, hf_token, torch_dtype, device):
            super().__init__()
            # Store config as module attributes (these can be serialized)
            self.model_id = model_id
            self.hf_token = hf_token
            self.torch_dtype = torch_dtype
            self.device = device
            # Create a dummy parameter so it's recognized as a valid nn.Module
            self.dummy_param = nn.Parameter(torch.zeros(1))
        
        def forward(self, *args, **kwargs):
            # This won't be called, but required for nn.Module
            raise NotImplementedError("This wrapper only stores config. Model is reloaded in service.")
    
    # Create wrapper with config
    wrapper = ModelConfigWrapper(
        model_id=MODEL_ID,
        hf_token=HF_TOKEN,
        torch_dtype="float16" if device == "cuda" else "float32",
        device=device
    )
    
    # Save using BentoML's PyTorch framework (wrapper is a valid nn.Module)
    import bentoml.pytorch
    
    bentoml.pytorch.save_model(
        "indic_conformer_600m_model",
        wrapper,
        labels={
            "hf_id": MODEL_ID,
            "model_type": "asr",
            "framework": "transformers",
            "requires_reload": "true",  # Flag to reload from HF
            "device": device,
        },
        signatures={
            "__call__": {
                "batchable": False,
                # Model signature: (wav_tensor, lang: str, strategy: str) -> transcription
            }
        },
    )
    
    # Also save config as JSON file in the artifact for easy access
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    config_file = os.path.join(temp_dir, "model_config.json")
    
    model_config = {
        "model_id": MODEL_ID,
        "hf_token": HF_TOKEN,
        "torch_dtype": "float16" if device == "cuda" else "float32",
        "device": device,
    }
    
    with open(config_file, 'w') as f:
        json.dump(model_config, f, indent=2)
    
    try:
        # Copy config file to model artifact directory
        model_artifact = bentoml.models.get("indic_conformer_600m_model:latest")
        artifact_path = model_artifact.path
        shutil.copy(config_file, artifact_path)
        print(f"Saved config file to: {artifact_path}/model_config.json")
    finally:
        shutil.rmtree(temp_dir)
    
    print("\n✅ Saved model configuration to Bento modelstore.")
    print("NOTE: Model will be reloaded from HuggingFace in the service (avoids serialization issues).")
    print(f"Device configuration: {device}")
    print("\nModel interface: model(wav_tensor, lang, strategy)")
    print("  - wav_tensor: audio waveform tensor")
    print("  - lang: language code (e.g., 'hi', 'en', 'ta')")
    print("  - strategy: 'ctc' or 'rnnt'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and save ASR model to BentoML modelstore")
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (required if HF_TOKEN env var not set)"
    )
    args = parser.parse_args()
    
    main(hf_token=args.token)
