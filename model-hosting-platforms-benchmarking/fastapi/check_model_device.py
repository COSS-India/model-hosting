#!/usr/bin/env python3
"""
Quick script to check where the model is actually loaded (CPU vs GPU)
"""
import os
import torch
from transformers import AutoModel

HF_TOKEN = os.environ.get("HF_TOKEN", "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")
MODEL_NAME = os.environ.get("MODEL_NAME", "ai4bharat/indic-conformer-600m-multilingual")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"DEVICE setting: {DEVICE}")

print(f"\nLoading model: {MODEL_NAME}")
model = AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None,
    token=HF_TOKEN,
    trust_remote_code=True
)

if DEVICE == "cpu":
    model = model.to(DEVICE)

print("\n=== Model Device Placement ===")
# Check where model parameters are
gpu_params = 0
cpu_params = 0
for name, param in model.named_parameters():
    if param.device.type == 'cuda':
        gpu_params += 1
    else:
        cpu_params += 1
    if gpu_params + cpu_params <= 5:  # Print first 5
        print(f"{name}: {param.device}")

print(f"\nTotal parameters on GPU: {gpu_params}")
print(f"Total parameters on CPU: {cpu_params}")

# Check if model has transcribe method
print(f"\nHas transcribe method: {hasattr(model, 'transcribe')}")
print(f"Has generate method: {hasattr(model, 'generate')}")
print(f"Is callable: {callable(model)}")


