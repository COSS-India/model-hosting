# Issues Fixed in download_model.py

## Problems Identified:

### 1. **❌ Trying to load a tokenizer that doesn't exist**
   - **Issue**: `AutoTokenizer.from_pretrained()` - This is an ASR (speech) model, not a text model
   - **Fix**: Removed tokenizer loading entirely
   - **Reason**: ASR models process audio directly, they don't use tokenizers

### 2. **❌ Missing HF_TOKEN authentication**
   - **Issue**: HF_TOKEN was commented out but the model requires authentication
   - **Fix**: Added required HF_TOKEN check with error if missing
   - **Reason**: The model is gated and requires HuggingFace authentication

### 3. **❌ No device/GPU configuration**
   - **Issue**: Model wasn't configured for GPU usage
   - **Fix**: Added `torch_dtype` and `device_map` parameters
   - **Reason**: For optimal performance, model should use GPU if available

### 4. **❌ Incorrect model interface understanding**
   - **Issue**: Code assumed standard PyTorch model interface
   - **Fix**: Added comments explaining actual interface: `model(wav_tensor, lang, strategy)`
   - **Reason**: This ONNX model has a custom interface, not standard forward()

### 5. **❌ Saving non-existent tokenizer**
   - **Issue**: Trying to save a tokenizer that was never successfully loaded
   - **Fix**: Removed tokenizer saving code
   - **Reason**: No tokenizer exists for this model

### 6. **❌ Missing model type information**
   - **Issue**: No indication this is an ONNX model, not pure PyTorch
   - **Fix**: Added comments explaining it's ONNX-based
   - **Reason**: Important for understanding model behavior and limitations

## Key Changes:

1. ✅ Removed `AutoTokenizer` import and usage
2. ✅ Added required `HF_TOKEN` validation
3. ✅ Added GPU/device configuration
4. ✅ Removed tokenizer saving code
5. ✅ Added documentation about model interface
6. ✅ Added proper error handling

## Usage:

```bash
export HF_TOKEN=your_huggingface_token_here
python download_model.py
```

## Model Interface:

The model uses a custom interface:
```python
transcription = model(wav_tensor, lang="hi", strategy="ctc")
# or
transcription = model(wav_tensor, lang="hi", strategy="rnnt")
```

Where:
- `wav_tensor`: Audio waveform as torch tensor
- `lang`: Language code (e.g., "hi", "en", "ta", "te", "mr")
- `strategy`: Decoding strategy - "ctc" or "rnnt"







