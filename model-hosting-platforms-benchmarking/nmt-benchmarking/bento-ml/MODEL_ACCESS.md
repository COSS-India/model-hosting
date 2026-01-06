# Model Access Instructions

## Current Status

✅ **Working**: Hindi to English (`indictrans2-indic-en-1B`) - You have access
❌ **Not Working**: English to Indic (`indictrans2-en-indic-1B`) - **Access needed**
❓ **Unknown**: Indic to Indic (`indictrans2-indic-indic-1B`) - Need to test

## How to Request Access

The IndicTrans2 models are **gated** on HuggingFace, meaning you need explicit access approval.

### Step 1: Visit Each Model Page

1. **English to Indic Model** (Required):
   - Go to: https://huggingface.co/ai4bharat/indictrans2-en-indic-1B
   - Click the **"Request access"** button
   - Wait for approval (usually quick, sometimes instant)

2. **Indic to Indic Model** (For Indic-Indic translations):
   - Go to: https://huggingface.co/ai4bharat/indictrans2-indic-indic-1B
   - Click the **"Request access"** button
   - Wait for approval

3. **Indic to English Model** (Already working):
   - Go to: https://huggingface.co/ai4bharat/indictrans2-indic-en-1B
   - Verify you have access (should show "You have access to this model")

### Step 2: Verify Access

After requesting access, you can test if it's been granted:

```bash
cd /home/ubuntu/nmt-benchmarking/bento-ml
source bento/bin/activate

# Test English to Indic access
python3 <<EOF
from transformers import AutoTokenizer
import os
os.environ['HF_TOKEN'] = 'YOUR_HF_TOKEN_HERE'
try:
    tokenizer = AutoTokenizer.from_pretrained(
        'ai4bharat/indictrans2-en-indic-1B',
        token=os.environ['HF_TOKEN']
    )
    print("✓ Access granted to En-Indic model!")
except Exception as e:
    print(f"✗ Access denied: {e}")
EOF
```

### Step 3: Restart Service After Access is Granted

Once you have access to all models:

```bash
cd /home/ubuntu/nmt-benchmarking/bento-ml
pkill -f "bentoml serve"
export HF_TOKEN=YOUR_HF_TOKEN_HERE
./start_service.sh
```

## Troubleshooting

### Still Getting 403 After Requesting Access?

1. **Wait a few minutes** - Access approval can take time
2. **Log out and log back in** to HuggingFace
3. **Verify your token** is still valid at https://huggingface.co/settings/tokens
4. **Check email** for any approval notifications

### Check Current Access Status

Visit your HuggingFace profile and check which models you have access to:
- https://huggingface.co/models?author=ai4bharat

## Current Working Endpoints

While waiting for access, you can use:

- ✅ **Hindi to English**: Works now
- ❌ **English to Hindi**: Needs En-Indic model access
- ❌ **Indic-Indic translations**: Needs Indic-Indic model access

