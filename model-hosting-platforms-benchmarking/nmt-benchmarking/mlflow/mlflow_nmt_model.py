"""
MLflow PyFunc model wrapper for IndicTrans2 NMT model
"""
import os
import sys
import torch
import mlflow
import mlflow.pyfunc
from mlflow.pyfunc import PythonModel
import json

# Add IndicTrans2 to path if cloned
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "IndicTrans2", "huggingface_interface"))

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
from mosestokenizer import MosesSentenceSplitter
from nltk import sent_tokenize
from nltk.data import find
import nltk
from indicnlp.tokenize.sentence_tokenize import sentence_split, DELIM_PAT_NO_DANDA

# Download NLTK data if not already present
try:
    find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK punkt_tab...")
    nltk.download('punkt_tab', quiet=True)
try:
    find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt...")
    nltk.download('punkt', quiet=True)

# FLORES language code mapping
flores_codes = {
    "asm_Beng": "as", "awa_Deva": "hi", "ben_Beng": "bn", "bho_Deva": "hi",
    "brx_Deva": "hi", "doi_Deva": "hi", "eng_Latn": "en", "gom_Deva": "kK",
    "guj_Gujr": "gu", "hin_Deva": "hi", "hne_Deva": "hi", "kan_Knda": "kn",
    "kas_Arab": "ur", "kas_Deva": "hi", "kha_Latn": "en", "lus_Latn": "en",
    "mag_Deva": "hi", "mai_Deva": "hi", "mal_Mlym": "ml", "mar_Deva": "mr",
    "mni_Beng": "bn", "mni_Mtei": "hi", "npi_Deva": "ne", "ory_Orya": "or",
    "pan_Guru": "pa", "san_Deva": "hi", "sat_Olck": "or", "snd_Arab": "ur",
    "snd_Deva": "hi", "tam_Taml": "ta", "tel_Telu": "te", "urd_Arab": "ur",
}

# Model configurations
EN_INDIC_MODEL = "ai4bharat/indictrans2-en-indic-1B"
INDIC_EN_MODEL = "ai4bharat/indictrans2-indic-en-1B"
INDIC_INDIC_MODEL = "ai4bharat/indictrans2-indic-indic-1B"

BATCH_SIZE = 4

def get_device():
    """Get device to use"""
    return os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

def split_sentences(input_text, lang):
    """Split text into sentences"""
    if lang == "eng_Latn":
        with MosesSentenceSplitter(flores_codes[lang]) as splitter:
            sents_moses = splitter([input_text])
        sents_nltk = sent_tokenize(input_text)
        if len(sents_nltk) < len(sents_moses):
            input_sentences = sents_nltk
        else:
            input_sentences = sents_moses
        input_sentences = [sent.replace("\xad", "") for sent in input_sentences]
    else:
        input_sentences = sentence_split(
            input_text, lang=flores_codes[lang], delim_pat=DELIM_PAT_NO_DANDA
        )
    return input_sentences

def initialize_model_and_tokenizer(ckpt_dir, device):
    """Initialize model and tokenizer"""
    hf_token = os.environ.get("HF_TOKEN", None)
    token_kwargs = {"token": hf_token} if hf_token else {}
    
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        **token_kwargs
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        **token_kwargs
    )
    
    if device == "cuda":
        model = model.to(device)
        model.half()
    
    model.eval()
    return tokenizer, model

class IndicTransNMTModel(PythonModel):
    """MLflow PyFunc model for IndicTrans2 NMT"""
    
    def load_context(self, context):
        """Load models when MLflow loads the model"""
        self.device = get_device()
        print(f"Loading IndicTrans2 models on device: {self.device}")
        
        # Authenticate with HuggingFace if token is provided
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            try:
                from huggingface_hub import login
                login(token=hf_token, add_to_git_credential=False)
                print(f"Authenticated with HuggingFace Hub")
            except Exception as e:
                print(f"Warning: Could not authenticate with HuggingFace: {e}")
        
        # Lazy loading - models will be loaded on first use
        self.en_indic_tokenizer = None
        self.en_indic_model = None
        self.indic_en_tokenizer = None
        self.indic_en_model = None
        self.indic_indic_tokenizer = None
        self.indic_indic_model = None
        self.ip = IndicProcessor(inference=True)
        print("Model wrapper initialized")
    
    def _load_models_for_pair(self, src_lang, tgt_lang):
        """Lazy load models based on language pair"""
        if src_lang == "eng_Latn" and tgt_lang != "eng_Latn":
            if self.en_indic_model is None:
                print(f"Loading En-Indic model: {EN_INDIC_MODEL}")
                self.en_indic_tokenizer, self.en_indic_model = initialize_model_and_tokenizer(
                    EN_INDIC_MODEL, self.device
                )
            return self.en_indic_model, self.en_indic_tokenizer
        elif src_lang != "eng_Latn" and tgt_lang == "eng_Latn":
            if self.indic_en_model is None:
                print(f"Loading Indic-En model: {INDIC_EN_MODEL}")
                self.indic_en_tokenizer, self.indic_en_model = initialize_model_and_tokenizer(
                    INDIC_EN_MODEL, self.device
                )
            return self.indic_en_model, self.indic_en_tokenizer
        elif src_lang != "eng_Latn" and tgt_lang != "eng_Latn":
            if self.indic_indic_model is None:
                print(f"Loading Indic-Indic model: {INDIC_INDIC_MODEL}")
                self.indic_indic_tokenizer, self.indic_indic_model = initialize_model_and_tokenizer(
                    INDIC_INDIC_MODEL, self.device
                )
            return self.indic_indic_model, self.indic_indic_tokenizer
        else:
            raise ValueError(f"Unsupported language pair: {src_lang} -> {tgt_lang}")
    
    def predict(self, context, model_input):
        """
        Predict translation
        
        model_input can be:
        - pandas DataFrame: columns should be "text", "src_lang", "tgt_lang"
        - Dict: {"text": "...", "src_lang": "...", "tgt_lang": "..."}
        - List of dicts
        """
        # Handle pandas DataFrame (MLflow converts inputs to DataFrame)
        try:
            import pandas as pd
            if isinstance(model_input, pd.DataFrame):
                # Extract values from DataFrame
                if len(model_input) == 0:
                    return []
                # Get first row (or handle multiple rows)
                row = model_input.iloc[0]
                text = str(row.get("text", ""))
                src_lang = str(row.get("src_lang", "eng_Latn"))
                tgt_lang = str(row.get("tgt_lang", "hin_Deva"))
            elif isinstance(model_input, dict):
                text = str(model_input.get("text", ""))
                src_lang = str(model_input.get("src_lang", "eng_Latn"))
                tgt_lang = str(model_input.get("tgt_lang", "hin_Deva"))
            elif isinstance(model_input, list):
                # Handle batch of requests
                if len(model_input) == 0:
                    return []
                if isinstance(model_input[0], dict):
                    return [self.predict(context, item) for item in model_input]
                else:
                    text = str(model_input[0]) if model_input else ""
                    src_lang = "eng_Latn"
                    tgt_lang = "hin_Deva"
            elif isinstance(model_input, str):
                try:
                    data = json.loads(model_input)
                    text = str(data.get("text", ""))
                    src_lang = str(data.get("src_lang", "eng_Latn"))
                    tgt_lang = str(data.get("tgt_lang", "hin_Deva"))
                except json.JSONDecodeError:
                    text = model_input
                    src_lang = "eng_Latn"
                    tgt_lang = "hin_Deva"
            else:
                # Try to convert to string and extract
                text = str(model_input)
                src_lang = "eng_Latn"
                tgt_lang = "hin_Deva"
        except Exception as e:
            # Fallback handling
            if isinstance(model_input, dict):
                text = str(model_input.get("text", ""))
                src_lang = str(model_input.get("src_lang", "eng_Latn"))
                tgt_lang = str(model_input.get("tgt_lang", "hin_Deva"))
            else:
                text = str(model_input)
                src_lang = "eng_Latn"
                tgt_lang = "hin_Deva"
        
        if not text:
            return {"translation": "", "src_lang": src_lang, "tgt_lang": tgt_lang}
        
        # Load appropriate model
        model, tokenizer = self._load_models_for_pair(src_lang, tgt_lang)
        
        # Split sentences
        input_sentences = split_sentences(text, src_lang)
        translations = []
        
        # Process in batches
        for i in range(0, len(input_sentences), BATCH_SIZE):
            batch = input_sentences[i : i + BATCH_SIZE]
            batch = self.ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)
            inputs = tokenizer(
                batch, truncation=True, padding="longest", return_tensors="pt", return_attention_mask=True,
            ).to(self.device)

            with torch.no_grad():
                generated_tokens = model.generate(
                    **inputs,
                    use_cache=False,
                    min_length=0,
                    max_length=256,
                    num_beams=5,
                    num_return_sequences=1,
                    early_stopping=True,
                )
            generated_tokens = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True,
            )
            translations += self.ip.postprocess_batch(generated_tokens, lang=tgt_lang)
            del inputs
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        translation_text = " ".join(translations)
        return {
            "translation": translation_text,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang
        }
