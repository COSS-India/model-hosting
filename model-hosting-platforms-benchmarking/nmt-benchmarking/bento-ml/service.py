# service.py - BentoML service for IndicTrans2 NMT model
import os
import sys
import torch
import asyncio
import concurrent.futures

# Add IndicTrans2 to path
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

# Authenticate with HuggingFace if token is provided
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    try:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
        # Also set as environment variable for transformers library
        os.environ["HF_TOKEN"] = hf_token
        print(f"Authenticated with HuggingFace Hub (token: {hf_token[:10]}...)")
    except Exception as e:
        print(f"Warning: Could not authenticate with HuggingFace: {e}")

import bentoml
from bentoml import Service

# Model configurations
EN_INDIC_MODEL = "ai4bharat/indictrans2-en-indic-1B"
INDIC_EN_MODEL = "ai4bharat/indictrans2-indic-en-1B"
INDIC_INDIC_MODEL = "ai4bharat/indictrans2-indic-indic-1B"

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

def get_device():
    """Get device to use"""
    return os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

def split_sentences(input_text, lang):
    """Split text into sentences"""
    if lang == "eng_Latn":
        input_sentences = sent_tokenize(input_text)
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

class IndicTransNMTModel:
    """Model runner for IndicTrans2"""
    
    def __init__(self):
        self.en_indic_model = None
        self.en_indic_tokenizer = None
        self.indic_en_model = None
        self.indic_en_tokenizer = None
        self.indic_indic_model = None
        self.indic_indic_tokenizer = None
        self.indicator_processor = None
        self.device = get_device()
        print(f"Using device: {self.device}")
    
    def _load_processor(self):
        """Load IndicProcessor"""
        if self.indicator_processor is None:
            self.indicator_processor = IndicProcessor(inference=True)
        return self.indicator_processor
    
    def _load_model(self, model_name, model_key):
        """Lazy load model"""
        # Get HuggingFace token from environment
        hf_token = os.environ.get("HF_TOKEN", None)
        token_kwargs = {"token": hf_token} if hf_token else {}
        
        if model_key == "en_indic" and self.en_indic_model is None:
            print(f"Loading {model_name}...")
            self.en_indic_tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True,
                **token_kwargs
            )
            self.en_indic_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                **token_kwargs
            )
            if self.device == "cuda":
                self.en_indic_model = self.en_indic_model.to(self.device)
                self.en_indic_model.half()
            self.en_indic_model.eval()
            print(f"Loaded {model_name}")
        
        elif model_key == "indic_en" and self.indic_en_model is None:
            print(f"Loading {model_name}...")
            self.indic_en_tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True,
                **token_kwargs
            )
            self.indic_en_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                **token_kwargs
            )
            if self.device == "cuda":
                self.indic_en_model = self.indic_en_model.to(self.device)
                self.indic_en_model.half()
            self.indic_en_model.eval()
            print(f"Loaded {model_name}")
        
        elif model_key == "indic_indic" and self.indic_indic_model is None:
            print(f"Loading {model_name}...")
            self.indic_indic_tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True,
                **token_kwargs
            )
            self.indic_indic_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                **token_kwargs
            )
            if self.device == "cuda":
                self.indic_indic_model = self.indic_indic_model.to(self.device)
                self.indic_indic_model.half()
            self.indic_indic_model.eval()
            print(f"Loaded {model_name}")
    
    def _determine_model_key(self, src_lang: str, tgt_lang: str) -> str:
        """Determine which model to use based on languages"""
        eng = "eng_Latn"
        if src_lang == eng:
            return "en_indic"
        elif tgt_lang == eng:
            return "indic_en"
        else:
            return "indic_indic"
    
    def _batch_translate(self, input_sentences, src_lang, tgt_lang, model, tokenizer, ip, batch_size=4):
        """Batch translate sentences"""
        translations = []
        for i in range(0, len(input_sentences), batch_size):
            batch = input_sentences[i : i + batch_size]
            
            # Preprocess the batch
            batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)
            
            # Tokenize
            inputs = tokenizer(
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                generated_tokens = model.generate(
                    **inputs,
                    use_cache=False,  # Disable cache to avoid past_key_values issues
                    min_length=0,
                    max_length=256,
                    num_beams=5,
                    num_return_sequences=1,
                    early_stopping=True,
                )
            
            # Decode
            generated_tokens = tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            
            # Postprocess
            translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)
            
            del inputs
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        return translations
    
    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Translate text"""
        ip = self._load_processor()
        model_key = self._determine_model_key(src_lang, tgt_lang)
        
        if model_key == "en_indic":
            self._load_model(EN_INDIC_MODEL, "en_indic")
            input_sentences = split_sentences(text, src_lang)
            translations = self._batch_translate(
                input_sentences, src_lang, tgt_lang,
                self.en_indic_model, self.en_indic_tokenizer, ip
            )
            return " ".join(translations)
        
        elif model_key == "indic_en":
            self._load_model(INDIC_EN_MODEL, "indic_en")
            input_sentences = split_sentences(text, src_lang)
            translations = self._batch_translate(
                input_sentences, src_lang, tgt_lang,
                self.indic_en_model, self.indic_en_tokenizer, ip
            )
            return " ".join(translations)
        
        else:  # indic_indic
            self._load_model(INDIC_INDIC_MODEL, "indic_indic")
            input_sentences = split_sentences(text, src_lang)
            translations = self._batch_translate(
                input_sentences, src_lang, tgt_lang,
                self.indic_indic_model, self.indic_indic_tokenizer, ip
            )
            return " ".join(translations)

# Create service class
class IndicTransNMT:
    def __init__(self):
        self.model_runner = None
    
    def _get_model_runner(self):
        if self.model_runner is None:
            self.model_runner = IndicTransNMTModel()
        return self.model_runner
    
    @bentoml.api(route="/translate")
    async def translate(self, text: str, src_lang: str, tgt_lang: str) -> dict:
        """
        Translate text from source language to target language.
        Supports: English ↔ Indic, Indic ↔ Indic
        
        Language codes format: "eng_Latn" (English), "hin_Deva" (Hindi), "mar_Deva" (Marathi), etc.
        
        Examples:
        - English to Hindi: src_lang="eng_Latn", tgt_lang="hin_Deva"
        - Hindi to English: src_lang="hin_Deva", tgt_lang="eng_Latn"
        - Hindi to Marathi: src_lang="hin_Deva", tgt_lang="mar_Deva"
        """
        runner = self._get_model_runner()
        
        loop = asyncio.get_event_loop()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        
        if runner.device == "cuda" and torch.cuda.is_available():
            _ = torch.cuda.current_device()
        
        translation = await loop.run_in_executor(
            executor, runner.translate, text, src_lang, tgt_lang
        )
        
        return {
            "translation": translation,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang
        }

# Create BentoML service
svc = Service("indictrans_nmt", inner=IndicTransNMT)
