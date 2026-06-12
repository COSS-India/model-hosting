import numpy as np
import triton_python_backend_utils as pb_utils
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

# short code -> IndicTrans2 FLORES code
SHORT_TO_FLORES = {
    "en": "eng_Latn", "as": "asm_Beng", "bn": "ben_Beng", "brx": "brx_Deva",
    "doi": "doi_Deva", "gu": "guj_Gujr", "hi": "hin_Deva", "kn": "kan_Knda",
    "ks": "kas_Arab", "kok": "gom_Deva", "mai": "mai_Deva", "ml": "mal_Mlym",
    "mni": "mni_Mtei", "mr": "mar_Deva", "ne": "npi_Deva", "or": "ory_Orya",
    "pa": "pan_Guru", "sa": "san_Deva", "sat": "sat_Olck", "sd": "snd_Arab",
    "ta": "tam_Taml", "te": "tel_Telu", "ur": "urd_Arab",
}
FLORES_SET = set(SHORT_TO_FLORES.values())
ENGLISH = "eng_Latn"

CKPTS = {
    "en-indic":    "ai4bharat/indictrans2-en-indic-dist-200M",
    "indic-en":    "ai4bharat/indictrans2-indic-en-dist-200M",
    "indic-indic": "ai4bharat/indictrans2-indic-indic-dist-320M",
}

# Beam width. 5 = best quality (slower on CPU). Lower to 1 for faster testing.
NUM_BEAMS = 5


def normalize(code):
    code = code.strip()
    if code in FLORES_SET:
        return code
    if code in SHORT_TO_FLORES:
        return SHORT_TO_FLORES[code]
    if code.lower() in SHORT_TO_FLORES:
        return SHORT_TO_FLORES[code.lower()]
    raise ValueError("Unknown language code: %r" % code)


class TritonPythonModel:
    def initialize(self, args):
        self.device = "cpu"
        self.ip = IndicProcessor(inference=True)
        self.loaded = {}  # direction -> (tokenizer, model), loaded lazily

    def _route(self, src, tgt):
        s, t = (src == ENGLISH), (tgt == ENGLISH)
        if s and not t:
            return "en-indic"
        if t and not s:
            return "indic-en"
        if not s and not t:
            return "indic-indic"
        return "identity"

    def _get(self, direction):
        if direction not in self.loaded:
            name = CKPTS[direction]
            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                name,
                trust_remote_code=True,
                torch_dtype=torch.float32,     # CPU needs float32, not float16
                attn_implementation="eager",   # not flash_attention_2 (GPU-only)
            ).to(self.device)
            model.eval()
            self.loaded[direction] = (tok, model)
        return self.loaded[direction]

    def _translate(self, text, src_raw, tgt_raw):
        src, tgt = normalize(src_raw), normalize(tgt_raw)
        direction = self._route(src, tgt)
        if direction == "identity":
            return text
        tok, model = self._get(direction)
        batch = self.ip.preprocess_batch([text], src_lang=src, tgt_lang=tgt)
        inputs = tok(batch, truncation=True, padding="longest",
                     return_tensors="pt", return_attention_mask=True).to(self.device)
        with torch.no_grad():
            gen = model.generate(**inputs, use_cache=True, min_length=0,
                                 max_length=256, num_beams=NUM_BEAMS, num_return_sequences=1)
        decoded = tok.batch_decode(gen, skip_special_tokens=True)
        return self.ip.postprocess_batch(decoded, lang=tgt)[0]

    @staticmethod
    def _as_str(np_val):
        # KServe clients may send shape [1, 1]; flatten to a single scalar string
        v = np_val.flatten()[0]
        return v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)

    def execute(self, requests):
        responses = []
        for request in requests:
            text = self._as_str(pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT").as_numpy())
            src  = self._as_str(pb_utils.get_input_tensor_by_name(request, "INPUT_LANGUAGE_ID").as_numpy())
            tgt  = self._as_str(pb_utils.get_input_tensor_by_name(request, "OUTPUT_LANGUAGE_ID").as_numpy())
            out = self._translate(text, src, tgt)
            out_tensor = pb_utils.Tensor(
                "OUTPUT_TEXT",
                np.array([[out.encode("utf-8")]], dtype=object),
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
        return responses
