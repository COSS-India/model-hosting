import json
import numpy as np
import triton_python_backend_utils as pb_utils
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import fasttext
import re
import os


class IndicLID_BERT_Model(nn.Module):
    """IndicLID BERT model for language identification."""
    
    def __init__(self, classes=47):
        super(IndicLID_BERT_Model, self).__init__()
        self.bert = AutoModel.from_pretrained("ai4bharat/IndicBERTv2-MLM-only")
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, classes)
        
    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return type('Outputs', (), {'logits': logits})()


class TritonPythonModel:
    """
    Triton Python backend model for Language Identification.
    Supports 47 language classes (24 native-script and 21 roman-script plus English and Others).
    """

    def initialize(self, args):
        """
        Called once when model is loaded.
        Loads the IndicLID models (FTN, FTR, and BERT).

        Args:
            args (dict): Contains model configuration
        """
        # Parse model configuration
        self.model_config = json.loads(args["model_config"])

        # Get output configuration for type conversion
        output_config = pb_utils.get_output_config_by_name(
            self.model_config, "OUTPUT_TEXT"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(
            output_config["data_type"]
        )

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[OK] Using device: {self.device}", flush=True)

        # Model paths
        model_dir = "/models/indiclid/1"
        self.IndicLID_FTN_path = os.path.join(model_dir, 'indiclid-ftn.bin')
        self.IndicLID_FTR_path = os.path.join(model_dir, 'indiclid-ftr.bin')
        self.IndicLID_BERT_path = os.path.join(model_dir, 'indiclid-bert.pt')

        # Load FastText models
        print(f"Loading IndicLID-FTN model from {self.IndicLID_FTN_path}...", flush=True)
        self.IndicLID_FTN = fasttext.load_model(self.IndicLID_FTN_path)
        print("[OK] IndicLID-FTN loaded successfully", flush=True)

        print(f"Loading IndicLID-FTR model from {self.IndicLID_FTR_path}...", flush=True)
        self.IndicLID_FTR = fasttext.load_model(self.IndicLID_FTR_path)
        print("[OK] IndicLID-FTR loaded successfully", flush=True)

        # Load BERT model
        print(f"Loading IndicLID-BERT model from {self.IndicLID_BERT_path}...", flush=True)
        self.IndicLID_BERT = torch.load(self.IndicLID_BERT_path, map_location=self.device)
        self.IndicLID_BERT.eval()
        print("[OK] IndicLID-BERT loaded successfully", flush=True)

        # Load tokenizer
        print("Loading IndicBERT tokenizer...", flush=True)
        self.IndicLID_BERT_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBERTv2-MLM-only")
        print("[OK] Tokenizer loaded successfully", flush=True)

        # Thresholds
        self.input_threshold = 0.5  # Threshold for roman vs native script detection
        self.model_threshold = 0.6  # Threshold for FTR confidence
        self.classes = 47

        # Language code mappings
        self.IndicLID_lang_code_dict_reverse = {
            0: 'asm_Latn', 1: 'ben_Latn', 2: 'brx_Latn', 3: 'guj_Latn',
            4: 'hin_Latn', 5: 'kan_Latn', 6: 'kas_Latn', 7: 'kok_Latn',
            8: 'mai_Latn', 9: 'mal_Latn', 10: 'mni_Latn', 11: 'mar_Latn',
            12: 'nep_Latn', 13: 'ori_Latn', 14: 'pan_Latn', 15: 'san_Latn',
            16: 'snd_Latn', 17: 'tam_Latn', 18: 'tel_Latn', 19: 'urd_Latn',
            20: 'eng_Latn', 21: 'other', 22: 'asm_Beng', 23: 'ben_Beng',
            24: 'brx_Deva', 25: 'doi_Deva', 26: 'guj_Gujr', 27: 'hin_Deva',
            28: 'kan_Knda', 29: 'kas_Arab', 30: 'kas_Deva', 31: 'kok_Deva',
            32: 'mai_Deva', 33: 'mal_Mlym', 34: 'mni_Beng', 35: 'mni_Meti',
            36: 'mar_Deva', 37: 'nep_Deva', 38: 'ori_Orya', 39: 'pan_Guru',
            40: 'san_Deva', 41: 'sat_Olch', 42: 'snd_Arab', 43: 'tam_Tamil',
            44: 'tel_Telu', 45: 'urd_Arab'
        }

        print(f"[OK] IndicLID model initialized with {self.classes} language classes", flush=True)

    def char_percent_check(self, input_text):
        """
        Check whether input has input_threshold of roman characters.
        
        Args:
            input_text: Input text to check
            
        Returns:
            float: Percentage of roman characters (0.0 to 1.0)
        """
        # Count total number of characters in string
        input_len = len(list(input_text))

        # Count special characters, spaces and newlines
        special_char_pattern = re.compile('[@_!#$%^&*()<>?/\|}{~:]')
        special_char_matches = special_char_pattern.findall(input_text)
        special_chars = len(special_char_matches)
        
        spaces = len(re.findall(r'\s', input_text))
        newlines = len(re.findall(r'\n', input_text))

        # Subtract total-special character counts
        total_chars = input_len - (special_chars + spaces + newlines)

        # Count the number of English characters and digits
        en_pattern = re.compile('[a-zA-Z0-9]')
        en_matches = en_pattern.findall(input_text)
        en_chars = len(en_matches)

        # Calculate the percentage of English characters
        if total_chars == 0:
            return 0
        return (en_chars / total_chars)

    def native_inference(self, input_text):
        """
        Perform inference using IndicLID-FTN for native script text.
        
        Args:
            input_text: Input text
            
        Returns:
            tuple: (predicted_language, confidence_score, model_name)
        """
        prediction = self.IndicLID_FTN.predict(input_text)
        pred_label = prediction[0][0][9:]  # Remove '__label__' prefix
        pred_score = float(prediction[1][0])
        return (pred_label, pred_score, 'IndicLID-FTN')

    def roman_inference(self, input_text):
        """
        Perform inference using IndicLID-FTR and IndicLID-BERT for roman script text.
        Two-stage approach: FTR first, then BERT if confidence is low.
        
        Args:
            input_text: Input text
            
        Returns:
            tuple: (predicted_language, confidence_score, model_name)
        """
        # Stage 1: FastText Roman model
        prediction = self.IndicLID_FTR.predict(input_text)
        pred_label = prediction[0][0][9:]  # Remove '__label__' prefix
        pred_score = float(prediction[1][0])
        
        # If confidence is high enough, return FTR result
        if pred_score > self.model_threshold:
            return (pred_label, pred_score, 'IndicLID-FTR')
        
        # Stage 2: BERT model for low-confidence cases
        return self.bert_inference(input_text)

    def bert_inference(self, input_text):
        """
        Perform inference using IndicLID-BERT model.
        
        Args:
            input_text: Input text
            
        Returns:
            tuple: (predicted_language, confidence_score, model_name)
        """
        with torch.no_grad():
            # Tokenize input
            word_embeddings = self.IndicLID_BERT_tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Move to device
            word_embeddings = {k: v.to(self.device) for k, v in word_embeddings.items()}
            
            # Get predictions
            outputs = self.IndicLID_BERT(
                word_embeddings['input_ids'],
                token_type_ids=word_embeddings['token_type_ids'],
                attention_mask=word_embeddings['attention_mask']
            )
            
            # Get predicted class and score
            _, predicted = torch.max(outputs.logits, 1)
            pred_label = self.IndicLID_lang_code_dict_reverse[predicted.item()]
            pred_score = float(outputs.logits[0][predicted.item()].item())
            
            return (pred_label, pred_score, 'IndicLID-BERT')

    def execute(self, requests):
        """
        Called for each inference request.

        Args:
            requests (list): List of pb_utils.InferenceRequest objects

        Returns:
            list: List of pb_utils.InferenceResponse objects
        """
        responses = []

        for request in requests:
            # Get input tensor
            input_text_tensor = pb_utils.get_input_tensor_by_name(
                request, "INPUT_TEXT"
            )

            # Convert to numpy array
            input_texts = input_text_tensor.as_numpy()

            # Process each text in the batch
            output_results = []

            for idx, input_text_bytes in enumerate(input_texts):
                try:
                    # Decode input text
                    input_text = input_text_bytes[0]
                    if isinstance(input_text, bytes):
                        input_text = input_text.decode('utf-8')

                    print(f"Processing language detection for text length: {len(input_text)} chars", flush=True)

                    # Determine if text is roman or native script
                    roman_percent = self.char_percent_check(input_text)
                    
                    if roman_percent > self.input_threshold:
                        # Roman script - use FTR/BERT pipeline
                        pred_label, pred_score, model_name = self.roman_inference(input_text)
                    else:
                        # Native script - use FTN model
                        pred_label, pred_score, model_name = self.native_inference(input_text)

                    print(f"Detected language: {pred_label} (score: {pred_score:.4f}, model: {model_name})", flush=True)

                    # Format output as JSON string
                    output_json = json.dumps({
                        "input": input_text,
                        "langCode": pred_label,
                        "confidence": pred_score,
                        "model": model_name
                    })

                    output_results.append([output_json])

                except Exception as e:
                    # If there's an error, return error message
                    import traceback
                    error_msg = f"Error processing language detection: {str(e)}"
                    print(error_msg, flush=True)
                    print(traceback.format_exc(), flush=True)
                    
                    # Return error in expected format
                    error_json = json.dumps({
                        "input": "",
                        "langCode": "error",
                        "confidence": 0.0,
                        "model": "error",
                        "error": error_msg
                    })
                    output_results.append([error_json])

            # Create output tensor
            output_array = np.array(output_results, dtype=self.output_dtype)

            out_tensor = pb_utils.Tensor(
                "OUTPUT_TEXT",
                output_array
            )

            # Create response
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[out_tensor])
            )

        return responses

    def finalize(self):
        """
        Called when model is unloaded.
        Clean up resources.
        """
        print("Cleaning up IndicLID model resources", flush=True)
        del self.IndicLID_FTN
        del self.IndicLID_FTR
        del self.IndicLID_BERT
        del self.IndicLID_BERT_tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

