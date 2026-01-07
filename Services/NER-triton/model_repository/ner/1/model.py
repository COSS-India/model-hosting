import json
import numpy as np
import triton_python_backend_utils as pb_utils
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import os
from huggingface_hub import login


class TritonPythonModel:
    """
    Triton Python backend model for Named Entity Recognition (NER).
    Supports 11 Indian languages using ai4bharat/IndicNER model.
    """

    def initialize(self, args):
        """
        Called once when model is loaded.
        Loads the IndicNER model and tokenizer.

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

        # Authenticate with HuggingFace if token is available (model is gated)
        hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            print("Authenticating with HuggingFace...", flush=True)
            login(token=hf_token)
            print("[OK] HuggingFace authentication successful", flush=True)
        else:
            print("WARNING: No HuggingFace token found. Model download may fail if gated.", flush=True)

        # Model name - IndicNER for 11 Indian languages
        model_name = "ai4bharat/IndicNER"

        # Load tokenizer and model
        print(f"Loading NER model: {model_name}", flush=True)
        print("This may take a few minutes on first run...", flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, 
            token=hf_token
        )

        # Move model to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"[OK] Model loaded successfully on device: {self.device}", flush=True)

        # Supported languages
        self.supported_languages = [
            "as",  # Assamese
            "bn",  # Bengali
            "gu",  # Gujarati
            "hi",  # Hindi
            "kn",  # Kannada
            "ml",  # Malayalam
            "mr",  # Marathi
            "or",  # Oriya
            "pa",  # Punjabi
            "ta",  # Tamil
            "te",  # Telugu
        ]
        print(f"[OK] Supported languages: {self.supported_languages}", flush=True)

        # Get label list from model config
        self.id2label = self.model.config.id2label
        print(f"[OK] NER labels: {list(self.id2label.values())}", flush=True)

    def aggregate_subword_entities(self, tokens, predictions, scores):
        """
        Aggregate subword tokens into complete words with their NER tags.
        
        Args:
            tokens: List of tokens from tokenizer
            predictions: List of predicted label IDs
            scores: List of confidence scores
            
        Returns:
            List of entities with format: {"entity": text, "class": label, "score": confidence}
        """
        entities = []
        current_entity = None
        current_tokens = []
        current_scores = []
        
        for token, pred_id, score in zip(tokens, predictions, scores):
            # Skip special tokens
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
                
            label = self.id2label[pred_id]
            
            # Handle BIO tagging scheme
            if label.startswith('B-'):
                # Save previous entity if exists
                if current_entity is not None:
                    entity_text = self.tokenizer.convert_tokens_to_string(current_tokens)
                    entity_text = entity_text.strip()
                    if entity_text:
                        entities.append({
                            "entity": entity_text,
                            "class": current_entity,
                            "score": float(np.mean(current_scores))
                        })
                
                # Start new entity
                current_entity = label[2:]  # Remove 'B-' prefix
                current_tokens = [token]
                current_scores = [score]
                
            elif label.startswith('I-') and current_entity == label[2:]:
                # Continue current entity
                current_tokens.append(token)
                current_scores.append(score)
                
            else:
                # 'O' tag or different entity - save previous entity
                if current_entity is not None:
                    entity_text = self.tokenizer.convert_tokens_to_string(current_tokens)
                    entity_text = entity_text.strip()
                    if entity_text:
                        entities.append({
                            "entity": entity_text,
                            "class": current_entity,
                            "score": float(np.mean(current_scores))
                        })
                    current_entity = None
                    current_tokens = []
                    current_scores = []
        
        # Don't forget the last entity
        if current_entity is not None:
            entity_text = self.tokenizer.convert_tokens_to_string(current_tokens)
            entity_text = entity_text.strip()
            if entity_text:
                entities.append({
                    "entity": entity_text,
                    "class": current_entity,
                    "score": float(np.mean(current_scores))
                })
        
        return entities

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
            # Get input tensors
            input_text_tensor = pb_utils.get_input_tensor_by_name(
                request, "INPUT_TEXT"
            )
            lang_id_tensor = pb_utils.get_input_tensor_by_name(
                request, "LANG_ID"
            )

            # Convert to numpy arrays
            input_texts = input_text_tensor.as_numpy()
            lang_ids = lang_id_tensor.as_numpy()

            # Process each text in the batch
            output_results = []

            for idx, input_text_bytes in enumerate(input_texts):
                try:
                    # Decode input text
                    input_text = input_text_bytes[0]
                    if isinstance(input_text, bytes):
                        input_text = input_text.decode('utf-8')

                    # Get language ID
                    lang_id = lang_ids[idx][0]
                    if isinstance(lang_id, bytes):
                        lang_id = lang_id.decode('utf-8')

                    # Validate language ID
                    if lang_id not in self.supported_languages:
                        print(f"WARNING: Language '{lang_id}' not in supported list. Supported: {self.supported_languages}. Proceeding anyway...", flush=True)

                    # Log processing (avoid printing Unicode to prevent encoding issues)
                    print(f"Processing NER for language: {lang_id}, text length: {len(input_text)} chars", flush=True)

                    # Tokenize input
                    inputs = self.tokenizer(
                        input_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                        padding=True
                    )
                    
                    # Move to device
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    # Perform inference
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        logits = outputs.logits

                    # Get predictions
                    predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()
                    
                    # Get confidence scores (softmax probabilities)
                    probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                    scores = np.max(probabilities, axis=-1)

                    # Get tokens for aggregation
                    tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].cpu().numpy())

                    # Aggregate subword tokens into entities
                    entities = self.aggregate_subword_entities(tokens, predictions, scores)

                    print(f"Found {len(entities)} entities", flush=True)

                    # Format output as JSON string matching the expected API format
                    output_json = json.dumps({
                        "source": input_text,
                        "nerPrediction": entities
                    })

                    output_results.append([output_json])

                except Exception as e:
                    # If there's an error, return error message
                    import traceback
                    error_msg = f"Error processing NER: {str(e)}"
                    print(error_msg, flush=True)
                    print(traceback.format_exc(), flush=True)
                    
                    # Return error in expected format
                    error_json = json.dumps({
                        "source": "",
                        "nerPrediction": [],
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
        print("Cleaning up NER model resources", flush=True)
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

