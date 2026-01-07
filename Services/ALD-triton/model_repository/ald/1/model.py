import json
import numpy as np
import triton_python_backend_utils as pb_utils
import torch
import torchaudio
import io
import base64
import os
import tempfile
import soundfile as sf
from speechbrain.inference.classifiers import EncoderClassifier


class TritonPythonModel:
    """
    Triton Python backend model for Audio Language Detection.
    Uses speechbrain/lang-id-voxlingua107-ecapa model to detect language from audio.
    Supports 107 languages.
    """

    def initialize(self, args):
        """
        Called once when model is loaded.
        Loads the VoxLingua107 ECAPA-TDNN language identification model.

        Args:
            args (dict): Contains model configuration
        """
        # Parse model configuration
        self.model_config = json.loads(args["model_config"])

        # Get output configurations for type conversion
        language_code_config = pb_utils.get_output_config_by_name(
            self.model_config, "LANGUAGE_CODE"
        )
        self.language_code_dtype = pb_utils.triton_string_to_numpy(
            language_code_config["data_type"]
        )

        confidence_config = pb_utils.get_output_config_by_name(
            self.model_config, "CONFIDENCE"
        )
        self.confidence_dtype = pb_utils.triton_string_to_numpy(
            confidence_config["data_type"]
        )

        all_scores_config = pb_utils.get_output_config_by_name(
            self.model_config, "ALL_SCORES"
        )
        self.all_scores_dtype = pb_utils.triton_string_to_numpy(
            all_scores_config["data_type"]
        )

        # Model name - VoxLingua107 ECAPA-TDNN for 107 languages
        model_name = "speechbrain/lang-id-voxlingua107-ecapa"

        # Load the language identification model
        print(f"Loading Audio Language Detection model: {model_name}", flush=True)
        print("This may take a few minutes on first run...", flush=True)

        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        run_opts = {"device": self.device} if self.device == "cuda" else {}

        # Load the model
        self.language_id = EncoderClassifier.from_hparams(
            source=model_name,
            savedir="tmp_ald_model",
            run_opts=run_opts
        )

        print(f"[OK] Model loaded successfully on device: {self.device}", flush=True)

        # The model automatically handles audio normalization (resampling to 16kHz, mono channel)
        print("[OK] Model ready for inference", flush=True)

    def decode_audio(self, audio_data_bytes):
        """
        Decode audio data from base64 encoded string or bytes.
        Saves to temp file and uses model's load_audio for proper normalization.
        
        Args:
            audio_data_bytes: Base64 encoded audio data or raw bytes
            
        Returns:
            torch.Tensor: Audio waveform tensor (normalized to 16kHz mono)
        """
        try:
            # Try to decode as base64 string first
            if isinstance(audio_data_bytes, bytes):
                audio_data_bytes = audio_data_bytes.decode('utf-8')
            
            # Decode base64
            audio_bytes = base64.b64decode(audio_data_bytes)
            
            # Save to temporary file and use model's load_audio method
            # This ensures proper normalization (16kHz, mono) as expected by the model
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file_path = tmp_file.name
            
            try:
                # Use model's load_audio method which handles all normalization
                waveform = self.language_id.load_audio(tmp_file_path)
                # Model normalizes to 16kHz mono, so sample_rate is always 16000
                sample_rate = 16000
                
                # Ensure proper shape: [channels, samples]
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                
                return waveform, sample_rate
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
            
        except Exception as e:
            print(f"Error decoding audio: {e}", flush=True)
            # Try loading as file path if base64 decode fails
            if isinstance(audio_data_bytes, str) and os.path.exists(audio_data_bytes):
                try:
                    # Use model's load_audio method for proper normalization
                    waveform = self.language_id.load_audio(audio_data_bytes)
                    return waveform, 16000  # Model normalizes to 16kHz
                except:
                    pass
            raise

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
            audio_data_tensor = pb_utils.get_input_tensor_by_name(
                request, "AUDIO_DATA"
            )

            # Convert to numpy array
            audio_data_array = audio_data_tensor.as_numpy()

            # Process each audio in the batch
            language_codes = []
            confidences = []
            all_scores_list = []

            for idx, audio_data_bytes in enumerate(audio_data_array):
                try:
                    # Get audio data
                    audio_data = audio_data_bytes[0]
                    if isinstance(audio_data, bytes):
                        audio_data = audio_data
                    elif isinstance(audio_data, str):
                        audio_data = audio_data.encode('utf-8')

                    # Decode and load audio
                    waveform, sample_rate = self.decode_audio(audio_data)
                    
                    print(f"Processing audio: shape={waveform.shape}, sample_rate={sample_rate}", flush=True)

                    # The model's classify_batch method handles normalization automatically
                    # It expects waveform tensor with shape [batch, channels, samples] or [channels, samples]
                    # Ensure proper shape: [channels, samples] or [batch, channels, samples]
                    if waveform.dim() == 1:
                        # [samples] -> [1, samples] (mono channel)
                        waveform = waveform.unsqueeze(0)
                    elif waveform.dim() == 2:
                        # Already in [channels, samples] format - this is what classify_batch expects
                        pass
                    else:
                        # If already has batch dimension, keep as is
                        pass

                    # Perform inference
                    # classify_batch expects [channels, samples] and handles normalization
                    prediction = self.language_id.classify_batch(waveform)

                    # Extract results
                    # prediction format: (logits, confidence, language_id, language_name)
                    # logits: tensor with scores for all languages
                    # confidence: tensor with confidence score
                    # language_id: tensor with predicted language index
                    # language_name: list with language code (e.g., ['th'])

                    logits = prediction[0]  # Shape: [batch, num_languages]
                    confidence_tensor = prediction[1]  # Shape: [batch]
                    language_code = prediction[3][0]  # Language code string

                    # Get confidence score
                    confidence_score = float(confidence_tensor[0].exp().item())

                    # Get all language scores (convert logits to probabilities)
                    all_scores = torch.softmax(logits[0], dim=0).cpu().numpy()
                    
                    # Create a dictionary with top languages (for debugging/info)
                    # Get top 5 languages
                    top_k = 5
                    top_scores, top_indices = torch.topk(torch.softmax(logits[0], dim=0), top_k)
                    
                    # Note: We don't have direct access to all language codes from the model
                    # So we'll just return the scores array as JSON
                    scores_dict = {
                        "predicted_language": language_code,
                        "confidence": float(confidence_score),
                        "top_scores": [float(s.item()) for s in top_scores]
                    }
                    
                    all_scores_json = json.dumps(scores_dict)

                    print(f"Detected language: {language_code}, confidence: {confidence_score:.4f}", flush=True)

                    language_codes.append([language_code])
                    confidences.append([confidence_score])
                    all_scores_list.append([all_scores_json])

                except Exception as e:
                    # If there's an error, return error message
                    import traceback
                    error_msg = f"Error processing audio: {str(e)}"
                    print(error_msg, flush=True)
                    print(traceback.format_exc(), flush=True)
                    
                    # Return error in expected format
                    language_codes.append([""])
                    confidences.append([0.0])
                    error_json = json.dumps({
                        "error": error_msg,
                        "predicted_language": "",
                        "confidence": 0.0
                    })
                    all_scores_list.append([error_json])

            # Create output tensors
            language_code_array = np.array(language_codes, dtype=self.language_code_dtype)
            confidence_array = np.array(confidences, dtype=self.confidence_dtype)
            all_scores_array = np.array(all_scores_list, dtype=self.all_scores_dtype)

            out_tensor_lang = pb_utils.Tensor(
                "LANGUAGE_CODE",
                language_code_array
            )
            out_tensor_conf = pb_utils.Tensor(
                "CONFIDENCE",
                confidence_array
            )
            out_tensor_scores = pb_utils.Tensor(
                "ALL_SCORES",
                all_scores_array
            )

            # Create response
            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[out_tensor_lang, out_tensor_conf, out_tensor_scores]
                )
            )

        return responses

    def finalize(self):
        """
        Called when model is unloaded.
        Clean up resources.
        """
        print("Cleaning up ALD model resources", flush=True)
        del self.language_id
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

