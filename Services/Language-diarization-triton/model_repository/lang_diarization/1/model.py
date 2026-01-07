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
    Triton Python backend model for Language Diarization.
    Performs end-to-end language diarization on audio by segmenting and detecting language.
    Uses Wav2Vec-based language identification for each segment.
    """

    def initialize(self, args):
        """
        Called once when model is loaded.
        Loads the language identification model for diarization.

        Args:
            args (dict): Contains model configuration
        """
        # Parse model configuration
        self.model_config = json.loads(args["model_config"])

        # Get output configuration for type conversion
        output_config = pb_utils.get_output_config_by_name(
            self.model_config, "DIARIZATION_RESULT"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(
            output_config["data_type"]
        )

        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        run_opts = {"device": self.device} if self.device == "cuda" else {}

        # Load language identification model (using VoxLingua107 for language detection)
        model_name = "speechbrain/lang-id-voxlingua107-ecapa"
        print(f"Loading Language Identification model: {model_name}", flush=True)
        print("This may take a few minutes on first run...", flush=True)

        self.language_id = EncoderClassifier.from_hparams(
            source=model_name,
            savedir="tmp_lang_diarization_model",
            run_opts=run_opts
        )

        print(f"[OK] Model loaded successfully on device: {self.device}", flush=True)

        # Diarization parameters
        self.segment_duration = 2.0  # Segment duration in seconds (2 seconds per segment)
        self.overlap = 0.5  # Overlap between segments in seconds
        self.min_segment_duration = 1.0  # Minimum segment duration

        print("[OK] Language Diarization model ready for inference", flush=True)

    def decode_audio(self, audio_data_bytes):
        """
        Decode audio data from base64 encoded string or bytes.
        
        Args:
            audio_data_bytes: Base64 encoded audio data or raw bytes
            
        Returns:
            tuple: (waveform tensor, sample_rate)
        """
        try:
            # Try to decode as base64 string first
            if isinstance(audio_data_bytes, bytes):
                audio_data_bytes = audio_data_bytes.decode('utf-8')
            
            # Decode base64
            audio_bytes = base64.b64decode(audio_data_bytes)
            
            # Save to temporary file and use model's load_audio method
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file_path = tmp_file.name
            
            try:
                # Use model's load_audio method which handles all normalization
                waveform = self.language_id.load_audio(tmp_file_path)
                sample_rate = 16000  # Model normalizes to 16kHz
                
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
            raise

    def segment_audio(self, waveform, sample_rate, segment_duration=2.0, overlap=0.5):
        """
        Segment audio into overlapping windows for diarization.
        
        Args:
            waveform: Audio waveform tensor [channels, samples]
            sample_rate: Sample rate of audio
            segment_duration: Duration of each segment in seconds
            overlap: Overlap between segments in seconds
            
        Returns:
            list: List of (start_time, end_time, segment_waveform) tuples
        """
        segments = []
        total_duration = waveform.shape[-1] / sample_rate
        segment_samples = int(segment_duration * sample_rate)
        overlap_samples = int(overlap * sample_rate)
        step_samples = segment_samples - overlap_samples
        
        start_sample = 0
        while start_sample < waveform.shape[-1]:
            end_sample = min(start_sample + segment_samples, waveform.shape[-1])
            
            # Extract segment
            segment = waveform[:, start_sample:end_sample]
            
            # Only process if segment is long enough
            if segment.shape[-1] >= int(self.min_segment_duration * sample_rate):
                start_time = start_sample / sample_rate
                end_time = end_sample / sample_rate
                segments.append((start_time, end_time, segment))
            
            start_sample += step_samples
        
        return segments

    def perform_diarization(self, waveform, sample_rate, target_language=None):
        """
        Perform language diarization on audio.
        
        Args:
            waveform: Audio waveform tensor
            sample_rate: Sample rate
            target_language: Optional target language code (e.g., 'ta', 'gu', 'te')
            
        Returns:
            list: List of diarization segments with language labels
        """
        # Segment the audio
        segments = self.segment_audio(
            waveform, 
            sample_rate, 
            self.segment_duration, 
            self.overlap
        )
        
        diarization_results = []
        
        for start_time, end_time, segment_waveform in segments:
            try:
                # Perform language identification on segment
                prediction = self.language_id.classify_batch(segment_waveform)
                
                # Extract results
                logits = prediction[0]
                confidence_tensor = prediction[1]
                language_code = prediction[3][0] if len(prediction[3]) > 0 else "unknown"
                
                # Get confidence score
                confidence_score = float(confidence_tensor[0].exp().item())
                
                # Filter by target language if specified
                if target_language and language_code != target_language:
                    # Check if language code starts with target (e.g., 'ta' matches 'ta: Tamil')
                    if not language_code.startswith(target_language):
                        continue
                
                diarization_results.append({
                    "start_time": round(start_time, 2),
                    "end_time": round(end_time, 2),
                    "duration": round(end_time - start_time, 2),
                    "language": language_code,
                    "confidence": round(confidence_score, 4)
                })
                
            except Exception as e:
                print(f"Error processing segment [{start_time:.2f}-{end_time:.2f}]: {e}", flush=True)
                continue
        
        return diarization_results

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
            audio_data_tensor = pb_utils.get_input_tensor_by_name(
                request, "AUDIO_DATA"
            )
            language_tensor = pb_utils.get_input_tensor_by_name(
                request, "LANGUAGE"
            )

            # Convert to numpy arrays
            audio_data_array = audio_data_tensor.as_numpy()
            language_array = language_tensor.as_numpy()

            # Process each audio in the batch
            diarization_results_list = []

            for idx, audio_data_bytes in enumerate(audio_data_array):
                try:
                    # Get audio data
                    audio_data = audio_data_bytes[0]
                    if isinstance(audio_data, bytes):
                        audio_data = audio_data
                    elif isinstance(audio_data, str):
                        audio_data = audio_data.encode('utf-8')

                    # Get target language (optional)
                    target_language = None
                    if language_array is not None and len(language_array) > idx:
                        lang_data = language_array[idx][0]
                        if isinstance(lang_data, bytes):
                            target_language = lang_data.decode('utf-8')
                        elif isinstance(lang_data, str):
                            target_language = lang_data
                        if target_language and target_language.strip() == "":
                            target_language = None

                    # Decode and load audio
                    waveform, sample_rate = self.decode_audio(audio_data)
                    
                    print(f"Processing audio for diarization: shape={waveform.shape}, sample_rate={sample_rate}, target_lang={target_language}", flush=True)

                    # Perform diarization
                    diarization_results = self.perform_diarization(
                        waveform, 
                        sample_rate, 
                        target_language
                    )

                    print(f"Found {len(diarization_results)} language segments", flush=True)

                    # Format output as JSON
                    output_json = json.dumps({
                        "total_segments": len(diarization_results),
                        "segments": diarization_results,
                        "target_language": target_language if target_language else "all"
                    })

                    diarization_results_list.append([output_json])

                except Exception as e:
                    # If there's an error, return error message
                    import traceback
                    error_msg = f"Error processing diarization: {str(e)}"
                    print(error_msg, flush=True)
                    print(traceback.format_exc(), flush=True)
                    
                    # Return error in expected format
                    error_json = json.dumps({
                        "total_segments": 0,
                        "segments": [],
                        "error": error_msg
                    })
                    diarization_results_list.append([error_json])

            # Create output tensor
            output_array = np.array(diarization_results_list, dtype=self.output_dtype)

            out_tensor = pb_utils.Tensor(
                "DIARIZATION_RESULT",
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
        print("Cleaning up Language Diarization model resources", flush=True)
        del self.language_id
        if torch.cuda.is_available():
            torch.cuda.empty_cache()













