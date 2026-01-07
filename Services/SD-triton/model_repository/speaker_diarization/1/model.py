import json
import numpy as np
import triton_python_backend_utils as pb_utils
import torch
import io
import base64
import os
import tempfile
from pyannote.audio import Pipeline
from huggingface_hub import login


class TritonPythonModel:
    """
    Triton Python backend model for Speaker Diarization.
    Uses pyannote.audio speaker diarization pipeline to identify and segment speakers in audio.
    """

    def initialize(self, args):
        """
        Called once when model is loaded.
        Loads the pyannote speaker diarization pipeline.

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[OK] Using device: {self.device}", flush=True)

        # Authenticate with HuggingFace if token is available (model is gated)
        # TOKEN USAGE: Token is retrieved from environment variable
        hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            print("Authenticating with HuggingFace...", flush=True)
            login(token=hf_token)
            print("[OK] HuggingFace authentication successful", flush=True)
        else:
            print("WARNING: No HuggingFace token found. Model download may fail if gated.", flush=True)
            print("Please set HUGGING_FACE_HUB_TOKEN environment variable.", flush=True)
            print("You also need to accept the model conditions at:", flush=True)
            print("  - https://huggingface.co/pyannote/speaker-diarization", flush=True)
            print("  - https://huggingface.co/pyannote/segmentation", flush=True)

        # Model name - pyannote speaker diarization
        model_name = "pyannote/speaker-diarization@2.1"

        # Load pyannote speaker diarization pipeline
        print(f"Loading Speaker Diarization model: {model_name}", flush=True)
        print("This may take a few minutes on first run...", flush=True)

        # TOKEN USAGE: Token is passed to from_pretrained() to authenticate
        self.pipeline = Pipeline.from_pretrained(
            model_name,
            use_auth_token=hf_token
        )

        # Note: pyannote.audio Pipeline handles device placement internally
        # We don't need to call .to() on the pipeline object
        # The pipeline will automatically use GPU if available

        print(f"[OK] Model loaded successfully on device: {self.device}", flush=True)
        print("[OK] Speaker Diarization model ready for inference", flush=True)

    def decode_audio(self, audio_data_bytes):
        """
        Decode audio data from base64 encoded string or bytes.
        
        Args:
            audio_data_bytes: Base64 encoded audio data or raw bytes
            
        Returns:
            str: Path to temporary audio file
        """
        try:
            # Try to decode as base64 string first
            if isinstance(audio_data_bytes, bytes):
                audio_data_bytes = audio_data_bytes.decode('utf-8')
            
            # Decode base64
            audio_bytes = base64.b64decode(audio_data_bytes)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file_path = tmp_file.name
            
            return tmp_file_path
            
        except Exception as e:
            print(f"Error decoding audio: {e}", flush=True)
            raise

    def perform_diarization(self, audio_path, num_speakers=None, min_speakers=None, max_speakers=None):
        """
        Perform speaker diarization on audio.
        
        Args:
            audio_path: Path to audio file
            num_speakers: Exact number of speakers (optional)
            min_speakers: Minimum number of speakers (optional)
            max_speakers: Maximum number of speakers (optional)
            
        Returns:
            dict: Diarization results
        """
        try:
            # Prepare pipeline parameters
            pipeline_kwargs = {}
            if num_speakers is not None:
                pipeline_kwargs['num_speakers'] = num_speakers
            if min_speakers is not None:
                pipeline_kwargs['min_speakers'] = min_speakers
            if max_speakers is not None:
                pipeline_kwargs['max_speakers'] = max_speakers
            
            # Run diarization
            diarization = self.pipeline(audio_path, **pipeline_kwargs)
            
            # Convert to list of segments
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "start_time": round(turn.start, 2),
                    "end_time": round(turn.end, 2),
                    "duration": round(turn.end - turn.start, 2),
                    "speaker": speaker
                })
            
            # Get unique speakers
            unique_speakers = sorted(set(seg["speaker"] for seg in segments))
            
            return {
                "total_segments": len(segments),
                "num_speakers": len(unique_speakers),
                "speakers": unique_speakers,
                "segments": segments
            }
            
        except Exception as e:
            print(f"Error performing diarization: {e}", flush=True)
            import traceback
            print(traceback.format_exc(), flush=True)
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
            # Get input tensors
            audio_data_tensor = pb_utils.get_input_tensor_by_name(
                request, "AUDIO_DATA"
            )
            num_speakers_tensor = pb_utils.get_input_tensor_by_name(
                request, "NUM_SPEAKERS"
            )

            # Convert to numpy arrays
            audio_data_array = audio_data_tensor.as_numpy()
            num_speakers_array = None
            if num_speakers_tensor is not None:
                num_speakers_array = num_speakers_tensor.as_numpy()

            # Process each audio in the batch
            diarization_results_list = []

            for idx, audio_data_bytes in enumerate(audio_data_array):
                tmp_file_path = None
                try:
                    # Get audio data
                    audio_data = audio_data_bytes[0]
                    if isinstance(audio_data, bytes):
                        audio_data = audio_data
                    elif isinstance(audio_data, str):
                        audio_data = audio_data.encode('utf-8')

                    # Get num_speakers parameter (optional)
                    num_speakers = None
                    if num_speakers_array is not None and len(num_speakers_array) > idx:
                        num_speakers_data = num_speakers_array[idx][0]
                        if isinstance(num_speakers_data, bytes):
                            num_speakers_str = num_speakers_data.decode('utf-8')
                        elif isinstance(num_speakers_data, str):
                            num_speakers_str = num_speakers_data
                        else:
                            num_speakers_str = ""
                        
                        if num_speakers_str and num_speakers_str.strip():
                            try:
                                num_speakers = int(num_speakers_str.strip())
                            except ValueError:
                                num_speakers = None

                    # Decode and save audio to temp file
                    tmp_file_path = self.decode_audio(audio_data)
                    
                    print(f"Processing audio for speaker diarization: {tmp_file_path}, num_speakers={num_speakers}", flush=True)

                    # Perform diarization
                    diarization_results = self.perform_diarization(
                        tmp_file_path,
                        num_speakers=num_speakers
                    )

                    print(f"Found {diarization_results['num_speakers']} speakers in {diarization_results['total_segments']} segments", flush=True)

                    # Format output as JSON
                    output_json = json.dumps(diarization_results)

                    diarization_results_list.append([output_json])

                except Exception as e:
                    # If there's an error, return error message
                    import traceback
                    error_msg = f"Error processing speaker diarization: {str(e)}"
                    print(error_msg, flush=True)
                    print(traceback.format_exc(), flush=True)
                    
                    # Return error in expected format
                    error_json = json.dumps({
                        "total_segments": 0,
                        "num_speakers": 0,
                        "speakers": [],
                        "segments": [],
                        "error": error_msg
                    })
                    diarization_results_list.append([error_json])
                
                finally:
                    # Clean up temp file
                    if tmp_file_path and os.path.exists(tmp_file_path):
                        try:
                            os.unlink(tmp_file_path)
                        except:
                            pass

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
        print("Cleaning up Speaker Diarization model resources", flush=True)
        del self.pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

