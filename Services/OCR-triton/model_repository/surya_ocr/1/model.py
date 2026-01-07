import json
import numpy as np
import triton_python_backend_utils as pb_utils
import torch
import base64
import io
import os
from PIL import Image
from surya.models import load_predictors


class TritonPythonModel:
    """
    Triton Python backend model for Surya OCR.
    Performs OCR on document images in 90+ languages.
    """

    def initialize(self, args):
        """
        Called once when model is loaded.
        Loads the Surya OCR models (Foundation, Detection, Recognition).

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

        # Get batch sizes from environment variables (for testing with reduced VRAM)
        self.recognition_batch_size = int(os.environ.get('RECOGNITION_BATCH_SIZE', '64'))
        self.detector_batch_size = int(os.environ.get('DETECTOR_BATCH_SIZE', '8'))
        
        print(f"[OK] Recognition batch size: {self.recognition_batch_size}", flush=True)
        print(f"[OK] Detector batch size: {self.detector_batch_size}", flush=True)

        # Load Surya predictors
        print("Loading Surya predictors...", flush=True)
        predictors = load_predictors(device=self.device)

        self.det_predictor = predictors['detection']
        self.rec_predictor = predictors['recognition']

        print("[OK] Detection predictor loaded successfully", flush=True)
        print("[OK] Recognition predictor loaded successfully", flush=True)
        print(f"[OK] Surya OCR model initialized successfully on {self.device}", flush=True)

    def decode_image(self, image_data):
        """
        Decode base64 encoded image data to PIL Image.
        
        Args:
            image_data: Base64 encoded image string or bytes
            
        Returns:
            PIL.Image: Decoded image
        """
        try:
            # Handle bytes input
            if isinstance(image_data, bytes):
                image_data = image_data.decode('utf-8')
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
        except Exception as e:
            raise ValueError(f"Failed to decode image: {str(e)}")

    def perform_ocr(self, image):
        """
        Perform OCR on a single image.

        Args:
            image: PIL Image

        Returns:
            dict: OCR results with text lines, bboxes, and confidence scores
        """
        try:
            # Run recognition with detection
            # The recognition predictor will call the detection predictor internally
            results = self.rec_predictor(
                images=[image],
                det_predictor=self.det_predictor,
                detection_batch_size=self.detector_batch_size,
                recognition_batch_size=self.recognition_batch_size,
                sort_lines=True  # Sort text lines in reading order
            )

            # Extract results from first (and only) prediction
            result = results[0]

            # Format output
            output = {
                "success": True,
                "text_lines": [],
                "full_text": "",
                "image_bbox": result.image_bbox
            }

            # Extract text lines with details
            all_text = []
            for line in result.text_lines:
                line_data = {
                    "text": line.text,
                    "confidence": float(line.confidence),
                    "bbox": line.bbox,
                    "polygon": line.polygon
                }
                output["text_lines"].append(line_data)
                all_text.append(line.text)

            # Combine all text
            output["full_text"] = "\n".join(all_text)

            return output

        except Exception as e:
            import traceback
            error_msg = f"OCR processing failed: {str(e)}"
            print(error_msg, flush=True)
            print(traceback.format_exc(), flush=True)

            return {
                "success": False,
                "error": error_msg,
                "text_lines": [],
                "full_text": "",
                "image_bbox": [0, 0, 0, 0]
            }

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
            input_tensor = pb_utils.get_input_tensor_by_name(
                request, "IMAGE_DATA"
            )

            # Convert to numpy array
            input_data = input_tensor.as_numpy()

            # Process each image in the batch
            output_results = []

            for idx, image_data_bytes in enumerate(input_data):
                try:
                    # Extract the image data
                    image_data = image_data_bytes[0]
                    
                    print(f"Processing OCR request {idx + 1}...", flush=True)

                    # Decode image
                    image = self.decode_image(image_data)
                    print(f"Image decoded: {image.size[0]}x{image.size[1]} pixels", flush=True)

                    # Perform OCR
                    result = self.perform_ocr(image)
                    
                    if result["success"]:
                        print(f"OCR completed: {len(result['text_lines'])} lines detected", flush=True)
                    else:
                        print(f"OCR failed: {result.get('error', 'Unknown error')}", flush=True)

                    # Format output as JSON string
                    output_json = json.dumps(result)
                    output_results.append([output_json])

                except Exception as e:
                    # If there's an error, return error message
                    import traceback
                    error_msg = f"Error processing OCR request: {str(e)}"
                    print(error_msg, flush=True)
                    print(traceback.format_exc(), flush=True)
                    
                    # Return error in expected format
                    error_json = json.dumps({
                        "success": False,
                        "error": error_msg,
                        "text_lines": [],
                        "full_text": "",
                        "image_bbox": [0, 0, 0, 0]
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
        print("Cleaning up Surya OCR model resources", flush=True)
        del self.foundation_predictor
        del self.detection_predictor
        del self.recognition_predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

