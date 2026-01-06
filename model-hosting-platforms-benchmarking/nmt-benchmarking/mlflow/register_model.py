#!/usr/bin/env python3
"""
Register IndicTrans2 NMT model with MLflow
"""
import os
import mlflow
import mlflow.pyfunc
from mlflow_nmt_model import IndicTransNMTModel

if __name__ == "__main__":
    # Set MLflow tracking URI (local file system by default)
    mlflow.set_tracking_uri("file:///home/ubuntu/nmt-benchmarking/mlflow/mlruns")
    
    # Authenticate with HuggingFace if token is provided
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
            print("Authenticated with HuggingFace Hub")
        except Exception as e:
            print(f"Warning: Could not authenticate with HuggingFace: {e}")
    
    # Create model artifact path
    artifact_path = "indictrans_nmt_model"
    
    # Start MLflow run
    with mlflow.start_run():
        # Log the model
        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=IndicTransNMTModel(),
            pip_requirements="requirements.txt",
            registered_model_name="IndicTransNMT"
        )
        
        # Log model metadata
        mlflow.log_param("model_type", "IndicTrans2")
        mlflow.log_param("en_indic_model", "ai4bharat/indictrans2-en-indic-1B")
        mlflow.log_param("indic_en_model", "ai4bharat/indictrans2-indic-en-1B")
        mlflow.log_param("indic_indic_model", "ai4bharat/indictrans2-indic-indic-1B")
        mlflow.log_param("device", os.environ.get("DEVICE", "cuda" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu"))
        
        print(f"\n✓ Model registered successfully!")
        print(f"  Model name: IndicTransNMT")
        print(f"  Run ID: {mlflow.active_run().info.run_id}")
        print(f"  Artifact path: {artifact_path}")
        print(f"\nTo serve the model, use:")
        print(f"  mlflow models serve -m runs:/{mlflow.active_run().info.run_id}/{artifact_path} --port 5000")

