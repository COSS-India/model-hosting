# ~/indic_serv/log_model.py
import mlflow, os
from mlflow import pyfunc
from mlflow_asr import IndicConformerWrapper

conda_env = {
    'name': 'mlflow-env',
    'channels': ['defaults'],
    'dependencies': [
        'python=3.10',
        {'pip': [
            'mlflow',
            'torch',
            'transformers',
            'torchaudio',
            'soundfile',
            'pandas'
        ]}
    ]
}

if __name__ == "__main__":
    # optional: if you downloaded the HF repo path, set hf_model_dir to that path and add to artifacts
    artifacts = {}
    # e.g., artifacts["hf_model_dir"] = "/home/ubuntu/indic_conformer_local"
    mlflow.pyfunc.log_model(
        artifact_path="indic_conformer_pyfunc",
        python_model=IndicConformerWrapper(),
        conda_env=conda_env,
        artifacts=artifacts
    )
    print("Model logged to mlruns/ (check mlflow ui or mlruns directory)")
