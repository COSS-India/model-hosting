# MLflow Quick Start Guide

## Setup Complete ✓

The MLflow model has been successfully registered!

## Serve the Model

```bash
cd /home/ubuntu/nmt-benchmarking/mlflow
source mlflow/bin/activate
export HF_TOKEN=YOUR_HF_TOKEN_HERE

# Serve using the latest run
./serve_model.sh 5000
```

Or manually:
```bash
export MLFLOW_TRACKING_URI="file:///home/ubuntu/nmt-benchmarking/mlflow/mlruns"
mlflow models serve -m "runs:/5c7c8011642c4282954c44cc49ed05f0/indictrans_nmt_model" --port 5000 --host 0.0.0.0
```

## Test the Service

```bash
# English to Hindi
curl -X POST http://localhost:5000/invocations \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello, how are you?","src_lang":"eng_Latn","tgt_lang":"hin_Deva"}' | python3 -m json.tool

# Hindi to English
curl -X POST http://localhost:5000/invocations \
  -H "Content-Type: application/json" \
  -d '{"text":"नमस्ते","src_lang":"hin_Deva","tgt_lang":"eng_Latn"}' | python3 -m json.tool
```

## Run Benchmark

```bash
python3 benchmark_nmt.py \
  --endpoint http://localhost:5000/invocations \
  --input_text "Hello, how are you?" \
  --src_lang eng_Latn \
  --tgt_lang hin_Deva \
  --outputdir ./bench_results \
  --rate 5.0 \
  --duration 30
```

## Current Model Info

- **Model Name**: IndicTransNMT
- **Run ID**: 5c7c8011642c4282954c44cc49ed05f0
- **Status**: Registered and ready to serve

