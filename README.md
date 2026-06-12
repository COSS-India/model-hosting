# ML Model Hosting – Benchmarking & Setup Guides

This repository provides a **practical reference for hosting and serving ML/NLP models at scale**.  
It covers **performance benchmarking across multiple model-serving platforms** and **step-by-step setup documentation for deploying NLP models using NVIDIA Triton Inference Server**.

The repo is intended for:
- ML / AI engineers
- Platform & DevOps teams
- Architects evaluating model-serving stacks
- Teams building scalable Language AI systems

---

## Repository Structure
```bash
.
├── nmt-triton/                          # CPU local NMT (IndicTrans2 + Triton)
├── model-hosting-platforms-benchmarking/
├── setup-docs/
└── README.md
```


## 📊 `model-hosting-platforms-benchmarking/`

This folder contains **benchmarking experiments and results** for hosting ML models across different serving platforms.

### Purpose
To help teams **compare model hosting approaches** based on real-world performance and operational characteristics.

### Platforms Covered (examples)
- NVIDIA Triton Inference Server
- MLflow Model Serving
- FastAPI-based custom serving
- Other popular serving frameworks (as added)

### Typical Benchmark Dimensions
- Inference latency (P50 / P90 / P99)
- Throughput (requests per second)
- Concurrency handling
- Resource utilization (CPU / GPU / memory)
- Scalability behavior under load
- Error rates during peak traffic

### What You’ll Find
- Benchmark configurations
- Load test scenarios
- Observed metrics & results
- Comparative analysis across platforms

This section is especially useful for **platform selection decisions** and **capacity planning**.

---

## 🚀 `nmt-triton/`

CPU-only **IndicTrans2** Triton deployment for local development and testing.

- Docker build + Triton model repository (`models/nmt/`)
- HTTP API on port **8000** (`/v2/models/nmt/infer`)
- No GPU required

See `nmt-triton/README.md` for build, run, and verification steps.

---

## 🛠️ `setup-docs/`

This folder contains **detailed setup guides** for deploying various NLP and Language AI models, primarily using **NVIDIA Triton Inference Server**.

### Purpose
To provide **repeatable, production-oriented deployment instructions** for common NLP workloads.

### Model Types Covered
- **NMT (Neural Machine Translation)**
- **OCR (Optical Character Recognition)**
- **Transliteration**
- Other NLP / language models (as added)

### What Each Setup Guide Typically Includes
- Model format & prerequisites
- Triton model repository structure
- `config.pbtxt` explanations
- Pre-processing & post-processing notes
- GPU / CPU configuration guidance
- Common pitfalls & troubleshooting tips

These guides are designed to reduce **time-to-deployment** and encourage **best practices for scalable inference**.

---

## 🧩 How to Use This Repository

- **Evaluating serving platforms?**  
  → Start with `model-hosting-platforms-benchmarking/`

- **Local CPU NMT on Triton?**  
  → Start with `nmt-triton/`

- **Deploying NLP models on Triton (GPU)?**  
  → Go to `setup-docs/`

- **Building a Language AI platform or sandbox?**  
  → Use benchmarking and setup guides together: benchmark first, then deploy with confidence.

---

## 📌 Notes & Scope

- Benchmarks reflect specific hardware, model sizes, and configurations—use them as **reference points**, not absolute numbers.
- Setup guides prioritize **clarity, reproducibility, and production-readiness**.
- Contributions and improvements are welcome.

---

## 🤝 Contributions

If you’d like to:
- Add new benchmarks
- Include additional serving platforms
- Extend setup guides to new models

Feel free to open a PR or raise an issue.

---

## 📄 License

MIT
