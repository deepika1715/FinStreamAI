# FinStreamAI — Real-Time Fraud Detection Platform

[![CI](https://github.com/deepika1715/FinStreamAI/actions/workflows/ci.yml/badge.svg)](https://github.com/deepika1715/FinStreamAI/actions/workflows/ci.yml)
[![Helm](https://img.shields.io/badge/helm-v0.1.0-blue)](https://deepika1715.github.io/FinStreamAI/)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

A production-grade MLOps + LLMOps platform that detects credit card fraud in real time and generates plain-English FCA-ready audit reports explaining every flagged transaction.

Built with XGBoost, FastAPI, Kafka, Kubernetes, RAG, and a full LLMOps observability layer.

---

## What This Project Does

FinStreamAI scores bank transactions for fraud in under 100ms. When a transaction is flagged as HIGH risk, it automatically generates a structured audit report explaining why — grounded in a curated fraud pattern knowledge base using RAG retrieval.

Trained on 284,807 real credit card transactions. Achieves 97.47% ROC-AUC.

---

## Quick Install via Helm

```bash
helm repo add finstreamai https://deepika1715.github.io/FinStreamAI
helm repo update
helm install my-finstreamai finstreamai/finstreamai
```

---

## Demo

```bash
# Terminal 1 — API
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Demo UI
streamlit run demo/app.py
```

Open http://localhost:8501 — score a transaction, generate an audit report, inspect traces.

---

## Architecture

```
Kafka stream
     │
     ▼
FastAPI /predict          XGBoost model · MLflow · Prometheus · Grafana
     │
     │ (if HIGH risk)
     ▼
FastAPI /explain          RAG retrieval (FAISS) → rule-based audit report
     │
     ├── traces/explain_traces.jsonl    (LLMOps tracing)
     ├── MLflow finstreamai-llmops      (prompt versioning + eval metrics)
     └── Streamlit demo/app.py          (interview demo UI)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | XGBoost (scale_pos_weight for class imbalance) |
| API | FastAPI + Uvicorn |
| Streaming | Apache Kafka |
| Containerisation | Docker + Docker Compose |
| Orchestration | Kubernetes · Helm chart |
| Monitoring | Prometheus + Grafana |
| CI/CD | GitHub Actions |
| Experiment Tracking | MLflow |
| RAG Knowledge Base | FAISS + TF-IDF vectorisation |
| LLMOps Tracing | Custom JSONL tracer |
| LLMOps Evaluation | DeepEval-style quality scoring |
| Prompt Versioning | MLflow artifact logging |
| Demo UI | Streamlit |

---

## Model Performance

| Metric | Score |
|---|---|
| ROC-AUC | 0.9747 |
| Fraud Precision | 0.78 |
| Fraud Recall | 0.84 |
| Fraud F1 | 0.81 |
| /predict latency | ~55ms |
| /explain latency | ~15ms |

---

## API Endpoints

### POST /predict

```json
{
  "is_fraud": true,
  "fraud_probability": 0.9988,
  "risk_level": "HIGH",
  "amount": 1.0,
  "api_model_version": "xgboost-v1"
}
```

### POST /explain

```json
{
  "transaction_id": "TXN-001",
  "explanation": "Transaction TXN-001 was flagged with 99.9% fraud probability...",
  "matched_pattern": "Card-Not-Present (CNP) Fraud. CVV mismatches are strong indicators.",
  "recommended_action": "Immediately block transaction, freeze card, and escalate.",
  "confidence": "HIGH",
  "prompt_version": "explain_v1"
}
```

---

## Project Structure

```
FinStreamAI/
├── src/
│   ├── api/
│   │   ├── main.py                # FastAPI — /predict, /explain, /health, /metrics
│   │   └── schemas.py             # Pydantic models inc. ExplainRequest/Response
│   ├── llmops/
│   │   ├── explainer.py           # FraudExplainer — RAG retrieval + audit report
│   │   ├── tracer.py              # JSONL trace logger
│   │   ├── eval_nightly.py        # Quality scoring → MLflow time-series
│   │   └── log_prompt.py          # Prompt versioning → MLflow artifact
│   ├── prompts/
│   │   └── explain_v1.txt         # Versioned prompt template
│   ├── producer/kafka_producer.py
│   └── consumer/kafka_consumer.py
├── knowledge_base/
│   ├── fraud_patterns.txt         # 12 curated fraud pattern descriptions
│   └── build_index.py             # TF-IDF vectoriser + FAISS index builder
├── demo/app.py                    # Streamlit demo UI
├── helm/finstreamai/              # Helm chart — one-command install
├── k8s/                           # Kubernetes manifests
├── monitoring/                    # Prometheus + Grafana config
├── tests/test_api.py              # 6 pytest tests
├── .github/workflows/
│   ├── ci.yml                     # CI — test + docker build
│   └── helm-release.yml           # Publish Helm chart to GitHub Pages
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Setup

```bash
git clone git@github.com:deepika1715/FinStreamAI.git
cd FinStreamAI
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
python3 knowledge_base/build_index.py
docker compose up -d
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

---

## LLMOps

```bash
# Run nightly evaluation
python3 -m src.llmops.eval_nightly

# Log new prompt version
python3 src/llmops/log_prompt.py

# View MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

---

## Monitoring

| Service | URL |
|---|---|
| Fraud API | http://localhost:8000 |
| Demo UI | http://localhost:8501 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |
| MLflow | http://localhost:5001 |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — Kaggle
284,807 transactions · 492 fraud cases (0.17%) · 30 PCA-anonymised features

---

## Author

**Deepika Sharma**
Senior Integration Engineer → AI/ML Engineer
IBM ACE · IBM MQ · Kafka · Kubernetes · MSc Artificial Intelligence, University of Bolton 2024

[github.com/deepika1715](https://github.com/deepika1715)

---

## Licence

Apache 2.0 — see [LICENSE](LICENSE)
