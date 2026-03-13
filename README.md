# FinStreamAI — Real-Time Fraud Detection Platform

A production-grade AI system that detects credit card fraud in real time.
Built with XGBoost, FastAPI, Kafka, Docker, and Kubernetes.

![CI](https://github.com/deepika1715/FinStreamAI/actions/workflows/ci.yml/badge.svg)

---

## What This Project Does

FinStreamAI scores bank transactions for fraud in under 100ms.
You send a transaction to the API, it returns a fraud probability and risk level instantly.

Trained on 284,807 real credit card transactions. Achieves 97.47% ROC-AUC.

---

## Results

| Metric | Score |
|---|---|
| ROC-AUC | 0.9747 |
| Fraud Precision | 0.78 |
| Fraud Recall | 0.84 |
| Fraud F1 | 0.81 |
| Average Latency | ~55ms |

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | XGBoost (scale_pos_weight for class imbalance) |
| API | FastAPI + Uvicorn |
| Streaming | Apache Kafka |
| Containerisation | Docker + Docker Compose |
| Orchestration | Kubernetes (Minikube) |
| Monitoring | Prometheus metrics |
| CI/CD | GitHub Actions |
| Experiment Tracking | MLflow |

---

## Project Structure
```
FinStreamAI/
├── src/
│   ├── api/
│   │   ├── main.py          # FastAPI app — /predict, /health, /metrics
│   │   └── schemas.py       # Pydantic request/response models
│   ├── producer/
│   │   └── kafka_producer.py
│   └── consumer/
│       └── kafka_consumer.py
├── models/                  # Trained model (not in git)
├── notebooks/
│   └── exploration.ipynb    # Training — XGBoost + Isolation Forest
├── tests/
│   └── test_api.py          # pytest test suite
├── .github/workflows/
│   └── ci.yml               # GitHub Actions CI/CD
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quickstart

### 1. Clone the repo
```bash
git clone git@github.com:deepika1715/FinStreamAI.git
cd FinStreamAI
```

### 2. Set up Python environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Download the dataset

Download creditcard.csv from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place it in the `data/` folder.

### 4. Train the model

Open and run `notebooks/exploration.ipynb` — this trains XGBoost and saves the model to `models/`.

### 5. Start Kafka and the API
```bash
docker compose up -d
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

---

## API Endpoints

### POST /predict

Score a transaction for fraud.

**Request:**
```json
{
  "Time": 406.0,
  "V1": -2.312227,
  "V2": 1.951992,
  "Amount": 1.0
}
```

**Response:**
```json
{
  "is_fraud": true,
  "fraud_probability": 0.9988,
  "risk_level": "HIGH",
  "amount": 1.0,
  "api_model_version": "xgboost-v1"
}
```

### GET /health
```json
{
  "status": "healthy",
  "model_loaded": true,
  "kafka_available": true
}
```

### GET /metrics

Prometheus metrics endpoint — scrape for live dashboards.

---

## Running Tests
```bash
pytest tests/ -v
```

6 tests covering health, prediction, validation, and metrics.

---

## How It Works

1. Transaction data streams into Kafka topic `transactions`
2. Consumer picks up each transaction and calls the FastAPI `/predict` endpoint
3. XGBoost scores the transaction — returns fraud probability in ~55ms
4. Risk level assigned — LOW, MEDIUM, or HIGH
5. Prometheus records every request and latency
6. Grafana (optional) visualises fraud rates on live dashboards

---

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — Kaggle

- 284,807 transactions
- 492 fraud cases (0.17%)
- 30 features (PCA-anonymised V1–V28 + Time + Amount)

---

## Author

**Deepika Sharma**
Senior Integration Engineer — IBM ACE, IBM MQ, Kafka, Kubernetes
MSc Artificial Intelligence — University of Bolton, 2024

[github.com/deepika1715](https://github.com/deepika1715)

---

## Licence

Apache 2.0 — see LICENSE