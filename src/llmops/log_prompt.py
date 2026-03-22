# Copyright (c) 2025 Deepika Sharma. Licensed under Apache 2.0.
"""
FinStreamAI - Week 1, Step 3
Logs the active prompt template as a versioned MLflow artifact.
Run once per prompt version change:
    python src/llmops/log_prompt.py
"""

import os
import mlflow
from pathlib import Path

PROMPT_VERSION = "explain_v1"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

PROMPT_TEMPLATE = """You are a fraud analyst at a UK bank regulated by the FCA.
A transaction has been flagged as potentially fraudulent.

Transaction details:
- Transaction ID: {transaction_id}
- Amount: GBP {amount:.2f}
- Fraud probability: {fraud_probability:.1%}
- Risk level: {risk_level}
- Key anomalous features: {top_features}

Relevant fraud pattern from our knowledge base:
{retrieved_context}

Write a concise fraud audit report with exactly three parts:
1. EXPLANATION: Why this transaction was flagged (2 sentences max)
2. PATTERN MATCH: Which fraud pattern this most closely resembles and why (1 sentence)
3. RECOMMENDED ACTION: What the compliance team should do next (1 sentence)

Keep the total response under 120 words. Be specific, not generic."""


def log_prompt():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("finstreamai-llmops")

    # Write prompt to a temp file then log as artifact
    prompt_path = Path(f"src/prompts/{PROMPT_VERSION}.txt")
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(PROMPT_TEMPLATE)

    with mlflow.start_run(run_name=f"prompt-{PROMPT_VERSION}"):
        mlflow.set_tag("type", "prompt_version")
        mlflow.set_tag("prompt_version", PROMPT_VERSION)
        mlflow.set_tag("model_version", "xgboost-v1")
        mlflow.set_tag("endpoint", "/explain")
        mlflow.log_param("prompt_version", PROMPT_VERSION)
        mlflow.log_param("max_tokens", 300)
        mlflow.log_param("retrieval_k", 2)
        mlflow.log_artifact(str(prompt_path), artifact_path="prompts")
        run_id = mlflow.active_run().info.run_id
        print(f"Logged prompt artifact")
        print(f"  Version:    {PROMPT_VERSION}")
        print(f"  Run ID:     {run_id}")
        print(f"  Artifact:   prompts/{PROMPT_VERSION}.txt")
        print(f"  Experiment: finstreamai-llmops")

    print(f"\nView in MLflow UI:")
    print(f"  mlflow ui --backend-store-uri mlruns")
    print(f"  Open: http://localhost:5001")


if __name__ == "__main__":
    log_prompt()
