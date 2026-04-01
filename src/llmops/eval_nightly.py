# Copyright (c) 2025 Deepika Sharma. Licensed under Apache 2.0.
"""
FinStreamAI - Week 2, Step 5
Nightly evaluation script.
Reads the last N traces, scores each explanation for quality,
and logs aggregate scores to MLflow as a time-series.

Run manually:     python src/llmops/eval_nightly.py
Run on a schedule: add to cron or GitHub Actions nightly workflow.

Metrics scored (no LLM API needed):
  - Relevance score:   does the explanation mention key transaction features?
  - Faithfulness score: does the explanation stay grounded in the retrieved pattern?
  - Completeness score: does the response have all 3 required sections?
  - Latency score:     is the response within acceptable speed thresholds?
"""

import json
import re
import logging
import mlflow
from datetime import datetime, timezone
from pathlib import Path

from src.llmops.tracer import read_traces, TRACES_FILE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_URI     = "sqlite:///mlflow.db"
EXPERIMENT     = "finstreamai-llmops"
MAX_LATENCY_MS = 500.0   # acceptable threshold


# ── Scoring functions (rule-based, no API needed) ─────────────────────────

def score_relevance(trace: dict) -> float:
    """
    Does the explanation reference the transaction's actual features and amount?
    Score: 0.0 to 1.0
    """
    explanation = trace["output"]["explanation"].lower()
    inp         = trace["input"]
    score       = 0.0

    # Check transaction ID mentioned
    if inp["transaction_id"].lower() in explanation:
        score += 0.3

    # Check amount mentioned (within explanation)
    amount_str = str(int(inp["amount"]))
    if amount_str in explanation:
        score += 0.3

    # Check fraud probability or risk level mentioned
    if str(int(inp["fraud_probability"] * 100)) in explanation:
        score += 0.2

    # Check at least one feature name mentioned
    for feat in inp["top_features"]:
        if feat.lower() in explanation:
            score += 0.2
            break

    return round(min(score, 1.0), 3)


def score_faithfulness(trace: dict) -> float:
    """
    Does the explanation stay grounded in the retrieved fraud pattern?
    Checks for overlap between key terms in the retrieved pattern
    and terms used in the explanation.
    Score: 0.0 to 1.0
    """
    pattern     = trace["retrieval"]["retrieved_pattern_preview"].lower()
    explanation = trace["output"]["explanation"].lower()
    matched     = trace["output"]["matched_pattern"].lower()

    # Extract meaningful words from pattern (>4 chars, not stopwords)
    stopwords = {"this","that","with","from","have","will","been","when","where",
                 "which","their","there","these","other","transaction","fraud"}
    pattern_words = {
        w for w in re.findall(r"[a-z]+", pattern)
        if len(w) > 4 and w not in stopwords
    }

    if not pattern_words:
        return 0.5

    # Count how many pattern words appear in explanation or matched pattern
    combined  = explanation + " " + matched
    overlap   = sum(1 for w in pattern_words if w in combined)
    score     = overlap / len(pattern_words)

    return round(min(score, 1.0), 3)


def score_completeness(trace: dict) -> float:
    """
    Does the response contain all 3 required sections?
    explanation, matched_pattern, recommended_action
    Score: 0.0 or 1.0 (binary)
    """
    out   = trace["output"]
    score = 0.0
    if out.get("explanation")     and len(out["explanation"])     > 20: score += 0.34
    if out.get("matched_pattern") and len(out["matched_pattern"]) > 10: score += 0.33
    # recommended_action is in explainer result but not in trace output
    # check explanation is not a fallback
    if "not available" not in out["explanation"].lower():               score += 0.33
    return round(score, 2)


def score_latency(trace: dict) -> float:
    """
    Is the response latency within the acceptable threshold?
    Returns 1.0 if under MAX_LATENCY_MS, scaled score otherwise.
    """
    latency = trace["performance"]["latency_ms"]
    if latency <= MAX_LATENCY_MS:
        return 1.0
    return round(MAX_LATENCY_MS / latency, 3)


def evaluate_traces(traces: list[dict]) -> dict:
    """Score all traces and return aggregate metrics."""
    if not traces:
        return {}

    relevance_scores    = [score_relevance(t)    for t in traces]
    faithfulness_scores = [score_faithfulness(t) for t in traces]
    completeness_scores = [score_completeness(t) for t in traces]
    latency_scores      = [score_latency(t)      for t in traces]

    return {
        "n_traces_evaluated":       len(traces),
        "avg_relevance":            round(sum(relevance_scores)    / len(traces), 3),
        "avg_faithfulness":         round(sum(faithfulness_scores) / len(traces), 3),
        "avg_completeness":         round(sum(completeness_scores) / len(traces), 3),
        "avg_latency_score":        round(sum(latency_scores)      / len(traces), 3),
        "overall_quality":          round((
            sum(relevance_scores) +
            sum(faithfulness_scores) +
            sum(completeness_scores) +
            sum(latency_scores)
        ) / (4 * len(traces)), 3),
        "min_relevance":            round(min(relevance_scores),    3),
        "min_faithfulness":         round(min(faithfulness_scores), 3),
        "pct_high_confidence":      round(
            sum(1 for t in traces if t["output"]["confidence"] == "HIGH") / len(traces), 3
        ),
        "avg_latency_ms":           round(
            sum(t["performance"]["latency_ms"] for t in traces) / len(traces), 1
        ),
    }


def log_to_mlflow(metrics: dict) -> str:
    """Log evaluation metrics to MLflow as a timestamped run."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    run_name = f"eval-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("type",          "nightly_eval")
        mlflow.set_tag("eval_date",     datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        mlflow.set_tag("prompt_version","explain_v1")
        mlflow.log_param("n_traces",    metrics["n_traces_evaluated"])

        for key, value in metrics.items():
            if key != "n_traces_evaluated":
                mlflow.log_metric(key, value)

        run_id = mlflow.active_run().info.run_id
        return run_id


def main():
    logger.info("Starting nightly evaluation...")

    if not TRACES_FILE.exists():
        logger.warning("No trace file found. Run some /explain requests first.")
        return

    traces = read_traces(last_n=100)
    if not traces:
        logger.warning("No traces found in file.")
        return

    logger.info(f"Evaluating {len(traces)} traces...")
    metrics = evaluate_traces(traces)

    print("\nEvaluation Results:")
    print("=" * 50)
    for key, value in metrics.items():
        print(f"  {key:<30} {value}")

    run_id = log_to_mlflow(metrics)
    print(f"\nLogged to MLflow")
    print(f"  Run ID:     {run_id}")
    print(f"  Experiment: {EXPERIMENT}")
    print(f"\nView: mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001")


if __name__ == "__main__":
    main()
