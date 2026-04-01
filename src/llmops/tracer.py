# Copyright (c) 2025 Deepika Sharma. Licensed under Apache 2.0.
"""
FinStreamAI - Week 2, Step 4
LLMOps observability tracer.
Logs every /explain call as a structured trace to traces/explain_traces.jsonl
One JSON object per line — easy to inspect, grep, and analyse.
No server required. No API key required.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

TRACES_DIR  = Path(__file__).parent.parent.parent / "traces"
TRACES_FILE = TRACES_DIR / "explain_traces.jsonl"


def _ensure_dir():
    TRACES_DIR.mkdir(exist_ok=True)


def log_trace(
    transaction_id:    str,
    fraud_probability: float,
    risk_level:        str,
    amount:            float,
    top_features:      dict,
    retrieved_pattern: str,
    explanation:       str,
    matched_pattern:   str,
    confidence:        str,
    prompt_version:    str,
    latency_ms:        float,
) -> str:
    """
    Write one trace record to explain_traces.jsonl.
    Returns the trace_id for the response log.
    """
    _ensure_dir()

    trace_id = f"trace-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"

    record = {
        "trace_id":          trace_id,
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "input": {
            "transaction_id":    transaction_id,
            "fraud_probability": round(fraud_probability, 4),
            "risk_level":        risk_level,
            "amount":            round(amount, 2),
            "top_features":      top_features,
        },
        "retrieval": {
            "retrieved_pattern_preview": retrieved_pattern[:120],
        },
        "output": {
            "explanation":     explanation,
            "matched_pattern": matched_pattern,
            "confidence":      confidence,
            "prompt_version":  prompt_version,
        },
        "performance": {
            "latency_ms": round(latency_ms, 2),
        },
    }

    with open(TRACES_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    logger.info(
        f"Trace logged — id: {trace_id}, "
        f"latency: {latency_ms:.0f}ms, "
        f"confidence: {confidence}"
    )
    return trace_id


def read_traces(last_n: int = 10) -> list[dict]:
    """Read the last N traces from the JSONL file."""
    if not TRACES_FILE.exists():
        return []
    lines = TRACES_FILE.read_text(encoding="utf-8").strip().split("\n")
    lines = [l for l in lines if l.strip()]
    return [json.loads(l) for l in lines[-last_n:]]


def trace_summary() -> dict:
    """Return summary stats across all logged traces."""
    traces = read_traces(last_n=10000)
    if not traces:
        return {"total": 0}

    latencies   = [t["performance"]["latency_ms"] for t in traces]
    confidences = [t["output"]["confidence"] for t in traces]
    patterns    = [t["output"]["matched_pattern"][:40] for t in traces]

    from collections import Counter
    return {
        "total":              len(traces),
        "avg_latency_ms":     round(sum(latencies) / len(latencies), 1),
        "max_latency_ms":     round(max(latencies), 1),
        "confidence_counts":  dict(Counter(confidences)),
        "top_patterns":       dict(Counter(t["output"]["matched_pattern"].split("resembles ")[-1][:40] for t in traces).most_common(3)),
        "first_trace":        traces[0]["timestamp"],
        "last_trace":         traces[-1]["timestamp"],
    }
