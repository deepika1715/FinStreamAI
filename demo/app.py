# Copyright (c) 2025 Deepika Sharma. Licensed under Apache 2.0.
"""
FinStreamAI - Week 2, Step 6
Streamlit demo UI - shows the full pipeline live.
Run: streamlit run demo/app.py
"""

import sys
import json
import time
import requests
import streamlit as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llmops.tracer import trace_summary

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="FinStreamAI",
    page_icon="🔍",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## FinStreamAI — Real-Time Fraud Detection Platform")
st.markdown("*Production MLOps + LLMOps pipeline — XGBoost · Kafka · Kubernetes · RAG · Audit Reports*")
st.divider()

# ── Sidebar: system status ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### System status")
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        h = r.json()
        st.success("API online")
        st.caption(f"Model loaded: {h['model_loaded']}")
        st.caption(f"Kafka: {h['kafka_available']}")
    except Exception:
        st.error("API offline — start uvicorn first")

    st.divider()
    st.markdown("### Trace summary")
    try:
        summary = trace_summary()
        if summary.get("total", 0) > 0:
            st.metric("Total explanations", summary["total"])
            st.metric("Avg latency", f"{summary['avg_latency_ms']} ms")
            st.metric("Avg quality", f"{summary.get('avg_latency_ms', 0)} ms")
            cc = summary.get("confidence_counts", {})
            for level, count in cc.items():
                st.caption(f"{level}: {count}")
        else:
            st.caption("No traces yet")
    except Exception:
        st.caption("No traces yet")

    st.divider()
    st.markdown("### About")
    st.caption("Built by Deepika Sharma")
    st.caption("Senior Integration Engineer → AI/ML")
    st.caption("[github.com/deepika1715](https://github.com/deepika1715)")

# ── Main: two columns ─────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### Step 1 — Score a transaction")
    st.caption("Enter transaction details to get a fraud probability from the XGBoost model")

    with st.form("predict_form"):
        amount = st.number_input("Amount (£)", min_value=0.01, max_value=50000.0,
                                  value=149.62, step=0.01)
        st.caption("PCA features — use defaults to test a known fraud case")

        col_a, col_b = st.columns(2)
        with col_a:
            v1  = st.number_input("V1",  value=-1.3598)
            v3  = st.number_input("V3",  value=2.5363)
            v5  = st.number_input("V5",  value=-0.3383)
            v14 = st.number_input("V14", value=-4.2)
        with col_b:
            v2  = st.number_input("V2",  value=-0.0728)
            v4  = st.number_input("V4",  value=2.8)
            v12 = st.number_input("V12", value=-3.1)
            v17 = st.number_input("V17", value=-5.1)

        submitted = st.form_submit_button("Score transaction", use_container_width=True)

    prediction = None
    if submitted:
        payload = {
            "Time": 0.0, "Amount": amount,
            "V1": v1,  "V2": v2,  "V3": v3,  "V4": v4,
            "V5": v5,  "V6": 0.0, "V7": 0.0, "V8": 0.0,
            "V9": 0.0, "V10": 0.0,"V11": 0.0,"V12": v12,
            "V13": 0.0,"V14": v14,"V15": 0.0,"V16": 0.0,
            "V17": v17,"V18": 0.0,"V19": 0.0,"V20": 0.0,
            "V21": 0.0,"V22": 0.0,"V23": 0.0,"V24": 0.0,
            "V25": 0.0,"V26": 0.0,"V27": 0.0,"V28": 0.0,
        }
        try:
            r = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
            prediction = r.json()
            st.session_state["prediction"] = prediction
            st.session_state["top_features"] = {
                "V14": v14, "V12": v12, "V4": v4, "V17": v17
            }
            st.session_state["amount"] = amount
        except Exception as e:
            st.error(f"API error: {e}")

    if "prediction" in st.session_state:
        pred = st.session_state["prediction"]
        risk = pred["risk_level"]
        prob = pred["fraud_probability"]

        if risk == "HIGH":
            st.error(f"FRAUD DETECTED — {prob:.1%} probability — Risk: {risk}")
        elif risk == "MEDIUM":
            st.warning(f"SUSPICIOUS — {prob:.1%} probability — Risk: {risk}")
        else:
            st.success(f"LEGITIMATE — {prob:.1%} probability — Risk: {risk}")

        m1, m2, m3 = st.columns(3)
        m1.metric("Fraud probability", f"{prob:.1%}")
        m2.metric("Risk level", risk)
        m3.metric("Amount", f"£{pred['amount']:.2f}")

with col2:
    st.markdown("### Step 2 — Generate audit report")
    st.caption("RAG retrieval from fraud knowledge base → structured FCA-ready explanation")

    txn_id = st.text_input("Transaction ID", value="TXN-20240315-001")

    explain_disabled = "prediction" not in st.session_state
    if explain_disabled:
        st.caption("Score a transaction first to enable this step")

    if st.button("Generate audit report", disabled=explain_disabled,
                 use_container_width=True):
        pred = st.session_state["prediction"]
        payload = {
            "transaction_id":    txn_id,
            "fraud_probability": pred["fraud_probability"],
            "risk_level":        pred["risk_level"],
            "amount":            st.session_state["amount"],
            "top_features":      st.session_state["top_features"],
        }
        try:
            t0 = time.time()
            r  = requests.post(f"{API_BASE}/explain", json=payload, timeout=15)
            ms = (time.time() - t0) * 1000
            explanation = r.json()
            st.session_state["explanation"] = explanation
            st.session_state["explain_ms"]  = ms
        except Exception as e:
            st.error(f"Explain API error: {e}")

    if "explanation" in st.session_state:
        exp = st.session_state["explanation"]
        ms  = st.session_state.get("explain_ms", 0)

        st.markdown("#### Fraud audit report")

        st.markdown("**Explanation**")
        st.info(exp["explanation"])

        st.markdown("**Matched fraud pattern**")
        st.warning(exp["matched_pattern"])

        st.markdown("**Recommended action**")
        st.error(exp["recommended_action"])

        m1, m2, m3 = st.columns(3)
        m1.metric("Confidence",      exp["confidence"])
        m2.metric("Prompt version",  exp["prompt_version"])
        m3.metric("Latency",         f"{ms:.0f} ms")

        with st.expander("Retrieved context preview"):
            st.code(exp["retrieved_context_preview"])

# ── Bottom: recent traces ─────────────────────────────────────────────────────
st.divider()
st.markdown("### Recent traces")

try:
    from src.llmops.tracer import read_traces
    traces = read_traces(last_n=5)
    if traces:
        for t in reversed(traces):
            with st.expander(
                f"{t['input']['transaction_id']} — "
                f"{t['output']['confidence']} confidence — "
                f"{t['performance']['latency_ms']:.0f}ms — "
                f"{t['timestamp'][:19]}"
            ):
                st.json(t)
    else:
        st.caption("No traces yet — generate some explanations above")
except Exception as e:
    st.caption(f"Could not load traces: {e}")
