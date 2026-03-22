# Copyright (c) 2025 Deepika Sharma. Licensed under Apache 2.0.
"""
FinStreamAI - Week 1, Step 2
RAG retrieval + rule-based explanation engine.
No API key required. Demonstrates full RAG pipeline.
"""

import math
import re
import pickle
import logging
from pathlib import Path
from collections import Counter

import faiss
import numpy as np

logger = logging.getLogger(__name__)

KB_DIR      = Path(__file__).parent.parent.parent / "knowledge_base"
INDEX_FILE  = KB_DIR / "index.faiss"
CHUNKS_FILE = KB_DIR / "chunks.pkl"
VOCAB_FILE  = KB_DIR / "vocab.pkl"

PROMPT_VERSION = "explain_v1"

# Recommended actions per risk level
ACTIONS = {
    "HIGH":   "Immediately block transaction, freeze card, and escalate to fraud operations team for same-day review.",
    "MEDIUM": "Place transaction on 24-hour hold and send real-time alert to cardholder for confirmation.",
    "LOW":    "Flag for routine end-of-day review by the compliance team; no immediate action required.",
}


class FraudExplainer:
    def __init__(self):
        self._index  = None
        self._chunks = None
        self._vocab  = None
        self._loaded = False
        self._load()

    def _load(self):
        try:
            self._index = faiss.read_index(str(INDEX_FILE))
            with open(CHUNKS_FILE, "rb") as f:
                self._chunks = pickle.load(f)
            with open(VOCAB_FILE, "rb") as f:
                self._vocab = pickle.load(f)
            self._loaded = True
            logger.info(
                f"FraudExplainer loaded - "
                f"{self._index.ntotal} patterns, vocab {len(self._vocab)}"
            )
        except Exception as e:
            logger.error(f"FraudExplainer failed to load: {e}")

    def _vectorise_query(self, query: str) -> np.ndarray:
        tokens   = re.findall(r"[a-z]+", query.lower())
        tf       = Counter(tokens)
        total    = len(tokens) if tokens else 1
        n_docs   = len(self._chunks)
        vec      = np.zeros(len(self._vocab), dtype="float32")
        word2idx = {w: i for i, w in enumerate(self._vocab)}
        for word, count in tf.items():
            if word in word2idx:
                tf_val  = count / total
                idf_val = math.log((n_docs + 1) / 2) + 1
                vec[word2idx[word]] = tf_val * idf_val
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.reshape(1, -1)

    def _retrieve(self, query: str, k: int = 2) -> list:
        vec          = self._vectorise_query(query)
        scores, idxs = self._index.search(vec, k)
        results      = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx >= 0 and score > 0:
                results.append(self._chunks[idx])
        return results

    @staticmethod
    def _extract_pattern_name(chunk: str) -> str:
        """Extract pattern name from the first line e.g. PATTERN: Card Testing"""
        first_line = chunk.split("\n")[0]
        match = re.search(r"PATTERN:\s*(.+)", first_line)
        return match.group(1).strip() if match else "Unknown Pattern"

    @staticmethod
    def _extract_indicators(chunk: str) -> str:
        """Pull the Risk indicators / Indicators sentence from the pattern."""
        for line in chunk.split("."):
            line = line.strip()
            if "indicator" in line.lower() or "risk signal" in line.lower():
                return line.strip(".").strip() + "."
        return ""

    @staticmethod
    def _build_query(fraud_probability, risk_level, amount, top_features):
        parts = [
            f"fraud probability {fraud_probability:.0%}",
            f"risk level {risk_level}",
            f"amount {amount:.2f}",
        ]
        for feat, val in top_features.items():
            parts.append(f"{feat} value {val:.2f}")
        return " ".join(parts)

    def explain(self, transaction_id, fraud_probability,
                risk_level, amount, top_features) -> dict:
        if not self._loaded:
            raise RuntimeError(
                "FraudExplainer not loaded - check knowledge_base files"
            )

        # 1. Retrieve closest fraud patterns via TF-IDF + FAISS
        query    = self._build_query(fraud_probability, risk_level, amount, top_features)
        patterns = self._retrieve(query, k=2)

        if not patterns:
            top_pattern      = "Unknown"
            top_chunk        = ""
            retrieved_context = "No matching pattern found in knowledge base."
        else:
            top_chunk         = patterns[0]
            top_pattern       = self._extract_pattern_name(top_chunk)
            retrieved_context = "\n\n".join(patterns)

        # 2. Build rule-based explanation from retrieved pattern
        indicators = self._extract_indicators(top_chunk) if top_chunk else ""

        feature_summary = (
            ", ".join(f"{k}: {v:.2f}" for k, v in list(top_features.items())[:3])
            if top_features else "anomalous feature values detected"
        )

        explanation = (
            f"Transaction {transaction_id} was flagged with {fraud_probability:.1%} fraud "
            f"probability based on anomalous feature values ({feature_summary}). "
            f"The transaction amount of GBP {amount:.2f} combined with the detected "
            f"feature pattern is consistent with known fraud behaviour."
        )

        matched_pattern = (
            f"This transaction most closely resembles {top_pattern}. "
            f"{indicators}"
        ).strip()

        # 3. Derive confidence
        if fraud_probability >= 0.85:
            confidence = "HIGH"
        elif fraud_probability >= 0.5:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        return {
            "transaction_id":            transaction_id,
            "explanation":               explanation,
            "matched_pattern":           matched_pattern,
            "recommended_action":        ACTIONS.get(risk_level, ACTIONS["MEDIUM"]),
            "confidence":                confidence,
            "prompt_version":            PROMPT_VERSION,
            "retrieved_context_preview": retrieved_context[:120],
        }
