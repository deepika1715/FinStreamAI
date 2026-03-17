# Copyright (c) 2025 Deepika Sharma. Licensed under Apache 2.0.

import numpy as np
import logging
import sqlite3
import os
from datetime import datetime
from src.monitoring.prediction_logger import get_recent_predictions, DB_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ── PSI Configuration ─────────────────────────────────────────────────────────
PSI_THRESHOLD = 0.2       # above this = significant drift
BASELINE_SIZE = 1000      # number of predictions to use as baseline
RECENT_SIZE   = 500       # number of recent predictions to compare against baseline
BINS          = 10        # number of buckets for PSI calculation

def calculate_psi(baseline_probs, recent_probs, bins=BINS):
    """
    Calculate Population Stability Index between baseline and recent distributions.
    PSI < 0.1  : No significant change
    PSI 0.1-0.2: Moderate change -- monitor closely
    PSI > 0.2  : Significant drift -- retrain recommended
    """
    # Create bin edges from baseline distribution
    bin_edges = np.percentile(baseline_probs, np.linspace(0, 100, bins + 1))
    bin_edges[0]  = 0.0
    bin_edges[-1] = 1.0
    # Remove duplicate edges
    bin_edges = np.unique(bin_edges)

    # Calculate proportions in each bin
    baseline_counts, _ = np.histogram(baseline_probs, bins=bin_edges)
    recent_counts,   _ = np.histogram(recent_probs,   bins=bin_edges)

    # Convert to proportions -- add small epsilon to avoid division by zero
    baseline_pct = (baseline_counts + 1e-6) / len(baseline_probs)
    recent_pct   = (recent_counts   + 1e-6) / len(recent_probs)

    # PSI formula
    psi = np.sum((recent_pct - baseline_pct) * np.log(recent_pct / baseline_pct))
    return float(psi)

def get_baseline_predictions():
    """Get the oldest BASELINE_SIZE predictions as the reference distribution."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
        SELECT fraud_probability FROM predictions
        ORDER BY timestamp ASC
        LIMIT ?
    ''', (BASELINE_SIZE,))
    rows = cursor.fetchall()
    conn.close()
    return [row['fraud_probability'] for row in rows]

def get_recent_probabilities():
    """Get the most recent RECENT_SIZE predictions for comparison."""
    predictions = get_recent_predictions(limit=RECENT_SIZE)
    return [p['fraud_probability'] for p in predictions]

def check_drift():
    """
    Run drift detection. Returns a dict with PSI score and drift status.
    """
    from src.monitoring.prediction_logger import get_prediction_count
    total = get_prediction_count()

    # Need enough data to compare
    if total < BASELINE_SIZE + RECENT_SIZE:
        logger.info(
            f"Not enough predictions for drift detection. "
            f"Have {total}, need {BASELINE_SIZE + RECENT_SIZE}."
        )
        return {
            'status': 'insufficient_data',
            'total_predictions': total,
            'required': BASELINE_SIZE + RECENT_SIZE,
            'psi': None,
            'drift_detected': False
        }

    baseline_probs = get_baseline_predictions()
    recent_probs   = get_recent_probabilities()

    psi = calculate_psi(baseline_probs, recent_probs)

    if psi < 0.1:
        status = 'stable'
        drift_detected = False
        logger.info(f"PSI: {psi:.4f} — Distribution stable. No action needed.")
    elif psi < PSI_THRESHOLD:
        status = 'monitoring'
        drift_detected = False
        logger.warning(f"PSI: {psi:.4f} — Moderate change detected. Monitoring.")
    else:
        status = 'drift_detected'
        drift_detected = True
        logger.warning(
            f"PSI: {psi:.4f} — Significant drift detected! "
            f"Retraining recommended."
        )

    return {
        'status': status,
        'total_predictions': total,
        'psi': round(psi, 4),
        'drift_detected': drift_detected,
        'checked_at': datetime.utcnow().isoformat()
    }

if __name__ == '__main__':
    print("Running drift detection...")
    result = check_drift()
    print(f"Result: {result}")