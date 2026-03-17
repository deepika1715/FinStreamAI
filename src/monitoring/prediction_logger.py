# Copyright (c) 2025 Deepika Sharma. Licensed under Apache 2.0.

import sqlite3
import os
from datetime import datetime

# ── Database path ─────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), '../../data/predictions.db')
DB_PATH = os.path.abspath(DB_PATH)

def init_db():
    """Create the predictions table if it does not exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            amount          REAL NOT NULL,
            fraud_probability REAL NOT NULL,
            risk_level      TEXT NOT NULL,
            is_fraud        INTEGER NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def log_prediction(amount, fraud_probability, risk_level, is_fraud):
    """Log a single prediction to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions
            (timestamp, amount, fraud_probability, risk_level, is_fraud)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        datetime.utcnow().isoformat(),
        amount,
        fraud_probability,
        risk_level,
        int(is_fraud)
    ))
    conn.commit()
    conn.close()

def get_recent_predictions(limit=1000):
    """Return the most recent N predictions as a list of dicts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM predictions
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (limit,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def get_prediction_count():
    """Return total number of predictions logged."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM predictions')
    count = cursor.fetchone()[0]
    conn.close()
    return count