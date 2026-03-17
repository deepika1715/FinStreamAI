# Copyright (c) 2025 Deepika Sharma. Licensed under Apache 2.0.

import json
import time
import requests
from kafka import KafkaConsumer

# ── Configuration ────────────────────────────────────────────────────────────
KAFKA_TOPIC = 'transactions'
KAFKA_BROKER = 'localhost:9092'
PREDICT_URL = 'http://localhost:8000/predict'
GROUP_ID = 'fraud-detection-group'

# ── Consumer ─────────────────────────────────────────────────────────────────
consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=[KAFKA_BROKER],
    group_id=GROUP_ID,
    auto_offset_reset='latest',       # on restart, only process new messages
    enable_auto_commit=True,          # commit offset after each message
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

print(f"Consumer started — listening on topic '{KAFKA_TOPIC}'")
print(f"Sending predictions to {PREDICT_URL}")
print("-" * 60)

# ── Main loop ────────────────────────────────────────────────────────────────
for message in consumer:
    transaction = message.value

    try:
        response = requests.post(
            PREDICT_URL,
            json=transaction,
            timeout=5       # never block longer than 5 seconds
        )

        if response.status_code == 200:
            result = response.json()
            amount = result.get('amount', 0)
            risk = result.get('risk_level', 'UNKNOWN')
            prob = result.get('fraud_probability', 0)
            is_fraud = result.get('is_fraud', False)

            flag = "FRAUD" if is_fraud else "OK"
            print(f"[{flag}]  Amount: £{amount:.2f}  "
                  f"Risk: {risk}  "
                  f"Probability: {prob:.4f}  "
                  f"Offset: {message.offset}")

        else:
            print(f"[SKIP]  /predict returned {response.status_code} "
                  f"— offset {message.offset} skipped")

    except requests.exceptions.Timeout:
        print(f"[SKIP]  /predict timed out — offset {message.offset} skipped")

    except requests.exceptions.ConnectionError:
        print(f"[WAIT]  API not reachable — waiting 3 seconds")
        time.sleep(3)

    except Exception as e:
        print(f"[ERROR] {e} — offset {message.offset} skipped")

    # Offset commits automatically after each iteration
    # No message is ever retried — no loop possible