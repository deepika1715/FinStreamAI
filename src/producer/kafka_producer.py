# Copyright (c) 2025 Deepika Sharma. Licensed under Apache 2.0.

from kafka import KafkaProducer
import pandas as pd
import json
import time
import logging

# Set up logging so we can see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def create_producer():
    """Create and return a Kafka producer."""
    return KafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        key_serializer=lambda k: k.encode('utf-8')
    )

def stream_transactions(csv_path: str, topic: str = 'transactions', delay: float = 0.01):
    """
    Read transactions from CSV and stream them to Kafka topic.
    delay = seconds between each message (0.01 = 100 transactions/sec)
    """
    logger.info(f"Loading transactions from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} transactions")

    producer = create_producer()
    sent = 0

    for _, row in df.iterrows():
        transaction = row.to_dict()
        transaction_id = f"txn_{sent}"

        producer.send(
            topic,
            key=transaction_id,
            value=transaction
        )

        sent += 1

        if sent % 1000 == 0:
            logger.info(f"Streamed {sent} transactions so far...")

        time.sleep(delay)

    producer.flush()
    logger.info(f"Done. Total transactions streamed: {sent}")

if __name__ == "__main__":
    stream_transactions('data/creditcard.csv')