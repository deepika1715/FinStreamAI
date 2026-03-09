# Copyright (c) 2025 Deepika Sharma. Licensed under Apache 2.0.

from kafka import KafkaConsumer
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def create_consumer(topic: str = 'transactions'):
    """Create and return a Kafka consumer."""
    return KafkaConsumer(
        topic,
        bootstrap_servers='localhost:9092',
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='earliest',
        group_id='fraud-detection-group',
        consumer_timeout_ms=10000
    )

def consume_transactions():
    """Consume transactions from Kafka and print each one."""
    logger.info("Starting consumer — waiting for transactions...")
    consumer = create_consumer()
    received = 0

    for message in consumer:
        transaction = message.value
        received += 1

        # For now just log every 100th transaction
        if received % 100 == 0:
            logger.info(f"Received {received} transactions. Latest: {transaction}")

    logger.info(f"Consumer finished. Total received: {received}")

if __name__ == "__main__":
    consume_transactions()