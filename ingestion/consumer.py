"""
Local streaming consumer.

Consumes transaction events, performs real-time ML risk scoring,
and writes both raw events and risk results into PostgreSQL.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from streaming.transaction_stream import stream_transactions
from src.ml.inference.realtime_scorer import RealTimeRiskScorer


load_dotenv(PROJECT_ROOT / ".env")


def get_database_url() -> str:
    """Build the PostgreSQL connection URL from .env."""

    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB")

    if not all([user, password, database]):
        raise ValueError(
            "Missing PostgreSQL configuration in .env"
        )

    return (
        f"postgresql+psycopg2://{user}:{password}"
        f"@{host}:{port}/{database}"
    )


def create_streaming_tables(engine) -> None:
    """Create tables required by the streaming pipeline."""

    query = text(
        """
        CREATE SCHEMA IF NOT EXISTS streaming;

        CREATE TABLE IF NOT EXISTS streaming.transaction_events (
            transaction_id VARCHAR(100) PRIMARY KEY,
            customer_id VARCHAR(100),
            merchant_id VARCHAR(100),
            timestamp TIMESTAMP,
            amount NUMERIC(12, 2),
            currency VARCHAR(10),
            payment_method VARCHAR(50),
            merchant_category VARCHAR(100),
            location VARCHAR(100),
            device_type VARCHAR(50),
            is_injected_anomaly INTEGER,
            anomaly_type VARCHAR(50),
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS streaming.transaction_risk_events (
            risk_event_id BIGSERIAL PRIMARY KEY,
            transaction_id VARCHAR(100) UNIQUE NOT NULL,
            anomaly_prediction INTEGER NOT NULL,
            is_anomaly INTEGER NOT NULL,
            risk_score NUMERIC(6, 2) NOT NULL,
            risk_level VARCHAR(20) NOT NULL,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS streaming.ingestion_runs (
            run_id BIGSERIAL PRIMARY KEY,
            started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            events_processed INTEGER DEFAULT 0,
            events_inserted INTEGER DEFAULT 0,
            risk_events_created INTEGER DEFAULT 0,
            status VARCHAR(20) DEFAULT 'running'
        );
        """
    )

    with engine.begin() as connection:
        connection.execute(query)


def consume_stream(
    batch_size: int = 100,
    delay_seconds: float = 0.05,
    max_events: int = 500,
) -> None:
    """
    Consume streaming events and perform real-time ML inference.
    """

    engine = create_engine(
        get_database_url(),
        pool_pre_ping=True,
    )

    create_streaming_tables(engine)

    # Load the trained ML model once.
    scorer = RealTimeRiskScorer()

    batch = []
    risk_batch = []

    total_processed = 0
    total_inserted = 0
    total_risk_events = 0

    print("Starting real-time transaction consumer...")
    print(f"Maximum events: {max_events}")
    print("ML model loaded successfully.")
    print("Press Ctrl+C to stop.\n")

    try:
        for event in stream_transactions(
            delay_seconds=delay_seconds
        ):

            # --------------------------------------------------
            # 1. Raw transaction event
            # --------------------------------------------------

            batch.append(
                {
                    "transaction_id": event["transaction_id"],
                    "customer_id": event["customer_id"],
                    "merchant_id": event["merchant_id"],
                    "timestamp": event["timestamp"],
                    "amount": event["amount"],
                    "currency": event["currency"],
                    "payment_method": event["payment_method"],
                    "merchant_category": event[
                        "merchant_category"
                    ],
                    "location": event["location"],
                    "device_type": event["device_type"],
                    "is_injected_anomaly": event[
                        "is_injected_anomaly"
                    ],
                    "anomaly_type": event["anomaly_type"],
                }
            )

            # --------------------------------------------------
            # 2. Real-time ML inference
            # --------------------------------------------------

            risk_result = scorer.score(event)

            risk_batch.append(
                {
                    "transaction_id": event["transaction_id"],
                    "anomaly_prediction": risk_result[
                        "anomaly_prediction"
                    ],
                    "is_anomaly": risk_result["is_anomaly"],
                    "risk_score": risk_result["risk_score"],
                    "risk_level": risk_result["risk_level"],
                }
            )

            total_processed += 1

            # --------------------------------------------------
            # 3. Batch persistence
            # --------------------------------------------------

            if len(batch) >= batch_size:

                batch_df = pd.DataFrame(batch)

                batch_df.to_sql(
                    "transaction_events",
                    engine,
                    schema="streaming",
                    if_exists="append",
                    index=False,
                    method="multi",
                )

                risk_df = pd.DataFrame(risk_batch)

                risk_df.to_sql(
                    "transaction_risk_events",
                    engine,
                    schema="streaming",
                    if_exists="append",
                    index=False,
                    method="multi",
                )

                total_inserted += len(batch_df)
                total_risk_events += len(risk_df)

                batch.clear()
                risk_batch.clear()

                print(
                    f"Processed: {total_processed:,} | "
                    f"Transactions: {total_inserted:,} | "
                    f"Risk events: {total_risk_events:,}"
                )

            if total_processed >= max_events:
                break

    except KeyboardInterrupt:
        print("\nConsumer stopped by user.")

    finally:

        # ------------------------------------------------------
        # Persist final partial batch
        # ------------------------------------------------------

        if batch:

            batch_df = pd.DataFrame(batch)

            batch_df.to_sql(
                "transaction_events",
                engine,
                schema="streaming",
                if_exists="append",
                index=False,
                method="multi",
            )

            total_inserted += len(batch_df)

        if risk_batch:

            risk_df = pd.DataFrame(risk_batch)

            risk_df.to_sql(
                "transaction_risk_events",
                engine,
                schema="streaming",
                if_exists="append",
                index=False,
                method="multi",
            )

            total_risk_events += len(risk_df)

        print("\nStreaming consumer finished.")
        print(f"Total processed: {total_processed:,}")
        print(f"Total transactions inserted: {total_inserted:,}")
        print(f"Total risk events: {total_risk_events:,}")


if __name__ == "__main__":
    consume_stream()