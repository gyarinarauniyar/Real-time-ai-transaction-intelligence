import json
import os
import time

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from src.ml.inference.realtime_scorer import RealTimeRiskScorer


load_dotenv()


DATABASE_URL = (
    f"postgresql+psycopg2://"
    f"{os.getenv('POSTGRES_USER')}:"
    f"{os.getenv('POSTGRES_PASSWORD')}@"
    f"{os.getenv('POSTGRES_HOST')}:"
    f"{os.getenv('POSTGRES_PORT')}/"
    f"{os.getenv('POSTGRES_DB')}"
)


POLL_INTERVAL = 1.0
BATCH_SIZE = 10


def get_engine():
    return create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
    )


def claim_events(engine):
    query = text(
        """
        SELECT
            event_id,
            transaction_id,
            event_payload
        FROM streaming.event_queue
        WHERE status = 'pending'
        ORDER BY event_id
        LIMIT :batch_size
        FOR UPDATE SKIP LOCKED;
        """
    )

    with engine.begin() as connection:

        rows = connection.execute(
            query,
            {"batch_size": BATCH_SIZE},
        ).fetchall()

        if not rows:
            return []

        event_ids = [row.event_id for row in rows]

        connection.execute(
            text(
                """
                UPDATE streaming.event_queue
                SET status = 'processing'
                WHERE event_id = ANY(:event_ids);
                """
            ),
            {"event_ids": event_ids},
        )

        return rows


def store_risk_event(engine, transaction_id, result):
    query = text(
        """
        INSERT INTO streaming.transaction_risk_events (
            transaction_id,
            anomaly_prediction,
            is_anomaly,
            risk_score,
            risk_level
        )
        VALUES (
            :transaction_id,
            :anomaly_prediction,
            :is_anomaly,
            :risk_score,
            :risk_level
        )
        ON CONFLICT (transaction_id)
        DO UPDATE SET
            anomaly_prediction = EXCLUDED.anomaly_prediction,
            is_anomaly = EXCLUDED.is_anomaly,
            risk_score = EXCLUDED.risk_score,
            risk_level = EXCLUDED.risk_level,
            processed_at = CURRENT_TIMESTAMP;
        """
    )

    with engine.begin() as connection:
        connection.execute(
            query,
            {
                "transaction_id": transaction_id,
                "anomaly_prediction": result["anomaly_prediction"],
                "is_anomaly": result["is_anomaly"],
                "risk_score": result["risk_score"],
                "risk_level": result["risk_level"],
            },
        )
def log_prediction(engine, transaction_id, result):
    query = text("""
        INSERT INTO mlops.prediction_logs (
            transaction_id,
            model_name,
            model_version,
            anomaly_prediction,
            is_anomaly,
            anomaly_score,
            risk_score,
            risk_level
        )
        VALUES (
            :transaction_id,
            :model_name,
            :model_version,
            :anomaly_prediction,
            :is_anomaly,
            :anomaly_score,
            :risk_score,
            :risk_level
        );
    """)

    with engine.begin() as connection:
        connection.execute(
            query,
            {
                "transaction_id": transaction_id,
                "model_name": result["model_name"],
                "model_version": result["model_version"],
                "anomaly_prediction": result["anomaly_prediction"],
                "is_anomaly": result["is_anomaly"],
                "anomaly_score": result["anomaly_score"],
                "risk_score": result["risk_score"],
                "risk_level": result["risk_level"],
            },
        )

def mark_processed(engine, event_id):
    query = text(
        """
        UPDATE streaming.event_queue
        SET
            status = 'processed',
            processed_at = CURRENT_TIMESTAMP,
            error_message = NULL
        WHERE event_id = :event_id;
        """
    )

    with engine.begin() as connection:
        connection.execute(
            query,
            {"event_id": event_id},
        )


def mark_failed(engine, event_id, error):
    query = text(
        """
        UPDATE streaming.event_queue
        SET
            status = 'failed',
            error_message = :error_message
        WHERE event_id = :event_id;
        """
    )

    with engine.begin() as connection:
        connection.execute(
            query,
            {
                "event_id": event_id,
                "error_message": str(error)[:1000],
            },
        )


def process_event(engine, scorer, event):
    event_id = event.event_id
    transaction_id = event.transaction_id

    try:
        payload = event.event_payload

        if isinstance(payload, str):
            payload = json.loads(payload)


        result = scorer.score(payload)

        store_risk_event(
            engine,
            transaction_id,
            result,
        )
        log_prediction(
            engine,
            transaction_id,
            result
        )
        mark_processed(
            engine,
            event_id,
        )

        print(
            f"[CONSUMER] "
            f"{transaction_id} | "
            f"risk={result['risk_score']:.2f} | "
            f"level={result['risk_level']} | "
            f"anomaly={result['is_anomaly']}"
        )

    except Exception as error:

        mark_failed(
            engine,
            event_id,
            error,
        )

        print(
            f"[CONSUMER ERROR] "
            f"{transaction_id}: {error}"
        )


def main():
    engine = get_engine()
    scorer = RealTimeRiskScorer()

    print()
    print("=" * 60)
    print("REAL-TIME TRANSACTION CONSUMER")
    print("=" * 60)
    print(f"Polling interval : {POLL_INTERVAL}s")
    print(f"Batch size       : {BATCH_SIZE}")
    print("=" * 60)
    print()
    print("[CONSUMER] Waiting for events...")
    print()

    while True:

        events = claim_events(engine)

        if not events:
            time.sleep(POLL_INTERVAL)
            continue

        for event in events:
            process_event(
                engine,
                scorer,
                event,
            )


if __name__ == "__main__":
    main()