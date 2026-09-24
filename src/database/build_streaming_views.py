"""
Build SQL views for monitoring streaming ingestion.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def get_database_url() -> str:
    return (
        f"postgresql+psycopg2://"
        f"{os.getenv('POSTGRES_USER')}:"
        f"{os.getenv('POSTGRES_PASSWORD')}@"
        f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
        f"{os.getenv('POSTGRES_PORT', '5432')}/"
        f"{os.getenv('POSTGRES_DB')}"
    )


def build_streaming_views() -> None:
    engine = create_engine(
        get_database_url(),
        pool_pre_ping=True,
    )

    query = text(
        """
        CREATE SCHEMA IF NOT EXISTS views;

        CREATE OR REPLACE VIEW views.streaming_health AS
        SELECT
            COUNT(*) AS total_events,
            COUNT(DISTINCT transaction_id) AS unique_transactions,
            MIN(ingested_at) AS first_ingested_at,
            MAX(ingested_at) AS last_ingested_at,
            COUNT(*) FILTER (
                WHERE anomaly_type <> 'normal'
            ) AS injected_anomaly_events,
            ROUND(
                AVG(amount)::numeric,
                2
            ) AS average_amount
        FROM streaming.transaction_events;

        CREATE OR REPLACE VIEW views.streaming_daily_volume AS
        SELECT
            DATE(ingested_at) AS ingestion_date,
            COUNT(*) AS event_count,
            ROUND(SUM(amount)::numeric, 2)
                AS total_transaction_value,
            ROUND(AVG(amount)::numeric, 2)
                AS average_transaction_amount
        FROM streaming.transaction_events
        GROUP BY DATE(ingested_at)
        ORDER BY ingestion_date;
        """
    )

    with engine.begin() as connection:
        connection.execute(query)

    print("Streaming views created successfully.")


if __name__ == "__main__":
    build_streaming_views()