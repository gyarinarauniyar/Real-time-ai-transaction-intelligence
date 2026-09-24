import json
import os
import time

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv


load_dotenv()


DATABASE_URL = (
    f"postgresql+psycopg2://"
    f"{os.getenv('POSTGRES_USER')}:"
    f"{os.getenv('POSTGRES_PASSWORD')}@"
    f"{os.getenv('POSTGRES_HOST')}:"
    f"{os.getenv('POSTGRES_PORT')}/"
    f"{os.getenv('POSTGRES_DB')}"
)


FEATURE_FILE = "data/processed/transactions_features.csv"

EVENTS_TO_SEND = 100
DELAY_SECONDS = 1.0

# Percentage of injected anomalies included in the stream.
ANOMALY_RATIO = 0.10


def get_engine():
    return create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
    )


def load_transactions():
    print(f"Loading transactions from: {FEATURE_FILE}")

    df = pd.read_csv(FEATURE_FILE)

    print(f"Loaded {len(df):,} transactions.")

    return df


def select_stream_transactions(df):
    """
    Select a realistic mixture of normal and injected-anomaly
    transactions for the real-time stream.
    """

    anomaly_df = df[
        df["is_injected_anomaly"] == 1
    ].copy()

    normal_df = df[
        df["is_injected_anomaly"] == 0
    ].copy()

    anomaly_count = max(
        1,
        int(EVENTS_TO_SEND * ANOMALY_RATIO)
    )

    normal_count = EVENTS_TO_SEND - anomaly_count

    anomaly_sample = anomaly_df.sample(
        n=min(anomaly_count, len(anomaly_df)),
        random_state=None,
    )

    normal_sample = normal_df.sample(
        n=min(normal_count, len(normal_df)),
        random_state=None,
    )

    selected = pd.concat(
        [
            anomaly_sample,
            normal_sample,
        ],
        ignore_index=True,
    )

    # Randomize the order so anomalies don't appear together.
    selected = selected.sample(
        frac=1,
        random_state=None,
    ).reset_index(drop=True)

    return selected


def publish_transaction(engine, row):
    transaction_id = str(row["transaction_id"])

    payload = {}

    for column, value in row.items():

        if pd.isna(value):
            payload[column] = None

        elif hasattr(value, "item"):
            payload[column] = value.item()

        else:
            payload[column] = value

    query = text(
        """
        INSERT INTO streaming.event_queue (
            transaction_id,
            event_payload,
            status
        )
        VALUES (
            :transaction_id,
            CAST(:event_payload AS JSONB),
            'pending'
        )
        ON CONFLICT (transaction_id)
        DO NOTHING;
        """
    )

    with engine.begin() as connection:

        result = connection.execute(
            query,
            {
                "transaction_id": transaction_id,
                "event_payload": json.dumps(payload),
            },
        )

    return result.rowcount


def main():

    engine = get_engine()

    df = load_transactions()

    stream_df = select_stream_transactions(df)

    sent = 0
    skipped = 0

    anomaly_events = int(
        stream_df["is_injected_anomaly"].sum()
    )

    normal_events = len(stream_df) - anomaly_events

    print()
    print("=" * 60)
    print("REAL-TIME TRANSACTION PRODUCER")
    print("=" * 60)
    print(f"Target events       : {EVENTS_TO_SEND}")
    print(f"Normal events       : {normal_events}")
    print(f"Injected anomalies  : {anomaly_events}")
    print(f"Delay               : {DELAY_SECONDS}s")
    print("=" * 60)
    print()

    for _, row in stream_df.iterrows():

        if sent >= EVENTS_TO_SEND:
            break

        transaction_id = row["transaction_id"]

        inserted = publish_transaction(
            engine,
            row,
        )

        if inserted:

            sent += 1

            anomaly_flag = int(
                row["is_injected_anomaly"]
            )

            anomaly_label = (
                "ANOMALY"
                if anomaly_flag
                else "normal"
            )

            print(
                f"[PRODUCER] "
                f"{sent:03d}/{EVENTS_TO_SEND} "
                f"{transaction_id} "
                f"[{anomaly_label}]"
            )

            time.sleep(DELAY_SECONDS)

        else:
            skipped += 1

    print()
    print(
        f"[PRODUCER] Finished. "
        f"Published {sent} events. "
        f"Skipped {skipped} duplicates."
    )


if __name__ == "__main__":
    main()