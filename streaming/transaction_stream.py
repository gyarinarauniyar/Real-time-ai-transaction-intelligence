"""
Local real-time transaction stream simulator.
"""

import time
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transactions_features.csv"
)


def stream_transactions(
    input_file: Path = INPUT_FILE,
    delay_seconds: float = 0.2,
):
    """
    Yield transactions one at a time as dictionaries.
    """

    df = pd.read_csv(input_file)

    # Normalize the timestamp column used by the consumer.
    timestamp_candidates = [
        "timestamp",
        "transaction_timestamp",
        "transaction_date",
        "datetime",
        "date",
    ]

    timestamp_column = next(
        (
            column
            for column in timestamp_candidates
            if column in df.columns
        ),
        None,
    )

    if timestamp_column is None:
        raise ValueError(
            "No timestamp column found. "
            f"Available columns: {df.columns.tolist()}"
        )

    if timestamp_column != "timestamp":
        df = df.rename(
            columns={timestamp_column: "timestamp"}
        )

    print(f"Loaded {len(df):,} transactions")
    print("Starting transaction stream...")
    print("Press Ctrl+C to stop.\n")

    for _, row in df.iterrows():
        event = row.to_dict()

        yield event

        time.sleep(delay_seconds)


if __name__ == "__main__":
    count = 0

    try:
        for transaction in stream_transactions():
            count += 1

            print(
                f"[EVENT {count}] "
                f"{transaction['transaction_id']} | "
                f"{transaction['customer_id']} | "
                f"₹{transaction['amount']:.2f} | "
                f"{transaction['location']}"
            )

    except KeyboardInterrupt:
        print(f"\nStream stopped after {count} events.")