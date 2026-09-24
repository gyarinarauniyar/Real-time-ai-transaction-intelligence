import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transactions_features.csv"
)


DATABASE_URL = (
    f"postgresql+psycopg2://"
    f"{os.getenv('POSTGRES_USER')}:"
    f"{os.getenv('POSTGRES_PASSWORD')}@"
    f"{os.getenv('POSTGRES_HOST')}:"
    f"{os.getenv('POSTGRES_PORT')}/"
    f"{os.getenv('POSTGRES_DB')}"
)


MODEL_NAME = "isolation_forest"
MODEL_VERSION = "v1.0.0"


def get_engine():
    return create_engine(
        DATABASE_URL,
        pool_pre_ping=True
    )


def load_ground_truth():
    """
    Load the synthetic ground-truth labels.

    is_injected_anomaly = 1
        -> confirmed suspicious/anomalous

    is_injected_anomaly = 0
        -> legitimate
    """

    df = pd.read_csv(
        FEATURE_FILE,
        usecols=[
            "transaction_id",
            "is_injected_anomaly",
        ],
    )

    df = df.rename(
        columns={
            "is_injected_anomaly": "actual_label"
        }
    )

    df["actual_label"] = (
        df["actual_label"]
        .astype(int)
    )

    return df


def load_predictions(engine):
    query = text("""
        SELECT
            prediction_id,
            transaction_id
        FROM mlops.prediction_logs
        WHERE model_name = :model_name
          AND model_version = :model_version
        ORDER BY prediction_id;
    """)

    with engine.connect() as connection:
        return pd.read_sql(
            query,
            connection,
            params={
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
            },
        )


def store_feedback(
    engine,
    feedback_df
):

    if feedback_df.empty:
        return 0

    query = text("""
        INSERT INTO mlops.prediction_feedback (
            prediction_id,
            transaction_id,
            actual_label,
            feedback_source,
            notes
        )
        VALUES (
            :prediction_id,
            :transaction_id,
            :actual_label,
            'synthetic_ground_truth',
            :notes
        )
        ON CONFLICT DO NOTHING;
    """)

    records = []

    for row in feedback_df.itertuples(
        index=False
    ):

        records.append(
            {
                "prediction_id": int(
                    row.prediction_id
                ),
                "transaction_id": str(
                    row.transaction_id
                ),
                "actual_label": int(
                    row.actual_label
                ),
                "notes": (
                    "Ground truth derived from "
                    "controlled synthetic anomaly injection."
                ),
            }
        )

    with engine.begin() as connection:

        result = connection.execute(
            query,
            records
        )

    return result.rowcount


def main():

    print()
    print("=" * 60)
    print("GENERATING MODEL FEEDBACK")
    print("=" * 60)

    engine = get_engine()

    # ---------------------------------------------------------
    # Load ground truth
    # ---------------------------------------------------------

    ground_truth = load_ground_truth()

    print(
        f"Ground-truth transactions: "
        f"{len(ground_truth):,}"
    )

    # ---------------------------------------------------------
    # Load actual model predictions
    # ---------------------------------------------------------

    predictions = load_predictions(
        engine
    )

    print(
        f"Logged predictions: "
        f"{len(predictions):,}"
    )

    if predictions.empty:

        print()
        print(
            "No prediction logs found."
        )

        return

    # ---------------------------------------------------------
    # Join predictions with ground truth
    # ---------------------------------------------------------

    feedback = predictions.merge(
        ground_truth,
        on="transaction_id",
        how="inner",
    )

    print(
        f"Matched predictions: "
        f"{len(feedback):,}"
    )

    if feedback.empty:

        print()
        print(
            "No predictions could be "
            "matched to ground truth."
        )

        return

    # ---------------------------------------------------------
    # Store feedback
    # ---------------------------------------------------------

    inserted = store_feedback(
        engine,
        feedback
    )

    print(
        f"Feedback rows inserted: "
        f"{inserted:,}"
    )

    # ---------------------------------------------------------
    # Show label distribution
    # ---------------------------------------------------------

    print()
    print("Ground-truth distribution:")

    print(
        feedback[
            "actual_label"
        ].value_counts()
        .rename(
            index={
                0: "legitimate",
                1: "anomalous",
            }
        )
    )

    print()
    print(
        "Feedback generation complete."
    )


if __name__ == "__main__":
    main()