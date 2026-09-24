import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from .config import (
    MODEL_NAME,
    MODEL_VERSION,
    DRIFT_SAMPLE_SIZE,
    NUMERIC_DRIFT_THRESHOLD,
    CATEGORICAL_DRIFT_THRESHOLD,
)


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASELINE_FILE = (
    PROJECT_ROOT
    / "mlops"
    / "monitoring"
    / "baseline.json"
)


DATABASE_URL = (
    f"postgresql+psycopg2://"
    f"{os.getenv('POSTGRES_USER')}:"
    f"{os.getenv('POSTGRES_PASSWORD')}@"
    f"{os.getenv('POSTGRES_HOST')}:"
    f"{os.getenv('POSTGRES_PORT')}/"
    f"{os.getenv('POSTGRES_DB')}"
)


def get_engine():

    return create_engine(
        DATABASE_URL,
        pool_pre_ping=True
    )


def load_baseline():

    with open(
        BASELINE_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def load_recent_events(engine):

    query = text("""
        SELECT
            event_payload,
            processed_at
        FROM streaming.event_queue
        WHERE status = 'processed'
        ORDER BY processed_at DESC
        LIMIT :limit;
    """)

    with engine.connect() as connection:

        rows = connection.execute(
            query,
            {
                "limit": DRIFT_SAMPLE_SIZE
            }
        ).fetchall()

    if not rows:
        return pd.DataFrame()

    records = []

    for row in rows:

        payload = row.event_payload

        if isinstance(payload, str):
            payload = json.loads(payload)

        records.append(payload)

    return pd.DataFrame(records)


def calculate_numeric_drift(
    baseline_stats,
    current_series
):

    current_series = pd.to_numeric(
        current_series,
        errors="coerce"
    ).dropna()

    if current_series.empty:
        return 0.0

    baseline_mean = baseline_stats["mean"]
    baseline_std = baseline_stats["std"]

    if baseline_std == 0:
        return abs(
            float(current_series.mean())
            - baseline_mean
        )

    # Standardized mean shift
    drift_score = abs(
        float(current_series.mean())
        - baseline_mean
    ) / baseline_std

    return float(drift_score)


def calculate_categorical_drift(
    baseline_distribution,
    current_series
):

    current = (
        current_series
        .fillna("NULL")
        .astype(str)
        .value_counts(normalize=True)
    )

    categories = set(
        baseline_distribution.keys()
    ) | set(current.index)

    psi = 0.0

    for category in categories:

        expected = baseline_distribution.get(
            category,
            0.0001
        )

        actual = current.get(
            category,
            0.0001
        )

        expected = max(
            float(expected),
            0.0001
        )

        actual = max(
            float(actual),
            0.0001
        )

        psi += (
            (actual - expected)
            * np.log(actual / expected)
        )

    return float(psi)


def store_drift_result(
    engine,
    feature_name,
    drift_type,
    drift_score,
    threshold,
    drift_detected,
    reference_size,
    current_size,
):

    query = text("""
        INSERT INTO mlops.drift_results (
            model_name,
            model_version,
            feature_name,
            drift_type,
            drift_score,
            threshold,
            drift_detected,
            reference_size,
            current_size
        )
        VALUES (
            :model_name,
            :model_version,
            :feature_name,
            :drift_type,
            :drift_score,
            :threshold,
            :drift_detected,
            :reference_size,
            :current_size
        );
    """)

    with engine.begin() as connection:

        connection.execute(
            query,
            {
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "feature_name": feature_name,
                "drift_type": drift_type,
                "drift_score": drift_score,
                "threshold": threshold,
                "drift_detected": drift_detected,
                "reference_size": reference_size,
                "current_size": current_size,
            }
        )


def main():

    print()
    print("=" * 60)
    print("DATA DRIFT DETECTION")
    print("=" * 60)

    baseline = load_baseline()

    engine = get_engine()

    current_df = load_recent_events(
        engine
    )

    if current_df.empty:

        print(
            "No recent processed events found."
        )

        return

    print(
        f"Current sample: "
        f"{len(current_df):,}"
    )

    drift_count = 0

    # ---------------------------------------------------------
    # Numeric drift
    # ---------------------------------------------------------

    for feature, stats in baseline[
        "numeric"
    ].items():

        if feature not in current_df.columns:
            continue

        drift_score = calculate_numeric_drift(
            stats,
            current_df[feature]
        )

        detected = (
            drift_score
            >= NUMERIC_DRIFT_THRESHOLD
        )

        store_drift_result(
            engine,
            feature,
            "numeric_mean_shift",
            drift_score,
            NUMERIC_DRIFT_THRESHOLD,
            detected,
            baseline["row_count"],
            len(current_df),
        )

        status = (
            "DRIFT"
            if detected
            else "OK"
        )

        print(
            f"{feature:40s} "
            f"score={drift_score:.4f} "
            f"[{status}]"
        )

        if detected:
            drift_count += 1

    # ---------------------------------------------------------
    # Categorical drift
    # ---------------------------------------------------------

    for feature, distribution in baseline[
        "categorical"
    ].items():

        if feature not in current_df.columns:
            continue

        drift_score = calculate_categorical_drift(
            distribution,
            current_df[feature]
        )

        detected = (
            drift_score
            >= CATEGORICAL_DRIFT_THRESHOLD
        )

        store_drift_result(
            engine,
            feature,
            "categorical_psi",
            drift_score,
            CATEGORICAL_DRIFT_THRESHOLD,
            detected,
            baseline["row_count"],
            len(current_df),
        )

        status = (
            "DRIFT"
            if detected
            else "OK"
        )

        print(
            f"{feature:40s} "
            f"PSI={drift_score:.4f} "
            f"[{status}]"
        )

        if detected:
            drift_count += 1

    print()
    print(
        f"Drifted features: {drift_count}"
    )


if __name__ == "__main__":
    main()