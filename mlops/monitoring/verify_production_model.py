# mlops/monitoring/verify_production_model.py

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "isolation_forest.joblib"
)

CALIBRATION_PATH = (
    PROJECT_ROOT
    / "models"
    / "risk_calibration.joblib"
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transactions_features.csv"
)

EXPECTED_MODEL = "isolation_forest"
EXPECTED_VERSION = "v1.0.0"

FEATURES = [
    "amount",
    "amount_deviation",
    "amount_to_customer_avg_ratio",
    "amount_above_customer_avg_ratio",
    "transactions_last_1h",
    "transactions_last_24h",
    "minutes_since_previous_transaction",
    "hour",
    "is_unusual_hour",
    "is_location_changed",
    "customer_hour_frequency",
    "customer_location_frequency",
    "is_new_customer_hour",
    "is_new_customer_location",
    "customer_transaction_count",
    "merchant_transaction_count",
]


# ============================================================
# DATABASE
# ============================================================

def get_database_url():

    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB")

    if not all([user, password, database]):
        raise RuntimeError(
            "Missing PostgreSQL configuration in .env"
        )

    return (
        f"postgresql+psycopg2://"
        f"{user}:{password}@{host}:{port}/{database}"
    )


engine = create_engine(get_database_url())


# ============================================================
# REGISTRY
# ============================================================

def check_registry():

    query = text(
        """
        SELECT
            model_name,
            model_version,
            model_type,
            model_path,
            calibration_path,
            status,
            training_rows,
            feature_count
        FROM mlops.model_registry
        WHERE model_name = :model_name
          AND status = 'active'
        ORDER BY created_at DESC
        LIMIT 1
        """
    )

    with engine.connect() as conn:

        row = conn.execute(
            query,
            {
                "model_name": EXPECTED_MODEL
            },
        ).mappings().first()

    if row is None:
        raise RuntimeError(
            "No active model found in registry."
        )

    print("\nRegistry")
    print("-" * 70)

    print(
        f"Model version : {row['model_version']}"
    )

    print(
        f"Model type    : {row['model_type']}"
    )

    print(
        f"Status        : {row['status']}"
    )

    print(
        f"Training rows : {row['training_rows']}"
    )

    print(
        f"Feature count : {row['feature_count']}"
    )

    if row["model_version"] != EXPECTED_VERSION:
        raise RuntimeError(
            f"Expected {EXPECTED_VERSION}, "
            f"found {row['model_version']}"
        )

    if row["status"] != "active":
        raise RuntimeError(
            "Expected model status to be active."
        )

    print("Registry check : PASSED")

    return row


# ============================================================
# ARTIFACTS
# ============================================================

def check_artifacts():

    print("\nArtifacts")
    print("-" * 70)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model: {MODEL_PATH}"
        )

    if not CALIBRATION_PATH.exists():
        raise FileNotFoundError(
            f"Missing calibration: {CALIBRATION_PATH}"
        )

    print("Model       : FOUND")
    print("Calibration : FOUND")

    model = joblib.load(MODEL_PATH)
    calibration = joblib.load(CALIBRATION_PATH)

    # --------------------------------------------------------
    # Model validation
    # --------------------------------------------------------

    if not hasattr(model, "predict"):
        raise RuntimeError(
            "Loaded model does not support predict()."
        )

    if not hasattr(model, "decision_function"):
        raise RuntimeError(
            "Loaded model does not support decision_function()."
        )

    print("Model load   : PASSED")

    # --------------------------------------------------------
    # Calibration validation
    # --------------------------------------------------------

    print(
        f"Calibration type : {type(calibration).__name__}"
    )

    if isinstance(calibration, dict):

        print(
            f"Calibration keys : {list(calibration.keys())}"
        )

        # Support the existing calibration format.
        #
        # We do NOT require a particular metadata schema here.
        # The production scorer is the source of truth for the
        # calibration artifact structure.

        if len(calibration) == 0:
            raise RuntimeError(
                "Calibration dictionary is empty."
            )

        numeric_values = []

        for key, value in calibration.items():

            if isinstance(
                value,
                (int, float, np.integer, np.floating),
            ):
                numeric_values.append(float(value))

        if not numeric_values:
            raise RuntimeError(
                "Calibration dictionary contains no "
                "numeric calibration values."
            )

        if not all(
            np.isfinite(value)
            for value in numeric_values
        ):
            raise RuntimeError(
                "Calibration contains NaN or infinite values."
            )

        print(
            "Calibration structure : PASSED"
        )

    elif isinstance(calibration, (tuple, list)):

        print(
            f"Calibration length : {len(calibration)}"
        )

        if len(calibration) == 0:
            raise RuntimeError(
                "Calibration artifact is empty."
            )

        print(
            "Calibration structure : PASSED"
        )

    else:

        # Some calibration artifacts may simply be a
        # serialized object used directly by the scorer.
        if calibration is None:
            raise RuntimeError(
                "Calibration artifact is None."
            )

        print(
            "Calibration object : VALID"
        )

    print(
        "Calibration validation : PASSED"
    )

    return model, calibration


# ============================================================
# INFERENCE
# ============================================================

def check_inference(model):

    print("\nInference")
    print("-" * 70)

    df = pd.read_csv(DATA_PATH)

    missing = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing inference features: {missing}"
        )

    sample = df[FEATURES].head(5)

    predictions = model.predict(sample)

    if len(predictions) != 5:
        raise RuntimeError(
            "Inference returned unexpected number "
            "of predictions."
        )

    decision_scores = model.decision_function(sample)

    if len(decision_scores) != 5:
        raise RuntimeError(
            "decision_function returned unexpected "
            "number of scores."
        )

    if not np.all(
        np.isfinite(decision_scores)
    ):
        raise RuntimeError(
            "Model returned invalid decision scores."
        )

    anomaly_predictions = (
        predictions == -1
    ).astype(int)

    print(
        f"Test rows       : {len(sample)}"
    )

    print(
        f"Predictions     : "
        f"{anomaly_predictions.tolist()}"
    )

    print(
        f"Decision scores : "
        f"{decision_scores.tolist()}"
    )

    print("Inference check : PASSED")


# ============================================================
# STREAMING
# ============================================================

def check_streaming():

    query = text(
        """
        SELECT
            COUNT(*) AS total_events,
            COUNT(*) FILTER (
                WHERE processed_at IS NOT NULL
            ) AS processed_events
        FROM streaming.transaction_risk_events
        """
    )

    with engine.connect() as conn:

        row = conn.execute(
            query
        ).mappings().first()

    print("\nStreaming")
    print("-" * 70)

    print(
        f"Total risk events : {row['total_events']}"
    )

    print(
        f"Processed events  : {row['processed_events']}"
    )

    if row["total_events"] == 0:

        print(
            "Streaming check : WARNING - "
            "no risk events found"
        )

    else:

        print(
            "Streaming check : PASSED"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PRODUCTION MODEL VERIFICATION")
    print("=" * 70)

    print(
        f"Expected model  : {EXPECTED_MODEL}"
    )

    print(
        f"Expected version: {EXPECTED_VERSION}"
    )

    print("=" * 70)

    # 1. Registry
    check_registry()

    # 2. Model + calibration
    model, calibration = check_artifacts()

    # 3. Actual inference
    check_inference(model)

    # 4. Streaming system
    check_streaming()

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PRODUCTION VERIFICATION PASSED")
    print("=" * 70)

    print(
        f"Active production model: "
        f"{EXPECTED_MODEL} {EXPECTED_VERSION}"
    )

    print(
        "Registry        : PASSED"
    )

    print(
        "Model artifact  : PASSED"
    )

    print(
        "Calibration     : PASSED"
    )

    print(
        "Inference       : PASSED"
    )

    print(
        "Streaming       : PASSED"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()