from pathlib import Path
from datetime import datetime
import os

import joblib
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transactions_features.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_NAME = "isolation_forest"
CANDIDATE_VERSION = "v1.1.0"

CANDIDATE_MODEL_FILE = (
    MODEL_DIR / "isolation_forest_v1.1.0.joblib"
)

CANDIDATE_CALIBRATION_FILE = (
    MODEL_DIR / "risk_calibration_v1.1.0.joblib"
)


# ============================================================
# DATABASE
# ============================================================

load_dotenv()


def get_engine():

    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB")

    missing = []

    if not user:
        missing.append("POSTGRES_USER")

    if not password:
        missing.append("POSTGRES_PASSWORD")

    if not database:
        missing.append("POSTGRES_DB")

    if missing:
        raise ValueError(
            "Missing database environment variables: "
            + ", ".join(missing)
        )

    database_url = (
        f"postgresql+psycopg2://"
        f"{user}:{password}@{host}:{port}/{database}"
    )

    return create_engine(database_url)


# ============================================================
# FEATURES
# ============================================================

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
# LOAD DATA
# ============================================================

def load_data():

    print("Loading feature data...")

    df = pd.read_csv(DATA_FILE)

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    X = df[FEATURES].copy()

    print(f"Rows       : {len(X):,}")
    print(f"Features   : {len(FEATURES)}")

    return X


# ============================================================
# LOAD CANDIDATE
# ============================================================

def load_candidate():

    if not CANDIDATE_MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Candidate model not found:\n"
            f"{CANDIDATE_MODEL_FILE}"
        )

    print()
    print("Loading candidate model...")
    print(CANDIDATE_MODEL_FILE)

    model = joblib.load(
        CANDIDATE_MODEL_FILE
    )

    return model


# ============================================================
# CREATE CALIBRATION
# ============================================================

def create_calibration(model, X):

    print()
    print("Creating candidate risk calibration...")

    # Isolation Forest decision_function:
    # larger values = more normal
    # smaller values = more anomalous
    #
    # Convert to anomaly score by multiplying by -1.

    anomaly_scores = -model.decision_function(X)

    min_score = float(
        np.min(anomaly_scores)
    )

    max_score = float(
        np.max(anomaly_scores)
    )

    if max_score <= min_score:

        raise ValueError(
            "Invalid calibration range: "
            "maximum score must be greater than minimum score."
        )

    calibration = {
        "min_score": min_score,
        "max_score": max_score,
        "created_at": datetime.now().isoformat(),
        "model_name": MODEL_NAME,
        "model_version": CANDIDATE_VERSION,
    }

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        calibration,
        CANDIDATE_CALIBRATION_FILE,
    )

    print()
    print("Calibration created:")
    print(
        f"Minimum anomaly score : {min_score:.6f}"
    )
    print(
        f"Maximum anomaly score : {max_score:.6f}"
    )

    print()
    print(
        f"Saved calibration:"
    )
    print(
        CANDIDATE_CALIBRATION_FILE
    )

    return calibration


# ============================================================
# VALIDATE CALIBRATION
# ============================================================

def validate_calibration(calibration):

    required_keys = [
        "min_score",
        "max_score",
        "model_name",
        "model_version",
    ]

    for key in required_keys:

        if key not in calibration:

            raise ValueError(
                f"Calibration missing key: {key}"
            )

    min_score = calibration["min_score"]
    max_score = calibration["max_score"]

    if max_score <= min_score:

        raise ValueError(
            "Calibration max_score must be "
            "greater than min_score."
        )

    if calibration["model_name"] != MODEL_NAME:

        raise ValueError(
            "Calibration model name does not match."
        )

    if (
        calibration["model_version"]
        != CANDIDATE_VERSION
    ):

        raise ValueError(
            "Calibration model version does not match "
            "candidate version."
        )

    print()
    print("Calibration validation: PASSED")


# ============================================================
# REGISTER MODEL
# ============================================================

def register_model(engine, training_rows):

    query = text("""
        INSERT INTO mlops.model_registry (
            model_name,
            model_version,
            model_type,
            model_path,
            calibration_path,
            training_date,
            training_rows,
            feature_count,
            status
        )
        VALUES (
            :model_name,
            :model_version,
            :model_type,
            :model_path,
            :calibration_path,
            :training_date,
            :training_rows,
            :feature_count,
            :status
        )
        ON CONFLICT (model_name, model_version)
        DO UPDATE SET
            model_type = EXCLUDED.model_type,
            model_path = EXCLUDED.model_path,
            calibration_path = EXCLUDED.calibration_path,
            training_date = EXCLUDED.training_date,
            training_rows = EXCLUDED.training_rows,
            feature_count = EXCLUDED.feature_count;
    """)

    with engine.begin() as connection:

        connection.execute(
            query,
            {
                "model_name": MODEL_NAME,
                "model_version": CANDIDATE_VERSION,
                "model_type": "IsolationForest",
                "model_path": str(
                    CANDIDATE_MODEL_FILE.relative_to(
                        PROJECT_ROOT
                    )
                ).replace("\\", "/"),
                "calibration_path": str(
                    CANDIDATE_CALIBRATION_FILE.relative_to(
                        PROJECT_ROOT
                    )
                ).replace("\\", "/"),
                "training_date": datetime.now(),
                "training_rows": training_rows,
                "feature_count": len(FEATURES),
                "status": "candidate",
            },
        )

    print()
    print("Model registered as candidate.")
    print(
        f"Model name    : {MODEL_NAME}"
    )
    print(
        f"Model version : {CANDIDATE_VERSION}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CANDIDATE MODEL CALIBRATION + REGISTRATION")
    print("=" * 70)

    print(
        f"Model       : {MODEL_NAME}"
    )

    print(
        f"Version     : {CANDIDATE_VERSION}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # 1. Connect database
    # --------------------------------------------------------

    engine = get_engine()

    # --------------------------------------------------------
    # 2. Check latest promotion decision
    # --------------------------------------------------------

    query = text("""
        SELECT
            decision,
            f1_score,
            recall_score,
            f1_improvement,
            recall_change
        FROM mlops.model_evaluations
        WHERE model_name = :model_name
          AND model_version = :model_version
        ORDER BY evaluated_at DESC
        LIMIT 1;
    """)

    with engine.connect() as connection:

        row = connection.execute(
            query,
            {
                "model_name": MODEL_NAME,
                "model_version": CANDIDATE_VERSION,
            },
        ).fetchone()

    if row is None:

        print()
        print(
            "No promotion evaluation found."
        )

        print(
            "Run retrain_and_promote first."
        )

        return

    print()
    print("Latest promotion result:")
    print(
        f"Decision        : {row.decision}"
    )
    print(
        f"F1 Score        : {row.f1_score}"
    )
    print(
        f"Recall          : {row.recall_score}"
    )
    print(
        f"F1 Improvement  : {row.f1_improvement}"
    )
    print(
        f"Recall Change   : {row.recall_change}"
    )

    # --------------------------------------------------------
    # 3. Stop if candidate was rejected
    # --------------------------------------------------------

    if row.decision != "promoted":

        print()
        print("=" * 70)
        print("REGISTRATION STOPPED")
        print("=" * 70)

        print()
        print(
            "Candidate did not pass the promotion gate."
        )

        print(
            "No calibration or registry update performed."
        )

        print(
            "Active v1.0.0 model remains unchanged."
        )

        print("=" * 70)

        return

    # --------------------------------------------------------
    # 4. Load data
    # --------------------------------------------------------

    X = load_data()

    # --------------------------------------------------------
    # 5. Load candidate
    # --------------------------------------------------------

    model = load_candidate()

    # --------------------------------------------------------
    # 6. Create calibration
    # --------------------------------------------------------

    calibration = create_calibration(
        model,
        X,
    )

    # --------------------------------------------------------
    # 7. Validate calibration
    # --------------------------------------------------------

    validate_calibration(
        calibration
    )

    # --------------------------------------------------------
    # 8. Register candidate
    # --------------------------------------------------------

    register_model(
        engine,
        training_rows=len(X),
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CANDIDATE REGISTRATION COMPLETE")
    print("=" * 70)

    print()
    print(
        "Candidate registered successfully."
    )

    print(
        "Active v1.0.0 model was NOT automatically replaced."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()