from pathlib import Path
from datetime import datetime
import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest


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

CURRENT_VERSION = "v1.0.0"
NEW_VERSION = "v1.1.0"

MODEL_NAME = "isolation_forest"

MODEL_FILE = MODEL_DIR / f"{MODEL_NAME}_{NEW_VERSION}.joblib"


# These are the same features used by the current model.
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

def load_training_data():

    print("Loading training data...")

    df = pd.read_csv(DATA_FILE)

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    X = df[FEATURES].copy()

    print(f"Total rows     : {len(X):,}")
    print(f"Feature count  : {len(FEATURES)}")

    return X


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(X):

    print()
    print("Training new Isolation Forest model...")
    print()

    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X)

    print("Training complete.")

    return model


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model):

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_FILE)

    print()
    print(f"Model saved:")
    print(MODEL_FILE)

    return MODEL_FILE


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MODEL RETRAINING")
    print("=" * 60)

    print(f"Model name       : {MODEL_NAME}")
    print(f"Current version  : {CURRENT_VERSION}")
    print(f"New version      : {NEW_VERSION}")
    print(f"Training date    : {datetime.now()}")
    print("=" * 60)

    X = load_training_data()

    model = train_model(X)

    save_model(model)

    print()
    print("=" * 60)
    print("RETRAINING COMPLETE")
    print("=" * 60)
    print(f"New model       : {MODEL_NAME}")
    print(f"New version     : {NEW_VERSION}")
    print(f"Training rows   : {len(X):,}")
    print(f"Feature count   : {len(FEATURES)}")
    print(f"Model path      : {MODEL_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()