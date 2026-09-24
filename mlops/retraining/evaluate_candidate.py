from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


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

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "isolation_forest_v1.1.0.joblib"
)


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

    df = pd.read_csv(DATA_FILE)

    X = df[FEATURES]

    # Controlled synthetic ground truth
    y_true = df["is_injected_anomaly"]

    return X, y_true


# ============================================================
# EVALUATE
# ============================================================

def main():

    print("=" * 60)
    print("CANDIDATE MODEL EVALUATION")
    print("=" * 60)

    print(f"Model: isolation_forest")
    print(f"Version: v1.1.0")
    print("=" * 60)

    X, y_true = load_data()

    _, X_test, _, y_test = train_test_split(
        X,
        y_true,
        test_size=0.20,
        random_state=42,
        stratify=y_true
    )

    X = X_test
    y_true = y_test

    print(f"Evaluation rows : {len(X):,}")

    model = joblib.load(MODEL_FILE)

    predictions = model.predict(X)

    # Isolation Forest:
    #  1  = normal
    # -1  = anomaly
    y_pred = (predictions == -1).astype(int)

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_true,
        y_pred
    )

    tn, fp, fn, tp = matrix.ravel()

    print()
    print("RESULTS")
    print("-" * 60)

    print(f"Accuracy     : {accuracy:.4f}")
    print(f"Precision    : {precision:.4f}")
    print(f"Recall       : {recall:.4f}")
    print(f"F1 Score     : {f1:.4f}")

    print()
    print("Confusion Matrix:")
    print(matrix)

    print()
    print(f"True Positive  : {tp}")
    print(f"True Negative  : {tn}")
    print(f"False Positive : {fp}")
    print(f"False Negative : {fn}")

    print("=" * 60)


if __name__ == "__main__":
    main()