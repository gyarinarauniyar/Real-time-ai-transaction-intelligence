from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split


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

BASELINE_MODEL = (
    MODEL_DIR / "isolation_forest.joblib"
)

CANDIDATE_MODEL = (
    MODEL_DIR / "isolation_forest_v1.1.0.joblib"
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

    X = df[FEATURES].copy()

    y = df["is_injected_anomaly"].astype(int)

    return X, y


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    y_pred = (predictions == -1).astype(int)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MODEL PROMOTION GATE")
    print("=" * 70)

    X, y = load_data()

    # --------------------------------------------------------
    # SAME TEST SET FOR BOTH MODELS
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"Training rows : {len(X_train):,}")
    print(f"Test rows     : {len(X_test):,}")

    # --------------------------------------------------------
    # Load baseline
    # --------------------------------------------------------

    print()
    print("Loading baseline model...")

    baseline_model = joblib.load(BASELINE_MODEL)

    # --------------------------------------------------------
    # Load candidate if available
    # --------------------------------------------------------

    if not CANDIDATE_MODEL.exists():

        print()
        print("Candidate model not found.")
        print("Promotion stopped.")

        return

    candidate_model = joblib.load(CANDIDATE_MODEL)

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    baseline = evaluate_model(
        baseline_model,
        X_test,
        y_test
    )

    candidate = evaluate_model(
        candidate_model,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BASELINE — v1.0.0")
    print("=" * 70)

    print(
        f"Precision : {baseline['precision']:.4f}"
    )

    print(
        f"Recall    : {baseline['recall']:.4f}"
    )

    print(
        f"F1        : {baseline['f1']:.4f}"
    )

    print()
    print("=" * 70)
    print("CANDIDATE — v1.1.0")
    print("=" * 70)

    print(
        f"Precision : {candidate['precision']:.4f}"
    )

    print(
        f"Recall    : {candidate['recall']:.4f}"
    )

    print(
        f"F1        : {candidate['f1']:.4f}"
    )

    # --------------------------------------------------------
    # Promotion criteria
    # --------------------------------------------------------

    f1_improved = (
        candidate["f1"] > baseline["f1"]
    )

    recall_not_decreased = (
        candidate["recall"] >= baseline["recall"]
    )

    print()
    print("=" * 70)
    print("PROMOTION CRITERIA")
    print("=" * 70)

    print(
        f"F1 improved          : "
        f"{f1_improved}"
    )

    print(
        f"Recall not decreased : "
        f"{recall_not_decreased}"
    )

    # --------------------------------------------------------
    # Promotion decision
    # --------------------------------------------------------

    promote = (
        f1_improved
        and recall_not_decreased
    )

    print()
    print("=" * 70)

    if promote:

        print("DECISION: PROMOTE CANDIDATE")
        print()
        print("Candidate model meets promotion criteria.")

    else:

        print("DECISION: REJECT CANDIDATE")
        print()
        print(
            "Candidate model does not improve "
            "the baseline sufficiently."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()