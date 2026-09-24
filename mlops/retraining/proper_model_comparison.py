from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
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
# EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    # Isolation Forest:
    #  1  = normal
    # -1  = anomaly

    y_pred = (predictions == -1).astype(int)

    accuracy = accuracy_score(y_test, y_pred)

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

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred
    ).ravel()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PROPER MODEL COMPARISON")
    print("=" * 70)

    X, y = load_data()

    print(f"Total transactions : {len(X):,}")
    print(f"Injected anomalies : {int(y.sum()):,}")

    # --------------------------------------------------------
    # Fixed train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print()
    print(f"Training rows : {len(X_train):,}")
    print(f"Test rows     : {len(X_test):,}")

    # --------------------------------------------------------
    # Train baseline model
    # --------------------------------------------------------

    print()
    print("Training baseline model...")

    baseline_model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )

    baseline_model.fit(X_train)

    # --------------------------------------------------------
    # Train candidate model
    # --------------------------------------------------------

    print("Training candidate model...")

    candidate_model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )

    candidate_model.fit(X_train)

    # --------------------------------------------------------
    # Evaluate both on SAME test set
    # --------------------------------------------------------

    baseline_results = evaluate_model(
        baseline_model,
        X_test,
        y_test
    )

    candidate_results = evaluate_model(
        candidate_model,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BASELINE MODEL — v1.0.0")
    print("=" * 70)

    print(f"Accuracy     : {baseline_results['accuracy']:.4f}")
    print(f"Precision    : {baseline_results['precision']:.4f}")
    print(f"Recall       : {baseline_results['recall']:.4f}")
    print(f"F1 Score     : {baseline_results['f1']:.4f}")

    print()
    print("Confusion Matrix:")
    print(
        f"[[{baseline_results['tn']:4d} {baseline_results['fp']:4d}]"
    )
    print(
        f" [{baseline_results['fn']:4d} {baseline_results['tp']:4d}]]"
    )

    print()
    print("=" * 70)
    print("CANDIDATE MODEL — v1.1.0")
    print("=" * 70)

    print(f"Accuracy     : {candidate_results['accuracy']:.4f}")
    print(f"Precision    : {candidate_results['precision']:.4f}")
    print(f"Recall       : {candidate_results['recall']:.4f}")
    print(f"F1 Score     : {candidate_results['f1']:.4f}")

    print()
    print("Confusion Matrix:")
    print(
        f"[[{candidate_results['tn']:4d} {candidate_results['fp']:4d}]"
    )
    print(
        f" [{candidate_results['fn']:4d} {candidate_results['tp']:4d}]]"
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        f"Precision: "
        f"{baseline_results['precision']:.4f}"
        f" -> "
        f"{candidate_results['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{baseline_results['recall']:.4f}"
        f" -> "
        f"{candidate_results['recall']:.4f}"
    )

    print(
        f"F1:        "
        f"{baseline_results['f1']:.4f}"
        f" -> "
        f"{candidate_results['f1']:.4f}"
    )

    print(
        f"False Positives: "
        f"{baseline_results['fp']}"
        f" -> "
        f"{candidate_results['fp']}"
    )

    print(
        f"False Negatives: "
        f"{baseline_results['fn']}"
        f" -> "
        f"{candidate_results['fn']}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()