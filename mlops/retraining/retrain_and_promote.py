# mlops/retraining/retrain_and_promote.py

import os
import itertools
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine, text


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transactions_features.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

CANDIDATE_VERSION = "v1.1.0"
ACTIVE_VERSION = "v1.0.0"

CANDIDATE_MODEL_PATH = (
    MODEL_DIR / f"isolation_forest_{CANDIDATE_VERSION}.joblib"
)

RANDOM_STATE = 42

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
# LOAD DATA
# ============================================================

def load_data():

    print("\nLoading dataset...")

    df = pd.read_csv(DATA_PATH)

    missing = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required features: {missing}"
        )

    if "is_injected_anomaly" not in df.columns:
        raise ValueError(
            "Column 'is_injected_anomaly' not found."
        )

    X = df[FEATURES].copy()
    y = df["is_injected_anomaly"].astype(int)

    X = X.replace([np.inf, -np.inf], np.nan)

    if X.isnull().any().any():
        raise ValueError(
            "Features contain NaN or infinite values."
        )

    print(f"Total rows           : {len(df):,}")
    print(f"Features             : {len(FEATURES)}")
    print(f"Injected anomalies   : {y.sum():,}")
    print(f"Normal rows          : {(y == 0).sum():,}")

    return X, y


# ============================================================
# SPLIT
# ============================================================

def create_split(X, y):

    return train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )


# ============================================================
# TRAIN
# ============================================================

def train_model(params, X_train):

    model = IsolationForest(
        n_estimators=params["n_estimators"],
        max_samples=params["max_samples"],
        max_features=params["max_features"],
        contamination=params["contamination"],
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train)

    return model


# ============================================================
# EVALUATE
# ============================================================

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    y_pred = np.where(
        predictions == -1,
        1,
        0,
    )

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
    }


# ============================================================
# SAVE EVALUATION
# ============================================================

def save_evaluation(
    metrics,
    training_rows,
    test_rows,
    decision,
    f1_improvement,
    recall_change,
    notes,
):

    query = text(
        """
        INSERT INTO mlops.model_evaluations (
            model_name,
            model_version,
            evaluation_type,
            training_rows,
            test_rows,
            accuracy,
            precision_score,
            recall_score,
            f1_score,
            true_positive,
            true_negative,
            false_positive,
            false_negative,
            decision,
            f1_improvement,
            recall_change,
            notes
        )
        VALUES (
            :model_name,
            :model_version,
            :evaluation_type,
            :training_rows,
            :test_rows,
            :accuracy,
            :precision_score,
            :recall_score,
            :f1_score,
            :true_positive,
            :true_negative,
            :false_positive,
            :false_negative,
            :decision,
            :f1_improvement,
            :recall_change,
            :notes
        )
        """
    )

    with engine.begin() as conn:

        conn.execute(
            query,
            {
                "model_name": "isolation_forest",
                "model_version": CANDIDATE_VERSION,
                "evaluation_type": "retraining_candidate_search",
                "training_rows": training_rows,
                "test_rows": test_rows,
                "accuracy": metrics["accuracy"],
                "precision_score": metrics["precision"],
                "recall_score": metrics["recall"],
                "f1_score": metrics["f1"],
                "true_positive": metrics["true_positive"],
                "true_negative": metrics["true_negative"],
                "false_positive": metrics["false_positive"],
                "false_negative": metrics["false_negative"],
                "decision": decision,
                "f1_improvement": f1_improvement,
                "recall_change": recall_change,
                "notes": notes,
            },
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MLOPS RETRAINING + RECALL-CONSTRAINED SEARCH")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    X, y = load_data()

    X_train, X_test, y_train, y_test = create_split(
        X,
        y,
    )

    print("\nTrain/Test split")
    print("-" * 70)
    print(f"Training rows : {len(X_train):,}")
    print(f"Testing rows  : {len(X_test):,}")

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BASELINE v1.0.0")
    print("=" * 70)

    baseline_params = {
        "n_estimators": 200,
        "max_samples": "auto",
        "max_features": 1.0,
        "contamination": "auto",
    }

    baseline_model = train_model(
        baseline_params,
        X_train,
    )

    baseline_metrics = evaluate_model(
        baseline_model,
        X_test,
        y_test,
    )

    print(
        f"Accuracy  : {baseline_metrics['accuracy']:.4f}"
    )
    print(
        f"Precision : {baseline_metrics['precision']:.4f}"
    )
    print(
        f"Recall    : {baseline_metrics['recall']:.4f}"
    )
    print(
        f"F1 Score  : {baseline_metrics['f1']:.4f}"
    )

    # --------------------------------------------------------
    # SEARCH SPACE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CANDIDATE SEARCH")
    print("=" * 70)

    parameter_grid = {
        "n_estimators": [200, 300, 500],
        "max_samples": ["auto", 0.8],
        "max_features": [0.7, 1.0],
        "contamination": ["auto", 0.03, 0.05, 0.08],
    }

    keys = list(parameter_grid.keys())

    configurations = [
        dict(zip(keys, values))
        for values in itertools.product(
            *parameter_grid.values()
        )
    ]

    print(
        f"Configurations to test: {len(configurations)}"
    )

    results = []

    # --------------------------------------------------------
    # TRAIN CANDIDATES
    # --------------------------------------------------------

    for index, params in enumerate(
        configurations,
        start=1,
    ):

        print(
            f"\n[{index}/{len(configurations)}] "
            f"{params}"
        )

        model = train_model(
            params,
            X_train,
        )

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        recall_change = (
            metrics["recall"]
            - baseline_metrics["recall"]
        )

        f1_change = (
            metrics["f1"]
            - baseline_metrics["f1"]
        )

        recall_pass = recall_change >= 0
        f1_pass = f1_change >= 0

        results.append(
            {
                **params,
                **metrics,
                "f1_change": f1_change,
                "recall_change": recall_change,
                "recall_pass": recall_pass,
                "f1_pass": f1_pass,
                "model": model,
            }
        )

        print(
            f"F1={metrics['f1']:.4f} | "
            f"Recall={metrics['recall']:.4f} | "
            f"Precision={metrics['precision']:.4f}"
        )

    # --------------------------------------------------------
    # FILTER VALID CANDIDATES
    # --------------------------------------------------------

    valid_candidates = [
        result
        for result in results
        if result["f1_pass"]
        and result["recall_pass"]
    ]

    print("\n" + "=" * 70)
    print("VALID CANDIDATES")
    print("=" * 70)

    print(
        f"Candidates passing both gates: "
        f"{len(valid_candidates)}"
    )

    # --------------------------------------------------------
    # SELECT BEST VALID CANDIDATE
    # --------------------------------------------------------

    if valid_candidates:

        valid_candidates.sort(
            key=lambda x: (
                x["f1"],
                x["recall"],
                x["precision"],
            ),
            reverse=True,
        )

        best = valid_candidates[0]

        best_model = best["model"]
        best_params = {
            key: best[key]
            for key in parameter_grid.keys()
        }

        best_metrics = {
            key: best[key]
            for key in [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "true_positive",
                "true_negative",
                "false_positive",
                "false_negative",
            ]
        }

    else:

        # No candidate satisfies both conditions.
        # Keep the best-F1 candidate only as an artifact,
        # but reject it from promotion.

        results.sort(
            key=lambda x: (
                x["f1"],
                x["recall"],
            ),
            reverse=True,
        )

        best = results[0]

        best_model = best["model"]

        best_params = {
            key: best[key]
            for key in parameter_grid.keys()
        }

        best_metrics = {
            key: best[key]
            for key in [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "true_positive",
                "true_negative",
                "false_positive",
                "false_negative",
            ]
        }

    # --------------------------------------------------------
    # FINAL COMPARISON
    # --------------------------------------------------------

    f1_improvement = (
        best_metrics["f1"]
        - baseline_metrics["f1"]
    )

    recall_change = (
        best_metrics["recall"]
        - baseline_metrics["recall"]
    )

    f1_criterion = f1_improvement >= 0
    recall_criterion = recall_change >= 0

    promote = (
        f1_criterion
        and recall_criterion
    )

    decision = (
        "promoted"
        if promote
        else "rejected"
    )

    # --------------------------------------------------------
    # SAVE CANDIDATE
    # --------------------------------------------------------

    joblib.dump(
        best_model,
        CANDIDATE_MODEL_PATH,
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SELECTED CANDIDATE")
    print("=" * 70)

    print(
        f"Parameters : {best_params}"
    )

    print(
        f"Accuracy   : "
        f"{best_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision  : "
        f"{best_metrics['precision']:.4f}"
    )

    print(
        f"Recall     : "
        f"{best_metrics['recall']:.4f}"
    )

    print(
        f"F1 Score   : "
        f"{best_metrics['f1']:.4f}"
    )

    print("\n" + "=" * 70)
    print("PROMOTION GATE")
    print("=" * 70)

    print(
        f"Baseline F1       : "
        f"{baseline_metrics['f1']:.4f}"
    )

    print(
        f"Candidate F1      : "
        f"{best_metrics['f1']:.4f}"
    )

    print(
        f"F1 Improvement    : "
        f"{f1_improvement:+.4f}"
    )

    print(
        f"\nBaseline Recall   : "
        f"{baseline_metrics['recall']:.4f}"
    )

    print(
        f"Candidate Recall  : "
        f"{best_metrics['recall']:.4f}"
    )

    print(
        f"Recall Change     : "
        f"{recall_change:+.4f}"
    )

    print("\nCriteria")
    print("-" * 70)
    print(
        f"F1 criterion      : {f1_criterion}"
    )
    print(
        f"Recall criterion  : {recall_criterion}"
    )

    print("\nCandidate artifact")
    print("-" * 70)
    print(CANDIDATE_MODEL_PATH)

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    notes = (
        "Candidate selected using Isolation Forest "
        "hyperparameter search on a fixed stratified "
        "test split. Promotion requires F1 to improve "
        "or remain equal and recall to not decrease. "
        f"Selected parameters: {best_params}"
    )

    save_evaluation(
        metrics=best_metrics,
        training_rows=len(X_train),
        test_rows=len(X_test),
        decision=decision,
        f1_improvement=f1_improvement,
        recall_change=recall_change,
        notes=notes,
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    if promote:

        print("PROMOTION APPROVED")
        print("=" * 70)

        print(
            f"Candidate {CANDIDATE_VERSION} "
            f"passed both promotion gates."
        )

        print(
            "\nNext commands:"
        )

        print(
            "python -m mlops.retraining.register_candidate"
        )

        print(
            "python -m mlops.retraining.promote_model"
        )

    else:

        print("PROMOTION REJECTED")
        print("=" * 70)

        print(
            "No candidate improved F1 without "
            "reducing recall."
        )

        print(
            f"Active model {ACTIVE_VERSION} "
            f"remains unchanged."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()