from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


FEATURES_PATH = Path(
    "data/processed/transactions_features.csv"
)

SCORED_PATH = Path(
    "data/processed/transactions_scored.csv"
)

REPORT_PATH = Path(
    "data/processed/anomaly_evaluation_report.csv"
)


def main():

    print("=" * 60)
    print("ANOMALY DETECTION MODEL EVALUATION")
    print("=" * 60)

    # ---------------------------------------------------------
    # Load ground truth
    # ---------------------------------------------------------

    features_df = pd.read_csv(
        FEATURES_PATH,
        parse_dates=["transaction_timestamp"],
    )

    # ---------------------------------------------------------
    # Load model predictions
    # ---------------------------------------------------------

    scored_df = pd.read_csv(
        SCORED_PATH,
        parse_dates=["transaction_timestamp"],
    )

    # ---------------------------------------------------------
    # Keep only the columns required for evaluation
    # ---------------------------------------------------------

    truth_df = features_df[
        [
            "transaction_id",
            "is_injected_anomaly",
            "anomaly_type",
        ]
    ].copy()

    predictions_df = scored_df[
        [
            "transaction_id",
            "is_anomaly",
        ]
    ].copy()

    # ---------------------------------------------------------
    # Join ground truth with predictions
    # ---------------------------------------------------------

    evaluation_df = truth_df.merge(
        predictions_df,
        on="transaction_id",
        how="inner",
    )

    if evaluation_df.empty:
        raise ValueError(
            "No transactions could be matched between "
            "features and scored data."
        )

    # ---------------------------------------------------------
    # Ground truth and predictions
    # ---------------------------------------------------------

    y_true = evaluation_df[
        "is_injected_anomaly"
    ]

    y_pred = evaluation_df[
        "is_anomaly"
    ]

    # ---------------------------------------------------------
    # Overall metrics
    # ---------------------------------------------------------

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    confusion = confusion_matrix(
        y_true,
        y_pred,
    )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print("\nEvaluation dataset:")
    print(f"Matched transactions: {len(evaluation_df):,}")

    print("\nOverall metrics:")
    print("-" * 60)

    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

    print("\nConfusion matrix:")
    print(
        confusion
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "normal",
                "anomaly",
            ],
            zero_division=0,
        )
    )

    # ---------------------------------------------------------
    # Evaluation by anomaly type
    # ---------------------------------------------------------

    print("\nPerformance by anomaly type:")
    print("-" * 60)

    anomaly_types = [
        "high_amount",
        "unusual_hour",
        "unusual_location",
        "high_velocity",
    ]

    type_results = []

    for anomaly_type in anomaly_types:

        actual_anomalies = (
            evaluation_df["anomaly_type"]
            == anomaly_type
        )

        actual_count = int(
            actual_anomalies.sum()
        )

        detected_count = int(
            (
                actual_anomalies
                & (evaluation_df["is_anomaly"] == 1)
            ).sum()
        )

        type_recall = (
            detected_count / actual_count
            if actual_count > 0
            else 0
        )

        type_results.append(
            {
                "anomaly_type": anomaly_type,
                "actual_count": actual_count,
                "detected_count": detected_count,
                "recall": type_recall,
            }
        )

        print(
            f"{anomaly_type:20s} "
            f"Actual: {actual_count:4d} | "
            f"Detected: {detected_count:4d} | "
            f"Recall: {type_recall:.4f}"
        )

    # ---------------------------------------------------------
    # Save evaluation report
    # ---------------------------------------------------------

    report_df = pd.DataFrame(
        type_results
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_df.to_csv(
        REPORT_PATH,
        index=False,
    )

    print(
        f"\n✓ Evaluation report saved to: "
        f"{REPORT_PATH}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()