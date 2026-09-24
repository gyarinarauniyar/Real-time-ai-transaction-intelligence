import pandas as pd

from behavioral_rules import calculate_behavioral_risk


def main():
    df = pd.read_csv(
        "data/processed/transactions_scored.csv"
    )

    scored_df = calculate_behavioral_risk(df)

    print("=" * 60)
    print("BEHAVIORAL RULE EVALUATION")
    print("=" * 60)

    print("\nBehavioral score by anomaly type:")
    print(
        scored_df.groupby("anomaly_type")[
            "behavioral_risk_score"
        ]
        .agg(
            [
                "count",
                "mean",
                "min",
                "max",
            ]
        )
        .round(2)
    )

    print("\nAverage behavioral score:")
    print(
        scored_df.groupby("is_injected_anomaly")[
            "behavioral_risk_score"
        ]
        .agg(
            [
                "count",
                "mean",
                "min",
                "max",
            ]
        )
        .round(2)
    )

    print("\nDetection at behavioral score >= 30:")

    scored_df["behavioral_prediction"] = (
        scored_df["behavioral_risk_score"] >= 30
    ).astype(int)

    actual = scored_df["is_injected_anomaly"]
    predicted = scored_df["behavioral_prediction"]

    true_positive = (
        ((actual == 1) & (predicted == 1))
        .sum()
    )

    false_positive = (
        ((actual == 0) & (predicted == 1))
        .sum()
    )

    false_negative = (
        ((actual == 1) & (predicted == 0))
        .sum()
    )

    precision = (
        true_positive
        / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0
    )

    recall = (
        true_positive
        / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print(f"True positives : {true_positive}")
    print(f"False positives: {false_positive}")
    print(f"False negatives: {false_negative}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1-score       : {f1:.4f}")


if __name__ == "__main__":
    main()