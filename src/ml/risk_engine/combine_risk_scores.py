"""
Combine Isolation Forest risk with explainable behavioral signals.
"""

from pathlib import Path

import pandas as pd

from .behavioral_rules import calculate_behavioral_risk


INPUT_PATH = Path(
    "data/processed/transactions_scored.csv"
)

OUTPUT_PATH = Path(
    "data/processed/transactions_final_risk.csv"
)


def classify_final_risk(row):
    is_anomaly = row["is_anomaly"]
    risk_score = row["risk_score"]

    # Strong ML anomaly.
    if is_anomaly == 1 and risk_score >= 30:
        risk_level = "high"
    elif is_anomaly == 1:
        risk_level = "review"
    elif risk_score >= 60:
        risk_level = "high"
    elif risk_score >= 30:
        risk_level = "medium"
    else:
        risk_level = "low"
    return risk_level


def main():
    print("=" * 60)
    print("COMBINED TRANSACTION RISK ENGINE")
    print("=" * 60)

    print("\nLoading scored transactions...")

    df = pd.read_csv(INPUT_PATH)

    print(f"Loaded transactions: {len(df):,}")

    print("\nCalculating behavioral risk...")

    df = calculate_behavioral_risk(df)

    print("\nAssigning final risk categories...")

    df["final_risk_level"] = df.apply(
        classify_final_risk,
        axis=1,
    )

    df["risk_explanation"] = (
        df["behavioral_reasons"]
        .replace("", "ml_or_baseline_signal")
    )

    print("\nFinal risk distribution:")
    print(
        df["final_risk_level"]
        .value_counts()
    )

    print("\nFinal risk distribution by injected label:")
    print(
        pd.crosstab(
            df["is_injected_anomaly"],
            df["final_risk_level"],
        )
    )

    print("\nHighest-risk transactions:")

    print(
        df[
            [
                "transaction_id",
                "anomaly_type",
                "amount",
                "risk_score",
                "behavioral_risk_score",
                "final_risk_level",
                "risk_explanation",
            ]
        ]
        .sort_values(
            [
                "final_risk_level",
                "risk_score",
                "behavioral_risk_score",
            ],
            ascending=False,
        )
        .head(15)
        .to_string(index=False)
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nSaved final risk dataset:")
    print(OUTPUT_PATH)

    print("=" * 60)


if __name__ == "__main__":
    main()