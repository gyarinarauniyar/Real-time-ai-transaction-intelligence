import pandas as pd

from behavioral_rules import calculate_behavioral_risk


def main():
    df = pd.read_csv(
        "data/processed/transactions_scored.csv"
    )

    scored_df = calculate_behavioral_risk(df)

    print("=" * 60)
    print("BEHAVIORAL RISK RULE TEST")
    print("=" * 60)

    print("\nBehavioral risk distribution:")
    print(
        scored_df["behavioral_risk_score"]
        .describe()
        .round(2)
    )

    print("\nRisk score counts:")
    print(
        scored_df["behavioral_risk_score"]
        .value_counts()
        .sort_index()
        .head(20)
    )

    print("\nHighest behavioral-risk transactions:")
    print(
        scored_df[
            [
                "transaction_id",
                "anomaly_type",
                "amount",
                "behavioral_risk_score",
                "behavioral_reasons",
            ]
        ]
        .sort_values(
            "behavioral_risk_score",
            ascending=False,
        )
        .head(15)
        .to_string(index=False)
    )

    print("\nMissed behavioral anomalies:")
    missed = scored_df[
        (scored_df["is_anomaly"] == 0)
        & (
            scored_df["anomaly_type"].isin(
                [
                    "unusual_hour",
                    "unusual_location",
                ]
            )
        )
    ]

    print(
        missed[
            [
                "transaction_id",
                "anomaly_type",
                "behavioral_risk_score",
                "behavioral_reasons",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()