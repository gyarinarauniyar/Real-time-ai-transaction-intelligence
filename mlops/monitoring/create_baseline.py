import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transactions_features.csv"
)

BASELINE_FILE = (
    PROJECT_ROOT
    / "mlops"
    / "monitoring"
    / "baseline.json"
)


NUMERIC_FEATURES = [
    "amount",
    "amount_deviation",
    "amount_to_customer_avg_ratio",
    "amount_above_customer_avg_ratio",
    "transactions_last_1h",
    "transactions_last_24h",
    "minutes_since_previous_transaction",
    "hour",
    "customer_hour_frequency",
    "customer_location_frequency",
    "customer_transaction_count",
    "merchant_transaction_count",
]


CATEGORICAL_FEATURES = [
    "payment_method",
    "merchant_category",
    "location",
    "device_type",
]


def create_baseline():

    print("=" * 60)
    print("CREATING MLOPS DATA BASELINE")
    print("=" * 60)

    df = pd.read_csv(FEATURE_FILE)

    print(f"Reference rows: {len(df):,}")

    baseline = {
        "source_file": str(FEATURE_FILE),
        "row_count": len(df),
        "numeric": {},
        "categorical": {},
    }

    # ---------------------------------------------------------
    # Numeric baseline
    # ---------------------------------------------------------

    for feature in NUMERIC_FEATURES:

        if feature not in df.columns:
            print(f"Skipping missing feature: {feature}")
            continue

        series = pd.to_numeric(
            df[feature],
            errors="coerce"
        ).dropna()

        baseline["numeric"][feature] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
        }

    # ---------------------------------------------------------
    # Categorical baseline
    # ---------------------------------------------------------

    for feature in CATEGORICAL_FEATURES:

        if feature not in df.columns:
            print(f"Skipping missing feature: {feature}")
            continue

        distribution = (
            df[feature]
            .fillna("NULL")
            .astype(str)
            .value_counts(normalize=True)
            .to_dict()
        )

        baseline["categorical"][feature] = {
            key: float(value)
            for key, value in distribution.items()
        }

    BASELINE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        BASELINE_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            baseline,
            file,
            indent=4
        )

    print()
    print(f"Baseline saved to:")
    print(BASELINE_FILE)
    print()
    print(
        f"Numeric features    : "
        f"{len(baseline['numeric'])}"
    )
    print(
        f"Categorical features: "
        f"{len(baseline['categorical'])}"
    )


if __name__ == "__main__":
    create_baseline()