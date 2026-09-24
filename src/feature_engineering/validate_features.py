import os

import numpy as np
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_PATH = "data/processed/transactions_features.csv"

OUTPUT_DIR = "data/processed"

REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "feature_quality_report.csv"
)


# --------------------------------------------------
# Expected ML features
# --------------------------------------------------

ML_FEATURES = [
    "amount",
    "hour",
    "day_of_week_num",
    "customer_avg_amount",
    "amount_deviation",
    "amount_to_customer_avg_ratio",
    "amount_above_customer_avg_ratio",
    "customer_transaction_count",
    "merchant_avg_amount",
    "amount_to_merchant_avg_ratio",
    "merchant_transaction_count",
    "transactions_last_1h",
    "transactions_last_24h",
    "minutes_since_previous_transaction",
    "customer_hour_frequency",
    "customer_location_frequency",
    "is_new_customer_hour",
    "is_new_customer_location",
    "customer_location_rarity",
    "is_unusual_hour",
    "is_location_changed",
]


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

def load_data():

    print("\nLoading feature dataset...")

    df = pd.read_csv(INPUT_PATH)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    return df


# --------------------------------------------------
# Validate required features
# --------------------------------------------------

def check_required_features(df):

    missing_features = [
        feature
        for feature in ML_FEATURES
        if feature not in df.columns
    ]

    return {
        "check": "required_features",
        "status": (
            "PASS"
            if not missing_features
            else "FAIL"
        ),
        "details": (
            "All required ML features exist"
            if not missing_features
            else f"Missing: {missing_features}"
        ),
    }


# --------------------------------------------------
# Missing values
# --------------------------------------------------

def check_missing_values(df):

    missing = int(
        df[ML_FEATURES]
        .isnull()
        .sum()
        .sum()
    )

    return {
        "check": "missing_feature_values",
        "status": (
            "PASS"
            if missing == 0
            else "FAIL"
        ),
        "details": f"Missing feature values: {missing}",
    }


# --------------------------------------------------
# Infinite values
# --------------------------------------------------

def check_infinite_values(df):

    numeric_features = df[ML_FEATURES].select_dtypes(
        include=["number"]
    )

    infinite_count = int(
        np.isinf(numeric_features)
        .sum()
        .sum()
    )

    return {
        "check": "infinite_values",
        "status": (
            "PASS"
            if infinite_count == 0
            else "FAIL"
        ),
        "details": (
            f"Infinite values: {infinite_count}"
        ),
    }


# --------------------------------------------------
# Amount validation
# --------------------------------------------------

def check_amounts(df):

    invalid = int(
        (df["amount"] <= 0).sum()
    )

    return {
        "check": "amount_range",
        "status": (
            "PASS"
            if invalid == 0
            else "FAIL"
        ),
        "details": (
            f"Non-positive amounts: {invalid}"
        ),
    }


# --------------------------------------------------
# Hour validation
# --------------------------------------------------

def check_hours(df):

    invalid = int(
        (
            (df["hour"] < 0)
            | (df["hour"] > 23)
        ).sum()
    )

    return {
        "check": "hour_range",
        "status": (
            "PASS"
            if invalid == 0
            else "FAIL"
        ),
        "details": (
            f"Invalid hour values: {invalid}"
        ),
    }


# --------------------------------------------------
# Day validation
# --------------------------------------------------

def check_days(df):

    invalid = int(
        (
            (df["day_of_week_num"] < 0)
            | (df["day_of_week_num"] > 6)
        ).sum()
    )

    return {
        "check": "day_of_week_range",
        "status": (
            "PASS"
            if invalid == 0
            else "FAIL"
        ),
        "details": (
            f"Invalid day values: {invalid}"
        ),
    }


# --------------------------------------------------
# Velocity validation
# --------------------------------------------------

def check_velocity(df):

    invalid_1h = int(
        (df["transactions_last_1h"] < 0).sum()
    )

    invalid_24h = int(
        (df["transactions_last_24h"] < 0).sum()
    )

    total_invalid = (
        invalid_1h + invalid_24h
    )

    return {
        "check": "velocity_features",
        "status": (
            "PASS"
            if total_invalid == 0
            else "FAIL"
        ),
        "details": (
            f"Invalid 1h values: {invalid_1h}; "
            f"Invalid 24h values: {invalid_24h}"
        ),
    }


# --------------------------------------------------
# Binary feature validation
# --------------------------------------------------

def check_binary_features(df):

    binary_features = [
        "is_unusual_hour",
        "is_location_changed",
        "is_new_customer_hour",
        "is_new_customer_location",
    ]

    invalid_counts = {}

    for feature in binary_features:

        invalid_counts[feature] = int(
            (~df[feature].isin([0, 1]))
            .sum()
        )

    total_invalid = sum(
        invalid_counts.values()
    )

    return {
        "check": "binary_features",
        "status": (
            "PASS"
            if total_invalid == 0
            else "FAIL"
        ),
        "details": (
            f"Invalid unusual-hour values: "
            f"{invalid_counts['is_unusual_hour']}; "
            f"Invalid location-change values: "
            f"{invalid_counts['is_location_changed']}; "
            f"Invalid new-customer-hour values: "
            f"{invalid_counts['is_new_customer_hour']}; "
            f"Invalid new-customer-location values: "
            f"{invalid_counts['is_new_customer_location']}"
        ),
    }


# --------------------------------------------------
# Run validation
# --------------------------------------------------

def main():

    print("=" * 60)
    print("FEATURE QUALITY VALIDATION")
    print("=" * 60)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    df = load_data()

    checks = [
        check_required_features(df),
        check_missing_values(df),
        check_infinite_values(df),
        check_amounts(df),
        check_hours(df),
        check_days(df),
        check_velocity(df),
        check_binary_features(df),
    ]

    report = pd.DataFrame(checks)

    print("\nValidation Results:")
    print("-" * 60)

    print(
        report[
            ["check", "status", "details"]
        ].to_string(index=False)
    )

    report.to_csv(
        REPORT_PATH,
        index=False
    )

    print("\nFeature Statistics:")
    print("-" * 60)

    print(
        df[ML_FEATURES]
        .describe()
        .T[
            ["count", "mean", "std", "min", "max"]
        ]
        .round(2)
        .to_string()
    )

    all_passed = (
        report["status"] == "PASS"
    ).all()

    print("\n" + "=" * 60)

    if all_passed:
        print("✓ ALL FEATURE QUALITY CHECKS PASSED")
    else:
        print("✗ FEATURE QUALITY ISSUES DETECTED")

    print(
        f"✓ Report saved to: {REPORT_PATH}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()