import os
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_PATH = "data/synthetic/transactions.csv"

OUTPUT_DIR = "data/processed"

CLEAN_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "transactions_clean.csv"
)

QUALITY_REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "data_quality_report.csv"
)


# --------------------------------------------------
# Expected schema
# --------------------------------------------------

EXPECTED_COLUMNS = [
    "transaction_id",
    "customer_id",
    "merchant_id",
    "transaction_timestamp",
    "amount",
    "currency",
    "merchant_category",
    "payment_method",
    "location",
    "device_type",
    "is_weekend",
    "hour",
    "day_of_week",
    "transaction_date",
    "is_injected_anomaly",
    "anomaly_type"
]


# --------------------------------------------------
# Load data
# --------------------------------------------------

def load_data(path):
    print("\nLoading dataset...")

    df = pd.read_csv(path)

    print(f"Loaded {len(df):,} rows")

    return df


# --------------------------------------------------
# Schema validation
# --------------------------------------------------

def check_schema(df):

    actual_columns = list(df.columns)

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in actual_columns
    ]

    unexpected_columns = [
        column
        for column in actual_columns
        if column not in EXPECTED_COLUMNS
    ]

    return {
        "check": "schema",
        "status": "PASS"
        if not missing_columns and not unexpected_columns
        else "FAIL",
        "details": (
            "Schema matches expected structure"
            if not missing_columns and not unexpected_columns
            else f"Missing: {missing_columns}; "
                 f"Unexpected: {unexpected_columns}"
        ),
    }


# --------------------------------------------------
# Missing-value validation
# --------------------------------------------------

def check_missing_values(df):

    missing_count = int(df.isnull().sum().sum())

    return {
        "check": "missing_values",
        "status": "PASS" if missing_count == 0 else "FAIL",
        "details": f"Total missing values: {missing_count}",
    }


# --------------------------------------------------
# Duplicate transaction validation
# --------------------------------------------------

def check_duplicates(df):

    duplicate_ids = int(
        df["transaction_id"].duplicated().sum()
    )

    return {
        "check": "duplicate_transaction_ids",
        "status": "PASS" if duplicate_ids == 0 else "FAIL",
        "details": (
            f"Duplicate transaction IDs: {duplicate_ids}"
        ),
    }


# --------------------------------------------------
# Amount validation
# --------------------------------------------------

def check_amounts(df):

    negative_amounts = int(
        (df["amount"] <= 0).sum()
    )

    return {
        "check": "transaction_amount",
        "status": "PASS"
        if negative_amounts == 0
        else "FAIL",
        "details": (
            f"Non-positive transaction amounts: "
            f"{negative_amounts}"
        ),
    }


# --------------------------------------------------
# Currency validation
# --------------------------------------------------

def check_currency(df):

    invalid_currency = int(
        (df["currency"] != "INR").sum()
    )

    return {
        "check": "currency",
        "status": "PASS"
        if invalid_currency == 0
        else "FAIL",
        "details": (
            f"Invalid currency records: "
            f"{invalid_currency}"
        ),
    }


# --------------------------------------------------
# Timestamp validation
# --------------------------------------------------

def check_timestamps(df):

    timestamps = pd.to_datetime(
        df["transaction_timestamp"],
        errors="coerce"
    )

    invalid_timestamps = int(
        timestamps.isnull().sum()
    )

    future_timestamps = int(
        (timestamps > pd.Timestamp.now()).sum()
    )

    total_issues = (
        invalid_timestamps +
        future_timestamps
    )

    return {
        "check": "transaction_timestamp",
        "status": "PASS"
        if total_issues == 0
        else "FAIL",
        "details": (
            f"Invalid timestamps: {invalid_timestamps}; "
            f"Future timestamps: {future_timestamps}"
        ),
    }


# --------------------------------------------------
# Categorical validation
# --------------------------------------------------

def check_categories(df):

    valid_payment_methods = {
        "credit_card",
        "debit_card",
        "upi",
        "digital_wallet",
    }

    valid_devices = {
        "mobile",
        "desktop",
        "tablet",
    }

    invalid_payment_methods = int(
        (~df["payment_method"].isin(valid_payment_methods)).sum()
    )

    invalid_devices = int(
        (~df["device_type"].isin(valid_devices)).sum()
    )

    total_issues = (
        invalid_payment_methods +
        invalid_devices
    )

    return {
        "check": "categorical_values",
        "status": "PASS"
        if total_issues == 0
        else "FAIL",
        "details": (
            f"Invalid payment methods: "
            f"{invalid_payment_methods}; "
            f"Invalid devices: {invalid_devices}"
        ),
    }

# ---------------------------------------------------------
# --------------------------------------------------
# Injected anomaly label validation
# --------------------------------------------------

def check_anomaly_labels(df):

    valid_anomaly_flags = {0, 1}

    invalid_anomaly_flags = int(
        (~df["is_injected_anomaly"].isin(valid_anomaly_flags)).sum()
    )

    valid_anomaly_types = {
        "normal",
        "high_amount",
        "unusual_hour",
        "unusual_location",
        "high_velocity",
    }

    invalid_anomaly_types = int(
        (~df["anomaly_type"].isin(valid_anomaly_types)).sum()
    )

    total_issues = (
        invalid_anomaly_flags +
        invalid_anomaly_types
    )

    return {
        "check": "injected_anomaly_labels",
        "status": "PASS"
        if total_issues == 0
        else "FAIL",
        "details": (
            f"Invalid anomaly flags: "
            f"{invalid_anomaly_flags}; "
            f"Invalid anomaly types: "
            f"{invalid_anomaly_types}"
        ),
    }

# --------------------------------------------------
# Run all quality checks
# --------------------------------------------------

def run_quality_checks(df):

    checks = [
        check_schema(df),
        check_missing_values(df),
        check_duplicates(df),
        check_amounts(df),
        check_currency(df),
        check_timestamps(df),
        check_categories(df),
        check_anomaly_labels(df),
    ]

    return pd.DataFrame(checks)


# --------------------------------------------------
# Main pipeline
# --------------------------------------------------

def main():

    print("=" * 60)
    print("TRANSACTION DATA QUALITY PIPELINE")
    print("=" * 60)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    df = load_data(INPUT_PATH)

    quality_report = run_quality_checks(df)

    print("\nData Quality Results:")
    print("-" * 60)

    print(
        quality_report[
            ["check", "status", "details"]
        ].to_string(index=False)
    )

    quality_report.to_csv(
        QUALITY_REPORT_PATH,
        index=False
    )

    # Only create the clean dataset if all checks pass
    all_passed = (
        quality_report["status"] == "PASS"
    ).all()

    if all_passed:

        df.to_csv(
            CLEAN_OUTPUT_PATH,
            index=False
        )

        print("\n✓ All quality checks passed")
        print(
            f"✓ Clean dataset saved to: "
            f"{CLEAN_OUTPUT_PATH}"
        )

    else:

        print(
            "\n✗ Data quality issues detected"
        )

        print(
            "Clean dataset was NOT created."
        )

    print(
        f"\n✓ Quality report saved to: "
        f"{QUALITY_REPORT_PATH}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()