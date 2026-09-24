import os

import numpy as np
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_PATH = "data/processed/transactions_clean.csv"

OUTPUT_DIR = "data/processed"

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "transactions_features.csv"
)


# --------------------------------------------------
# Load data
# --------------------------------------------------

def load_data(path):

    print("\nLoading clean transaction data...")

    df = pd.read_csv(path)

    df["transaction_timestamp"] = pd.to_datetime(
        df["transaction_timestamp"]
    )

    df = df.sort_values(
        "transaction_timestamp"
    ).reset_index(drop=True)

    print(f"Loaded {len(df):,} transactions")

    return df


# --------------------------------------------------
# Basic time features
# --------------------------------------------------

def create_time_features(df):

    df["hour"] = (
        df["transaction_timestamp"].dt.hour
    )

    df["day_of_week_num"] = (
        df["transaction_timestamp"].dt.dayofweek
    )

    # Transactions between midnight and 5 AM
    df["is_unusual_hour"] = (
        (df["hour"] >= 0)
        & (df["hour"] <= 5)
    ).astype(int)

    return df


# --------------------------------------------------
# Customer behavior features
# --------------------------------------------------

def create_customer_features(df):

    df = df.sort_values(
        ["customer_id", "transaction_timestamp"]
    ).copy()

    # Previous transaction count for this customer
    df["customer_transaction_count"] = (
        df.groupby("customer_id")
        .cumcount()
    )

    # Historical customer statistics
    customer_group = df.groupby("customer_id")["amount"]

    df["customer_avg_amount"] = (
        customer_group
        .transform(
            lambda x: x.shift(1).expanding().mean()
        )
    )

    df["customer_median_amount"] = (
        customer_group
        .transform(
            lambda x: x.shift(1).expanding().median()
        )
    )

    df["customer_std_amount"] = (
        customer_group
        .transform(
            lambda x: x.shift(1).expanding().std()
        )
    )

    # Amount compared with customer's historical behavior
    df["amount_deviation"] = (
        df["amount"] - df["customer_avg_amount"]
    )

    df["amount_to_customer_avg_ratio"] = (
        df["amount"]
        / df["customer_avg_amount"].replace(0, np.nan)
    )

    df["amount_to_customer_avg_ratio"] = (
        df["amount_to_customer_avg_ratio"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    df["amount_above_customer_avg_ratio"] = (
        df["amount_to_customer_avg_ratio"] - 1
    ).clip(lower=0)

    # Restore chronological order
    df = df.sort_values(
        "transaction_timestamp"
    ).reset_index(drop=True)

    return df

# --------------------------------------------------
# Merchant behavior features
# --------------------------------------------------

def create_merchant_features(df):

    df = df.sort_values(
        ["merchant_id", "transaction_timestamp"]
    ).copy()

    # Previous transaction count for this merchant
    df["merchant_transaction_count"] = (
        df.groupby("merchant_id")
        .cumcount()
    )

    # Historical merchant statistics
    merchant_group = df.groupby("merchant_id")["amount"]

    df["merchant_avg_amount"] = (
        merchant_group
        .transform(
            lambda x: x.shift(1).expanding().mean()
        )
    )

    df["merchant_median_amount"] = (
        merchant_group
        .transform(
            lambda x: x.shift(1).expanding().median()
        )
    )

    # Amount compared with merchant's historical behavior
    df["amount_to_merchant_avg_ratio"] = (
        df["amount"]
        / df["merchant_avg_amount"].replace(0, np.nan)
    )

    df["amount_to_merchant_avg_ratio"] = (
        df["amount_to_merchant_avg_ratio"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # Restore chronological order
    df = df.sort_values(
        "transaction_timestamp"
    ).reset_index(drop=True)

    return df

# --------------------------------------------------
# Transaction velocity features
# --------------------------------------------------

def create_velocity_features(df):

    # Work on a sorted copy so time-window calculations
    # are performed chronologically.
    df = df.sort_values(
        ["customer_id", "transaction_timestamp"]
    ).copy()

    # Create a timestamp index for time-based rolling.
    df = df.set_index("transaction_timestamp")

    # Count transactions for each customer in the
    # previous 1 hour.
    transactions_1h = (
        df.groupby("customer_id")["transaction_id"]
        .rolling("1h")
        .count()
        .reset_index(name="transactions_last_1h")
    )

    # Count transactions for each customer in the
    # previous 24 hours.
    transactions_24h = (
        df.groupby("customer_id")["transaction_id"]
        .rolling("24h")
        .count()
        .reset_index(name="transactions_last_24h")
    )

    # The rolling results contain customer_id and timestamp.
    # Merge them back to the original transaction rows.
    transactions_1h = transactions_1h.rename(
        columns={
            "transaction_timestamp": "rolling_timestamp"
        }
    )

    transactions_24h = transactions_24h.rename(
        columns={
            "transaction_timestamp": "rolling_timestamp"
        }
    )

    # Restore timestamp as a normal column.
    df = df.reset_index()

    # Create a stable row identifier for merging.
    df["_feature_row_id"] = range(len(df))

    # Merge 1-hour counts by customer + timestamp.
    df = df.merge(
        transactions_1h,
        left_on=["customer_id", "transaction_timestamp"],
        right_on=["customer_id", "rolling_timestamp"],
        how="left",
    )

    df = df.drop(
        columns=["rolling_timestamp"],
        errors="ignore",
    )

    # Merge 24-hour counts.
    df = df.merge(
        transactions_24h,
        left_on=["customer_id", "transaction_timestamp"],
        right_on=["customer_id", "rolling_timestamp"],
        how="left",
    )

    df = df.drop(
        columns=["rolling_timestamp"],
        errors="ignore",
    )

    # Remove temporary identifier.
    df = df.drop(
        columns=["_feature_row_id"],
        errors="ignore",
    )

    # A transaction is included in its own rolling window,
    # so subtract 1 to measure previous transactions.
    df["transactions_last_1h"] = (
        df["transactions_last_1h"] - 1
    ).clip(lower=0)

    df["transactions_last_24h"] = (
        df["transactions_last_24h"] - 1
    ).clip(lower=0)

    # Restore global chronological order.
    df = df.sort_values(
        "transaction_timestamp"
    ).reset_index(drop=True)

    return df


# --------------------------------------------------
# Location behavior
# --------------------------------------------------

# --------------------------------------------------
# Customer behavioral pattern features
# --------------------------------------------------

def create_behavioral_features(df):

    df = df.sort_values(
        ["customer_id", "transaction_timestamp"]
    ).copy()

    # ----------------------------------------------
    # Time since customer's previous transaction
    # ----------------------------------------------

    df["previous_transaction_timestamp"] = (
        df.groupby("customer_id")[
            "transaction_timestamp"
        ].shift(1)
    )

    df["minutes_since_previous_transaction"] = (
        (
            df["transaction_timestamp"]
            - df["previous_transaction_timestamp"]
        ).dt.total_seconds()
        / 60
    )

    # ----------------------------------------------
    # Customer's historical transaction count
    # ----------------------------------------------

    df["customer_previous_transaction_count"] = (
        df.groupby("customer_id")
        .cumcount()
    )

    # ----------------------------------------------
    # Customer hour frequency
    #
    # How frequently has this customer used this
    # hour BEFORE the current transaction?
    # ----------------------------------------------

    customer_hour_count = (
        df.groupby(
            ["customer_id", "hour"]
        ).cumcount()
    )

    df["customer_hour_frequency"] = (
        customer_hour_count
        / df["customer_previous_transaction_count"]
        .replace(0, np.nan)
    )

    df["customer_hour_frequency"] = (
        df["customer_hour_frequency"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    df["is_new_customer_hour"] = (
        df["customer_hour_frequency"] == 0
    ).astype(int)

    # ----------------------------------------------
    # Customer location frequency
    #
    # How frequently has this customer used this
    # location BEFORE the current transaction?
    # ----------------------------------------------

    customer_location_count = (
        df.groupby(
            ["customer_id", "location"]
        ).cumcount()
    )

    df["customer_location_frequency"] = (
        customer_location_count
        / df["customer_previous_transaction_count"]
        .replace(0, np.nan)
    )

    df["customer_location_frequency"] = (
        df["customer_location_frequency"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    df["customer_location_rarity"]=(
        1-df["customer_location_frequency"]
    )

    df["is_new_customer_location"] = (
        df["customer_location_frequency"] == 0
    ).astype(int)

    # ----------------------------------------------
    # Location change indicator
    # ----------------------------------------------

    df["previous_location"] = (
        df.groupby("customer_id")["location"]
        .shift(1)
    )

    df["is_location_changed"] = (
        (
            df["location"]
            != df["previous_location"]
        )
        & df["previous_location"].notna()
    ).astype(int)

    # Restore global chronological order
    df = df.sort_values(
        "transaction_timestamp"
    ).reset_index(drop=True)

    return df


# --------------------------------------------------
# Clean feature dataset
# --------------------------------------------------

def clean_features(df):

    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns

    df[numeric_columns] = (
        df[numeric_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # We don't need this intermediate column
    # for ML.
    temporary_columns = [
        "previous_location",
        "previous_transaction_timestamp",
        "customer_previous_transaction_count",
    ]

    df = df.drop(
        columns=temporary_columns,
        errors="ignore",
    )
    
    return df


# --------------------------------------------------
# Main feature pipeline
# --------------------------------------------------

def main():

    print("=" * 60)
    print("TRANSACTION FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    df = load_data(INPUT_PATH)

    print("\nCreating time features...")
    df = create_time_features(df)

    print("Creating customer behavior features...")
    df = create_customer_features(df)

    print("Creating merchant behavior features...")
    df = create_merchant_features(df)

    print("Creating transaction velocity features...")
    df = create_velocity_features(df)

    print("Creating customer behavior features...")
    df = create_behavioral_features(df)

    print("Cleaning feature dataset...")
    df = clean_features(df)

    # Save
    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nML-relevant features:")

    feature_columns = [
        "amount",
        "hour",
        "day_of_week_num",

        "customer_avg_amount",
        "amount_deviation",
        "amount_to_customer_avg_ratio",
        "customer_transaction_count",

        "merchant_avg_amount",
        "amount_to_merchant_avg_ratio",
        "merchant_transaction_count",

        "transactions_last_1h",
        "transactions_last_24h",
        "minutes_since_previous_transaction",

        "is_unusual_hour",
        "is_location_changed",

        "customer_hour_frequency",
        "customer_location_frequency",
        "customer_location_rarity",
    ]

    for column in feature_columns:
        if column in df.columns:
            print(f" - {column}")

    print(
        f"\nFeature dataset saved to:\n"
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()