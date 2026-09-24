from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest


INPUT_PATH = Path("data/processed/transactions_features.csv")

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "isolation_forest.joblib"
CALIBRATION_PATH = MODEL_DIR / "risk_calibration.joblib"

OUTPUT_PATH = Path("data/processed/transactions_scored.csv")


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


def classify_risk(score):

    if score >= 80:
        return "critical"

    if score >= 60:
        return "high"

    if score >= 30:
        return "medium"

    return "low"


def main():

    print("=" * 60)
    print("ISOLATION FOREST ANOMALY DETECTION")
    print("=" * 60)

    # ---------------------------------------------------------
    # Load feature dataset
    # ---------------------------------------------------------

    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["transaction_timestamp"]
    )

    print(f"\nTotal transactions: {len(df):,}")

    # ---------------------------------------------------------
    # Validate required features
    # ---------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    # ---------------------------------------------------------
    # Sort chronologically
    # ---------------------------------------------------------

    df = df.sort_values(
        "transaction_timestamp"
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Chronological train/test split
    # ---------------------------------------------------------

    split_index = int(len(df) * 0.80)

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    print(f"Training rows: {len(train_df):,}")
    print(f"Testing rows:  {len(test_df):,}")

    # ---------------------------------------------------------
    # Prepare feature matrices
    # ---------------------------------------------------------

    X_train = train_df[FEATURES]
    X_test = test_df[FEATURES]

    # ---------------------------------------------------------
    # Train Isolation Forest
    # ---------------------------------------------------------

    model = IsolationForest(
        n_estimators=200,
        contamination=0.03,
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining Isolation Forest...")

    model.fit(X_train)

    print("Model training complete.")

    # ---------------------------------------------------------
    # CALIBRATION
    # ---------------------------------------------------------
    # Build the risk-score calibration from TRAINING data only.
    #
    # Isolation Forest:
    # lower decision_function = more anomalous
    # ---------------------------------------------------------

    train_anomaly_scores = (
        -model.decision_function(X_train)
    )

    calibration_min = float(
        train_anomaly_scores.min()
    )

    calibration_max = float(
        train_anomaly_scores.max()
    )

    calibration = {
        "min_score": calibration_min,
        "max_score": calibration_max,
    }

    joblib.dump(
        calibration,
        CALIBRATION_PATH
    )

    print("\nRisk calibration:")
    print(
        f"Minimum anomaly score : "
        f"{calibration_min:.4f}"
    )

    print(
        f"Maximum anomaly score : "
        f"{calibration_max:.4f}"
    )

    print(
        f"\nSaved calibration:"
        f"\n{CALIBRATION_PATH}"
    )

    # ---------------------------------------------------------
    # Score test transactions
    # ---------------------------------------------------------

    test_df["anomaly_prediction"] = (
        model.predict(X_test)
    )

    test_df["is_anomaly"] = (
        test_df["anomaly_prediction"] == -1
    ).astype(int)

    test_df["anomaly_score"] = (
        -model.decision_function(X_test)
    )

    # ---------------------------------------------------------
    # Convert anomaly score to 0–100 risk score
    # ---------------------------------------------------------

    if calibration_max == calibration_min:

        test_df["risk_score"] = 0.0

    else:

        test_df["risk_score"] = (
            (
                test_df["anomaly_score"]
                - calibration_min
            )
            /
            (
                calibration_max
                - calibration_min
            )
            * 100
        )

    # Keep score inside 0–100.

    test_df["risk_score"] = (
        test_df["risk_score"]
        .clip(0, 100)
        .round(2)
    )

    # ---------------------------------------------------------
    # Risk levels
    # ---------------------------------------------------------

    test_df["risk_level"] = (
        test_df["risk_score"]
        .apply(classify_risk)
    )

    # ---------------------------------------------------------
    # Save scored transactions
    # ---------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    test_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    anomaly_count = test_df["is_anomaly"].sum()

    anomaly_rate = (
        anomaly_count
        / len(test_df)
        * 100
    )

    print("\n" + "=" * 60)
    print("MODEL RESULTS")
    print("=" * 60)

    print(
        f"Test transactions  : "
        f"{len(test_df):,}"
    )

    print(
        f"Anomalies detected : "
        f"{anomaly_count:,}"
    )

    print(
        f"Anomaly rate       : "
        f"{anomaly_rate:.2f}%"
    )

    print("\nRisk distribution:")

    print(
        test_df["risk_level"]
        .value_counts()
        .sort_index()
    )

    print("\nSaved model:")
    print(MODEL_PATH)

    print("\nSaved calibration:")
    print(CALIBRATION_PATH)

    print("\nSaved scored transactions:")
    print(OUTPUT_PATH)

    print("\nTop 10 highest-risk transactions:")

    columns_to_show = [
        "transaction_id",
        "transaction_timestamp",
        "amount",
        "anomaly_score",
        "risk_score",
        "risk_level",
    ]

    print(
        test_df
        .sort_values(
            "risk_score",
            ascending=False
        )[columns_to_show]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()