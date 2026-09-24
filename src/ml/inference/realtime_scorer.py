"""
Real-time transaction risk scoring.

Loads the trained Isolation Forest model and the
training-time risk calibration, then scores individual
transaction events.
"""

from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_NAME = "isolation_forest"
MODEL_VERSION = "v1.0.0"
MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "isolation_forest.joblib"
)

CALIBRATION_FILE = (
    PROJECT_ROOT
    / "models"
    / "risk_calibration.joblib"
)


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


class RealTimeRiskScorer:
    """Score individual transaction events."""

    def __init__(
        self,
        model_file: Path = MODEL_FILE,
        calibration_file: Path = CALIBRATION_FILE,
    ):

        self.model = joblib.load(
            model_file
        )

        self.calibration = joblib.load(
            calibration_file
        )

        self.min_score = float(
            self.calibration["min_score"]
        )

        self.max_score = float(
            self.calibration["max_score"]
        )

    def score(self, event: dict) -> dict:
        """Return anomaly prediction and calibrated risk score."""

        row = pd.DataFrame([event])

        missing_features = [
            feature
            for feature in FEATURES
            if feature not in row.columns
        ]

        if missing_features:
            raise ValueError(
                f"Missing model features: {missing_features}"
            )

        X = row[FEATURES]

        # -----------------------------------------------------
        # Isolation Forest prediction
        # -----------------------------------------------------

        prediction = int(
            self.model.predict(X)[0]
        )

        # -----------------------------------------------------
        # Raw anomaly score
        #
        # Lower Isolation Forest decision_function
        # values indicate more anomalous observations.
        # Negating it makes higher = more anomalous.
        # -----------------------------------------------------

        anomaly_score = float(
            -self.model.decision_function(X)[0]
        )

        # -----------------------------------------------------
        # Calibrate to 0–100 using training-time calibration
        # -----------------------------------------------------

        if self.max_score == self.min_score:

            risk_score = 0.0

        else:

            risk_score = (
                (
                    anomaly_score
                    - self.min_score
                )
                /
                (
                    self.max_score
                    - self.min_score
                )
                * 100
            )

        risk_score = max(
            0,
            min(
                100,
                risk_score,
            ),
        )

        # -----------------------------------------------------
        # Risk level
        # -----------------------------------------------------

        if risk_score >= 80:

            risk_level = "critical"

        elif risk_score >= 60:

            risk_level = "high"

        elif risk_score >= 30:

            risk_level = "medium"

        else:

            risk_level = "low"

        return {
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "anomaly_prediction": prediction,
            "is_anomaly": int(
                prediction == -1
            ),
            "anomaly_score": round(
                anomaly_score,
                6,
            ),
            "risk_score": round(
                risk_score,
                2,
            ),
            "risk_level": risk_level,
        }