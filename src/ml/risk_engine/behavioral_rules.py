"""
High-confidence, explainable behavioral risk rules.
"""

import pandas as pd


def calculate_behavioral_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate a conservative behavioral risk score.

    Individual weak behavioral signals are not scored alone.
    Risk is assigned mainly to stronger combinations.
    """

    df = df.copy()

    df["behavioral_risk_score"] = 0.0
    df["behavioral_reasons"] = ""

    def add_risk(mask, points, reason):
        df.loc[mask, "behavioral_risk_score"] += points

        existing_reasons = df.loc[mask, "behavioral_reasons"]

        df.loc[mask, "behavioral_reasons"] = (
            existing_reasons
            .where(
                existing_reasons.eq(""),
                existing_reasons + "; ",
            )
            + reason
        )

    # Strong signal: high transaction velocity.
    add_risk(
        df["transactions_last_1h"] >= 3,
        points=25,
        reason="high_velocity",
    )

    # Strong behavioral combination:
    # unusual hour and location change.
    add_risk(
        (df["is_unusual_hour"] == 1)
        & (df["is_location_changed"] == 1),
        points=30,
        reason="unusual_hour_and_location_change",
    )

    # Strong behavioral combination:
    # unusual hour and previously unseen location.
    add_risk(
        (df["is_unusual_hour"] == 1)
        & (df["is_new_customer_location"] == 1),
        points=30,
        reason="unusual_hour_and_new_location",
    )

    # New location combined with high velocity.
    add_risk(
        (df["is_new_customer_location"] == 1)
        & (df["transactions_last_1h"] >= 3),
        points=35,
        reason="new_location_and_high_velocity",
    )

    # Very high amount relative to the customer's history.
    add_risk(
        df["amount_above_customer_avg_ratio"] >= 5,
        points=35,
        reason="very_high_customer_relative_amount",
    )

    # Cap score at 100.
    df["behavioral_risk_score"] = (
        df["behavioral_risk_score"]
        .clip(lower=0, upper=100)
    )

    return df