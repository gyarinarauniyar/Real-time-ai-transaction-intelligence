from pathlib import Path

import pandas as pd

from src.ml.inference.realtime_scorer import (
    FEATURES,
    RealTimeRiskScorer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transactions_features.csv"
)


def test_realtime_scorer():
    df = pd.read_csv(FEATURE_FILE)

    event = df.iloc[0].to_dict()

    scorer = RealTimeRiskScorer()

    result = scorer.score(event)

    assert "anomaly_prediction" in result
    assert "is_anomaly" in result
    assert "risk_score" in result
    assert "risk_level" in result

    assert result["anomaly_prediction"] in [-1, 1]
    assert result["is_anomaly"] in [0, 1]
    assert 0 <= result["risk_score"] <= 100
    assert result["risk_level"] in [
        "low",
        "medium",
        "high",
        "critical",
    ]