MODEL_NAME = "isolation_forest"
MODEL_VERSION = "v1.0.0"

# Monitoring window
WINDOW_MINUTES = 5

# Minimum events required before calculating a window
MIN_PREDICTIONS = 10

# Operational thresholds
MAX_ANOMALY_RATE = 15.0
MAX_AVERAGE_RISK = 60.0

# Risk thresholds
HIGH_RISK_THRESHOLD = 60.0
CRITICAL_RISK_THRESHOLD = 80.0

# Drift thresholds
NUMERIC_DRIFT_THRESHOLD = 0.20
CATEGORICAL_DRIFT_THRESHOLD = 0.20

# Number of recent events used for drift detection
DRIFT_SAMPLE_SIZE = 500