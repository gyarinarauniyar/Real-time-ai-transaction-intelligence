# Transaction Risk Rules

## Risk Engine

The platform combines machine-learning anomaly detection with behavioral risk signals.

The final risk score ranges from 0 to 100.

## Risk Levels

### Low

Risk score below 30 when no stronger anomaly signal is present.

### Medium

Risk score from 30 up to but not including 60.

### High

A transaction can become high risk when:

- The transaction is classified as an ML anomaly and the risk score is at least 30.
- Or the final risk score is at least 60.

### Critical

Real-time scoring uses a critical threshold of 80 or above.

## Behavioral Signals

The behavioral risk layer can consider:

- Unusual transaction amount
- Unusual transaction hour
- Location change
- New customer location
- New customer hour
- High transaction velocity
- Customer behavioral deviation
- Merchant behavioral signals

## Important Interpretation

A risk score is a risk signal, not proof that a transaction is fraudulent.

The system is designed to prioritize transactions for investigation and monitoring.