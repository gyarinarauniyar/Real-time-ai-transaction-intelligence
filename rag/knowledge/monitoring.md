# MLOps Monitoring

## Monitoring Windows

The platform collects prediction metrics over monitoring windows.

Tracked metrics include:

- Total predictions
- Anomaly count
- Anomaly rate
- Average risk score
- High-risk count
- Critical-risk count

## Data Drift

The monitoring system checks numerical and categorical feature distributions.

Numerical features use a drift score threshold.

Categorical features use PSI-based monitoring.

## Alerts

The alert engine can create alerts when configured production thresholds are exceeded.

Examples include:

- High anomaly rate
- High average risk
- Critical-risk transactions

## Current Production Monitoring

The production model is:

Isolation Forest v1.0.0

The monitoring pipeline supports:

1. Prediction logging
2. Monitoring windows
3. Data drift detection
4. Alert detection
5. Model lifecycle monitoring

## Interpretation

A monitoring alert indicates that a configured operational or model-risk threshold has been breached.

It does not automatically indicate system failure or confirmed fraud.