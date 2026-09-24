# Anomaly Detection Model

## Production Model

Model name:

Isolation Forest

Production version:

v1.0.0

## Model Purpose

The anomaly detection model identifies transactions that differ from learned transaction behavior.

It is an unsupervised anomaly detection model.

## Model Features

The anomaly detection model uses 16 production features for transaction-level anomaly detection.

The 16 features used by the production anomaly detection model are:

1. amount
2. amount_deviation
3. amount_to_customer_avg_ratio
4. amount_above_customer_avg_ratio
5. transactions_last_1h
6. transactions_last_24h
7. minutes_since_previous_transaction
8. hour
9. is_unusual_hour
10. is_location_changed
11. customer_hour_frequency
12. customer_location_frequency
13. is_new_customer_hour
14. is_new_customer_location
15. customer_transaction_count
16. merchant_transaction_count

These anomaly detection features capture:

- Transaction amount behavior
- Customer spending behavior
- Transaction velocity
- Time and hour behavior
- Location behavior
- Customer behavior
- Merchant activity

These are the production features used by the Isolation Forest anomaly detection model.

## Model Output

The model produces:

- anomaly prediction
- anomaly score

The platform combines the model signal with behavioral rules to calculate the final risk score.

## Production Model Management

The project maintains:

- Model registry
- Model versions
- Calibration artifacts
- Model promotion
- Model rollback
- Prediction logging
- Monitoring
- Drift detection
- Alerting

## Interpretation

An anomaly prediction indicates that a transaction differs from learned behavior.

It does not independently establish that the transaction is fraudulent.