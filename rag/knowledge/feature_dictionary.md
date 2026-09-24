# Transaction Intelligence Feature Dictionary

## Amount Features

### amount
The monetary value of the transaction.

### amount_deviation
Difference between the current transaction amount and the customer's historical average transaction amount.

### amount_to_customer_avg_ratio
Ratio between the current transaction amount and the customer's historical average transaction amount.

### amount_above_customer_avg_ratio
Measures how far the transaction amount is above the customer's historical average.

## Transaction Velocity Features

### transactions_last_1h
Number of transactions made by the customer during the previous one-hour window.

### transactions_last_24h
Number of transactions made by the customer during the previous 24-hour window.

### minutes_since_previous_transaction
Time elapsed since the customer's previous transaction.

## Time Features

### hour
Hour of the transaction.

### is_unusual_hour
Indicates whether the transaction occurred at an unusual hour for the customer.

### customer_hour_frequency
Historical frequency of transactions by the customer during the transaction hour.

### is_new_customer_hour
Indicates whether the customer is transacting at an hour not previously observed in their history.

## Location Features

### is_location_changed
Indicates whether the transaction location differs from the customer's previous transaction location.

### customer_location_frequency
Historical frequency of the customer's transactions at the current location.

### is_new_customer_location
Indicates whether the customer is using a location not previously observed.

## Customer Features

### customer_transaction_count
Total number of transactions associated with the customer.

## Merchant Features

### merchant_transaction_count
Total number of transactions associated with the merchant.

## Risk Interpretation

These features are used by the anomaly detection and behavioral risk layers.

A high value or unusual pattern does not automatically mean fraud. It represents a signal that may contribute to transaction risk.