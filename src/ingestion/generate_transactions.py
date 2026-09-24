import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker


fake = Faker()

# -----------------------------
# Configuration
# -----------------------------

NUM_TRANSACTIONS = 10000
NUM_CUSTOMERS = 1000
NUM_MERCHANTS = 200

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
Faker.seed(RANDOM_SEED)


# -----------------------------
# Reference data
# -----------------------------

MERCHANT_CATEGORIES = [
    "grocery",
    "restaurant",
    "electronics",
    "travel",
    "fashion",
    "utilities",
    "healthcare",
    "entertainment",
    "fuel",
    "online_services",
]

PAYMENT_METHODS = [
    "credit_card",
    "debit_card",
    "upi",
    "digital_wallet",
]

DEVICE_TYPES = [
    "mobile",
    "desktop",
    "tablet",
]

LOCATIONS = [
    "Mumbai",
    "Pune",
    "Delhi",
    "Bangalore",
    "Hyderabad",
    "Chennai",
    "Kolkata",
    "Ahmedabad",
]


# -----------------------------
# Create customers and merchants
# -----------------------------

customers = [
    f"CUST_{i:05d}"
    for i in range(1, NUM_CUSTOMERS + 1)
]

merchants = [
    f"MERCH_{i:04d}"
    for i in range(1, NUM_MERCHANTS + 1)
]


# Give each customer a normal spending profile
customer_profiles = {}

for customer_id in customers:
    customer_profiles[customer_id] = {
        "avg_amount": random.uniform(300, 3000),
        "preferred_location": random.choice(LOCATIONS),
        "preferred_category": random.choice(MERCHANT_CATEGORIES),
    }


# -----------------------------
# Generate transactions
# -----------------------------

start_date = datetime.now() - timedelta(days=90)

transactions = []

for i in range(1, NUM_TRANSACTIONS + 1):

    transaction_id = f"TXN_{i:08d}"

    customer_id = random.choice(customers)
    merchant_id = random.choice(merchants)

    profile = customer_profiles[customer_id]

    timestamp = start_date + timedelta(
        minutes=random.randint(0, 90 * 24 * 60)
    )

    # Normal transaction amount around customer's spending profile
    amount = np.random.normal(
        loc=profile["avg_amount"],
        scale=profile["avg_amount"] * 0.35,
    )

    amount = max(10, round(amount, 2))

    merchant_category = random.choice(MERCHANT_CATEGORIES)

    payment_method = random.choice(PAYMENT_METHODS)

    # Customers are more likely to transact from their preferred location
    if random.random() < 0.75:
        location = profile["preferred_location"]
    else:
        location = random.choice(LOCATIONS)

    device_type = random.choice(DEVICE_TYPES)

    is_weekend = timestamp.weekday() >= 5

    transactions.append(
        {
            "transaction_id": transaction_id,
            "customer_id": customer_id,
            "merchant_id": merchant_id,
            "transaction_timestamp": timestamp,
            "amount": amount,
            "currency": "INR",
            "merchant_category": merchant_category,
            "payment_method": payment_method,
            "location": location,
            "device_type": device_type,
            "is_weekend": is_weekend,

            "is_injected_anomaly": 0,
            "anomaly_type": "normal",
        }
    )


df = pd.DataFrame(transactions)


# -----------------------------
# Inject controlled anomalies
# -----------------------------

NUM_ANOMALIES = int(NUM_TRANSACTIONS * 0.03)

df["is_injected_anomaly"] = 0
df["anomaly_type"] = "normal"

anomaly_indices = np.random.choice(
    df.index,
    size=NUM_ANOMALIES,
    replace=False,
)

for index in anomaly_indices:

    anomaly_type = random.choice(
        [
            "high_amount",
            "unusual_hour",
            "unusual_location",
            "high_velocity",
        ]
    )

    df.loc[index,"is_injected_anomaly"] = 1
    df.loc[index, "anomaly_type"] = anomaly_type

    if anomaly_type == "high_amount":

        df.loc[index, "amount"] = round(
            df.loc[index, "amount"] * random.uniform(8, 20),
            2,
        )

    elif anomaly_type == "unusual_hour":

        timestamp = df.loc[index, "transaction_timestamp"]

        unusual_hour = random.choice([1, 2, 3, 4])

        df.loc[index, "transaction_timestamp"] = timestamp.replace(
            hour=unusual_hour,
            minute=random.randint(0, 59),
        )

    elif anomaly_type == "unusual_location":

        current_location = df.loc[index, "location"]

        other_locations = [
            location
            for location in LOCATIONS
            if location != current_location
        ]

        df.loc[index, "location"] = random.choice(other_locations)

    elif anomaly_type == "high_velocity":

        customer_id = df.loc[index, "customer_id"]

        timestamp = df.loc[index, "transaction_timestamp"]

        # Create several transactions for the same customer
        # within a very short period.
        for j in range(1, 4):

            new_transaction = df.loc[index].copy()

            new_transaction["transaction_id"] = (
                f"TXN_VEL_{index}_{j}"
            )

            new_transaction["transaction_timestamp"] = (
                timestamp + timedelta(seconds=j * 20)
            )

            new_transaction["amount"] = round(
                random.uniform(100, 5000),
                2,
            )

            new_transaction["is_injected_anomaly"]=1
            new_transaction["anomaly_type"] = "high_velocity"

            df = pd.concat(
                [df, pd.DataFrame([new_transaction])],
                ignore_index=True,
            )


# -----------------------------
# Derived columns
# -----------------------------

df["hour"] = df["transaction_timestamp"].dt.hour

df["day_of_week"] = df["transaction_timestamp"].dt.day_name()

df["transaction_date"] = (
    df["transaction_timestamp"].dt.date
)


# Sort by timestamp
df = df.sort_values(
    "transaction_timestamp"
).reset_index(drop=True)


# -----------------------------
# Save dataset
# -----------------------------

output_path = "data/synthetic/transactions.csv"

df.to_csv(
    output_path,
    index=False,
)

print("=" * 60)
print("Synthetic transaction dataset generated successfully")
print("=" * 60)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Output: {output_path}")

print("\nColumns:")
for column in df.columns:
    print(f" - {column}")

print("\nSample:")
print(df.head())

print("\nBasic statistics:")
print(df["amount"].describe())

print("\nMissing values:")
print(df.isnull().sum())