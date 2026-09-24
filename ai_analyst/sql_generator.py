import requests

from .config import OLLAMA_URL, MODEL_NAME


SCHEMA = """
Database: transaction_intelligence

Schema: streaming

Table: streaming.transaction_risk_events
Columns:
- risk_event_id
- transaction_id
- anomaly_prediction
- is_anomaly
- risk_score
- risk_level
- processed_at

Table: streaming.event_queue
Columns:
- event_id
- transaction_id
- event_payload
- status
- created_at
- processed_at
- error_message

The event_payload JSONB contains transaction-level information such as:
- amount
- customer_id
- merchant_id
- merchant_category
- payment_method
- device_type
- location
- transaction_date
- hour
"""

SYSTEM_PROMPT = """
You are a SQL generation engine for a transaction intelligence platform.

Generate PostgreSQL SQL only.

Rules:

1. Use ONLY tables and columns provided in the schema.
2. ALWAYS use the complete schema-qualified table name.
3. For risk questions, use streaming.transaction_risk_events.
4. For transaction attributes, use streaming.event_queue.
5. Never modify data.
6. Never use INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE or CREATE.
7. Only generate SELECT queries.
8. Always include a reasonable LIMIT unless the query is an aggregate such as COUNT, SUM, AVG, MIN or MAX.
9. Return SQL only.
"""



def generate_sql(question):

    prompt = f"""
{SYSTEM_PROMPT}

DATABASE SCHEMA:

{SCHEMA}

USER QUESTION:

{question}

Return only the PostgreSQL SELECT query.
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 256
            }
        },
        timeout=180
    )

    response.raise_for_status()

    sql = response.json()["response"].strip()

    sql = sql.replace("```sql", "")
    sql = sql.replace("```", "")
    sql = sql.strip()

    return sql

def generate_transaction_sql(question):
    """
    Generate a deterministic SQL query for a specific transaction.

    Only extracts fields needed for risk explanation.
    """

    import re

    match = re.search(
        r"\bTXN_[A-Za-z0-9_]+\b",
        question,
        re.IGNORECASE
    )

    if not match:
        raise ValueError(
            "No transaction ID found. "
            "Please provide a transaction ID such as TXN_VEL_922_2."
        )

    transaction_id = match.group(0).replace("'", "''")

    sql = f"""
SELECT
    t.transaction_id,
    t.anomaly_prediction,
    t.is_anomaly,
    t.risk_score,
    t.risk_level,
    t.processed_at,

    e.event_payload ->> 'amount' AS amount,
    e.event_payload ->> 'currency' AS currency,
    e.event_payload ->> 'customer_id' AS customer_id,
    e.event_payload ->> 'merchant_id' AS merchant_id,
    e.event_payload ->> 'merchant_category' AS merchant_category,
    e.event_payload ->> 'payment_method' AS payment_method,
    e.event_payload ->> 'device_type' AS device_type,
    e.event_payload ->> 'location' AS location,
    e.event_payload ->> 'transaction_date' AS transaction_date,
    e.event_payload ->> 'hour' AS hour,
    e.event_payload ->> 'anomaly_type' AS anomaly_type,

    e.event_payload ->> 'amount_deviation' AS amount_deviation,
    e.event_payload ->> 'amount_to_customer_avg_ratio'
        AS amount_to_customer_avg_ratio,
    e.event_payload ->> 'amount_above_customer_avg_ratio'
        AS amount_above_customer_avg_ratio,
    e.event_payload ->> 'transactions_last_1h'
        AS transactions_last_1h,
    e.event_payload ->> 'transactions_last_24h'
        AS transactions_last_24h,
    e.event_payload ->> 'minutes_since_previous_transaction'
        AS minutes_since_previous_transaction,
    e.event_payload ->> 'is_unusual_hour'
        AS is_unusual_hour,
    e.event_payload ->> 'is_location_changed'
        AS is_location_changed

FROM streaming.transaction_risk_events AS t

LEFT JOIN streaming.event_queue AS e
    ON t.transaction_id = e.transaction_id

WHERE t.transaction_id = '{transaction_id}'

LIMIT 1
""".strip()

    return sql