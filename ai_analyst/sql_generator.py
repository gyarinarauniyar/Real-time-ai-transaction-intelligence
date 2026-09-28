import re


TRANSACTION_ID_PATTERN = re.compile(
    r"\bTXN_[A-Za-z0-9_]+\b",
    re.IGNORECASE
)


def extract_transaction_id(question: str):
    """
    Extract a transaction ID from the user's question.
    """

    match = TRANSACTION_ID_PATTERN.search(question)

    if not match:
        return None

    return match.group(0)


def generate_transaction_sql(question: str):
    """
    Generate deterministic SQL for a specific transaction.

    Risk information comes from:
        streaming.transaction_risk_events

    Transaction information comes from:
        streaming.transaction_events

    The risk table is the primary table so that risk events
    without matching transaction details are still returned.
    """

    transaction_id = extract_transaction_id(question)

    if not transaction_id:
        raise ValueError(
            "No transaction ID found. "
            "Please provide a transaction ID such as TXN_VEL_922_2."
        )

    transaction_id = transaction_id.replace("'", "''")

    sql = f"""
SELECT
    r.transaction_id,

    e.customer_id,
    e.merchant_id,
    e.amount,
    e.currency,
    e.merchant_category,
    e.payment_method,
    e.location,
    e.device_type,
    e.timestamp,
    e.is_injected_anomaly,
    e.anomaly_type,

    r.anomaly_prediction,
    r.is_anomaly,
    r.risk_score,
    r.risk_level,
    r.processed_at

FROM streaming.transaction_risk_events AS r

LEFT JOIN streaming.transaction_events AS e
    ON r.transaction_id = e.transaction_id

WHERE r.transaction_id = '{transaction_id}'

LIMIT 1
""".strip()

    return sql


def generate_sql(question: str):
    """
    Generate deterministic SQL for common platform data questions.

    LLM-generated SQL is intentionally avoided so that the model
    cannot invent tables or columns.
    """

    q = question.lower().strip()

    # --------------------------------------------------
    # TOTAL TRANSACTION COUNT
    # --------------------------------------------------

    if (
        "how many transactions" in q
        or "total transactions" in q
        or "transaction count" in q
        or "count of transactions" in q
    ):
        return """
SELECT COUNT(*) AS transaction_count
FROM streaming.transaction_events
""".strip()

    # --------------------------------------------------
    # ANOMALY COUNT
    # --------------------------------------------------

    if (
        "how many anomalies" in q
        or "anomaly count" in q
        or "count of anomalies" in q
        or "number of anomalies" in q
    ):
        return """
SELECT COUNT(*) AS anomaly_count
FROM streaming.transaction_risk_events
WHERE is_anomaly = 1
""".strip()

    # --------------------------------------------------
    # HIGH-RISK COUNT
    # --------------------------------------------------

    if (
        "how many high risk" in q
        or "how many high-risk" in q
        or "high risk transactions" in q
        or "high-risk transactions" in q
    ):
        return """
SELECT COUNT(*) AS high_risk_count
FROM streaming.transaction_risk_events
WHERE risk_level = 'high'
""".strip()

    # --------------------------------------------------
    # CRITICAL-RISK COUNT
    # --------------------------------------------------

    if (
        "how many critical" in q
        or "critical transactions" in q
        or "critical risk" in q
        or "critical-risk" in q
    ):
        return """
SELECT COUNT(*) AS critical_count
FROM streaming.transaction_risk_events
WHERE risk_level = 'critical'
""".strip()

    # --------------------------------------------------
    # AVERAGE TRANSACTION AMOUNT
    # --------------------------------------------------

    if (
        "average transaction amount" in q
        or "average amount" in q
        or "avg transaction amount" in q
        or "avg amount" in q
    ):
        return """
SELECT ROUND(AVG(amount), 2) AS average_transaction_amount
FROM streaming.transaction_events
""".strip()

    # --------------------------------------------------
    # TOTAL TRANSACTION AMOUNT
    # --------------------------------------------------

    if (
        "total transaction amount" in q
        or "total amount" in q
        or "sum of transactions" in q
    ):
        return """
SELECT ROUND(SUM(amount), 2) AS total_transaction_amount
FROM streaming.transaction_events
""".strip()

    # --------------------------------------------------
    # LATEST TRANSACTIONS
    # --------------------------------------------------

    if (
        "latest transactions" in q
        or "recent transactions" in q
        or "show me transactions" in q
        or "show transactions" in q
    ):
        return """
SELECT
    e.transaction_id,
    e.customer_id,
    e.merchant_id,
    e.amount,
    e.currency,
    e.merchant_category,
    e.payment_method,
    e.location,
    e.device_type,
    e.timestamp,
    r.risk_score,
    r.risk_level,
    r.is_anomaly

FROM streaming.transaction_events AS e

LEFT JOIN streaming.transaction_risk_events AS r
    ON e.transaction_id = r.transaction_id

ORDER BY e.timestamp DESC

LIMIT 20
""".strip()

    # --------------------------------------------------
    # HIGH-RISK TRANSACTIONS
    # --------------------------------------------------

    if (
        "show high risk" in q
        or "show high-risk" in q
        or "list high risk" in q
        or "list high-risk" in q
    ):
        return """
SELECT
    e.transaction_id,
    e.customer_id,
    e.merchant_id,
    e.amount,
    e.merchant_category,
    e.location,
    r.risk_score,
    r.risk_level,
    r.is_anomaly,
    r.processed_at

FROM streaming.transaction_events AS e

JOIN streaming.transaction_risk_events AS r
    ON e.transaction_id = r.transaction_id

WHERE r.risk_level = 'high'

ORDER BY r.risk_score DESC

LIMIT 20
""".strip()

    # --------------------------------------------------
    # CRITICAL TRANSACTIONS
    # --------------------------------------------------

    if (
        "show critical" in q
        or "list critical" in q
    ):
        return """
SELECT
    e.transaction_id,
    e.customer_id,
    e.merchant_id,
    e.amount,
    e.merchant_category,
    e.location,
    r.risk_score,
    r.risk_level,
    r.is_anomaly,
    r.processed_at

FROM streaming.transaction_events AS e

JOIN streaming.transaction_risk_events AS r
    ON e.transaction_id = r.transaction_id

WHERE r.risk_level = 'critical'

ORDER BY r.risk_score DESC

LIMIT 20
""".strip()

    # --------------------------------------------------
    # LOCATION SUMMARY
    # --------------------------------------------------

    if (
        "transactions by location" in q
        or "transaction count by location" in q
        or "locations" in q
    ):
        return """
SELECT
    location,
    COUNT(*) AS transaction_count,
    ROUND(AVG(amount), 2) AS average_amount

FROM streaming.transaction_events

GROUP BY location

ORDER BY transaction_count DESC

LIMIT 20
""".strip()

    # --------------------------------------------------
    # PAYMENT METHOD SUMMARY
    # --------------------------------------------------

    if (
        "payment methods" in q
        or "transactions by payment method" in q
        or "transaction count by payment" in q
    ):
        return """
SELECT
    payment_method,
    COUNT(*) AS transaction_count,
    ROUND(AVG(amount), 2) AS average_amount

FROM streaming.transaction_events

GROUP BY payment_method

ORDER BY transaction_count DESC

LIMIT 20
""".strip()

    # --------------------------------------------------
    # MERCHANT CATEGORY SUMMARY
    # --------------------------------------------------

    if (
        "merchant categories" in q
        or "transactions by merchant category" in q
        or "transaction count by category" in q
    ):
        return """
SELECT
    merchant_category,
    COUNT(*) AS transaction_count,
    ROUND(AVG(amount), 2) AS average_amount

FROM streaming.transaction_events

GROUP BY merchant_category

ORDER BY transaction_count DESC

LIMIT 20
""".strip()

    raise ValueError(
        "I couldn't map that data question to a supported "
        "database query. Try asking about transaction counts, "
        "anomalies, risk levels, transaction amounts, recent "
        "transactions, locations, payment methods, or merchant categories."
    )