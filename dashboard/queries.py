import pandas as pd
from sqlalchemy import text

from .db import get_engine


def load_dataframe(query: str, params=None) -> pd.DataFrame:
    engine = get_engine()

    with engine.connect() as connection:
        return pd.read_sql(
            text(query),
            connection,
            params=params or {},
        )


def get_transaction_date_range():
    query = """
        SELECT
            MIN(transaction_timestamp::timestamp)::date AS min_date,
            MAX(transaction_timestamp::timestamp)::date AS max_date
        FROM analytics.transaction_risk;
    """

    df = load_dataframe(query)

    return df.iloc[0]["min_date"], df.iloc[0]["max_date"]


def build_filter_conditions(
    risk_levels=None,
    payment_methods=None,
    merchant_categories=None,
    locations=None,
    start_date=None,
    end_date=None,
):
    conditions = []
    params = {}

    if risk_levels:
        conditions.append("final_risk_level = ANY(:risk_levels)")
        params["risk_levels"] = list(risk_levels)

    if payment_methods:
        conditions.append("payment_method = ANY(:payment_methods)")
        params["payment_methods"] = list(payment_methods)

    if merchant_categories:
        conditions.append("merchant_category = ANY(:merchant_categories)")
        params["merchant_categories"] = list(merchant_categories)

    if locations:
        conditions.append("location = ANY(:locations)")
        params["locations"] = list(locations)

    if start_date:
        conditions.append("transaction_timestamp::date >= :start_date")
        params["start_date"] = start_date

    if end_date:
        conditions.append("transaction_timestamp::date <= :end_date")
        params["end_date"] = end_date

    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)
    else:
        where_clause = ""

    return where_clause, params


def get_overall_kpis(
    risk_levels=None,
    payment_methods=None,
    merchant_categories=None,
    locations=None,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:

    where_clause, params = build_filter_conditions(
        risk_levels,
        payment_methods,
        merchant_categories,
        locations,
        start_date,
        end_date,
    )

    query = f"""
        SELECT
            COUNT(*) AS total_transactions,
            COALESCE(SUM(amount), 0) AS total_transaction_value,
            COALESCE(AVG(amount), 0) AS average_transaction_amount,
            COALESCE(AVG(risk_score), 0) AS average_risk_score,
            COUNT(DISTINCT customer_id) AS unique_customers,
            COUNT(DISTINCT merchant_id) AS unique_merchants,

            SUM(
                CASE
                    WHEN is_anomaly = 1 THEN 1
                    ELSE 0
                END
            ) AS ml_anomalies,

            SUM(
                CASE
                    WHEN final_risk_level IN ('high', 'critical')
                    THEN 1
                    ELSE 0
                END
            ) AS high_risk_transactions

        FROM analytics.transaction_risk
        {where_clause};
    """

    return load_dataframe(query, params)


def get_risk_distribution(
    risk_levels=None,
    payment_methods=None,
    merchant_categories=None,
    locations=None,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:

    where_clause, params = build_filter_conditions(
        risk_levels,
        payment_methods,
        merchant_categories,
        locations,
        start_date,
        end_date,
    )

    query = f"""
        SELECT
            final_risk_level,
            COUNT(*) AS transaction_count,
            SUM(amount) AS transaction_value,
            AVG(risk_score) AS average_risk_score
        FROM analytics.transaction_risk
        {where_clause}
        GROUP BY final_risk_level
        ORDER BY transaction_count DESC;
    """

    return load_dataframe(query, params)


def get_daily_risk(
    risk_levels=None,
    payment_methods=None,
    merchant_categories=None,
    locations=None,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:

    where_clause, params = build_filter_conditions(
        risk_levels,
        payment_methods,
        merchant_categories,
        locations,
        start_date,
        end_date,
    )

    query = f"""
        SELECT
            transaction_timestamp::date AS transaction_date,

            COUNT(*) AS transaction_count,

            SUM(amount) AS transaction_value,

            AVG(amount) AS average_transaction_amount,

            AVG(risk_score) AS average_risk_score,

            SUM(
                CASE
                    WHEN is_anomaly = 1 THEN 1
                    ELSE 0
                END
            ) AS ml_anomaly_count,

            SUM(
                CASE
                    WHEN final_risk_level IN ('high', 'critical')
                    THEN 1
                    ELSE 0
                END
            ) AS high_risk_count,

            SUM(
                CASE
                    WHEN final_risk_level = 'critical'
                    THEN 1
                    ELSE 0
                END
            ) AS critical_count

        FROM analytics.transaction_risk
        {where_clause}

        GROUP BY transaction_timestamp::date
        ORDER BY transaction_date;
    """

    return load_dataframe(query, params)


def get_top_customers(
    limit=10,
    risk_levels=None,
    payment_methods=None,
    merchant_categories=None,
    locations=None,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:

    where_clause, params = build_filter_conditions(
        risk_levels,
        payment_methods,
        merchant_categories,
        locations,
        start_date,
        end_date,
    )

    query = f"""
        SELECT
            customer_id,
            COUNT(*) AS transaction_count,
            SUM(amount) AS total_transaction_value,
            AVG(amount) AS average_transaction_amount,
            AVG(risk_score) AS average_risk_score,
            MAX(risk_score) AS maximum_risk_score,

            SUM(
                CASE
                    WHEN final_risk_level IN ('high', 'critical')
                    THEN 1
                    ELSE 0
                END
            ) AS high_risk_transactions,

            SUM(
                CASE
                    WHEN final_risk_level = 'critical'
                    THEN 1
                    ELSE 0
                END
            ) AS critical_transactions,

            SUM(
                CASE
                    WHEN is_anomaly = 1
                    THEN 1
                    ELSE 0
                END
            ) AS ml_anomalies

        FROM analytics.transaction_risk
        {where_clause}

        GROUP BY customer_id

        ORDER BY average_risk_score DESC

        LIMIT {int(limit)};
    """

    return load_dataframe(query, params)


def get_top_merchants(
    limit=10,
    risk_levels=None,
    payment_methods=None,
    merchant_categories=None,
    locations=None,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:

    where_clause, params = build_filter_conditions(
        risk_levels,
        payment_methods,
        merchant_categories,
        locations,
        start_date,
        end_date,
    )

    query = f"""
        SELECT
            merchant_id,
            merchant_category,
            COUNT(*) AS transaction_count,
            SUM(amount) AS total_transaction_value,
            AVG(amount) AS average_transaction_amount,
            AVG(risk_score) AS average_risk_score,
            MAX(risk_score) AS maximum_risk_score,

            SUM(
                CASE
                    WHEN final_risk_level IN ('high', 'critical')
                    THEN 1
                    ELSE 0
                END
            ) AS high_risk_transactions,

            SUM(
                CASE
                    WHEN final_risk_level = 'critical'
                    THEN 1
                    ELSE 0
                END
            ) AS critical_transactions

        FROM analytics.transaction_risk
        {where_clause}

        GROUP BY
            merchant_id,
            merchant_category

        ORDER BY average_risk_score DESC

        LIMIT {int(limit)};
    """

    return load_dataframe(query, params)


def get_filtered_transactions(
    risk_levels=None,
    payment_methods=None,
    merchant_categories=None,
    locations=None,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:

    where_clause, params = build_filter_conditions(
        risk_levels,
        payment_methods,
        merchant_categories,
        locations,
        start_date,
        end_date,
    )

    query = f"""
        SELECT
            transaction_id,
            transaction_timestamp,
            customer_id,
            merchant_id,
            amount,
            merchant_category,
            payment_method,
            location,
            risk_score,
            final_risk_level,
            risk_explanation

        FROM analytics.transaction_risk

        {where_clause}

        ORDER BY transaction_timestamp DESC

        LIMIT 500;
    """

    return load_dataframe(query, params)


def get_transaction_details(transaction_id: str) -> pd.DataFrame:

    query = """
        SELECT
            transaction_id,
            transaction_timestamp,
            customer_id,
            merchant_id,
            amount,
            currency,
            merchant_category,
            payment_method,
            location,
            device_type,
            risk_score,
            behavioral_risk_score,
            anomaly_prediction,
            is_anomaly,
            final_risk_level,
            risk_explanation,
            is_injected_anomaly,
            anomaly_type

        FROM analytics.transaction_risk

        WHERE transaction_id = :transaction_id;
    """

    return load_dataframe(
        query,
        {"transaction_id": transaction_id},
    )
def get_live_risk_feed(limit=50) -> pd.DataFrame:
    query = f"""
        SELECT
            transaction_id,
            processed_at,
            risk_score,
            risk_level,
            is_anomaly
        FROM streaming.transaction_risk_events
        ORDER BY processed_at DESC
        LIMIT {int(limit)};
    """

    return load_dataframe(query)