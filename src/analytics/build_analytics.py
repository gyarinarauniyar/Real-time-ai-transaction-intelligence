"""
Build DuckDB analytics views from the final transaction risk dataset.
"""

from pathlib import Path

import duckdb


INPUT_PATH = Path(
    "data/processed/transactions_final_risk.csv"
)

DB_PATH = Path(
    "data/processed/analytics.duckdb"
)


def main():

    print("=" * 60)
    print("TRANSACTION ANALYTICS LAYER")
    print("=" * 60)

    print("\nConnecting to DuckDB...")

    con = duckdb.connect(str(DB_PATH))

    print("\nLoading final risk dataset...")

    con.execute(
        f"""
        CREATE OR REPLACE TABLE transactions AS
        SELECT *
        FROM read_csv_auto(
            '{INPUT_PATH.as_posix()}',
            header=True
        )
        """
    )

    count = con.execute(
        "SELECT COUNT(*) FROM transactions"
    ).fetchone()[0]

    print(f"Loaded transactions: {count:,}")

    print("\nCreating analytics views...")

    # ---------------------------------------------------------
    # 1. Overall transaction summary
    # ---------------------------------------------------------

    con.execute(
        """
        CREATE OR REPLACE VIEW vw_transaction_summary AS

        SELECT
            COUNT(*) AS total_transactions,

            ROUND(SUM(amount), 2)
                AS total_transaction_value,

            ROUND(AVG(amount), 2)
                AS average_transaction_amount,

            COUNT(
                CASE
                    WHEN is_anomaly = 1
                    THEN 1
                END
            ) AS ml_anomaly_count,

            COUNT(
                CASE
                    WHEN is_injected_anomaly = 1
                    THEN 1
                END
            ) AS injected_anomaly_count,

            COUNT(
                CASE
                    WHEN final_risk_level IN
                        ('high', 'critical')
                    THEN 1
                END
            ) AS high_risk_transactions,

            COUNT(
                CASE
                    WHEN final_risk_level = 'critical'
                    THEN 1
                END
            ) AS critical_transactions

        FROM transactions
        """
    )

    # ---------------------------------------------------------
    # 2. Daily risk
    # ---------------------------------------------------------

    con.execute(
        """
        CREATE OR REPLACE VIEW vw_daily_risk AS

        SELECT
            transaction_date,

            COUNT(*) AS transaction_count,

            ROUND(SUM(amount), 2)
                AS transaction_value,

            ROUND(AVG(amount), 2)
                AS average_amount,

            SUM(
                CASE
                    WHEN is_anomaly = 1
                    THEN 1
                    ELSE 0
                END
            ) AS ml_anomaly_count,

            SUM(
                CASE
                    WHEN final_risk_level
                        IN ('high', 'critical')
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

        FROM transactions

        GROUP BY transaction_date

        ORDER BY transaction_date
        """
    )

    # ---------------------------------------------------------
    # 3. Risk distribution
    # ---------------------------------------------------------

    con.execute(
        """
        CREATE OR REPLACE VIEW vw_risk_distribution AS

        SELECT
            final_risk_level,

            COUNT(*) AS transaction_count,

            ROUND(
                100.0 * COUNT(*) /
                SUM(COUNT(*)) OVER (),
                2
            ) AS percentage_of_transactions,

            ROUND(
                SUM(amount),
                2
            ) AS transaction_value

        FROM transactions

        GROUP BY final_risk_level

        ORDER BY
            CASE final_risk_level
                WHEN 'critical' THEN 1
                WHEN 'high' THEN 2
                WHEN 'behavioral_review' THEN 3
                WHEN 'medium' THEN 4
                WHEN 'low' THEN 5
            END
        """
    )

    # ---------------------------------------------------------
    # 4. Payment method risk
    # ---------------------------------------------------------

    con.execute(
        """
        CREATE OR REPLACE VIEW vw_risk_by_payment_method AS

        SELECT
            payment_method,

            COUNT(*) AS transaction_count,

            ROUND(
                SUM(amount),
                2
            ) AS transaction_value,

            ROUND(
                AVG(risk_score),
                2
            ) AS average_ml_risk_score,

            SUM(
                CASE
                    WHEN final_risk_level
                        IN ('high', 'critical')
                    THEN 1
                    ELSE 0
                END
            ) AS high_risk_count,

            ROUND(
                100.0 *
                SUM(
                    CASE
                        WHEN final_risk_level
                            IN ('high', 'critical')
                        THEN 1
                        ELSE 0
                    END
                ) / COUNT(*),
                2
            ) AS high_risk_percentage

        FROM transactions

        GROUP BY payment_method

        ORDER BY high_risk_percentage DESC
        """
    )

    # ---------------------------------------------------------
    # 5. Location risk
    # ---------------------------------------------------------

    con.execute(
        """
        CREATE OR REPLACE VIEW vw_risk_by_location AS

        SELECT
            location,

            COUNT(*) AS transaction_count,

            ROUND(
                SUM(amount),
                2
            ) AS transaction_value,

            ROUND(
                AVG(risk_score),
                2
            ) AS average_ml_risk_score,

            SUM(
                CASE
                    WHEN final_risk_level
                        IN ('high', 'critical')
                    THEN 1
                    ELSE 0
                END
            ) AS high_risk_count,

            SUM(
                CASE
                    WHEN is_location_changed = 1
                    THEN 1
                    ELSE 0
                END
            ) AS location_change_count

        FROM transactions

        GROUP BY location

        ORDER BY high_risk_count DESC
        """
    )

    # ---------------------------------------------------------
    # 6. Merchant category risk
    # ---------------------------------------------------------

    con.execute(
        """
        CREATE OR REPLACE VIEW vw_risk_by_merchant_category AS

        SELECT
            merchant_category,

            COUNT(*) AS transaction_count,

            ROUND(
                SUM(amount),
                2
            ) AS transaction_value,

            ROUND(
                AVG(amount),
                2
            ) AS average_transaction_amount,

            ROUND(
                AVG(risk_score),
                2
            ) AS average_ml_risk_score,

            SUM(
                CASE
                    WHEN final_risk_level
                        IN ('high', 'critical')
                    THEN 1
                    ELSE 0
                END
            ) AS high_risk_count

        FROM transactions

        GROUP BY merchant_category

        ORDER BY high_risk_count DESC
        """
    )

    # ---------------------------------------------------------
    # 7. High-risk customers
    # ---------------------------------------------------------

    con.execute(
        """
        CREATE OR REPLACE VIEW vw_high_risk_customers AS

        SELECT
            customer_id,

            COUNT(*) AS transaction_count,

            ROUND(
                SUM(amount),
                2
            ) AS total_transaction_value,

            ROUND(
                AVG(risk_score),
                2
            ) AS average_risk_score,

            MAX(risk_score)
                AS maximum_risk_score,

            SUM(
                CASE
                    WHEN final_risk_level
                        IN ('high', 'critical')
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

        FROM transactions

        GROUP BY customer_id

        HAVING
            high_risk_count > 0

        ORDER BY
            critical_count DESC,
            high_risk_count DESC,
            maximum_risk_score DESC
        """
    )

    # ---------------------------------------------------------
    # 8. High-risk merchants
    # ---------------------------------------------------------

    con.execute(
        """
        CREATE OR REPLACE VIEW vw_high_risk_merchants AS

        SELECT
            merchant_id,

            merchant_category,

            COUNT(*) AS transaction_count,

            ROUND(
                SUM(amount),
                2
            ) AS total_transaction_value,

            ROUND(
                AVG(risk_score),
                2
            ) AS average_risk_score,

            SUM(
                CASE
                    WHEN final_risk_level
                        IN ('high', 'critical')
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

        FROM transactions

        GROUP BY
            merchant_id,
            merchant_category

        HAVING
            high_risk_count > 0

        ORDER BY
            critical_count DESC,
            high_risk_count DESC
        """
    )

    # ---------------------------------------------------------
    # 9. Hourly risk
    # ---------------------------------------------------------

    con.execute(
        """
        CREATE OR REPLACE VIEW vw_hourly_risk AS

        SELECT
            hour,

            COUNT(*) AS transaction_count,

            ROUND(
                AVG(amount),
                2
            ) AS average_amount,

            ROUND(
                AVG(risk_score),
                2
            ) AS average_risk_score,

            SUM(
                CASE
                    WHEN is_unusual_hour = 1
                    THEN 1
                    ELSE 0
                END
            ) AS unusual_hour_count,

            SUM(
                CASE
                    WHEN final_risk_level
                        IN ('high', 'critical')
                    THEN 1
                    ELSE 0
                END
            ) AS high_risk_count

        FROM transactions

        GROUP BY hour

        ORDER BY hour
        """
    )

    # ---------------------------------------------------------
    # 10. Anomaly types
    # ---------------------------------------------------------

    con.execute(
        """
        CREATE OR REPLACE VIEW vw_anomaly_types AS

        SELECT
            anomaly_type,

            COUNT(*) AS transaction_count,

            ROUND(
                100.0 * COUNT(*) /
                SUM(COUNT(*)) OVER (),
                2
            ) AS percentage_of_transactions,

            ROUND(
                AVG(risk_score),
                2
            ) AS average_risk_score,

            SUM(
                CASE
                    WHEN final_risk_level
                        IN ('high', 'critical')
                    THEN 1
                    ELSE 0
                END
            ) AS high_risk_count

        FROM transactions

        GROUP BY anomaly_type

        ORDER BY transaction_count DESC
        """
    )

    print("\nAnalytics views created successfully.")

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    views = [
        "vw_transaction_summary",
        "vw_daily_risk",
        "vw_risk_distribution",
        "vw_risk_by_payment_method",
        "vw_risk_by_location",
        "vw_risk_by_merchant_category",
        "vw_high_risk_customers",
        "vw_high_risk_merchants",
        "vw_hourly_risk",
        "vw_anomaly_types",
    ]

    print("\nValidating views...")

    for view in views:

        result = con.execute(
            f"SELECT COUNT(*) FROM {view}"
        ).fetchone()[0]

        print(f"{view}: {result:,} rows")

    print("\nOverall summary:")

    summary = con.execute(
        """
        SELECT *
        FROM vw_transaction_summary
        """
    ).fetchdf()

    print(summary.to_string(index=False))

    con.close()

    print("\nDuckDB database saved:")
    print(DB_PATH)

    print("=" * 60)


if __name__ == "__main__":
    main()