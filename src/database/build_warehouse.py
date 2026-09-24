"""
Build the PostgreSQL analytics warehouse layer.
"""

from pathlib import Path
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


load_dotenv()


DATABASE_URL = (
    f"postgresql+psycopg2://"
    f"{os.getenv('POSTGRES_USER')}:"
    f"{os.getenv('POSTGRES_PASSWORD')}@"
    f"{os.getenv('POSTGRES_HOST')}:"
    f"{os.getenv('POSTGRES_PORT')}/"
    f"{os.getenv('POSTGRES_DB')}"
)


def main():

    print("=" * 60)
    print("POSTGRESQL ANALYTICS WAREHOUSE")
    print("=" * 60)

    engine = create_engine(DATABASE_URL)

    with engine.begin() as connection:

        print("\nCreating schemas...")

        connection.execute(
            text("""
                CREATE SCHEMA IF NOT EXISTS raw;
            """)
        )

        connection.execute(
            text("""
                CREATE SCHEMA IF NOT EXISTS analytics;
            """)
        )

        connection.execute(
            text("""
                CREATE SCHEMA IF NOT EXISTS views;
            """)
        )

        print("Schemas created.")

        # -----------------------------------------------------
        # RAW LAYER
        # -----------------------------------------------------

        print("\nBuilding raw layer...")

        connection.execute(
            text("""
                DROP TABLE IF EXISTS
                    raw.transactions;
            """)
        )

        connection.execute(
            text("""
                CREATE TABLE raw.transactions AS
                SELECT *
                FROM public.transactions;
            """)
        )

        print("raw.transactions created.")

        # -----------------------------------------------------
        # ANALYTICS: TRANSACTION RISK
        # -----------------------------------------------------

        print("\nBuilding transaction risk table...")

        connection.execute(
            text("""
                DROP TABLE IF EXISTS
                    analytics.transaction_risk;
            """)
        )

        connection.execute(
            text("""
                CREATE TABLE analytics.transaction_risk AS

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

                FROM raw.transactions;
            """)
        )

        print(
            "analytics.transaction_risk created."
        )

        # -----------------------------------------------------
        # ANALYTICS: CUSTOMER RISK
        # -----------------------------------------------------

        print("\nBuilding customer risk table...")

        connection.execute(
            text("""
                DROP TABLE IF EXISTS
                    analytics.customer_risk;
            """)
        )

        connection.execute(
            text("""
                CREATE TABLE analytics.customer_risk AS

                SELECT

                    customer_id,

                    COUNT(*) AS transaction_count,

                    ROUND(
                        SUM(amount)::numeric,
                        2
                    ) AS total_transaction_value,

                    ROUND(
                        AVG(amount)::numeric,
                        2
                    ) AS average_transaction_amount,

                    ROUND(
                        AVG(risk_score)::numeric,
                        2
                    ) AS average_risk_score,

                    ROUND(
                        MAX(risk_score)::numeric,
                        2
                    ) AS maximum_risk_score,

                    SUM(
                        CASE
                            WHEN final_risk_level
                                IN ('high', 'critical')
                            THEN 1
                            ELSE 0
                        END
                    ) AS high_risk_transactions,

                    SUM(
                        CASE
                            WHEN final_risk_level =
                                'critical'
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

                FROM raw.transactions

                GROUP BY customer_id;
            """)
        )

        print(
            "analytics.customer_risk created."
        )

        # -----------------------------------------------------
        # ANALYTICS: MERCHANT RISK
        # -----------------------------------------------------

        print("\nBuilding merchant risk table...")

        connection.execute(
            text("""
                DROP TABLE IF EXISTS
                    analytics.merchant_risk;
            """)
        )

        connection.execute(
            text("""
                CREATE TABLE analytics.merchant_risk AS

                SELECT

                    merchant_id,

                    merchant_category,

                    COUNT(*) AS transaction_count,

                    ROUND(
                        SUM(amount)::numeric,
                        2
                    ) AS total_transaction_value,

                    ROUND(
                        AVG(amount)::numeric,
                        2
                    ) AS average_transaction_amount,

                    ROUND(
                        AVG(risk_score)::numeric,
                        2
                    ) AS average_risk_score,

                    ROUND(
                        MAX(risk_score)::numeric,
                        2
                    ) AS maximum_risk_score,

                    SUM(
                        CASE
                            WHEN final_risk_level
                                IN ('high', 'critical')
                            THEN 1
                            ELSE 0
                        END
                    ) AS high_risk_transactions,

                    SUM(
                        CASE
                            WHEN final_risk_level =
                                'critical'
                            THEN 1
                            ELSE 0
                        END
                    ) AS critical_transactions

                FROM raw.transactions

                GROUP BY
                    merchant_id,
                    merchant_category;
            """)
        )

        print(
            "analytics.merchant_risk created."
        )

        # -----------------------------------------------------
        # ANALYTICS: DAILY RISK
        # -----------------------------------------------------

        print("\nBuilding daily risk table...")

        connection.execute(
            text("""
                DROP TABLE IF EXISTS
                    analytics.daily_risk;
            """)
        )

        connection.execute(
            text("""
                CREATE TABLE analytics.daily_risk AS

                SELECT

                    transaction_date,

                    COUNT(*) AS transaction_count,

                    ROUND(
                        SUM(amount)::numeric,
                        2
                    ) AS transaction_value,

                    ROUND(
                        AVG(amount)::numeric,
                        2
                    ) AS average_transaction_amount,

                    ROUND(
                        AVG(risk_score)::numeric,
                        2
                    ) AS average_risk_score,

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
                            WHEN final_risk_level =
                                'critical'
                            THEN 1
                            ELSE 0
                        END
                    ) AS critical_count

                FROM raw.transactions

                GROUP BY transaction_date

                ORDER BY transaction_date;
            """)
        )

        print(
            "analytics.daily_risk created."
        )

        # -----------------------------------------------------
        # INDEXES
        # -----------------------------------------------------

        print("\nCreating warehouse indexes...")

        indexes = [
            """
            CREATE INDEX IF NOT EXISTS
            idx_risk_customer
            ON analytics.transaction_risk(customer_id);
            """,

            """
            CREATE INDEX IF NOT EXISTS
            idx_risk_merchant
            ON analytics.transaction_risk(merchant_id);
            """,

            """
            CREATE INDEX IF NOT EXISTS
            idx_risk_timestamp
            ON analytics.transaction_risk(transaction_timestamp);
            """,

            """
            CREATE INDEX IF NOT EXISTS
            idx_risk_level
            ON analytics.transaction_risk(final_risk_level);
            """,

            """
            CREATE INDEX IF NOT EXISTS
            idx_customer_risk
            ON analytics.customer_risk(average_risk_score);
            """,

            """
            CREATE INDEX IF NOT EXISTS
            idx_merchant_risk
            ON analytics.merchant_risk(average_risk_score);
            """,

            """
            CREATE INDEX IF NOT EXISTS
            idx_daily_risk_date
            ON analytics.daily_risk(transaction_date);
            """
        ]

        for index_sql in indexes:
            connection.execute(text(index_sql))

        print("Indexes created.")

        # -----------------------------------------------------
        # VIEWS
        # -----------------------------------------------------

        print("\nBuilding analytical views...")

        connection.execute(
            text("""
                CREATE OR REPLACE VIEW
                views.risk_summary AS

                SELECT

                    COUNT(*) AS total_transactions,

                    ROUND(
                        SUM(amount)::numeric,
                        2
                    ) AS total_transaction_value,

                    ROUND(
                        AVG(amount)::numeric,
                        2
                    ) AS average_transaction_amount,

                    ROUND(
                        AVG(risk_score)::numeric,
                        2
                    ) AS average_risk_score,

                    COUNT(
                        CASE
                            WHEN is_anomaly = 1
                            THEN 1
                        END
                    ) AS ml_anomaly_count,

                    COUNT(
                        CASE
                            WHEN final_risk_level
                                IN ('high', 'critical')
                            THEN 1
                        END
                    ) AS high_risk_count,

                    COUNT(
                        CASE
                            WHEN final_risk_level =
                                'critical'
                            THEN 1
                        END
                    ) AS critical_count

                FROM raw.transactions;
            """)
        )

        connection.execute(
            text("""
                CREATE OR REPLACE VIEW
                views.customer_risk_summary AS

                SELECT *

                FROM analytics.customer_risk

                WHERE high_risk_transactions > 0

                ORDER BY
                    critical_transactions DESC,
                    high_risk_transactions DESC,
                    maximum_risk_score DESC;
            """)
        )

        connection.execute(
            text("""
                CREATE OR REPLACE VIEW
                views.merchant_risk_summary AS

                SELECT *

                FROM analytics.merchant_risk

                WHERE high_risk_transactions > 0

                ORDER BY
                    critical_transactions DESC,
                    high_risk_transactions DESC,
                    maximum_risk_score DESC;
            """)
        )

        print("Analytical views created.")

        # -----------------------------------------------------
        # VALIDATION
        # -----------------------------------------------------

        print("\nValidating warehouse...")

        tables = [
            "raw.transactions",
            "analytics.transaction_risk",
            "analytics.customer_risk",
            "analytics.merchant_risk",
            "analytics.daily_risk",
        ]

        for table in tables:

            count = connection.execute(
                text(
                    f"SELECT COUNT(*) FROM {table};"
                )
            ).fetchone()[0]

            print(
                f"{table}: {count:,} rows"
            )

        print("\nRisk summary:")

        summary = connection.execute(
            text("""
                SELECT *
                FROM views.risk_summary;
            """)
        ).fetchone()

        print(
            f"Transactions: {summary[0]:,}"
        )

        print(
            f"Transaction value: ₹{summary[1]:,.2f}"
        )

        print(
            f"Average amount: ₹{summary[2]:,.2f}"
        )

        print(
            f"Average ML risk: {summary[3]:.2f}"
        )

        print(
            f"ML anomalies: {summary[4]:,}"
        )

        print(
            f"High risk: {summary[5]:,}"
        )

        print(
            f"Critical: {summary[6]:,}"
        )

    print("\nWarehouse build completed successfully.")

    print("=" * 60)


if __name__ == "__main__":
    main()