"""
Load the final transaction risk dataset into PostgreSQL.
"""

from pathlib import Path
from dotenv import load_dotenv
import os

import pandas as pd
from sqlalchemy import create_engine, text


INPUT_PATH = Path(
    "data/processed/transactions_final_risk.csv"
)

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
    print("POSTGRESQL TRANSACTION DATA LOAD")
    print("=" * 60)

    print("\nLoading final risk dataset...")

    df = pd.read_csv(INPUT_PATH)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nConnecting to PostgreSQL...")

    engine = create_engine(DATABASE_URL)

    print("\nTesting connection...")

    with engine.connect() as connection:

        result = connection.execute(
            text("SELECT current_database();")
        )

        database_name = result.fetchone()[0]

        print(
            f"Connected to database: "
            f"{database_name}"
        )

    print("\nLoading transaction data...")

    df.to_sql(
        "transactions",
        engine,
        if_exists="replace",
        index=False,
        chunksize=1000,
        method="multi",
    )

    print(
        "\nTransaction table created successfully."
    )

    print("\nCreating indexes...")

    with engine.begin() as connection:

        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS
                idx_transactions_customer
                ON transactions(customer_id);
                """
            )
        )

        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS
                idx_transactions_merchant
                ON transactions(merchant_id);
                """
            )
        )

        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS
                idx_transactions_timestamp
                ON transactions(transaction_timestamp);
                """
            )
        )

        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS
                idx_transactions_risk
                ON transactions(final_risk_level);
                """
            )
        )

    print("Indexes created successfully.")

    print("\nValidating PostgreSQL table...")

    with engine.connect() as connection:

        row_count = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM transactions;
                """
            )
        ).fetchone()[0]

        customer_count = connection.execute(
            text(
                """
                SELECT COUNT(DISTINCT customer_id)
                FROM transactions;
                """
            )
        ).fetchone()[0]

        high_risk_count = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM transactions
                WHERE final_risk_level
                    IN ('high', 'critical');
                """
            )
        ).fetchone()[0]

    print(
        f"PostgreSQL rows: {row_count:,}"
    )

    print(
        f"Unique customers: "
        f"{customer_count:,}"
    )

    print(
        f"High/critical transactions: "
        f"{high_risk_count:,}"
    )

    print("\nPostgreSQL load completed.")

    print("=" * 60)


if __name__ == "__main__":
    main()