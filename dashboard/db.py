"""
PostgreSQL connection utilities for the Streamlit dashboard.
"""

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


def get_engine():
    """Create and return a SQLAlchemy PostgreSQL engine."""
    return create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
    )


def test_connection():
    """Test the PostgreSQL connection."""
    engine = get_engine()

    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT current_database(), current_user")
        )
        return result.fetchone()