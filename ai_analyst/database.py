import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


load_dotenv()


_engine = None


def get_engine():

    global _engine

    if _engine is not None:
        return _engine

    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB")

    if not user:
        raise RuntimeError(
            "POSTGRES_USER is not configured in .env"
        )

    if not password:
        raise RuntimeError(
            "POSTGRES_PASSWORD is not configured in .env"
        )

    if not database:
        raise RuntimeError(
            "POSTGRES_DB is not configured in .env"
        )

    database_url = (
        "postgresql+psycopg2://"
        f"{user}:{password}@{host}:{port}/{database}"
    )

    _engine = create_engine(
        database_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5
    )

    return _engine


def execute_query(sql):

    engine = get_engine()

    with engine.connect() as connection:

        result = connection.execute(
            text(sql)
        )

        rows = result.fetchall()
        columns = list(result.keys())

    return columns, rows