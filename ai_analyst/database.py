import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


load_dotenv()


def get_engine():

    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB")

    if not user:
        raise RuntimeError("POSTGRES_USER is not configured in .env")

    if not password:
        raise RuntimeError("POSTGRES_PASSWORD is not configured in .env")

    if not database:
        raise RuntimeError("POSTGRES_DB is not configured in .env")

    database_url = (
        f"postgresql+psycopg2://"
        f"{user}:{password}@{host}:{port}/{database}"
    )

    return create_engine(
        database_url,
        pool_pre_ping=True
    )


def execute_query(sql):

    engine = get_engine()

    with engine.connect() as connection:

        result = connection.execute(
            text(sql)
        )

        rows = result.fetchall()
        columns = result.keys()

    return columns, rows