import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

MODEL_NAME = "isolation_forest"
ACTIVE_VERSION = "v1.0.0"
ACTUAL_FEATURE_COUNT = 16


# ============================================================
# DATABASE
# ============================================================

def get_database_url():

    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB")

    if not all([user, password, database]):
        raise RuntimeError(
            "Missing PostgreSQL configuration in .env"
        )

    return (
        f"postgresql+psycopg2://"
        f"{user}:{password}@{host}:{port}/{database}"
    )


engine = create_engine(get_database_url())


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MODEL REGISTRY METADATA FIX")
    print("=" * 70)

    print(f"Model           : {MODEL_NAME}")
    print(f"Active version  : {ACTIVE_VERSION}")
    print(f"Actual features : {ACTUAL_FEATURE_COUNT}")

    # --------------------------------------------------------
    # Read current metadata
    # --------------------------------------------------------

    select_query = text(
        """
        SELECT
            model_name,
            model_version,
            status,
            training_rows,
            feature_count
        FROM mlops.model_registry
        WHERE model_name = :model_name
          AND model_version = :model_version
        """
    )

    with engine.connect() as conn:

        row = conn.execute(
            select_query,
            {
                "model_name": MODEL_NAME,
                "model_version": ACTIVE_VERSION,
            },
        ).mappings().first()

    if row is None:
        raise RuntimeError(
            "Active model registry entry was not found."
        )

    print("\nCurrent registry metadata")
    print("-" * 70)

    print(
        f"Model version : {row['model_version']}"
    )

    print(
        f"Status        : {row['status']}"
    )

    print(
        f"Training rows : {row['training_rows']}"
    )

    print(
        f"Feature count : {row['feature_count']}"
    )

    if row["status"] != "active":
        raise RuntimeError(
            f"{ACTIVE_VERSION} is not currently active."
        )

    # --------------------------------------------------------
    # Update
    # --------------------------------------------------------

    update_query = text(
        """
        UPDATE mlops.model_registry
        SET feature_count = :feature_count
        WHERE model_name = :model_name
          AND model_version = :model_version
          AND status = 'active'
        """
    )

    with engine.begin() as conn:

        result = conn.execute(
            update_query,
            {
                "feature_count": ACTUAL_FEATURE_COUNT,
                "model_name": MODEL_NAME,
                "model_version": ACTIVE_VERSION,
            },
        )

    if result.rowcount != 1:
        raise RuntimeError(
            "Registry metadata update failed."
        )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    with engine.connect() as conn:

        updated = conn.execute(
            select_query,
            {
                "model_name": MODEL_NAME,
                "model_version": ACTIVE_VERSION,
            },
        ).mappings().first()

    print("\nUpdated registry metadata")
    print("-" * 70)

    print(
        f"Model version : {updated['model_version']}"
    )

    print(
        f"Status        : {updated['status']}"
    )

    print(
        f"Training rows : {updated['training_rows']}"
    )

    print(
        f"Feature count : {updated['feature_count']}"
    )

    if updated["feature_count"] != ACTUAL_FEATURE_COUNT:
        raise RuntimeError(
            "Feature count verification failed."
        )

    print("\n" + "=" * 70)
    print("REGISTRY METADATA FIX COMPLETE")
    print("=" * 70)

    print(
        f"Active model {ACTIVE_VERSION} "
        f"now correctly reports "
        f"{ACTUAL_FEATURE_COUNT} inference features."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()