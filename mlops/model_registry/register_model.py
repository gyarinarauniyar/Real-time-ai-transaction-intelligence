import os
from datetime import datetime

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

MODEL_NAME = "isolation_forest"
MODEL_VERSION = "v1.0.0"

MODEL_PATH = "models/isolation_forest.joblib"
CALIBRATION_PATH = "models/risk_calibration.joblib"

TRAINING_ROWS = 8175
FEATURE_COUNT = 38


def get_engine():
    return create_engine(
        DATABASE_URL,
        pool_pre_ping=True
    )


def register_model():
    engine = get_engine()

    query = text("""
        INSERT INTO mlops.model_registry (
            model_name,
            model_version,
            model_type,
            model_path,
            calibration_path,
            training_date,
            training_rows,
            feature_count,
            status
        )
        VALUES (
            :model_name,
            :model_version,
            :model_type,
            :model_path,
            :calibration_path,
            :training_date,
            :training_rows,
            :feature_count,
            'active'
        )
        ON CONFLICT (model_name, model_version)
        DO UPDATE SET
            model_path = EXCLUDED.model_path,
            calibration_path = EXCLUDED.calibration_path,
            training_rows = EXCLUDED.training_rows,
            feature_count = EXCLUDED.feature_count,
            status = EXCLUDED.status;
    """)

    with engine.begin() as connection:
        connection.execute(
            query,
            {
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "model_type": "Isolation Forest",
                "model_path": MODEL_PATH,
                "calibration_path": CALIBRATION_PATH,
                "training_date": datetime.now(),
                "training_rows": TRAINING_ROWS,
                "feature_count": FEATURE_COUNT,
            }
        )

    print("Model registered successfully.")
    print(f"Model name    : {MODEL_NAME}")
    print(f"Model version : {MODEL_VERSION}")
    print(f"Model path    : {MODEL_PATH}")
    print(f"Calibration   : {CALIBRATION_PATH}")


if __name__ == "__main__":
    register_model()