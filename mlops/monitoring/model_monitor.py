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


MODEL_NAME = "isolation_forest"
MODEL_VERSION = "v1.0.0"


def get_engine():
    return create_engine(
        DATABASE_URL,
        pool_pre_ping=True
    )


def calculate_monitoring_metrics(engine):
    query = text("""
        SELECT
            COUNT(*) AS total_predictions,

            COUNT(*) FILTER (
                WHERE is_anomaly = 1
            ) AS anomaly_count,

            ROUND(
                (
                    COUNT(*) FILTER (
                        WHERE is_anomaly = 1
                    )::NUMERIC
                    / NULLIF(COUNT(*), 0)
                ) * 100,
                4
            ) AS anomaly_rate,

            ROUND(
                AVG(risk_score),
                4
            ) AS average_risk_score,

            COUNT(*) FILTER (
                WHERE risk_level = 'high'
            ) AS high_risk_count,

            COUNT(*) FILTER (
                WHERE risk_level = 'critical'
            ) AS critical_risk_count

        FROM mlops.prediction_logs

        WHERE model_name = :model_name
          AND model_version = :model_version;
    """)

    with engine.connect() as connection:
        result = connection.execute(
            query,
            {
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
            }
        )

        return result.fetchone()


def store_monitoring_metrics(engine, metrics):
    query = text("""
        INSERT INTO mlops.model_monitoring (
            model_name,
            model_version,
            total_predictions,
            anomaly_count,
            anomaly_rate,
            average_risk_score,
            high_risk_count,
            critical_risk_count
        )
        VALUES (
            :model_name,
            :model_version,
            :total_predictions,
            :anomaly_count,
            :anomaly_rate,
            :average_risk_score,
            :high_risk_count,
            :critical_risk_count
        );
    """)

    with engine.begin() as connection:
        connection.execute(
            query,
            {
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "total_predictions": metrics.total_predictions,
                "anomaly_count": metrics.anomaly_count,
                "anomaly_rate": metrics.anomaly_rate,
                "average_risk_score": metrics.average_risk_score,
                "high_risk_count": metrics.high_risk_count,
                "critical_risk_count": metrics.critical_risk_count,
            }
        )


def main():

    engine = get_engine()

    print()
    print("=" * 60)
    print("MODEL MONITORING")
    print("=" * 60)
    print(f"Model name    : {MODEL_NAME}")
    print(f"Model version : {MODEL_VERSION}")
    print("=" * 60)

    metrics = calculate_monitoring_metrics(engine)

    if metrics.total_predictions == 0:
        print()
        print("No prediction logs found.")
        return

    store_monitoring_metrics(
        engine,
        metrics
    )

    print()
    print(f"Total predictions : {metrics.total_predictions:,}")
    print(f"Anomalies         : {metrics.anomaly_count:,}")
    print(f"Anomaly rate      : {metrics.anomaly_rate:.2f}%")
    print(f"Average risk      : {metrics.average_risk_score:.2f}")
    print(f"High risk         : {metrics.high_risk_count:,}")
    print(f"Critical risk     : {metrics.critical_risk_count:,}")
    print()
    print("Monitoring metrics stored successfully.")


if __name__ == "__main__":
    main()