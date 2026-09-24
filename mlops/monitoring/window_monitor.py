import os
from datetime import datetime, timedelta

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from .config import (
    MODEL_NAME,
    MODEL_VERSION,
    WINDOW_MINUTES,
    MIN_PREDICTIONS,
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


def get_engine():
    return create_engine(
        DATABASE_URL,
        pool_pre_ping=True
    )


def calculate_window_metrics(engine):

    window_end = datetime.now()
    window_start = (
        window_end
        - timedelta(minutes=WINDOW_MINUTES)
    )

    query = text("""
        SELECT
            COUNT(*) AS total_predictions,

            COUNT(*) FILTER (
                WHERE is_anomaly = 1
            ) AS anomaly_count,

            AVG(risk_score) AS average_risk_score,

            COUNT(*) FILTER (
                WHERE risk_score >= 60
            ) AS high_risk_count,

            COUNT(*) FILTER (
                WHERE risk_score >= 80
            ) AS critical_risk_count

        FROM mlops.prediction_logs

        WHERE model_name = :model_name
          AND model_version = :model_version
          AND prediction_timestamp >= :window_start
          AND prediction_timestamp <= :window_end;
    """)

    with engine.connect() as connection:

        result = connection.execute(
            query,
            {
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "window_start": window_start,
                "window_end": window_end,
            }
        ).fetchone()

    total = result.total_predictions or 0
    anomalies = result.anomaly_count or 0

    if total > 0:
        anomaly_rate = (
            anomalies / total
        ) * 100
    else:
        anomaly_rate = 0.0

    return {
        "window_start": window_start,
        "window_end": window_end,
        "total_predictions": total,
        "anomaly_count": anomalies,
        "anomaly_rate": anomaly_rate,
        "average_risk_score": (
            float(result.average_risk_score)
            if result.average_risk_score is not None
            else 0.0
        ),
        "high_risk_count": (
            result.high_risk_count or 0
        ),
        "critical_risk_count": (
            result.critical_risk_count or 0
        ),
    }


def store_window_metrics(
    engine,
    metrics
):

    query = text("""
        INSERT INTO mlops.monitoring_windows (
            model_name,
            model_version,
            window_start,
            window_end,
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
            :window_start,
            :window_end,
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
                "window_start": metrics["window_start"],
                "window_end": metrics["window_end"],
                "total_predictions": metrics["total_predictions"],
                "anomaly_count": metrics["anomaly_count"],
                "anomaly_rate": metrics["anomaly_rate"],
                "average_risk_score": metrics[
                    "average_risk_score"
                ],
                "high_risk_count": metrics[
                    "high_risk_count"
                ],
                "critical_risk_count": metrics[
                    "critical_risk_count"
                ],
            }
        )


def main():

    engine = get_engine()

    print()
    print("=" * 60)
    print("TIME-WINDOW MODEL MONITORING")
    print("=" * 60)

    metrics = calculate_window_metrics(engine)

    print(
        f"Window: "
        f"{metrics['window_start']} → "
        f"{metrics['window_end']}"
    )

    print()
    print(
        f"Predictions : "
        f"{metrics['total_predictions']:,}"
    )

    print(
        f"Anomalies   : "
        f"{metrics['anomaly_count']:,}"
    )

    print(
        f"Anomaly rate: "
        f"{metrics['anomaly_rate']:.2f}%"
    )

    print(
        f"Average risk: "
        f"{metrics['average_risk_score']:.2f}"
    )

    print(
        f"High risk   : "
        f"{metrics['high_risk_count']:,}"
    )

    print(
        f"Critical    : "
        f"{metrics['critical_risk_count']:,}"
    )

    if (
        metrics["total_predictions"]
        >= MIN_PREDICTIONS
    ):

        store_window_metrics(
            engine,
            metrics
        )

        print()
        print(
            "Monitoring window stored."
        )

    else:

        print()
        print(
            f"Not enough predictions. "
            f"Need at least {MIN_PREDICTIONS}."
        )


if __name__ == "__main__":
    main()