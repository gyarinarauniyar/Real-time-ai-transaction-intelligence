import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from .config import (
    MODEL_NAME,
    MODEL_VERSION,
    MAX_ANOMALY_RATE,
    MAX_AVERAGE_RISK,
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


def create_alert(
    engine,
    alert_type,
    severity,
    metric_name,
    metric_value,
    threshold,
    message,
):

    query = text("""
        INSERT INTO mlops.alerts (
            model_name,
            model_version,
            alert_type,
            severity,
            metric_name,
            metric_value,
            threshold,
            message
        )
        VALUES (
            :model_name,
            :model_version,
            :alert_type,
            :severity,
            :metric_name,
            :metric_value,
            :threshold,
            :message
        );
    """)

    with engine.begin() as connection:

        connection.execute(
            query,
            {
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "alert_type": alert_type,
                "severity": severity,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "threshold": threshold,
                "message": message,
            }
        )


def get_latest_window(engine):

    query = text("""
        SELECT
            total_predictions,
            anomaly_rate,
            average_risk_score,
            high_risk_count,
            critical_risk_count,
            window_end
        FROM mlops.monitoring_windows
        WHERE model_name = :model_name
          AND model_version = :model_version
        ORDER BY window_end DESC
        LIMIT 1;
    """)

    with engine.connect() as connection:

        return connection.execute(
            query,
            {
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
            }
        ).fetchone()


def get_latest_drift(engine):

    query = text("""
        SELECT
            feature_name,
            drift_type,
            drift_score,
            threshold
        FROM mlops.drift_results
        WHERE model_name = :model_name
          AND model_version = :model_version
          AND drift_detected = TRUE
        ORDER BY detected_at DESC;
    """)

    with engine.connect() as connection:

        return connection.execute(
            query,
            {
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
            }
        ).fetchall()


def main():

    engine = get_engine()

    print()
    print("=" * 60)
    print("MODEL ALERT ENGINE")
    print("=" * 60)

    window = get_latest_window(
        engine
    )

    if window is None:

        print(
            "No monitoring window available."
        )

        return

    alerts_created = 0

    # ---------------------------------------------------------
    # Anomaly-rate alert
    # ---------------------------------------------------------

    if (
        window.anomaly_rate
        > MAX_ANOMALY_RATE
    ):

        create_alert(
            engine,
            "anomaly_rate",
            "high",
            "anomaly_rate",
            float(window.anomaly_rate),
            MAX_ANOMALY_RATE,
            (
                f"Anomaly rate reached "
                f"{window.anomaly_rate:.2f}%, "
                f"above threshold "
                f"{MAX_ANOMALY_RATE:.2f}%."
            ),
        )

        print(
            "[ALERT] High anomaly rate"
        )

        alerts_created += 1

    # ---------------------------------------------------------
    # Average-risk alert
    # ---------------------------------------------------------

    if (
        window.average_risk_score
        > MAX_AVERAGE_RISK
    ):

        create_alert(
            engine,
            "average_risk",
            "high",
            "average_risk_score",
            float(
                window.average_risk_score
            ),
            MAX_AVERAGE_RISK,
            (
                f"Average risk score reached "
                f"{window.average_risk_score:.2f}, "
                f"above threshold "
                f"{MAX_AVERAGE_RISK:.2f}."
            ),
        )

        print(
            "[ALERT] High average risk"
        )

        alerts_created += 1

    # ---------------------------------------------------------
    # Critical-risk alert
    # ---------------------------------------------------------

    if window.critical_risk_count > 0:

        create_alert(
            engine,
            "critical_risk",
            "critical",
            "critical_risk_count",
            float(
                window.critical_risk_count
            ),
            0,
            (
                f"{window.critical_risk_count} "
                f"critical-risk transactions "
                f"were detected in the latest "
                f"monitoring window."
            ),
        )

        print(
            "[ALERT] Critical-risk transactions"
        )

        alerts_created += 1

    # ---------------------------------------------------------
    # Drift alerts
    # ---------------------------------------------------------

    drift_results = get_latest_drift(
        engine
    )

    for drift in drift_results:

        create_alert(
            engine,
            "data_drift",
            "medium",
            drift.feature_name,
            float(drift.drift_score),
            float(drift.threshold),
            (
                f"Data drift detected for "
                f"feature '{drift.feature_name}'. "
                f"Drift score: "
                f"{drift.drift_score:.4f}; "
                f"threshold: "
                f"{drift.threshold:.4f}."
            ),
        )

        print(
            f"[ALERT] Drift detected: "
            f"{drift.feature_name}"
        )

        alerts_created += 1

    print()
    print(
        f"Alerts created: {alerts_created}"
    )


if __name__ == "__main__":
    main()