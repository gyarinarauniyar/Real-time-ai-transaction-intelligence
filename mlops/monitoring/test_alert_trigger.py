from sqlalchemy import text

from .window_monitor import get_engine


def main():

    engine = get_engine()

    print()
    print("=" * 70)
    print("CONTROLLED ALERT TEST")
    print("=" * 70)

    with engine.begin() as conn:

        conn.execute(
            text(
                """
                INSERT INTO mlops.monitoring_windows
                (
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
                VALUES
                (
                    'isolation_forest',
                    'v1.0.0',
                    CURRENT_TIMESTAMP - INTERVAL '1 minute',
                    CURRENT_TIMESTAMP,
                    20,
                    10,
                    50.0,
                    85.0,
                    8,
                    4
                )
                """
            )
        )

    print()
    print("Controlled high-risk monitoring window inserted.")
    print("Expected: alert should be created.")
    print()


if __name__ == "__main__":
    main()