from .window_monitor import calculate_window_metrics
from .window_monitor import get_engine
from .window_monitor import store_window_metrics

from .drift_detector import main as run_drift_detection
from .alert_manager import main as run_alert_engine

from .config import MIN_PREDICTIONS


def main():

    print()
    print("=" * 70)
    print("MLOPS MONITORING PIPELINE")
    print("=" * 70)

    # ---------------------------------------------------------
    # Step 1: Time-window monitoring
    # ---------------------------------------------------------

    print()
    print("STEP 1 — WINDOW MONITORING")

    engine = get_engine()

    metrics = calculate_window_metrics(
        engine
    )

    print(
        f"Predictions : "
        f"{metrics['total_predictions']}"
    )

    print(
        f"Anomaly rate: "
        f"{metrics['anomaly_rate']:.2f}%"
    )

    print(
        f"Average risk: "
        f"{metrics['average_risk_score']:.2f}"
    )

    if (
        metrics["total_predictions"]
        >= MIN_PREDICTIONS
    ):

        store_window_metrics(
            engine,
            metrics
        )

        print(
            "Window metrics stored."
        )

    else:

        print(
            "Not enough predictions "
            "for monitoring window."
        )

    # ---------------------------------------------------------
    # Step 2: Drift detection
    # ---------------------------------------------------------

    print()
    print("STEP 2 — DRIFT DETECTION")

    run_drift_detection()

    # ---------------------------------------------------------
    # Step 3: Alerts
    # ---------------------------------------------------------

    print()
    print("STEP 3 — ALERT DETECTION")

    run_alert_engine()

    print()
    print("=" * 70)
    print("MLOPS MONITORING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()