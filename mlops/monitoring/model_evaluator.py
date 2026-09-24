import os

from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
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


def load_evaluation_data(engine):

    query = text("""
        SELECT
            p.prediction_id,
            p.transaction_id,
            p.is_anomaly AS predicted_label,
            f.actual_label

        FROM mlops.prediction_logs p

        INNER JOIN mlops.prediction_feedback f
            ON p.prediction_id = f.prediction_id

        WHERE p.model_name = :model_name
          AND p.model_version = :model_version;
    """)

    with engine.connect() as connection:

        rows = connection.execute(
            query,
            {
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
            }
        ).fetchall()

    return rows


def main():
    engine = get_engine()

    rows = load_evaluation_data(engine)

    if not rows:
        print("No labelled predictions found.")
        return

    y_true = [row.actual_label for row in rows]
    y_pred = [row.predicted_label for row in rows]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    matrix = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = matrix.ravel()

    print()
    print("=" * 60)
    print("MODEL PERFORMANCE EVALUATION")
    print("=" * 60)
    print(f"Model   : {MODEL_NAME}")
    print(f"Version : {MODEL_VERSION}")
    print("=" * 60)

    print(f"Labelled predictions : {len(rows)}")
    print(f"Accuracy             : {accuracy:.4f}")
    print(f"Precision            : {precision:.4f}")
    print(f"Recall               : {recall:.4f}")
    print(f"F1 Score             : {f1:.4f}")

    print()
    print("Confusion Matrix:")
    print(matrix)

    # Store evaluation results
    query = text("""
        INSERT INTO mlops.model_evaluation (
            model_name,
            model_version,
            sample_size,
            accuracy,
            precision_score,
            recall_score,
            f1_score,
            true_positive,
            true_negative,
            false_positive,
            false_negative
        )
        VALUES (
            :model_name,
            :model_version,
            :sample_size,
            :accuracy,
            :precision_score,
            :recall_score,
            :f1_score,
            :true_positive,
            :true_negative,
            :false_positive,
            :false_negative
        );
    """)

    with engine.begin() as connection:
        connection.execute(
            query,
            {
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "sample_size": len(rows),
                "accuracy": accuracy,
                "precision_score": precision,
                "recall_score": recall,
                "f1_score": f1,
                "true_positive": int(tp),
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
            }
        )

    print()
    print("Evaluation results stored in mlops.model_evaluation.")
    print("=" * 60)


if __name__ == "__main__":
    main()