from pathlib import Path
from datetime import datetime
import os
import shutil

import joblib

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_NAME = "isolation_forest"

ACTIVE_VERSION = "v1.0.0"
CANDIDATE_VERSION = "v1.1.0"

ACTIVE_MODEL_FILE = (
    MODEL_DIR / "isolation_forest.joblib"
)

ACTIVE_CALIBRATION_FILE = (
    MODEL_DIR / "risk_calibration.joblib"
)

CANDIDATE_MODEL_FILE = (
    MODEL_DIR / "isolation_forest_v1.1.0.joblib"
)

CANDIDATE_CALIBRATION_FILE = (
    MODEL_DIR / "risk_calibration_v1.1.0.joblib"
)

BACKUP_DIR = MODEL_DIR / "backups"


# ============================================================
# DATABASE
# ============================================================

load_dotenv()


def get_engine():

    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB")

    missing = []

    if not user:
        missing.append("POSTGRES_USER")

    if not password:
        missing.append("POSTGRES_PASSWORD")

    if not database:
        missing.append("POSTGRES_DB")

    if missing:
        raise ValueError(
            "Missing database environment variables: "
            + ", ".join(missing)
        )

    database_url = (
        f"postgresql+psycopg2://"
        f"{user}:{password}@{host}:{port}/{database}"
    )

    return create_engine(database_url)


# ============================================================
# CHECK FILES
# ============================================================

def check_candidate_files():

    print()
    print("Checking candidate artifacts...")

    if not CANDIDATE_MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Candidate model not found:\n"
            f"{CANDIDATE_MODEL_FILE}"
        )

    if not CANDIDATE_CALIBRATION_FILE.exists():

        raise FileNotFoundError(
            f"Candidate calibration not found:\n"
            f"{CANDIDATE_CALIBRATION_FILE}"
        )

    print("Candidate model      : FOUND")
    print("Candidate calibration: FOUND")


# ============================================================
# CHECK PROMOTION DECISION
# ============================================================

def get_promotion_decision(engine):

    query = text("""
        SELECT
            decision,
            f1_score,
            recall_score,
            f1_improvement,
            recall_change,
            evaluated_at
        FROM mlops.model_evaluations
        WHERE model_name = :model_name
          AND model_version = :model_version
        ORDER BY evaluated_at DESC
        LIMIT 1;
    """)

    with engine.connect() as connection:

        row = connection.execute(
            query,
            {
                "model_name": MODEL_NAME,
                "model_version": CANDIDATE_VERSION,
            },
        ).fetchone()

    if row is None:

        raise ValueError(
            "No evaluation record found for candidate "
            f"{CANDIDATE_VERSION}."
        )

    print()
    print("Latest candidate evaluation:")
    print(
        f"Decision       : {row.decision}"
    )
    print(
        f"F1 Score       : {row.f1_score}"
    )
    print(
        f"Recall         : {row.recall_score}"
    )
    print(
        f"F1 Improvement : {row.f1_improvement}"
    )
    print(
        f"Recall Change  : {row.recall_change}"
    )
    print(
        f"Evaluated At   : {row.evaluated_at}"
    )

    return row


# ============================================================
# VALIDATE CALIBRATION
# ============================================================

def validate_calibration():

    print()
    print("Validating candidate calibration...")

    calibration = joblib.load(
        CANDIDATE_CALIBRATION_FILE
    )

    required_keys = [
        "min_score",
        "max_score",
        "model_name",
        "model_version",
    ]

    for key in required_keys:

        if key not in calibration:

            raise ValueError(
                f"Calibration missing key: {key}"
            )

    if calibration["model_name"] != MODEL_NAME:

        raise ValueError(
            "Calibration model name does not match."
        )

    if (
        calibration["model_version"]
        != CANDIDATE_VERSION
    ):

        raise ValueError(
            "Calibration model version does not match."
        )

    if (
        calibration["max_score"]
        <= calibration["min_score"]
    ):

        raise ValueError(
            "Invalid calibration score range."
        )

    print("Calibration validation: PASSED")

    return calibration


# ============================================================
# BACKUP ACTIVE MODEL
# ============================================================

def backup_active_artifacts():

    print()
    print("Backing up active artifacts...")

    BACKUP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    model_backup = (
        BACKUP_DIR
        / f"isolation_forest_{ACTIVE_VERSION}_{timestamp}.joblib"
    )

    calibration_backup = (
        BACKUP_DIR
        / f"risk_calibration_{ACTIVE_VERSION}_{timestamp}.joblib"
    )

    if ACTIVE_MODEL_FILE.exists():

        shutil.copy2(
            ACTIVE_MODEL_FILE,
            model_backup,
        )

    if ACTIVE_CALIBRATION_FILE.exists():

        shutil.copy2(
            ACTIVE_CALIBRATION_FILE,
            calibration_backup,
        )

    print(
        f"Model backup       : {model_backup}"
    )

    print(
        f"Calibration backup : {calibration_backup}"
    )


# ============================================================
# PROMOTE FILES
# ============================================================

def promote_artifacts():

    print()
    print("Promoting candidate artifacts...")

    # Copy candidate model over active model.
    shutil.copy2(
        CANDIDATE_MODEL_FILE,
        ACTIVE_MODEL_FILE,
    )

    # Copy matching candidate calibration.
    shutil.copy2(
        CANDIDATE_CALIBRATION_FILE,
        ACTIVE_CALIBRATION_FILE,
    )

    print()
    print("Active model updated:")
    print(ACTIVE_MODEL_FILE)

    print()
    print("Active calibration updated:")
    print(ACTIVE_CALIBRATION_FILE)


# ============================================================
# UPDATE MODEL REGISTRY
# ============================================================

def update_registry(engine):

    print()
    print("Updating model registry...")

    with engine.begin() as connection:

        # Deactivate current active versions.
        connection.execute(
            text("""
                UPDATE mlops.model_registry
                SET status = 'archived'
                WHERE model_name = :model_name
                  AND status = 'active';
            """),
            {
                "model_name": MODEL_NAME,
            },
        )

        # Promote candidate.
        connection.execute(
            text("""
                UPDATE mlops.model_registry
                SET status = 'active'
                WHERE model_name = :model_name
                  AND model_version = :model_version;
            """),
            {
                "model_name": MODEL_NAME,
                "model_version": CANDIDATE_VERSION,
            },
        )

    print(
        f"{CANDIDATE_VERSION} is now ACTIVE "
        "in the model registry."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MODEL PROMOTION")
    print("=" * 70)

    print(
        f"Model          : {MODEL_NAME}"
    )

    print(
        f"Current active : {ACTIVE_VERSION}"
    )

    print(
        f"Candidate      : {CANDIDATE_VERSION}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # 1. Database
    # --------------------------------------------------------

    engine = get_engine()

    # --------------------------------------------------------
    # 2. Verify promotion decision
    # --------------------------------------------------------

    evaluation = get_promotion_decision(
        engine
    )

    if evaluation.decision != "promoted":

        print()
        print("=" * 70)
        print("PROMOTION BLOCKED")
        print("=" * 70)

        print()
        print(
            f"Candidate {CANDIDATE_VERSION} "
            f"is marked '{evaluation.decision}'."
        )

        print(
            "Active model remains unchanged."
        )

        print("=" * 70)

        return

    # --------------------------------------------------------
    # 3. Check candidate artifacts
    # --------------------------------------------------------

    check_candidate_files()

    # --------------------------------------------------------
    # 4. Validate candidate calibration
    # --------------------------------------------------------

    validate_calibration()

    # --------------------------------------------------------
    # 5. Backup active artifacts
    # --------------------------------------------------------

    backup_active_artifacts()

    # --------------------------------------------------------
    # 6. Promote files
    # --------------------------------------------------------

    promote_artifacts()

    # --------------------------------------------------------
    # 7. Update registry
    # --------------------------------------------------------

    update_registry(
        engine
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MODEL PROMOTION COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Active model version: {CANDIDATE_VERSION}"
    )

    print(
        "Model and calibration were promoted together."
    )

    print(
        "Previous active artifacts were backed up."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()