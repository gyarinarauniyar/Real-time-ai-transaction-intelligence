import os
import shutil
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"
BACKUP_DIR = MODEL_DIR / "backups"

ACTIVE_MODEL = MODEL_DIR / "isolation_forest.joblib"
ACTIVE_CALIBRATION = MODEL_DIR / "risk_calibration.joblib"

MODEL_NAME = "isolation_forest"

TARGET_ROLLBACK_VERSION = "v1.0.0"


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
# FIND BACKUPS
# ============================================================

def find_latest_backup(version):

    model_backups = sorted(
        BACKUP_DIR.glob(
            f"isolation_forest_{version}_*.joblib"
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    calibration_backups = sorted(
        BACKUP_DIR.glob(
            f"risk_calibration_{version}_*.joblib"
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not model_backups:
        raise FileNotFoundError(
            f"No model backup found for {version}"
        )

    if not calibration_backups:
        raise FileNotFoundError(
            f"No calibration backup found for {version}"
        )

    return model_backups[0], calibration_backups[0]


# ============================================================
# VERIFY REGISTRY
# ============================================================

def get_active_version():

    query = text(
        """
        SELECT model_version
        FROM mlops.model_registry
        WHERE model_name = :model_name
          AND status = 'active'
        ORDER BY created_at DESC
        LIMIT 1
        """
    )

    with engine.connect() as conn:

        result = conn.execute(
            query,
            {
                "model_name": MODEL_NAME
            },
        ).fetchone()

    if result is None:
        raise RuntimeError(
            "No active model found in registry."
        )

    return result[0]


# ============================================================
# ROLLBACK
# ============================================================

def main():

    print("=" * 70)
    print("MODEL ROLLBACK")
    print("=" * 70)

    active_version = get_active_version()

    print(f"Model             : {MODEL_NAME}")
    print(f"Current active    : {active_version}")
    print(f"Rollback target   : {TARGET_ROLLBACK_VERSION}")

    print("=" * 70)

    if active_version == TARGET_ROLLBACK_VERSION:

        print("\nROLLBACK NOT REQUIRED")
        print(
            f"{TARGET_ROLLBACK_VERSION} "
            "is already active."
        )
        return

    # --------------------------------------------------------
    # Find backup
    # --------------------------------------------------------

    print("\nFinding backup artifacts...")

    model_backup, calibration_backup = find_latest_backup(
        TARGET_ROLLBACK_VERSION
    )

    print(f"Model backup       : {model_backup}")
    print(f"Calibration backup : {calibration_backup}")

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    if not model_backup.exists():
        raise FileNotFoundError(
            "Model backup does not exist."
        )

    if not calibration_backup.exists():
        raise FileNotFoundError(
            "Calibration backup does not exist."
        )

    print("\nBackup validation: PASSED")

    # --------------------------------------------------------
    # Backup current active artifacts
    # --------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    current_model_backup = (
        BACKUP_DIR
        / f"isolation_forest_{active_version}_rollback_{timestamp}.joblib"
    )

    current_calibration_backup = (
        BACKUP_DIR
        / f"risk_calibration_{active_version}_rollback_{timestamp}.joblib"
    )

    shutil.copy2(
        ACTIVE_MODEL,
        current_model_backup,
    )

    shutil.copy2(
        ACTIVE_CALIBRATION,
        current_calibration_backup,
    )

    print("\nCurrent active artifacts backed up:")
    print(
        f"Model       : {current_model_backup}"
    )
    print(
        f"Calibration : {current_calibration_backup}"
    )

    # --------------------------------------------------------
    # Restore target
    # --------------------------------------------------------

    shutil.copy2(
        model_backup,
        ACTIVE_MODEL,
    )

    shutil.copy2(
        calibration_backup,
        ACTIVE_CALIBRATION,
    )

    print("\nRollback artifacts restored.")

    # --------------------------------------------------------
    # Update registry
    # --------------------------------------------------------

    with engine.begin() as conn:

        conn.execute(
            text(
                """
                UPDATE mlops.model_registry
                SET status = 'archived'
                WHERE model_name = :model_name
                  AND status = 'active'
                """
            ),
            {
                "model_name": MODEL_NAME
            },
        )

        conn.execute(
            text(
                """
                UPDATE mlops.model_registry
                SET status = 'active'
                WHERE model_name = :model_name
                  AND model_version = :version
                """
            ),
            {
                "model_name": MODEL_NAME,
                "version": TARGET_ROLLBACK_VERSION,
            },
        )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    new_active_version = get_active_version()

    print("\n" + "=" * 70)
    print("ROLLBACK COMPLETE")
    print("=" * 70)

    print(
        f"Active model version: {new_active_version}"
    )

    print(
        "Model and calibration restored together."
    )

    print(
        "Previous active artifacts were backed up."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()