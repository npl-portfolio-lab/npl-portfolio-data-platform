from pathlib import Path
from npl_portfolio.core.paths import PROJECT_ROOT, ML_DATA_DIR

from npl_portfolio.core.duckdb_manager import DuckDBManager


ROOT = PROJECT_ROOT

SOURCE = (
    ROOT
    / "data"
    / "processed"
    / "features"
    / "client_features_train.parquet"
)

ML_DIR = ML_DATA_DIR

TRAIN = ML_DIR / "ml_train.parquet"
VALIDATION = ML_DIR / "ml_validation.parquet"


def main() -> None:
    for path in (SOURCE, TRAIN, VALIDATION):
        if not path.exists():
            raise FileNotFoundError(
                f"No existe el archivo requerido: {path}"
            )

    connection = DuckDBManager().connect()

    try:
        source = str(SOURCE.resolve()).replace("'", "''")
        train = str(TRAIN.resolve()).replace("'", "''")
        validation = str(VALIDATION.resolve()).replace("'", "''")

        source_count = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{source}')
            """
        ).fetchone()[0]

        train_count = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{train}')
            """
        ).fetchone()[0]

        validation_count = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{validation}')
            """
        ).fetchone()[0]

        overlap = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{train}') AS t
            INNER JOIN read_parquet('{validation}') AS v
                USING (SK_ID_CURR)
            """
        ).fetchone()[0]

        train_unique = connection.execute(
            f"""
            SELECT COUNT(DISTINCT SK_ID_CURR)
            FROM read_parquet('{train}')
            """
        ).fetchone()[0]

        validation_unique = connection.execute(
            f"""
            SELECT COUNT(DISTINCT SK_ID_CURR)
            FROM read_parquet('{validation}')
            """
        ).fetchone()[0]

        train_target_rate = connection.execute(
            f"""
            SELECT AVG(TARGET)
            FROM read_parquet('{train}')
            """
        ).fetchone()[0]

        validation_target_rate = connection.execute(
            f"""
            SELECT AVG(TARGET)
            FROM read_parquet('{validation}')
            """
        ).fetchone()[0]

        source_schema = connection.execute(
            f"""
            DESCRIBE
            SELECT *
            FROM read_parquet('{source}')
            """
        ).fetchall()

        train_schema = connection.execute(
            f"""
            DESCRIBE
            SELECT *
            FROM read_parquet('{train}')
            """
        ).fetchall()

        validation_schema = connection.execute(
            f"""
            DESCRIBE
            SELECT *
            FROM read_parquet('{validation}')
            """
        ).fetchall()

        print("=" * 80)
        print("FASE 9.2 - VALIDACION SPLIT ML")
        print("=" * 80)

        assert train_count + validation_count == source_count
        print(
            f"[PASS] Cobertura: "
            f"{train_count:,} + {validation_count:,} "
            f"= {source_count:,}"
        )

        assert train_unique == train_count
        print("[PASS] TRAIN: SK_ID_CURR único.")

        assert validation_unique == validation_count
        print("[PASS] VALIDATION: SK_ID_CURR único.")

        assert overlap == 0
        print("[PASS] TRAIN y VALIDATION no comparten clientes.")

        assert source_schema == train_schema == validation_schema
        print("[PASS] Schemas idénticos.")

        assert abs(
            train_target_rate - validation_target_rate
        ) < 0.001

        print(
            "[PASS] Estratificación TARGET: "
            f"TRAIN={train_target_rate * 100:.4f}% | "
            f"VALIDATION={validation_target_rate * 100:.4f}%"
        )

        print()
        print("=" * 80)
        print("RESULTADO FINAL: PASS")
        print("FASE 9.2 VALIDADA CORRECTAMENTE")
        print("=" * 80)

    finally:
        connection.close()


if __name__ == "__main__":
    main()
