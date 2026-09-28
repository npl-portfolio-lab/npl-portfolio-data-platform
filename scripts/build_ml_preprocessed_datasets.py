from pathlib import Path

import joblib
import pandas as pd
from scipy import sparse

from npl_portfolio.core.duckdb_manager import DuckDBManager
from npl_portfolio.ml.batch_preprocessor import BatchPreprocessor
from npl_portfolio.ml.preprocessing_contract import (
    PreprocessingContractBuilder,
)


ROOT = Path(__file__).resolve().parents[1]

ML_DIR = ROOT / "data" / "processed" / "ml"
ARTIFACTS_DIR = ROOT / "data" / "artifacts" / "ml"

TRAIN_PATH = ML_DIR / "ml_train.parquet"
VALIDATION_PATH = ML_DIR / "ml_validation.parquet"

CONTRACT_PATH = ARTIFACTS_DIR / "preprocessing_contract.joblib"

BATCH_SIZE = 20_000

EXCLUDED_COLUMNS = {
    "SK_ID_CURR",
    "TARGET",
}


def get_schema(
    parquet_path: Path,
) -> list[tuple]:
    connection = DuckDBManager().connect()

    try:
        return connection.execute(
            """
            DESCRIBE
            SELECT *
            FROM read_parquet(?)
            """,
            [str(parquet_path.resolve())],
        ).fetchall()

    finally:
        connection.close()


def get_feature_columns(
    parquet_path: Path,
) -> tuple[list[str], list[str]]:
    """
    Obtiene el contrato de columnas directamente desde
    el schema Parquet sin cargar el dataset en Pandas.
    """
    schema = get_schema(parquet_path)

    numeric_types = {
        "BIGINT",
        "INTEGER",
        "DOUBLE",
        "FLOAT",
        "REAL",
        "SMALLINT",
        "TINYINT",
        "HUGEINT",
        "DECIMAL",
    }

    numeric_columns = []
    categorical_columns = []

    for column, dtype, *_ in schema:
        if column in EXCLUDED_COLUMNS:
            continue

        base_type = dtype.split("(")[0]

        if base_type in numeric_types:
            numeric_columns.append(column)

        elif base_type == "VARCHAR":
            categorical_columns.append(column)

        else:
            raise ValueError(
                f"Tipo no soportado: {column} -> {dtype}"
            )

    return (
        numeric_columns,
        categorical_columns,
    )


def get_row_count(
    parquet_path: Path,
) -> int:
    connection = DuckDBManager().connect()

    try:
        return connection.execute(
            """
            SELECT COUNT(*)
            FROM read_parquet(?)
            """,
            [str(parquet_path.resolve())],
        ).fetchone()[0]

    finally:
        connection.close()


def load_batch(
    parquet_path: Path,
    offset: int,
    batch_size: int,
) -> pd.DataFrame:
    """
    Materializa únicamente un lote en Pandas.
    """
    connection = DuckDBManager().connect()

    try:
        return connection.execute(
            """
            SELECT *
            FROM read_parquet(?)
            ORDER BY SK_ID_CURR
            LIMIT ?
            OFFSET ?
            """,
            [
                str(parquet_path.resolve()),
                batch_size,
                offset,
            ],
        ).fetchdf()

    finally:
        connection.close()


def transform_in_batches(
    parquet_path: Path,
    output_prefix: str,
    preprocessor: BatchPreprocessor,
) -> None:
    """
    Transforma el dataset por lotes y persiste cada
    resultado como matriz CSR.
    """
    total_rows = get_row_count(
        parquet_path
    )

    batch_number = 0

    for offset in range(
        0,
        total_rows,
        BATCH_SIZE,
    ):
        batch_number += 1

        batch = load_batch(
            parquet_path=parquet_path,
            offset=offset,
            batch_size=BATCH_SIZE,
        )

        ids = batch[
            "SK_ID_CURR"
        ].to_numpy()

        target = batch[
            "TARGET"
        ].to_numpy()

        features = batch.drop(
            columns=[
                "SK_ID_CURR",
                "TARGET",
            ]
        )

        transformed = preprocessor.transform(
            features
        )

        matrix_path = (
            ML_DIR
            / f"{output_prefix}_X_{batch_number:03d}.npz"
        )

        metadata_path = (
            ML_DIR
            / f"{output_prefix}_meta_{batch_number:03d}.parquet"
        )

        sparse.save_npz(
            matrix_path,
            transformed,
        )

        pd.DataFrame(
            {
                "SK_ID_CURR": ids,
                "TARGET": target,
            }
        ).to_parquet(
            metadata_path,
            index=False,
        )

        print(
            f"[OK] {output_prefix.upper()} "
            f"batch {batch_number:03d} | "
            f"filas={len(batch):,} | "
            f"features={transformed.shape[1]}"
        )

        del batch
        del features
        del transformed


def main() -> None:
    for path in (
        TRAIN_PATH,
        VALIDATION_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"No existe: {path}"
            )

    ML_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("FASE 9.3 - PREPROCESSING ML OUT-OF-CORE")
    print("=" * 80)

    numeric_columns, categorical_columns = (
        get_feature_columns(TRAIN_PATH)
    )

    print(
        f"Features numéricas:   {len(numeric_columns)}"
    )
    print(
        f"Features categóricas: {len(categorical_columns)}"
    )

    print()
    print(
        "[CONTRACT] Aprendiendo estadísticas "
        "exclusivamente desde TRAIN..."
    )

    contract_builder = PreprocessingContractBuilder(
        parquet_path=str(
            TRAIN_PATH.resolve()
        ),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

    contract = contract_builder.build()

    joblib.dump(
        contract,
        CONTRACT_PATH,
    )

    print(
        f"[OK] Contrato guardado: {CONTRACT_PATH}"
    )

    preprocessor = BatchPreprocessor(
        contract
    )

    print()
    print(
        f"[BATCH] Tamaño configurado: "
        f"{BATCH_SIZE:,} filas"
    )

    print()
    print("[TRANSFORM] TRAIN")

    transform_in_batches(
        parquet_path=TRAIN_PATH,
        output_prefix="train",
        preprocessor=preprocessor,
    )

    print()
    print("[TRANSFORM] VALIDATION")

    transform_in_batches(
        parquet_path=VALIDATION_PATH,
        output_prefix="validation",
        preprocessor=preprocessor,
    )

    print()
    print("=" * 80)
    print("FASE 9.3 PREPROCESSING FINALIZADO")
    print("=" * 80)


if __name__ == "__main__":
    main()