from pathlib import Path
from npl_portfolio.core.paths import PROJECT_ROOT, ML_DATA_DIR, ML_ARTIFACTS_DIR

import joblib
import numpy as np
import pandas as pd
from scipy import sparse


ROOT = PROJECT_ROOT

ML_DIR = ML_DATA_DIR
ARTIFACTS_DIR = ML_ARTIFACTS_DIR

CONTRACT_PATH = (
    ARTIFACTS_DIR
    / "preprocessing_contract.joblib"
)

EXPECTED_TRAIN_ROWS = 246_008
EXPECTED_VALIDATION_ROWS = 61_503


def validate_dataset(
    prefix: str,
    expected_rows: int,
) -> tuple[int, int, set[int]]:
    matrix_files = sorted(
        ML_DIR.glob(f"{prefix}_X_*.npz")
    )

    metadata_files = sorted(
        ML_DIR.glob(f"{prefix}_meta_*.parquet")
    )

    if not matrix_files:
        raise AssertionError(
            f"No existen matrices para {prefix}."
        )

    if len(matrix_files) != len(metadata_files):
        raise AssertionError(
            f"{prefix}: cantidad de matrices "
            "y metadatos diferente."
        )

    total_rows = 0
    expected_features = None
    all_ids: set[int] = set()

    for matrix_path, metadata_path in zip(
        matrix_files,
        metadata_files,
        strict=True,
    ):
        matrix = sparse.load_npz(
            matrix_path
        )

        metadata = pd.read_parquet(
            metadata_path
        )

        if not sparse.isspmatrix_csr(matrix):
            raise AssertionError(
                f"{matrix_path.name} no es CSR."
            )

        if matrix.shape[0] != len(metadata):
            raise AssertionError(
                f"{matrix_path.name}: filas de X "
                "no coinciden con metadata."
            )

        if expected_features is None:
            expected_features = matrix.shape[1]

        elif matrix.shape[1] != expected_features:
            raise AssertionError(
                f"{matrix_path.name}: número "
                "inconsistente de features."
            )

        if np.isnan(matrix.data).any():
            raise AssertionError(
                f"{matrix_path.name} contiene NaN."
            )

        if np.isinf(matrix.data).any():
            raise AssertionError(
                f"{matrix_path.name} contiene infinito."
            )

        required_metadata = {
            "SK_ID_CURR",
            "TARGET",
        }

        if set(metadata.columns) != required_metadata:
            raise AssertionError(
                f"{metadata_path.name}: columnas "
                "de metadata inválidas."
            )

        if metadata["SK_ID_CURR"].isna().any():
            raise AssertionError(
                f"{metadata_path.name}: "
                "SK_ID_CURR contiene NULL."
            )

        if metadata["TARGET"].isna().any():
            raise AssertionError(
                f"{metadata_path.name}: "
                "TARGET contiene NULL."
            )

        if not set(
            metadata["TARGET"].unique()
        ).issubset({0, 1}):
            raise AssertionError(
                f"{metadata_path.name}: "
                "TARGET inválido."
            )

        batch_ids = set(
            metadata["SK_ID_CURR"].astype(int)
        )

        if len(batch_ids) != len(metadata):
            raise AssertionError(
                f"{metadata_path.name}: "
                "IDs duplicados dentro del batch."
            )

        if not all_ids.isdisjoint(batch_ids):
            raise AssertionError(
                f"{prefix}: existen IDs repetidos "
                "entre batches."
            )

        all_ids.update(batch_ids)

        total_rows += matrix.shape[0]

        print(
            f"[PASS] {matrix_path.name} | "
            f"filas={matrix.shape[0]:,} | "
            f"features={matrix.shape[1]} | "
            f"nnz={matrix.nnz:,}"
        )

    if total_rows != expected_rows:
        raise AssertionError(
            f"{prefix}: se esperaban "
            f"{expected_rows:,} filas y se "
            f"obtuvieron {total_rows:,}."
        )

    if len(all_ids) != expected_rows:
        raise AssertionError(
            f"{prefix}: cantidad de IDs "
            "no coincide con filas esperadas."
        )

    print(
        f"[PASS] {prefix.upper()}: "
        f"{total_rows:,} filas completas."
    )

    return (
        total_rows,
        expected_features,
        all_ids,
    )


def main() -> None:
    print("=" * 80)
    print("FASE 9.3 - VALIDACION PREPROCESSING ML")
    print("=" * 80)

    if not CONTRACT_PATH.exists():
        raise FileNotFoundError(
            f"No existe el contrato: "
            f"{CONTRACT_PATH}"
        )

    contract = joblib.load(
        CONTRACT_PATH
    )

    print(
        "[PASS] Contrato de preprocessing cargado."
    )

    print(
        f"[PASS] Medianas numéricas: "
        f"{len(contract.numeric_medians)}"
    )

    print(
        f"[PASS] Modas categóricas: "
        f"{len(contract.categorical_modes)}"
    )

    print()
    print("VALIDANDO TRAIN")
    print("-" * 80)

    (
        train_rows,
        train_features,
        train_ids,
    ) = validate_dataset(
        prefix="train",
        expected_rows=EXPECTED_TRAIN_ROWS,
    )

    print()
    print("VALIDANDO VALIDATION")
    print("-" * 80)

    (
        validation_rows,
        validation_features,
        validation_ids,
    ) = validate_dataset(
        prefix="validation",
        expected_rows=EXPECTED_VALIDATION_ROWS,
    )

    if train_features != validation_features:
        raise AssertionError(
            "TRAIN y VALIDATION tienen "
            "distinto número de features."
        )

    print(
        f"[PASS] TRAIN y VALIDATION: "
        f"{train_features} features."
    )

    if not train_ids.isdisjoint(
        validation_ids
    ):
        raise AssertionError(
            "TRAIN y VALIDATION comparten clientes."
        )

    print(
        "[PASS] TRAIN y VALIDATION "
        "no comparten clientes."
    )

    if (
        train_rows + validation_rows
        != 307_511
    ):
        raise AssertionError(
            "Cobertura total incorrecta."
        )

    print(
        "[PASS] Cobertura total: "
        f"{train_rows + validation_rows:,} clientes."
    )

    print()
    print("=" * 80)
    print("RESULTADO FINAL: PASS")
    print(
        "FASE 9.3 - PREPROCESSING ML "
        "VALIDADO CORRECTAMENTE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
