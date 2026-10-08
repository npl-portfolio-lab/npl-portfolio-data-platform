"""Fase 10.6: preprocesamiento V2 aislado y out-of-core."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from npl_portfolio.core.paths import ML_DATA_DIR, ML_ARTIFACTS_DIR
from npl_portfolio.ml.batch_preprocessor import BatchPreprocessor
from npl_portfolio.ml.preprocessing_contract import (
    PreprocessingContractBuilder,
)

from build_ml_preprocessed_datasets import (
    get_feature_columns,
    get_row_count,
    load_batch,
)


DATA_DIR = ML_DATA_DIR / "v2"
ARTIFACT_DIR = ML_ARTIFACTS_DIR / "v2"

TRAIN_PATH = DATA_DIR / "ml_train.parquet"
SPLITS = ("train", "validation", "test")

CONTRACT_PATH = ARTIFACT_DIR / "preprocessing_contract.joblib"
MANIFEST_PATH = DATA_DIR / "preprocessing_manifest.json"

BATCH_SIZE = 20_000


def validate_contract(contract, numeric_columns, categorical_columns):
    missing_numeric = sorted(
        set(numeric_columns) - set(contract.numeric_medians)
    )
    missing_categorical = sorted(
        set(categorical_columns) - set(contract.categorical_modes)
    )
    empty_categories = [
        name
        for name, values in contract.categorical_values.items()
        if not values
    ]

    if missing_numeric or missing_categorical or empty_categories:
        raise ValueError(
            "Contrato incompleto. "
            f"Numéricas: {missing_numeric}; "
            f"Categóricas: {missing_categorical}; "
            f"Categorías vacías: {empty_categories}"
        )


def validate_no_existing_outputs():
    existing = []

    for split in SPLITS:
        existing.extend(DATA_DIR.glob(f"{split}_X_*.npz"))
        existing.extend(DATA_DIR.glob(f"{split}_meta_*.parquet"))

    for path in (CONTRACT_PATH, MANIFEST_PATH):
        if path.exists():
            existing.append(path)

    if existing:
        raise FileExistsError(
            "Existen artefactos V2. No se sobrescribirán:\n"
            + "\n".join(str(path) for path in existing)
        )


def transform_split(split, preprocessor, expected_features):
    parquet_path = DATA_DIR / f"ml_{split}.parquet"
    total_rows = get_row_count(parquet_path)

    processed = 0
    batches = 0

    for offset in range(0, total_rows, BATCH_SIZE):
        batches += 1

        batch = load_batch(
            parquet_path=parquet_path,
            offset=offset,
            batch_size=BATCH_SIZE,
        )

        ids = batch["SK_ID_CURR"].to_numpy()
        labels = batch["TARGET"].to_numpy()

        features = batch.drop(
            columns=["SK_ID_CURR", "TARGET"]
        )

        matrix = preprocessor.transform(features)

        if matrix.shape != (len(batch), expected_features):
            raise ValueError(
                f"Dimensiones incompatibles en {split}: "
                f"{matrix.shape}"
            )

        if not np.isfinite(matrix.data).all():
            raise ValueError(
                f"Valores no finitos en {split}, lote {batches}"
            )

        matrix_path = (
            DATA_DIR / f"{split}_X_{batches:03d}.npz"
        )
        metadata_path = (
            DATA_DIR / f"{split}_meta_{batches:03d}.parquet"
        )

        sparse.save_npz(matrix_path, matrix)

        pd.DataFrame({
            "SK_ID_CURR": ids,
            "TARGET": labels,
        }).to_parquet(metadata_path, index=False)

        processed += len(batch)

        print(
            f"[OK] {split.upper()} "
            f"lote={batches:03d} "
            f"filas={len(batch):,} "
            f"features={matrix.shape[1]}"
        )

    if processed != total_rows:
        raise ValueError(
            f"Conteo incorrecto en {split}: "
            f"{processed} != {total_rows}"
        )

    return {
        "rows": processed,
        "batches": batches,
        "features": expected_features,
    }


def main():
    print("=" * 70)
    print("FASE 10.6 - PREPROCESAMIENTO V2")
    print("=" * 70)

    for split in SPLITS:
        path = DATA_DIR / f"ml_{split}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)

    validate_no_existing_outputs()

    numeric_columns, categorical_columns = (
        get_feature_columns(TRAIN_PATH)
    )

    print(f"Numéricas: {len(numeric_columns)}")
    print(f"Categóricas: {len(categorical_columns)}")

    print("[CONTRACT] Aprendiendo exclusivamente desde TRAIN")

    builder = PreprocessingContractBuilder(
        parquet_path=str(TRAIN_PATH.resolve()),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

    contract = builder.build()

    validate_contract(
        contract,
        numeric_columns,
        categorical_columns,
    )

    preprocessor = BatchPreprocessor(contract)

    expected_features = (
        len(contract.numeric_medians)
        + sum(
            len(values)
            for values in contract.categorical_values.values()
        )
    )

    print(f"[CONTRACT] Features esperadas: {expected_features}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(contract, CONTRACT_PATH)

    results = {}

    for split in SPLITS:
        print(f"\n[TRANSFORM] {split.upper()}")

        results[split] = transform_split(
            split,
            preprocessor,
            expected_features,
        )

    manifest = {
        "phase": "10.6",
        "version": "v2",
        "contract_source": "ml_train.parquet",
        "contract_path": str(CONTRACT_PATH),
        "batch_size": BATCH_SIZE,
        "numeric_columns": len(numeric_columns),
        "categorical_columns": len(categorical_columns),
        "features": expected_features,
        "partitions": results,
        "test_evaluated": False,
    }

    temporary = MANIFEST_PATH.with_suffix(".json.tmp")

    temporary.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    temporary.replace(MANIFEST_PATH)

    print(f"\n[OK] Contrato: {CONTRACT_PATH}")
    print(f"[OK] Manifiesto: {MANIFEST_PATH}")
    print("[OK] PREPROCESAMIENTO V2 FINALIZADO")


if __name__ == "__main__":
    main()
