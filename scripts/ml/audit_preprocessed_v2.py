"""Fase 10.6: auditoría de matrices y metadatos V2."""

import json

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from npl_portfolio.core.paths import ML_DATA_DIR, ML_ARTIFACTS_DIR


DATA_DIR = ML_DATA_DIR / "v2"
ARTIFACT_DIR = ML_ARTIFACTS_DIR / "v2"
SPLITS = ("train", "validation", "test")


def main():
    print("=" * 70)
    print("FASE 10.6 - AUDITORIA DE MATRICES V2")
    print("=" * 70)

    manifest_path = DATA_DIR / "preprocessing_manifest.json"
    split_manifest_path = DATA_DIR / "split_manifest.json"
    contract_path = ARTIFACT_DIR / "preprocessing_contract.joblib"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_manifest = json.loads(
        split_manifest_path.read_text(encoding="utf-8")
    )
    contract = joblib.load(contract_path)

    expected_features = (
        len(contract.numeric_medians)
        + sum(len(v) for v in contract.categorical_values.values())
    )

    assert expected_features == manifest["features"]

    all_ids = {}

    for split in SPLITS:
        expected = manifest["partitions"][split]
        expected_source = split_manifest["partitions"][split]

        matrix_files = sorted(DATA_DIR.glob(f"{split}_X_*.npz"))
        metadata_files = sorted(DATA_DIR.glob(f"{split}_meta_*.parquet"))

        assert len(matrix_files) == expected["batches"]
        assert len(metadata_files) == expected["batches"]

        total_rows = 0
        positive_count = 0
        ids = set()

        for matrix_path, metadata_path in zip(
            matrix_files, metadata_files, strict=True
        ):
            matrix = sparse.load_npz(matrix_path)
            metadata = pd.read_parquet(metadata_path)

            assert sparse.isspmatrix_csr(matrix)
            assert matrix.shape[0] == len(metadata)
            assert matrix.shape[1] == expected_features
            assert np.isfinite(matrix.data).all()
            assert list(metadata.columns) == ["SK_ID_CURR", "TARGET"]
            assert not metadata.isna().any().any()
            assert metadata["TARGET"].isin([0, 1]).all()

            batch_ids = metadata["SK_ID_CURR"].tolist()
            assert len(batch_ids) == len(set(batch_ids))
            assert ids.isdisjoint(batch_ids)

            ids.update(batch_ids)
            total_rows += len(metadata)
            positive_count += int(metadata["TARGET"].sum())

        assert total_rows == expected["rows"]
        assert total_rows == expected_source["rows"]
        assert positive_count == expected_source["positives"]

        all_ids[split] = ids

        print(
            f"[PASS] {split.upper():10} "
            f"filas={total_rows:,} "
            f"lotes={len(matrix_files)} "
            f"features={expected_features}"
        )

    assert all_ids["train"].isdisjoint(all_ids["validation"])
    assert all_ids["train"].isdisjoint(all_ids["test"])
    assert all_ids["validation"].isdisjoint(all_ids["test"])

    print("[PASS] Sin clientes compartidos entre matrices")
    print("[PASS] Contrato y manifiestos consistentes")
    print("[OK] AUDITORIA DE MATRICES V2 COMPLETADA")


if __name__ == "__main__":
    main()
