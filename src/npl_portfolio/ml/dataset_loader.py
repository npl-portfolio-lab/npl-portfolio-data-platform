from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(frozen=True)
class MLDataset:
    X: sparse.csr_matrix
    y: np.ndarray
    ids: np.ndarray


class SparseDatasetLoader:
    """
    Carga los batches generados durante preprocessing.

    Mantiene X como matriz CSR y valida la correspondencia
    entre matrices y metadata.
    """

    def __init__(
        self,
        ml_dir: Path,
    ) -> None:
        self.ml_dir = ml_dir

    def load(
        self,
        prefix: str,
    ) -> MLDataset:
        matrix_files = sorted(
            self.ml_dir.glob(
                f"{prefix}_X_*.npz"
            )
        )

        metadata_files = sorted(
            self.ml_dir.glob(
                f"{prefix}_meta_*.parquet"
            )
        )

        if not matrix_files:
            raise FileNotFoundError(
                f"No existen matrices para {prefix}."
            )

        if len(matrix_files) != len(metadata_files):
            raise ValueError(
                f"{prefix}: cantidad de matrices "
                "y metadatos diferente."
            )

        matrices = []
        targets = []
        ids = []

        expected_features = None

        for matrix_path, metadata_path in zip(
            matrix_files,
            metadata_files,
            strict=True,
        ):
            matrix = sparse.load_npz(
                matrix_path
            ).tocsr()

            metadata = pd.read_parquet(
                metadata_path,
                columns=[
                    "SK_ID_CURR",
                    "TARGET",
                ],
            )

            if matrix.shape[0] != len(metadata):
                raise ValueError(
                    f"{matrix_path.name}: X y metadata "
                    "tienen diferente número de filas."
                )

            if expected_features is None:
                expected_features = matrix.shape[1]

            elif matrix.shape[1] != expected_features:
                raise ValueError(
                    f"{matrix_path.name}: número "
                    "inconsistente de features."
                )

            matrices.append(matrix)

            targets.append(
                metadata["TARGET"].to_numpy(
                    dtype=np.int8
                )
            )

            ids.append(
                metadata["SK_ID_CURR"].to_numpy()
            )

        X = sparse.vstack(
            matrices,
            format="csr",
        )

        y = np.concatenate(
            targets
        )

        client_ids = np.concatenate(
            ids
        )

        if X.shape[0] != len(y):
            raise ValueError(
                "X e y tienen diferente número de filas."
            )

        if len(client_ids) != len(y):
            raise ValueError(
                "IDs e y tienen diferente número de filas."
            )

        return MLDataset(
            X=X,
            y=y,
            ids=client_ids,
        )
