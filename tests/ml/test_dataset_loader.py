from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from npl_portfolio.ml.dataset_loader import (
    SparseDatasetLoader,
)


def test_sparse_dataset_loader(
    tmp_path: Path,
) -> None:
    X_1 = sparse.csr_matrix(
        [
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
        ],
        dtype=np.float32,
    )

    X_2 = sparse.csr_matrix(
        [
            [4.0, 0.0, 5.0],
        ],
        dtype=np.float32,
    )

    sparse.save_npz(
        tmp_path / "train_X_001.npz",
        X_1,
    )

    sparse.save_npz(
        tmp_path / "train_X_002.npz",
        X_2,
    )

    pd.DataFrame(
        {
            "SK_ID_CURR": [
                100001,
                100002,
            ],
            "TARGET": [
                0,
                1,
            ],
        }
    ).to_parquet(
        tmp_path / "train_meta_001.parquet",
        index=False,
    )

    pd.DataFrame(
        {
            "SK_ID_CURR": [
                100003,
            ],
            "TARGET": [
                0,
            ],
        }
    ).to_parquet(
        tmp_path / "train_meta_002.parquet",
        index=False,
    )

    loader = SparseDatasetLoader(
        ml_dir=tmp_path
    )

    dataset = loader.load(
        "train"
    )

    assert sparse.isspmatrix_csr(
        dataset.X
    )

    assert dataset.X.shape == (
        3,
        3,
    )

    assert dataset.y.tolist() == [
        0,
        1,
        0,
    ]

    assert dataset.ids.tolist() == [
        100001,
        100002,
        100003,
    ]
