from pathlib import Path

import pandas as pd

from npl_portfolio.ml.preprocessing_contract import (
    PreprocessingContractBuilder,
)


def test_preprocessing_contract(
    tmp_path: Path,
) -> None:
    path = tmp_path / "train.parquet"

    data = pd.DataFrame(
        {
            "NUMERIC_A": [
                10.0,
                20.0,
                None,
                40.0,
            ],
            "CATEGORY_A": [
                "A",
                "A",
                None,
                "B",
            ],
        }
    )

    data.to_parquet(
        path,
        index=False,
    )

    builder = PreprocessingContractBuilder(
        parquet_path=str(path),
        numeric_columns=["NUMERIC_A"],
        categorical_columns=["CATEGORY_A"],
    )

    contract = builder.build()

    assert contract.numeric_medians[
        "NUMERIC_A"
    ] == 20.0

    assert contract.categorical_modes[
        "CATEGORY_A"
    ] == "A"

    assert contract.categorical_values[
        "CATEGORY_A"
    ] == [
        "A",
        "B",
    ]
