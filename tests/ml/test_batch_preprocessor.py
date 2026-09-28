import numpy as np
import pandas as pd
from scipy import sparse

from npl_portfolio.ml.batch_preprocessor import (
    BatchPreprocessor,
)
from npl_portfolio.ml.preprocessing_contract import (
    PreprocessingContract,
)


def test_batch_preprocessor() -> None:
    contract = PreprocessingContract(
        numeric_medians={
            "NUMERIC_A": 20.0,
        },
        categorical_modes={
            "CATEGORY_A": "A",
        },
        categorical_values={
            "CATEGORY_A": ["A", "B"],
        },
    )

    data = pd.DataFrame(
        {
            "NUMERIC_A": [
                10.0,
                np.nan,
            ],
            "CATEGORY_A": [
                "B",
                None,
            ],
        }
    )

    preprocessor = BatchPreprocessor(
        contract
    )

    result = preprocessor.transform(
        data
    )

    assert sparse.isspmatrix_csr(result)

    assert result.shape == (
        2,
        3,
    )

    dense = result.toarray()

    # Primera columna = NUMERIC_A
    assert dense[0, 0] == 10.0

    # NaN debe utilizar la mediana aprendida de TRAIN.
    assert dense[1, 0] == 20.0

    assert not np.isnan(
        result.data
    ).any()
