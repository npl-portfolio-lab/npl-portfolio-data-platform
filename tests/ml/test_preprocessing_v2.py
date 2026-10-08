import numpy as np
import pandas as pd
import pytest

from npl_portfolio.ml.batch_preprocessor import BatchPreprocessor
from npl_portfolio.ml.preprocessing_contract import PreprocessingContract


@pytest.fixture
def preprocessor():
    contract = PreprocessingContract(
        numeric_medians={"INCOME": 100.0},
        categorical_modes={"SEGMENT": "A"},
        categorical_values={"SEGMENT": ["A", "B"]},
    )
    return BatchPreprocessor(contract)


def test_output_dimensions(preprocessor):
    data = pd.DataFrame({
        "INCOME": [100.0, 200.0],
        "SEGMENT": ["A", "B"],
    })

    matrix = preprocessor.transform(data)

    assert matrix.shape == (2, 3)


def test_numeric_imputation_uses_contract(preprocessor):
    data = pd.DataFrame({
        "INCOME": [np.nan],
        "SEGMENT": ["A"],
    })

    matrix = preprocessor.transform(data).toarray()

    assert matrix[0, 0] == 100.0


def test_categorical_imputation_uses_contract(preprocessor):
    data = pd.DataFrame({
        "INCOME": [50.0],
        "SEGMENT": [None],
    })

    matrix = preprocessor.transform(data).toarray()

    np.testing.assert_array_equal(
        matrix[0],
        np.array([50.0, 1.0, 0.0]),
    )


def test_unknown_category_is_ignored(preprocessor):
    data = pd.DataFrame({
        "INCOME": [50.0],
        "SEGMENT": ["UNKNOWN"],
    })

    matrix = preprocessor.transform(data).toarray()

    np.testing.assert_array_equal(
        matrix[0],
        np.array([50.0, 0.0, 0.0]),
    )


def test_repeated_transform_is_deterministic(preprocessor):
    data = pd.DataFrame({
        "INCOME": [10.0, np.nan],
        "SEGMENT": ["B", "A"],
    })

    first = preprocessor.transform(data).toarray()
    second = preprocessor.transform(data).toarray()

    np.testing.assert_array_equal(first, second)


def test_transform_does_not_change_input(preprocessor):
    data = pd.DataFrame({
        "INCOME": [np.nan],
        "SEGMENT": [None],
    })

    original = data.copy(deep=True)

    preprocessor.transform(data)

    pd.testing.assert_frame_equal(data, original)
