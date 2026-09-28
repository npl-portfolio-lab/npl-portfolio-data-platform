import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder

from npl_portfolio.ml.preprocessing_contract import (
    PreprocessingContract,
)


class BatchPreprocessor:
    """
    Transforma lotes usando únicamente estadísticas
    aprendidas previamente desde TRAIN.

    No aprende información durante transform().
    """

    def __init__(
        self,
        contract: PreprocessingContract,
    ) -> None:
        self.contract = contract

        self.numeric_columns = list(
            contract.numeric_medians.keys()
        )

        self.categorical_columns = list(
            contract.categorical_modes.keys()
        )

        self.encoder = OneHotEncoder(
            categories=[
                contract.categorical_values[column]
                for column in self.categorical_columns
            ],
            handle_unknown="ignore",
            sparse_output=True,
            dtype="float32",
        )

        encoder_seed = pd.DataFrame(
            {
                column: [
                    contract.categorical_modes[column]
                ]
                for column in self.categorical_columns
            }
        )

        self.encoder.fit(encoder_seed)

    def transform(
        self,
        dataframe: pd.DataFrame,
    ):
        numeric = dataframe[
            self.numeric_columns
        ].copy()

        numeric = numeric.fillna(
            self.contract.numeric_medians
        )

        numeric_matrix = sparse.csr_matrix(
            numeric.to_numpy(
                dtype="float32",
            )
        )

        categorical = dataframe[
            self.categorical_columns
        ].copy()

        categorical = categorical.fillna(
            self.contract.categorical_modes
        )

        categorical_matrix = self.encoder.transform(
            categorical
        )

        return sparse.hstack(
            [
                numeric_matrix,
                categorical_matrix,
            ],
            format="csr",
        )
