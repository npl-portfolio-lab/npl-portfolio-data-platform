from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetSplit:
    train_ids: np.ndarray
    validation_ids: np.ndarray


class StratifiedDatasetSplitter:
    """
    Divide clientes entre TRAIN y VALIDATION preservando
    aproximadamente la distribución de TARGET.

    Solo utiliza SK_ID_CURR y TARGET para realizar el split.

    No realiza:
    - imputación;
    - encoding;
    - escalado;
    - selección de features;
    - entrenamiento.

    De esta manera el split ocurre antes de cualquier
    transformación aprendida y se evita data leakage.
    """

    def __init__(
        self,
        validation_size: float = 0.20,
        random_state: int = 42,
    ) -> None:
        if not 0 < validation_size < 1:
            raise ValueError(
                "validation_size debe estar entre 0 y 1."
            )

        self.validation_size = validation_size
        self.random_state = random_state

    def split(
        self,
        clients: pd.DataFrame,
    ) -> DatasetSplit:
        required_columns = {
            "SK_ID_CURR",
            "TARGET",
        }

        missing_columns = (
            required_columns - set(clients.columns)
        )

        if missing_columns:
            raise ValueError(
                "Faltan columnas requeridas: "
                f"{sorted(missing_columns)}"
            )

        if clients["SK_ID_CURR"].isna().any():
            raise ValueError(
                "SK_ID_CURR contiene valores NULL."
            )

        if not clients["SK_ID_CURR"].is_unique:
            raise ValueError(
                "SK_ID_CURR contiene duplicados."
            )

        if clients["TARGET"].isna().any():
            raise ValueError(
                "TARGET contiene valores NULL."
            )

        target_values = set(
            clients["TARGET"].unique()
        )

        if not target_values.issubset({0, 1}):
            raise ValueError(
                "TARGET debe contener únicamente 0 y 1."
            )

        train_ids, validation_ids = train_test_split(
            clients["SK_ID_CURR"].to_numpy(),
            test_size=self.validation_size,
            random_state=self.random_state,
            stratify=clients["TARGET"].to_numpy(),
        )

        return DatasetSplit(
            train_ids=train_ids,
            validation_ids=validation_ids,
        )