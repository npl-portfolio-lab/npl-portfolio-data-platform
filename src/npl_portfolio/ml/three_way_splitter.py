"""División estratificada reproducible TRAIN / VALIDATION / TEST."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class ThreeWayDatasetSplit:
    train_ids: np.ndarray
    validation_ids: np.ndarray
    test_ids: np.ndarray


class StratifiedThreeWaySplitter:
    """Divide clientes en tres particiones sin compartir identificadores."""

    def __init__(
        self,
        train_size: float = 0.64,
        validation_size: float = 0.16,
        test_size: float = 0.20,
        random_state: int = 42,
    ) -> None:
        sizes = (train_size, validation_size, test_size)

        if any(not 0 < size < 1 for size in sizes):
            raise ValueError("Las proporciones deben estar entre 0 y 1.")

        if not np.isclose(sum(sizes), 1.0, atol=1e-10):
            raise ValueError("Las proporciones deben sumar 1.")

        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = test_size
        self.random_state = random_state

    def split(self, clients: pd.DataFrame) -> ThreeWayDatasetSplit:
        required = {"SK_ID_CURR", "TARGET"}

        if not required.issubset(clients.columns):
            raise ValueError(
                f"Faltan columnas: {sorted(required - set(clients.columns))}"
            )

        if clients["SK_ID_CURR"].isna().any():
            raise ValueError("SK_ID_CURR contiene valores nulos.")

        if not clients["SK_ID_CURR"].is_unique:
            raise ValueError("SK_ID_CURR contiene duplicados.")

        if clients["TARGET"].isna().any():
            raise ValueError("TARGET contiene valores nulos.")

        if not clients["TARGET"].isin([0, 1]).all():
            raise ValueError("TARGET debe contener únicamente 0 y 1.")

        ids = clients["SK_ID_CURR"].to_numpy()
        target = clients["TARGET"].to_numpy()

        development_ids, test_ids, development_y, _ = train_test_split(
            ids,
            target,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=target,
        )

        relative_validation_size = (
            self.validation_size / (self.train_size + self.validation_size)
        )

        train_ids, validation_ids = train_test_split(
            development_ids,
            test_size=relative_validation_size,
            random_state=self.random_state,
            stratify=development_y,
        )

        return ThreeWayDatasetSplit(
            train_ids=train_ids,
            validation_ids=validation_ids,
            test_ids=test_ids,
        )
