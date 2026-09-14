from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class ValueQualityRule(ABC):
    """
    Contrato base para las reglas de calidad de valores.
    """

    def __init__(
        self,
        column: str,
        allow_null: bool = True,
    ) -> None:
        self.column = column
        self.allow_null = allow_null

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def invalid_mask(
        self,
        series: pd.Series,
    ) -> pd.Series:
        """
        Retorna True para cada valor inválido.
        """
        raise NotImplementedError


class AllowedValuesRule(ValueQualityRule):
    """
    Valida que los valores pertenezcan
    a un dominio permitido.
    """

    def __init__(
        self,
        column: str,
        allowed_values: set[Any],
        allow_null: bool = True,
    ) -> None:
        super().__init__(
            column=column,
            allow_null=allow_null,
        )

        self.allowed_values = allowed_values

    def invalid_mask(
        self,
        series: pd.Series,
    ) -> pd.Series:

        invalid = ~series.isin(self.allowed_values)

        if self.allow_null:
            invalid &= series.notna()

        return invalid


class PositiveValueRule(ValueQualityRule):
    """
    Valida valores estrictamente mayores que cero.
    """

    def invalid_mask(
        self,
        series: pd.Series,
    ) -> pd.Series:

        numeric = pd.to_numeric(
            series,
            errors="coerce",
        )

        invalid = numeric <= 0

        conversion_error = series.notna() & numeric.isna()

        invalid |= conversion_error

        if not self.allow_null:
            invalid |= series.isna()

        return invalid


class NonNegativeValueRule(ValueQualityRule):
    """
    Valida valores mayores o iguales a cero.
    """

    def invalid_mask(
        self,
        series: pd.Series,
    ) -> pd.Series:

        numeric = pd.to_numeric(
            series,
            errors="coerce",
        )

        invalid = numeric < 0

        conversion_error = series.notna() & numeric.isna()

        invalid |= conversion_error

        if not self.allow_null:
            invalid |= series.isna()

        return invalid
