from abc import ABC, abstractmethod
from typing import Any

from npl_portfolio.data_quality.models import QualityCheckResult


class DataQualityRule(ABC):
    """Contrato base para todas las reglas de calidad."""

    @abstractmethod
    def validate(
        self,
        table_name: str,
        table_profile: dict[str, Any],
    ) -> QualityCheckResult:
        raise NotImplementedError


class RequiredColumnRule(DataQualityRule):
    """Valida que una columna obligatoria exista."""

    def __init__(
        self,
        column: str,
    ) -> None:
        self.column = column

    def validate(
        self,
        table_name: str,
        table_profile: dict[str, Any],
    ) -> QualityCheckResult:

        columns = {column["name"] for column in table_profile["columns"]}

        passed = self.column in columns

        return QualityCheckResult(
            table=table_name,
            rule="required_column",
            column=self.column,
            passed=passed,
            message=(
                f"La columna {self.column} existe."
                if passed
                else f"Falta la columna obligatoria {self.column}."
            ),
        )


class NullPercentageRule(DataQualityRule):
    """Valida el porcentaje máximo permitido de nulos."""

    def __init__(
        self,
        column: str,
        max_percentage: float = 0.0,
    ) -> None:
        self.column = column
        self.max_percentage = max_percentage

    def validate(
        self,
        table_name: str,
        table_profile: dict[str, Any],
    ) -> QualityCheckResult:

        column = self._find_column(
            table_profile,
            self.column,
        )

        if column is None:
            return QualityCheckResult(
                table=table_name,
                rule="null_percentage",
                column=self.column,
                passed=False,
                message=(
                    f"No se puede validar nulos porque " f"{self.column} no existe."
                ),
            )

        percentage = float(column["null_percentage"])

        passed = percentage <= self.max_percentage

        return QualityCheckResult(
            table=table_name,
            rule="null_percentage",
            column=self.column,
            passed=passed,
            message=(
                f"Nulos: {percentage:.2f}% "
                f"(máximo permitido: "
                f"{self.max_percentage:.2f}%)."
            ),
        )

    @staticmethod
    def _find_column(
        table_profile: dict[str, Any],
        column_name: str,
    ) -> dict[str, Any] | None:

        for column in table_profile["columns"]:
            if column["name"] == column_name:
                return column

        return None


class UniqueColumnRule(DataQualityRule):
    """Valida que una columna actúe como identificador único."""

    def __init__(
        self,
        column: str,
    ) -> None:
        self.column = column

    def validate(
        self,
        table_name: str,
        table_profile: dict[str, Any],
    ) -> QualityCheckResult:

        column = self._find_column(
            table_profile,
            self.column,
        )

        if column is None:
            return QualityCheckResult(
                table=table_name,
                rule="unique_column",
                column=self.column,
                passed=False,
                message=(f"No existe la columna {self.column}."),
            )

        row_count = int(table_profile["row_count"])

        unique_count = int(column["unique_count"])

        null_count = int(column["null_count"])

        passed = unique_count == row_count and null_count == 0

        return QualityCheckResult(
            table=table_name,
            rule="unique_column",
            column=self.column,
            passed=passed,
            message=(
                f"Filas={row_count:,}, "
                f"únicos={unique_count:,}, "
                f"nulos={null_count:,}."
            ),
        )

    @staticmethod
    def _find_column(
        table_profile: dict[str, Any],
        column_name: str,
    ) -> dict[str, Any] | None:

        for column in table_profile["columns"]:
            if column["name"] == column_name:
                return column

        return None


class DuplicateRowsRule(DataQualityRule):
    """Valida que no se hayan detectado filas duplicadas."""

    def validate(
        self,
        table_name: str,
        table_profile: dict[str, Any],
    ) -> QualityCheckResult:

        duplicates = int(table_profile["duplicate_rows"])

        passed = duplicates == 0

        return QualityCheckResult(
            table=table_name,
            rule="duplicate_rows",
            column=None,
            passed=passed,
            message=(f"Duplicados detectados: " f"{duplicates:,}."),
        )
