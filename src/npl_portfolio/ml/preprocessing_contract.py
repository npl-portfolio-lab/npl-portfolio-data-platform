from dataclasses import dataclass

from npl_portfolio.core.duckdb_manager import DuckDBManager


@dataclass(frozen=True)
class PreprocessingContract:
    numeric_medians: dict[str, float]
    categorical_modes: dict[str, str]
    categorical_values: dict[str, list[str]]


class PreprocessingContractBuilder:
    """
    Aprende estadísticas de preprocessing exclusivamente desde TRAIN
    utilizando DuckDB.

    No materializa el dataset completo en Pandas.
    """

    def __init__(
        self,
        parquet_path: str,
        numeric_columns: list[str],
        categorical_columns: list[str],
    ) -> None:
        self.parquet_path = parquet_path
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns

    @staticmethod
    def _quote_identifier(column: str) -> str:
        return '"' + column.replace('"', '""') + '"'

    def build(self) -> PreprocessingContract:
        connection = DuckDBManager().connect()

        try:
            numeric_medians = self._get_numeric_medians(connection)

            categorical_modes = self._get_categorical_modes(connection)

            categorical_values = self._get_categorical_values(connection)

            return PreprocessingContract(
                numeric_medians=numeric_medians,
                categorical_modes=categorical_modes,
                categorical_values=categorical_values,
            )

        finally:
            connection.close()

    def _get_numeric_medians(
        self,
        connection,
    ) -> dict[str, float]:
        """
        Calcula todas las medianas numéricas en una sola
        consulta DuckDB sobre TRAIN.
        """
        if not self.numeric_columns:
            return {}

        expressions = [
            (
                f"MEDIAN({self._quote_identifier(column)}) "
                f"AS {self._quote_identifier(column)}"
            )
            for column in self.numeric_columns
        ]

        query = f"""
            SELECT
                {", ".join(expressions)}
            FROM read_parquet(?)
        """

        row = connection.execute(
            query,
            [self.parquet_path],
        ).fetchone()

        return {
            column: float(value)
            for column, value in zip(
                self.numeric_columns,
                row,
                strict=True,
            )
            if value is not None
        }

    def _get_categorical_modes(
        self,
        connection,
    ) -> dict[str, str]:
        result = {}

        for column in self.categorical_columns:
            quoted = self._quote_identifier(column)

            row = connection.execute(
                f"""
                SELECT {quoted}
                FROM read_parquet(?)
                WHERE {quoted} IS NOT NULL
                GROUP BY {quoted}
                ORDER BY COUNT(*) DESC, {quoted}
                LIMIT 1
                """,
                [self.parquet_path],
            ).fetchone()

            if row is not None:
                result[column] = str(row[0])

        return result

    def _get_categorical_values(
        self,
        connection,
    ) -> dict[str, list[str]]:
        result = {}

        for column in self.categorical_columns:
            quoted = self._quote_identifier(column)

            rows = connection.execute(
                f"""
                SELECT DISTINCT {quoted}
                FROM read_parquet(?)
                WHERE {quoted} IS NOT NULL
                ORDER BY {quoted}
                """,
                [self.parquet_path],
            ).fetchall()

            result[column] = [str(row[0]) for row in rows]

        return result
