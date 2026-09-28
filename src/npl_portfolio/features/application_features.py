from pathlib import Path

from npl_portfolio.core.duckdb_manager import DuckDBManager
from npl_portfolio.features.base_feature_builder import (
    BaseDuckDBFeatureBuilder,
)


class ApplicationFeatureBuilder(BaseDuckDBFeatureBuilder):
    """
    Construye features determinísticas de application_train
    y application_test.

    Responsabilidad:
    - Preservar una fila por SK_ID_CURR.
    - Excluir TARGET de las features.
    - Tratar DAYS_EMPLOYED=365243 como sentinel.
    - Crear variables determinísticas de aplicación.
    - No realizar imputación aprendida.
    - No realizar encoding aprendido.
    - No utilizar información de TARGET.

    Para producción puede escribir directamente a Parquet
    mediante build_to_parquet(), evitando materializar
    el dataset completo en Pandas.
    """

    DAYS_EMPLOYED_SENTINEL = 365243

    REQUIRED_COLUMNS = [
        "SK_ID_CURR",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
    ]

    DERIVED_FEATURES = [
        "DAYS_EMPLOYED_ANOMALY",
        "AGE_YEARS",
        "EMPLOYMENT_YEARS",
        "CREDIT_INCOME_RATIO",
        "ANNUITY_INCOME_RATIO",
        "CREDIT_ANNUITY_RATIO",
        "GOODS_CREDIT_RATIO",
    ]

    def __init__(
        self,
        parquet_path: Path,
    ) -> None:
        self.parquet_path = parquet_path

        if not self.parquet_path.exists():
            raise FileNotFoundError(f"No existe el archivo: {self.parquet_path}")

        self._validate_schema()

    def _get_schema_columns(self) -> list[str]:
        connection = DuckDBManager().connect()

        try:
            rows = connection.execute(
                """
                DESCRIBE
                SELECT *
                FROM read_parquet(?)
                """,
                [str(self.parquet_path.resolve())],
            ).fetchall()

            return [row[0] for row in rows]

        finally:
            connection.close()

    def _validate_schema(self) -> None:
        columns = self._get_schema_columns()

        missing_columns = [
            column for column in self.REQUIRED_COLUMNS if column not in columns
        ]

        if missing_columns:
            raise ValueError("Faltan columnas requeridas: " f"{missing_columns}")

        self.has_target = "TARGET" in columns

    def _get_parameters(self) -> list[str]:
        return [
            str(self.parquet_path.resolve()),
        ]

    def _get_query(self) -> str:
        if self.has_target:
            base_columns = "* EXCLUDE (TARGET, DAYS_EMPLOYED)"
        else:
            base_columns = "* EXCLUDE (DAYS_EMPLOYED)"

        return f"""
            SELECT
                {base_columns},

                CASE
                    WHEN DAYS_EMPLOYED = {self.DAYS_EMPLOYED_SENTINEL}
                        THEN NULL
                    ELSE DAYS_EMPLOYED
                END AS DAYS_EMPLOYED,

                CASE
                    WHEN DAYS_EMPLOYED = {self.DAYS_EMPLOYED_SENTINEL}
                        THEN 1
                    ELSE 0
                END AS DAYS_EMPLOYED_ANOMALY,

                -DAYS_BIRTH / 365.25
                    AS AGE_YEARS,

                CASE
                    WHEN DAYS_EMPLOYED = {self.DAYS_EMPLOYED_SENTINEL}
                        THEN NULL
                    ELSE -DAYS_EMPLOYED / 365.25
                END AS EMPLOYMENT_YEARS,

                CASE
                    WHEN AMT_INCOME_TOTAL IS NULL
                      OR AMT_INCOME_TOTAL = 0
                        THEN NULL
                    ELSE AMT_CREDIT / AMT_INCOME_TOTAL
                END AS CREDIT_INCOME_RATIO,

                CASE
                    WHEN AMT_INCOME_TOTAL IS NULL
                      OR AMT_INCOME_TOTAL = 0
                        THEN NULL
                    ELSE AMT_ANNUITY / AMT_INCOME_TOTAL
                END AS ANNUITY_INCOME_RATIO,

                CASE
                    WHEN AMT_ANNUITY IS NULL
                      OR AMT_ANNUITY = 0
                        THEN NULL
                    ELSE AMT_CREDIT / AMT_ANNUITY
                END AS CREDIT_ANNUITY_RATIO,

                CASE
                    WHEN AMT_CREDIT IS NULL
                      OR AMT_CREDIT = 0
                        THEN NULL
                    ELSE AMT_GOODS_PRICE / AMT_CREDIT
                END AS GOODS_CREDIT_RATIO

            FROM read_parquet(?)
        """
