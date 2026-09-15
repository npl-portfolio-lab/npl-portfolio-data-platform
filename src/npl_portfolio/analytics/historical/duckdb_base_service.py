from pathlib import Path
from typing import Any

import duckdb


class DuckDBBaseService:
    """
    Servicio base para análisis de tablas históricas
    almacenadas en formato Parquet mediante DuckDB.

    Contiene operaciones genéricas reutilizables por
    POS, Credit Card, Installments y Bureau Balance.
    """

    def __init__(self, parquet_path: Path) -> None:
        self.parquet_path = parquet_path

        if not self.parquet_path.exists():
            raise FileNotFoundError(f"No existe el archivo: {self.parquet_path}")

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(database=":memory:")

    def get_dataset_overview(self) -> dict[str, Any]:
        connection = self._get_connection()

        try:
            total_rows = connection.execute(
                """
                SELECT COUNT(*)
                FROM read_parquet(?)
                """,
                [str(self.parquet_path)],
            ).fetchone()[0]

            schema = connection.execute(
                """
                DESCRIBE
                SELECT *
                FROM read_parquet(?)
                """,
                [str(self.parquet_path)],
            ).fetchdf()

        finally:
            connection.close()

        return {
            "file": self.parquet_path.name,
            "rows": int(total_rows),
            "columns": int(len(schema)),
            "column_names": schema["column_name"].tolist(),
        }

    def get_unique_clients(
        self,
        client_column: str = "SK_ID_CURR",
    ) -> dict[str, Any]:
        connection = self._get_connection()

        try:
            result = connection.execute(
                f"""
                SELECT
                    COUNT(*) AS total_rows,
                    COUNT(DISTINCT {client_column})
                        AS unique_clients
                FROM read_parquet(?)
                """,
                [str(self.parquet_path)],
            ).fetchone()

        finally:
            connection.close()

        total_rows = int(result[0])
        unique_clients = int(result[1])

        average = total_rows / unique_clients if unique_clients else 0.0

        return {
            "total_rows": total_rows,
            "unique_clients": unique_clients,
            "average_records_per_client": round(
                average,
                2,
            ),
        }

    def get_records_per_client_summary(
        self,
        client_column: str = "SK_ID_CURR",
    ) -> dict[str, Any]:
        connection = self._get_connection()

        try:
            result = connection.execute(
                f"""
                WITH records_per_client AS (
                    SELECT
                        {client_column},
                        COUNT(*) AS record_count
                    FROM read_parquet(?)
                    GROUP BY {client_column}
                )

                SELECT
                    COUNT(*) AS clients,
                    AVG(record_count) AS mean_records,
                    MEDIAN(record_count) AS median_records,
                    MIN(record_count) AS min_records,
                    QUANTILE_CONT(
                        record_count,
                        0.25
                    ) AS p25_records,
                    QUANTILE_CONT(
                        record_count,
                        0.75
                    ) AS p75_records,
                    QUANTILE_CONT(
                        record_count,
                        0.90
                    ) AS p90_records,
                    QUANTILE_CONT(
                        record_count,
                        0.95
                    ) AS p95_records,
                    QUANTILE_CONT(
                        record_count,
                        0.99
                    ) AS p99_records,
                    MAX(record_count) AS max_records
                FROM records_per_client
                """,
                [str(self.parquet_path)],
            ).fetchone()

        finally:
            connection.close()

        return {
            "clients": int(result[0]),
            "mean_records": float(result[1]),
            "median_records": float(result[2]),
            "min_records": int(result[3]),
            "p25_records": float(result[4]),
            "p75_records": float(result[5]),
            "p90_records": float(result[6]),
            "p95_records": float(result[7]),
            "p99_records": float(result[8]),
            "max_records": int(result[9]),
        }

    def get_categorical_summary(
        self,
        columns: list[str],
        top_n: int = 10,
    ) -> dict[str, Any]:
        connection = self._get_connection()

        statistics = []

        try:
            total_rows = connection.execute(
                """
                SELECT COUNT(*)
                FROM read_parquet(?)
                """,
                [str(self.parquet_path)],
            ).fetchone()[0]

            for column in columns:
                unique_values = connection.execute(
                    f"""
                    SELECT COUNT(DISTINCT {column})
                    FROM read_parquet(?)
                    """,
                    [str(self.parquet_path)],
                ).fetchone()[0]

                null_count = connection.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM read_parquet(?)
                    WHERE {column} IS NULL
                    """,
                    [str(self.parquet_path)],
                ).fetchone()[0]

                categories = connection.execute(
                    f"""
                    SELECT
                        COALESCE(
                            CAST({column} AS VARCHAR),
                            '<MISSING>'
                        ) AS value,
                        COUNT(*) AS category_count
                    FROM read_parquet(?)
                    GROUP BY 1
                    ORDER BY category_count DESC
                    LIMIT ?
                    """,
                    [
                        str(self.parquet_path),
                        top_n,
                    ],
                ).fetchall()

                top_categories = []

                for value, count in categories:
                    top_categories.append(
                        {
                            "value": value,
                            "count": int(count),
                            "percentage": round(
                                count / total_rows * 100,
                                2,
                            ),
                        }
                    )

                statistics.append(
                    {
                        "column": column,
                        "unique_values": int(unique_values),
                        "null_count": int(null_count),
                        "null_percentage": round(
                            null_count / total_rows * 100,
                            2,
                        ),
                        "top_categories": (top_categories),
                    }
                )

        finally:
            connection.close()

        return {
            "statistics": statistics,
        }

    def get_numeric_summary(
        self,
        columns: list[str],
    ) -> dict[str, Any]:
        connection = self._get_connection()

        statistics = []

        try:
            total_rows = connection.execute(
                """
                SELECT COUNT(*)
                FROM read_parquet(?)
                """,
                [str(self.parquet_path)],
            ).fetchone()[0]

            for column in columns:
                result = connection.execute(
                    f"""
                    SELECT
                        COUNT({column}),
                        COUNT(*)
                            - COUNT({column}),
                        AVG({column}),
                        MEDIAN({column}),
                        QUANTILE_CONT(
                            {column},
                            0.25
                        ),
                        QUANTILE_CONT(
                            {column},
                            0.75
                        ),
                        MIN({column}),
                        MAX({column})
                    FROM read_parquet(?)
                    """,
                    [str(self.parquet_path)],
                ).fetchone()

                count = int(result[0])
                null_count = int(result[1])

                statistics.append(
                    {
                        "column": column,
                        "count": count,
                        "null_count": null_count,
                        "null_percentage": round(
                            null_count / total_rows * 100,
                            2,
                        ),
                        "mean": (float(result[2]) if result[2] is not None else None),
                        "median": (float(result[3]) if result[3] is not None else None),
                        "p25": (float(result[4]) if result[4] is not None else None),
                        "p75": (float(result[5]) if result[5] is not None else None),
                        "min": (float(result[6]) if result[6] is not None else None),
                        "max": (float(result[7]) if result[7] is not None else None),
                    }
                )

        finally:
            connection.close()

        return {
            "statistics": statistics,
        }
