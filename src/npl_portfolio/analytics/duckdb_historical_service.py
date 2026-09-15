from pathlib import Path
from typing import Any

import duckdb


class DuckDBHistoricalService:
    """
    Servicio analítico para tablas históricas grandes
    almacenadas en formato Parquet.

    DuckDB ejecuta las consultas directamente sobre
    los archivos Parquet sin cargar todo el dataset
    en memoria mediante pandas.
    """

    def __init__(
        self,
        parquet_path: Path,
    ) -> None:
        self.parquet_path = parquet_path

        if not self.parquet_path.exists():
            raise FileNotFoundError(f"No existe el archivo: {self.parquet_path}")

    def _get_connection(
        self,
    ) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(database=":memory:")

    def get_dataset_overview(
        self,
    ) -> dict[str, Any]:
        connection = self._get_connection()

        try:
            query = """
                SELECT
                    COUNT(*) AS total_rows
                FROM read_parquet(?)
            """

            result = connection.execute(
                query,
                [str(self.parquet_path)],
            ).fetchone()

            columns = connection.execute(
                """
                DESCRIBE
                SELECT *
                FROM read_parquet(?)
                """,
                [str(self.parquet_path)],
            ).fetchall()

            return {
                "file": self.parquet_path.name,
                "rows": int(result[0]),
                "columns": len(columns),
            }

        finally:
            connection.close()

    def get_unique_clients(
        self,
        client_column: str = "SK_ID_CURR",
    ) -> dict[str, Any]:
        connection = self._get_connection()

        try:
            query = f"""
                SELECT
                    COUNT(*) AS total_rows,
                    COUNT(
                        DISTINCT {client_column}
                    ) AS unique_clients
                FROM read_parquet(?)
            """

            result = connection.execute(
                query,
                [str(self.parquet_path)],
            ).fetchone()

            total_rows = int(result[0])
            unique_clients = int(result[1])

            average_records = total_rows / unique_clients if unique_clients else 0.0

            return {
                "total_rows": total_rows,
                "unique_clients": unique_clients,
                "average_records_per_client": round(
                    average_records,
                    4,
                ),
            }

        finally:
            connection.close()

    def get_records_per_client_summary(
        self,
        client_column: str = "SK_ID_CURR",
    ) -> dict[str, Any]:
        connection = self._get_connection()

        try:
            query = f"""
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
            """

            result = connection.execute(
                query,
                [str(self.parquet_path)],
            ).fetchone()

            return {
                "clients": int(result[0]),
                "mean_records": round(
                    float(result[1]),
                    4,
                ),
                "median_records": round(
                    float(result[2]),
                    4,
                ),
                "min_records": int(result[3]),
                "p25_records": round(
                    float(result[4]),
                    4,
                ),
                "p75_records": round(
                    float(result[5]),
                    4,
                ),
                "p90_records": round(
                    float(result[6]),
                    4,
                ),
                "p95_records": round(
                    float(result[7]),
                    4,
                ),
                "p99_records": round(
                    float(result[8]),
                    4,
                ),
                "max_records": int(result[9]),
            }

        finally:
            connection.close()

    def get_categorical_summary(
        self,
        columns: list[str],
        top_n: int = 10,
    ) -> dict[str, Any]:
        if not columns:
            raise ValueError("Debe proporcionar al menos una columna.")

        connection = self._get_connection()

        try:
            results = []

            total_rows = connection.execute(
                """
                SELECT COUNT(*)
                FROM read_parquet(?)
                """,
                [str(self.parquet_path)],
            ).fetchone()[0]

            for column in columns:
                metadata_query = f"""
                    SELECT
                        COUNT(
                            DISTINCT {column}
                        ) AS unique_values,
                        SUM(
                            CASE
                                WHEN {column} IS NULL
                                THEN 1
                                ELSE 0
                            END
                        ) AS null_count
                    FROM read_parquet(?)
                """

                metadata = connection.execute(
                    metadata_query,
                    [str(self.parquet_path)],
                ).fetchone()

                category_query = f"""
                    SELECT
                        CAST(
                            {column} AS VARCHAR
                        ) AS value,
                        COUNT(*) AS count
                    FROM read_parquet(?)
                    WHERE {column} IS NOT NULL
                    GROUP BY {column}
                    ORDER BY count DESC
                    LIMIT ?
                """

                categories_raw = connection.execute(
                    category_query,
                    [
                        str(self.parquet_path),
                        top_n,
                    ],
                ).fetchall()

                categories = []

                for value, count in categories_raw:
                    percentage = count / total_rows * 100 if total_rows else 0.0

                    categories.append(
                        {
                            "value": str(value),
                            "count": int(count),
                            "percentage": round(
                                percentage,
                                4,
                            ),
                        }
                    )

                null_count = int(metadata[1] or 0)

                results.append(
                    {
                        "column": column,
                        "unique_values": int(metadata[0]),
                        "null_count": null_count,
                        "null_percentage": round(
                            (null_count / total_rows * 100) if total_rows else 0.0,
                            4,
                        ),
                        "top_categories": categories,
                    }
                )

            return {
                "columns_analyzed": len(results),
                "statistics": results,
            }

        finally:
            connection.close()

    def get_numeric_summary(
        self,
        columns: list[str],
    ) -> dict[str, Any]:
        if not columns:
            raise ValueError("Debe proporcionar al menos una columna.")

        connection = self._get_connection()

        try:
            results = []

            total_rows = connection.execute(
                """
                SELECT COUNT(*)
                FROM read_parquet(?)
                """,
                [str(self.parquet_path)],
            ).fetchone()[0]

            for column in columns:
                query = f"""
                    SELECT
                        COUNT({column}) AS valid_count,
                        AVG({column}) AS mean_value,
                        MEDIAN({column}) AS median_value,
                        STDDEV_SAMP(
                            {column}
                        ) AS std_value,
                        MIN({column}) AS min_value,
                        QUANTILE_CONT(
                            {column},
                            0.25
                        ) AS p25_value,
                        QUANTILE_CONT(
                            {column},
                            0.75
                        ) AS p75_value,
                        MAX({column}) AS max_value
                    FROM read_parquet(?)
                """

                result = connection.execute(
                    query,
                    [str(self.parquet_path)],
                ).fetchone()

                valid_count = int(result[0])
                null_count = total_rows - valid_count

                def safe_float(
                    value: Any,
                ) -> float | None:
                    if value is None:
                        return None

                    return round(
                        float(value),
                        4,
                    )

                results.append(
                    {
                        "column": column,
                        "count": valid_count,
                        "null_count": null_count,
                        "null_percentage": round(
                            (null_count / total_rows * 100) if total_rows else 0.0,
                            4,
                        ),
                        "mean": safe_float(result[1]),
                        "median": safe_float(result[2]),
                        "std": safe_float(result[3]),
                        "min": safe_float(result[4]),
                        "p25": safe_float(result[5]),
                        "p75": safe_float(result[6]),
                        "max": safe_float(result[7]),
                    }
                )

            return {
                "columns_analyzed": len(results),
                "statistics": results,
            }

        finally:
            connection.close()

    def get_pos_cash_target_analysis(
        self,
        application_path: Path,
        target_column: str = "TARGET",
    ) -> dict[str, Any]:
        """
        Agrega POS_CASH_balance por cliente mediante DuckDB
        y compara las métricas resultantes entre TARGET=0
        y TARGET=1.

        TARGET se incorpora únicamente después de agregar
        el historial para evitar utilizarlo en la construcción
        de las features.
        """

        if not application_path.exists():
            raise FileNotFoundError(f"No existe el archivo: {application_path}")

        connection = self._get_connection()

        try:
            query = f"""
                WITH pos_aggregated AS (
                    SELECT
                        SK_ID_CURR,

                        COUNT(*) AS POS_RECORD_COUNT,

                        COUNT(
                            DISTINCT SK_ID_PREV
                        ) AS POS_CONTRACT_COUNT,

                        SUM(
                            CASE
                                WHEN NAME_CONTRACT_STATUS = 'Active'
                                THEN 1
                                ELSE 0
                            END
                        ) AS POS_ACTIVE_RECORD_COUNT,

                        SUM(
                            CASE
                                WHEN NAME_CONTRACT_STATUS = 'Completed'
                                THEN 1
                                ELSE 0
                            END
                        ) AS POS_COMPLETED_RECORD_COUNT,

                        SUM(
                            CASE
                                WHEN SK_DPD > 0
                                THEN 1
                                ELSE 0
                            END
                        ) AS POS_DPD_RECORD_COUNT,

                        SUM(
                            CASE
                                WHEN SK_DPD_DEF > 0
                                THEN 1
                                ELSE 0
                            END
                        ) AS POS_DPD_DEF_RECORD_COUNT,

                        MAX(SK_DPD)
                            AS POS_MAX_DPD,

                        AVG(SK_DPD)
                            AS POS_AVG_DPD,

                        MAX(SK_DPD_DEF)
                            AS POS_MAX_DPD_DEF,

                        AVG(SK_DPD_DEF)
                            AS POS_AVG_DPD_DEF,

                        AVG(CNT_INSTALMENT)
                            AS POS_AVG_INSTALMENT,

                        AVG(CNT_INSTALMENT_FUTURE)
                            AS POS_AVG_INSTALMENT_FUTURE,

                        MIN(MONTHS_BALANCE)
                            AS POS_OLDEST_MONTH,

                        MAX(MONTHS_BALANCE)
                            AS POS_RECENT_MONTH

                    FROM read_parquet(?)
                    GROUP BY SK_ID_CURR
                ),

                application AS (
                    SELECT
                        SK_ID_CURR,
                        {target_column}
                    FROM read_parquet(?)
                ),

                merged AS (
                    SELECT
                        application.SK_ID_CURR,
                        application.{target_column},

                        pos_aggregated.* EXCLUDE (
                            SK_ID_CURR
                        ),

                        CASE
                            WHEN pos_aggregated.SK_ID_CURR
                                IS NOT NULL
                            THEN 1
                            ELSE 0
                        END AS HAS_POS_HISTORY

                    FROM application

                    LEFT JOIN pos_aggregated
                        ON application.SK_ID_CURR
                        = pos_aggregated.SK_ID_CURR
                ),

                features AS (
                    SELECT
                        *,

                        CASE
                            WHEN POS_RECORD_COUNT > 0
                            THEN
                                POS_DPD_RECORD_COUNT
                                * 1.0
                                / POS_RECORD_COUNT
                        END AS POS_DPD_RATE,

                        CASE
                            WHEN POS_RECORD_COUNT > 0
                            THEN
                                POS_DPD_DEF_RECORD_COUNT
                                * 1.0
                                / POS_RECORD_COUNT
                        END AS POS_DPD_DEF_RATE

                    FROM merged
                )

                SELECT *
                FROM features
            """

            dataframe = connection.execute(
                query,
                [
                    str(self.parquet_path),
                    str(application_path),
                ],
            ).fetchdf()

        finally:
            connection.close()

        feature_columns = [
            "POS_RECORD_COUNT",
            "POS_CONTRACT_COUNT",
            "POS_ACTIVE_RECORD_COUNT",
            "POS_COMPLETED_RECORD_COUNT",
            "POS_DPD_RECORD_COUNT",
            "POS_DPD_DEF_RECORD_COUNT",
            "POS_MAX_DPD",
            "POS_AVG_DPD",
            "POS_MAX_DPD_DEF",
            "POS_AVG_DPD_DEF",
            "POS_AVG_INSTALMENT",
            "POS_AVG_INSTALMENT_FUTURE",
            "POS_OLDEST_MONTH",
            "POS_RECENT_MONTH",
            "POS_DPD_RATE",
            "POS_DPD_DEF_RATE",
        ]

        clients_with_history = int(dataframe["HAS_POS_HISTORY"].sum())

        clients_without_history = int(len(dataframe) - clients_with_history)

        target_results = []

        for target_value in sorted(dataframe[target_column].dropna().unique()):
            target_data = dataframe.loc[dataframe[target_column] == target_value]

            target_clients = len(target_data)

            target_with_history = int(target_data["HAS_POS_HISTORY"].sum())

            feature_statistics = []

            for column in feature_columns:
                series = target_data[column].dropna()

                if series.empty:
                    continue

                feature_statistics.append(
                    {
                        "feature": column,
                        "count": int(series.count()),
                        "mean": round(
                            float(series.mean()),
                            4,
                        ),
                        "median": round(
                            float(series.median()),
                            4,
                        ),
                        "p25": round(
                            float(series.quantile(0.25)),
                            4,
                        ),
                        "p75": round(
                            float(series.quantile(0.75)),
                            4,
                        ),
                        "min": round(
                            float(series.min()),
                            4,
                        ),
                        "max": round(
                            float(series.max()),
                            4,
                        ),
                    }
                )

            target_results.append(
                {
                    "target": int(target_value),
                    "clients": int(target_clients),
                    "clients_with_history": (target_with_history),
                    "history_coverage_percentage": round(
                        (
                            (target_with_history / target_clients * 100)
                            if target_clients
                            else 0.0
                        ),
                        4,
                    ),
                    "features": feature_statistics,
                }
            )

        return {
            "application_clients": int(len(dataframe)),
            "clients_with_history": (clients_with_history),
            "clients_without_history": (clients_without_history),
            "history_coverage_percentage": round(
                (
                    (clients_with_history / len(dataframe) * 100)
                    if len(dataframe)
                    else 0.0
                ),
                4,
            ),
            "aggregated_features": (feature_columns),
            "target_statistics": (target_results),
        }

    def get_credit_card_target_analysis(
        self,
        application_path: Path,
        target_column: str = "TARGET",
    ) -> dict[str, Any]:
        """
        Agrega credit_card_balance por cliente mediante DuckDB
        y compara las características históricas entre TARGET=0
        y TARGET=1.

        La agregación se realiza antes de incorporar TARGET.
        """

        if not application_path.exists():
            raise FileNotFoundError(f"No existe el archivo: {application_path}")

        connection = self._get_connection()

        try:
            query = f"""
                WITH credit_card_aggregated AS (
                    SELECT
                        SK_ID_CURR,

                        COUNT(*) AS CC_RECORD_COUNT,

                        COUNT(
                            DISTINCT SK_ID_PREV
                        ) AS CC_CONTRACT_COUNT,

                        SUM(
                            CASE
                                WHEN NAME_CONTRACT_STATUS = 'Active'
                                THEN 1
                                ELSE 0
                            END
                        ) AS CC_ACTIVE_RECORD_COUNT,

                        SUM(
                            CASE
                                WHEN NAME_CONTRACT_STATUS = 'Completed'
                                THEN 1
                                ELSE 0
                            END
                        ) AS CC_COMPLETED_RECORD_COUNT,

                        AVG(AMT_BALANCE)
                            AS CC_AVG_BALANCE,

                        MAX(AMT_BALANCE)
                            AS CC_MAX_BALANCE,

                        AVG(AMT_CREDIT_LIMIT_ACTUAL)
                            AS CC_AVG_CREDIT_LIMIT,

                        MAX(AMT_CREDIT_LIMIT_ACTUAL)
                            AS CC_MAX_CREDIT_LIMIT,

                        AVG(AMT_DRAWINGS_CURRENT)
                            AS CC_AVG_DRAWINGS,

                        SUM(AMT_DRAWINGS_CURRENT)
                            AS CC_TOTAL_DRAWINGS,

                        AVG(AMT_PAYMENT_CURRENT)
                            AS CC_AVG_PAYMENT_CURRENT,

                        AVG(AMT_PAYMENT_TOTAL_CURRENT)
                            AS CC_AVG_PAYMENT_TOTAL,

                        SUM(AMT_PAYMENT_TOTAL_CURRENT)
                            AS CC_TOTAL_PAYMENT,

                        AVG(AMT_TOTAL_RECEIVABLE)
                            AS CC_AVG_RECEIVABLE,

                        AVG(CNT_DRAWINGS_CURRENT)
                            AS CC_AVG_DRAWING_COUNT,

                        AVG(CNT_INSTALMENT_MATURE_CUM)
                            AS CC_AVG_MATURE_INSTALMENTS,

                        SUM(
                            CASE
                                WHEN SK_DPD > 0
                                THEN 1
                                ELSE 0
                            END
                        ) AS CC_DPD_RECORD_COUNT,

                        SUM(
                            CASE
                                WHEN SK_DPD_DEF > 0
                                THEN 1
                                ELSE 0
                            END
                        ) AS CC_DPD_DEF_RECORD_COUNT,

                        MAX(SK_DPD)
                            AS CC_MAX_DPD,

                        AVG(SK_DPD)
                            AS CC_AVG_DPD,

                        MAX(SK_DPD_DEF)
                            AS CC_MAX_DPD_DEF,

                        AVG(SK_DPD_DEF)
                            AS CC_AVG_DPD_DEF,

                        MIN(MONTHS_BALANCE)
                            AS CC_OLDEST_MONTH,

                        MAX(MONTHS_BALANCE)
                            AS CC_RECENT_MONTH,

                        AVG(
                            CASE
                                WHEN AMT_CREDIT_LIMIT_ACTUAL > 0
                                THEN
                                    AMT_BALANCE
                                    / AMT_CREDIT_LIMIT_ACTUAL
                                ELSE NULL
                            END
                        ) AS CC_AVG_UTILIZATION,

                        MAX(
                            CASE
                                WHEN AMT_CREDIT_LIMIT_ACTUAL > 0
                                THEN
                                    AMT_BALANCE
                                    / AMT_CREDIT_LIMIT_ACTUAL
                                ELSE NULL
                            END
                        ) AS CC_MAX_UTILIZATION

                    FROM read_parquet(?)

                    GROUP BY SK_ID_CURR
                ),

                application AS (
                    SELECT
                        SK_ID_CURR,
                        {target_column}
                    FROM read_parquet(?)
                ),

                merged AS (
                    SELECT
                        application.SK_ID_CURR,
                        application.{target_column},

                        credit_card_aggregated.* EXCLUDE (
                            SK_ID_CURR
                        ),

                        CASE
                            WHEN credit_card_aggregated.SK_ID_CURR
                                IS NOT NULL
                            THEN 1
                            ELSE 0
                        END AS HAS_CREDIT_CARD_HISTORY

                    FROM application

                    LEFT JOIN credit_card_aggregated
                        ON application.SK_ID_CURR
                        = credit_card_aggregated.SK_ID_CURR
                ),

                features AS (
                    SELECT
                        *,

                        CASE
                            WHEN CC_RECORD_COUNT > 0
                            THEN
                                CC_DPD_RECORD_COUNT
                                * 1.0
                                / CC_RECORD_COUNT
                        END AS CC_DPD_RATE,

                        CASE
                            WHEN CC_RECORD_COUNT > 0
                            THEN
                                CC_DPD_DEF_RECORD_COUNT
                                * 1.0
                                / CC_RECORD_COUNT
                        END AS CC_DPD_DEF_RATE

                    FROM merged
                )

                SELECT *
                FROM features
            """

            dataframe = connection.execute(
                query,
                [
                    str(self.parquet_path),
                    str(application_path),
                ],
            ).fetchdf()

        finally:
            connection.close()

        feature_columns = [
            "CC_RECORD_COUNT",
            "CC_CONTRACT_COUNT",
            "CC_ACTIVE_RECORD_COUNT",
            "CC_COMPLETED_RECORD_COUNT",
            "CC_AVG_BALANCE",
            "CC_MAX_BALANCE",
            "CC_AVG_CREDIT_LIMIT",
            "CC_MAX_CREDIT_LIMIT",
            "CC_AVG_DRAWINGS",
            "CC_TOTAL_DRAWINGS",
            "CC_AVG_PAYMENT_CURRENT",
            "CC_AVG_PAYMENT_TOTAL",
            "CC_TOTAL_PAYMENT",
            "CC_AVG_RECEIVABLE",
            "CC_AVG_DRAWING_COUNT",
            "CC_AVG_MATURE_INSTALMENTS",
            "CC_DPD_RECORD_COUNT",
            "CC_DPD_DEF_RECORD_COUNT",
            "CC_MAX_DPD",
            "CC_AVG_DPD",
            "CC_MAX_DPD_DEF",
            "CC_AVG_DPD_DEF",
            "CC_OLDEST_MONTH",
            "CC_RECENT_MONTH",
            "CC_AVG_UTILIZATION",
            "CC_MAX_UTILIZATION",
            "CC_DPD_RATE",
            "CC_DPD_DEF_RATE",
        ]

        clients_with_history = int(dataframe["HAS_CREDIT_CARD_HISTORY"].sum())

        clients_without_history = int(len(dataframe) - clients_with_history)

        target_results = []

        for target_value in sorted(dataframe[target_column].dropna().unique()):
            target_data = dataframe.loc[dataframe[target_column] == target_value]

            target_clients = len(target_data)

            target_with_history = int(target_data["HAS_CREDIT_CARD_HISTORY"].sum())

            feature_statistics = []

            for column in feature_columns:
                series = target_data[column].dropna()

                if series.empty:
                    continue

                feature_statistics.append(
                    {
                        "feature": column,
                        "count": int(series.count()),
                        "mean": round(
                            float(series.mean()),
                            4,
                        ),
                        "median": round(
                            float(series.median()),
                            4,
                        ),
                        "p25": round(
                            float(series.quantile(0.25)),
                            4,
                        ),
                        "p75": round(
                            float(series.quantile(0.75)),
                            4,
                        ),
                        "min": round(
                            float(series.min()),
                            4,
                        ),
                        "max": round(
                            float(series.max()),
                            4,
                        ),
                    }
                )

            target_results.append(
                {
                    "target": int(target_value),
                    "clients": int(target_clients),
                    "clients_with_history": (target_with_history),
                    "history_coverage_percentage": round(
                        (
                            (target_with_history / target_clients * 100)
                            if target_clients
                            else 0.0
                        ),
                        4,
                    ),
                    "features": (feature_statistics),
                }
            )

        return {
            "application_clients": int(len(dataframe)),
            "clients_with_history": (clients_with_history),
            "clients_without_history": (clients_without_history),
            "history_coverage_percentage": round(
                (
                    (clients_with_history / len(dataframe) * 100)
                    if len(dataframe)
                    else 0.0
                ),
                4,
            ),
            "aggregated_features": (feature_columns),
            "target_statistics": (target_results),
        }
