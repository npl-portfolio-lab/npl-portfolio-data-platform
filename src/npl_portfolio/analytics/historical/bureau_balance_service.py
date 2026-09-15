from pathlib import Path
from typing import Any

from npl_portfolio.analytics.historical.duckdb_base_service import (
    DuckDBBaseService,
)


class BureauBalanceService(DuckDBBaseService):
    """
    Servicio especializado para bureau_balance.

    bureau_balance no contiene SK_ID_CURR.

    La relación necesaria es:

        bureau_balance.SK_ID_BUREAU
                    |
                    v
        bureau.SK_ID_BUREAU
                    |
                    v
        bureau.SK_ID_CURR
                    |
                    v
        application_train.SK_ID_CURR

    Los registros de bureau_balance sin correspondencia en bureau
    se contabilizan explícitamente y no se consideran registros
    mapeables a un cliente.
    """

    def get_mapping_diagnostics(
        self,
        bureau_path: Path,
    ) -> dict[str, Any]:
        """
        Cuantifica la cobertura de SK_ID_BUREAU entre
        bureau_balance y bureau.
        """
        if not bureau_path.exists():
            raise FileNotFoundError(f"No existe el archivo: {bureau_path}")

        connection = self._get_connection()

        try:
            query = """
                SELECT
                    COUNT(*) AS total_rows,

                    SUM(
                        CASE
                            WHEN bureau.SK_ID_BUREAU IS NOT NULL
                            THEN 1
                            ELSE 0
                        END
                    ) AS mapped_rows,

                    SUM(
                        CASE
                            WHEN bureau.SK_ID_BUREAU IS NULL
                            THEN 1
                            ELSE 0
                        END
                    ) AS unmapped_rows,

                    COUNT(
                        DISTINCT CASE
                            WHEN bureau.SK_ID_BUREAU IS NULL
                            THEN bureau_balance.SK_ID_BUREAU
                        END
                    ) AS unmapped_bureau_ids

                FROM read_parquet(?) AS bureau_balance

                LEFT JOIN read_parquet(?) AS bureau
                    ON bureau_balance.SK_ID_BUREAU
                    = bureau.SK_ID_BUREAU
            """

            result = connection.execute(
                query,
                [
                    str(self.parquet_path),
                    str(bureau_path),
                ],
            ).fetchone()

        finally:
            connection.close()

        total_rows = int(result[0])
        mapped_rows = int(result[1] or 0)
        unmapped_rows = int(result[2] or 0)
        unmapped_bureau_ids = int(result[3] or 0)

        return {
            "total_rows": total_rows,
            "mapped_rows": mapped_rows,
            "unmapped_rows": unmapped_rows,
            "unmapped_bureau_ids": (unmapped_bureau_ids),
            "mapped_percentage": round(
                (mapped_rows / total_rows * 100) if total_rows else 0.0,
                4,
            ),
            "unmapped_percentage": round(
                (unmapped_rows / total_rows * 100) if total_rows else 0.0,
                4,
            ),
        }

    def get_bureau_balance_target_analysis(
        self,
        bureau_path: Path,
        application_path: Path,
        target_column: str = "TARGET",
    ) -> dict[str, Any]:
        """
        Mapea bureau_balance hacia SK_ID_CURR mediante bureau,
        agrega el historial mensual por cliente y posteriormente
        incorpora TARGET.

        Los STATUS 1-5 se tratan como estados numéricos de
        morosidad para construir métricas descriptivas.

        STATUS C y X se conservan como categorías independientes.
        """
        if not bureau_path.exists():
            raise FileNotFoundError(f"No existe el archivo: {bureau_path}")

        if not application_path.exists():
            raise FileNotFoundError(f"No existe el archivo: {application_path}")

        mapping_diagnostics = self.get_mapping_diagnostics(
            bureau_path=bureau_path,
        )

        connection = self._get_connection()

        try:
            query = f"""
                WITH mapped_history AS (
                    SELECT
                        bureau.SK_ID_CURR,
                        bureau_balance.SK_ID_BUREAU,
                        bureau_balance.MONTHS_BALANCE,
                        bureau_balance.STATUS

                    FROM read_parquet(?) AS bureau_balance

                    INNER JOIN read_parquet(?) AS bureau
                        ON bureau_balance.SK_ID_BUREAU
                        = bureau.SK_ID_BUREAU
                ),

                bureau_balance_aggregated AS (
                    SELECT
                        SK_ID_CURR,

                        COUNT(*)
                            AS BB_RECORD_COUNT,

                        COUNT(
                            DISTINCT SK_ID_BUREAU
                        ) AS BB_BUREAU_CREDIT_COUNT,

                        MIN(MONTHS_BALANCE)
                            AS BB_OLDEST_MONTH,

                        MAX(MONTHS_BALANCE)
                            AS BB_RECENT_MONTH,

                        SUM(
                            CASE
                                WHEN STATUS = '0'
                                THEN 1
                                ELSE 0
                            END
                        ) AS BB_STATUS_0_COUNT,

                        SUM(
                            CASE
                                WHEN STATUS = '1'
                                THEN 1
                                ELSE 0
                            END
                        ) AS BB_STATUS_1_COUNT,

                        SUM(
                            CASE
                                WHEN STATUS = '2'
                                THEN 1
                                ELSE 0
                            END
                        ) AS BB_STATUS_2_COUNT,

                        SUM(
                            CASE
                                WHEN STATUS = '3'
                                THEN 1
                                ELSE 0
                            END
                        ) AS BB_STATUS_3_COUNT,

                        SUM(
                            CASE
                                WHEN STATUS = '4'
                                THEN 1
                                ELSE 0
                            END
                        ) AS BB_STATUS_4_COUNT,

                        SUM(
                            CASE
                                WHEN STATUS = '5'
                                THEN 1
                                ELSE 0
                            END
                        ) AS BB_STATUS_5_COUNT,

                        SUM(
                            CASE
                                WHEN STATUS = 'C'
                                THEN 1
                                ELSE 0
                            END
                        ) AS BB_STATUS_C_COUNT,

                        SUM(
                            CASE
                                WHEN STATUS = 'X'
                                THEN 1
                                ELSE 0
                            END
                        ) AS BB_STATUS_X_COUNT,

                        SUM(
                            CASE
                                WHEN STATUS IN (
                                    '1',
                                    '2',
                                    '3',
                                    '4',
                                    '5'
                                )
                                THEN 1
                                ELSE 0
                            END
                        ) AS BB_DPD_RECORD_COUNT,

                        SUM(
                            CASE
                                WHEN STATUS IN (
                                    '3',
                                    '4',
                                    '5'
                                )
                                THEN 1
                                ELSE 0
                            END
                        ) AS BB_SEVERE_DPD_RECORD_COUNT,

                        MAX(
                            CASE
                                WHEN STATUS IN (
                                    '0',
                                    '1',
                                    '2',
                                    '3',
                                    '4',
                                    '5'
                                )
                                THEN CAST(
                                    STATUS AS INTEGER
                                )
                                ELSE NULL
                            END
                        ) AS BB_MAX_STATUS_LEVEL

                    FROM mapped_history

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

                        bureau_balance_aggregated.* EXCLUDE (
                            SK_ID_CURR
                        ),

                        CASE
                            WHEN bureau_balance_aggregated.SK_ID_CURR
                                IS NOT NULL
                            THEN 1
                            ELSE 0
                        END AS HAS_BUREAU_BALANCE_HISTORY

                    FROM application

                    LEFT JOIN bureau_balance_aggregated
                        ON application.SK_ID_CURR
                        = bureau_balance_aggregated.SK_ID_CURR
                ),

                features AS (
                    SELECT
                        *,

                        CASE
                            WHEN BB_RECORD_COUNT > 0
                            THEN
                                BB_DPD_RECORD_COUNT
                                * 1.0
                                / BB_RECORD_COUNT
                        END AS BB_DPD_RATE,

                        CASE
                            WHEN BB_RECORD_COUNT > 0
                            THEN
                                BB_SEVERE_DPD_RECORD_COUNT
                                * 1.0
                                / BB_RECORD_COUNT
                        END AS BB_SEVERE_DPD_RATE,

                        CASE
                            WHEN BB_RECORD_COUNT > 0
                            THEN
                                BB_STATUS_C_COUNT
                                * 1.0
                                / BB_RECORD_COUNT
                        END AS BB_CLOSED_STATUS_RATE,

                        CASE
                            WHEN BB_RECORD_COUNT > 0
                            THEN
                                BB_STATUS_X_COUNT
                                * 1.0
                                / BB_RECORD_COUNT
                        END AS BB_UNKNOWN_STATUS_RATE

                    FROM merged
                )

                SELECT *
                FROM features
            """

            dataframe = connection.execute(
                query,
                [
                    str(self.parquet_path),
                    str(bureau_path),
                    str(application_path),
                ],
            ).fetchdf()

        finally:
            connection.close()

        feature_columns = [
            "BB_RECORD_COUNT",
            "BB_BUREAU_CREDIT_COUNT",
            "BB_OLDEST_MONTH",
            "BB_RECENT_MONTH",
            "BB_STATUS_0_COUNT",
            "BB_STATUS_1_COUNT",
            "BB_STATUS_2_COUNT",
            "BB_STATUS_3_COUNT",
            "BB_STATUS_4_COUNT",
            "BB_STATUS_5_COUNT",
            "BB_STATUS_C_COUNT",
            "BB_STATUS_X_COUNT",
            "BB_DPD_RECORD_COUNT",
            "BB_SEVERE_DPD_RECORD_COUNT",
            "BB_MAX_STATUS_LEVEL",
            "BB_DPD_RATE",
            "BB_SEVERE_DPD_RATE",
            "BB_CLOSED_STATUS_RATE",
            "BB_UNKNOWN_STATUS_RATE",
        ]

        clients_with_history = int(dataframe["HAS_BUREAU_BALANCE_HISTORY"].sum())

        clients_without_history = int(len(dataframe) - clients_with_history)

        target_results = []

        target_values = sorted(dataframe[target_column].dropna().unique())

        for target_value in target_values:
            target_data = dataframe.loc[dataframe[target_column] == target_value]

            target_clients = len(target_data)

            target_with_history = int(target_data["HAS_BUREAU_BALANCE_HISTORY"].sum())

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
            "mapping_diagnostics": (mapping_diagnostics),
            "aggregated_features": (feature_columns),
            "target_statistics": (target_results),
        }
