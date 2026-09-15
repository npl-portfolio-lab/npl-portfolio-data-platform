from pathlib import Path
from typing import Any

from npl_portfolio.analytics.historical.duckdb_base_service import (
    DuckDBBaseService,
)


class POSCashService(DuckDBBaseService):
    """
    Servicio especializado para el análisis de POS_CASH_balance.

    Hereda las operaciones descriptivas generales de
    DuckDBBaseService e incorpora las agregaciones específicas
    del historial POS por cliente.
    """

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
        el historial.
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

        clients_without_history = len(dataframe) - clients_with_history

        target_results = []

        target_values = sorted(dataframe[target_column].dropna().unique())

        for target_value in target_values:
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
            "clients_without_history": int(clients_without_history),
            "history_coverage_percentage": round(
                (
                    (clients_with_history / len(dataframe) * 100)
                    if len(dataframe)
                    else 0.0
                ),
                4,
            ),
            "aggregated_features": feature_columns,
            "target_statistics": target_results,
        }
