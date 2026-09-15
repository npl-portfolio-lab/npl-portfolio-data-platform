from pathlib import Path
from typing import Any

from npl_portfolio.analytics.historical.duckdb_base_service import (
    DuckDBBaseService,
)


class CreditCardService(DuckDBBaseService):
    """
    Servicio especializado para credit_card_balance.
    """

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

        target_values = sorted(dataframe[target_column].dropna().unique())

        for target_value in target_values:
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
