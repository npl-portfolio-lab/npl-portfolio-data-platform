from pathlib import Path
from typing import Any

from npl_portfolio.analytics.historical.duckdb_base_service import (
    DuckDBBaseService,
)


class InstallmentsService(DuckDBBaseService):
    """
    Servicio especializado para installments_payments.

    Permite analizar el comportamiento histórico de pagos
    y construir agregaciones por cliente mediante DuckDB.

    La tabla histórica se agrega por SK_ID_CURR antes de
    incorporar TARGET.
    """

    def get_installments_target_analysis(
        self,
        application_path: Path,
        target_column: str = "TARGET",
    ) -> dict[str, Any]:
        """
        Agrega installments_payments por cliente y compara
        las métricas históricas entre TARGET=0 y TARGET=1.

        PAYMENT_DELAY_DAYS:
            DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT

            > 0  -> pago tardío
            = 0  -> pago en fecha
            < 0  -> pago anticipado

        PAYMENT_DIFFERENCE:
            AMT_INSTALMENT - AMT_PAYMENT

            > 0  -> pago inferior al importe esperado
            = 0  -> pago completo
            < 0  -> pago superior al importe esperado
        """
        if not application_path.exists():
            raise FileNotFoundError(f"No existe el archivo: {application_path}")

        connection = self._get_connection()

        try:
            query = f"""
                WITH installments_enriched AS (
                    SELECT
                        SK_ID_CURR,
                        SK_ID_PREV,
                        NUM_INSTALMENT_VERSION,
                        NUM_INSTALMENT_NUMBER,
                        DAYS_INSTALMENT,
                        DAYS_ENTRY_PAYMENT,
                        AMT_INSTALMENT,
                        AMT_PAYMENT,

                        CASE
                            WHEN DAYS_ENTRY_PAYMENT IS NOT NULL
                            THEN
                                DAYS_ENTRY_PAYMENT
                                - DAYS_INSTALMENT
                            ELSE NULL
                        END AS PAYMENT_DELAY_DAYS,

                        CASE
                            WHEN AMT_PAYMENT IS NOT NULL
                            THEN
                                AMT_INSTALMENT
                                - AMT_PAYMENT
                            ELSE NULL
                        END AS PAYMENT_DIFFERENCE

                    FROM read_parquet(?)
                ),

                installments_aggregated AS (
                    SELECT
                        SK_ID_CURR,

                        COUNT(*)
                            AS INST_RECORD_COUNT,

                        COUNT(
                            DISTINCT SK_ID_PREV
                        ) AS INST_CONTRACT_COUNT,

                        AVG(NUM_INSTALMENT_VERSION)
                            AS INST_AVG_VERSION,

                        MAX(NUM_INSTALMENT_VERSION)
                            AS INST_MAX_VERSION,

                        AVG(NUM_INSTALMENT_NUMBER)
                            AS INST_AVG_NUMBER,

                        MAX(NUM_INSTALMENT_NUMBER)
                            AS INST_MAX_NUMBER,

                        AVG(AMT_INSTALMENT)
                            AS INST_AVG_INSTALMENT_AMOUNT,

                        MAX(AMT_INSTALMENT)
                            AS INST_MAX_INSTALMENT_AMOUNT,

                        SUM(AMT_INSTALMENT)
                            AS INST_TOTAL_INSTALMENT_AMOUNT,

                        AVG(AMT_PAYMENT)
                            AS INST_AVG_PAYMENT_AMOUNT,

                        MAX(AMT_PAYMENT)
                            AS INST_MAX_PAYMENT_AMOUNT,

                        SUM(AMT_PAYMENT)
                            AS INST_TOTAL_PAYMENT_AMOUNT,

                        AVG(PAYMENT_DELAY_DAYS)
                            AS INST_AVG_PAYMENT_DELAY,

                        MAX(PAYMENT_DELAY_DAYS)
                            AS INST_MAX_PAYMENT_DELAY,

                        MIN(PAYMENT_DELAY_DAYS)
                            AS INST_MIN_PAYMENT_DELAY,

                        SUM(
                            CASE
                                WHEN PAYMENT_DELAY_DAYS > 0
                                THEN 1
                                ELSE 0
                            END
                        ) AS INST_LATE_PAYMENT_COUNT,

                        SUM(
                            CASE
                                WHEN PAYMENT_DELAY_DAYS <= 0
                                THEN 1
                                ELSE 0
                            END
                        ) AS INST_ON_TIME_OR_EARLY_COUNT,

                        SUM(
                            CASE
                                WHEN PAYMENT_DELAY_DAYS IS NULL
                                THEN 1
                                ELSE 0
                            END
                        ) AS INST_MISSING_PAYMENT_DATE_COUNT,

                        AVG(PAYMENT_DIFFERENCE)
                            AS INST_AVG_PAYMENT_DIFFERENCE,

                        MAX(PAYMENT_DIFFERENCE)
                            AS INST_MAX_PAYMENT_DIFFERENCE,

                        SUM(
                            CASE
                                WHEN PAYMENT_DIFFERENCE > 0
                                THEN 1
                                ELSE 0
                            END
                        ) AS INST_UNDERPAYMENT_COUNT,

                        SUM(
                            CASE
                                WHEN PAYMENT_DIFFERENCE < 0
                                THEN 1
                                ELSE 0
                            END
                        ) AS INST_OVERPAYMENT_COUNT,

                        MIN(DAYS_INSTALMENT)
                            AS INST_OLDEST_SCHEDULED_DAY,

                        MAX(DAYS_INSTALMENT)
                            AS INST_RECENT_SCHEDULED_DAY,

                        MIN(DAYS_ENTRY_PAYMENT)
                            AS INST_OLDEST_PAYMENT_DAY,

                        MAX(DAYS_ENTRY_PAYMENT)
                            AS INST_RECENT_PAYMENT_DAY

                    FROM installments_enriched

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

                        installments_aggregated.* EXCLUDE (
                            SK_ID_CURR
                        ),

                        CASE
                            WHEN installments_aggregated.SK_ID_CURR
                                IS NOT NULL
                            THEN 1
                            ELSE 0
                        END AS HAS_INSTALLMENTS_HISTORY

                    FROM application

                    LEFT JOIN installments_aggregated
                        ON application.SK_ID_CURR
                        = installments_aggregated.SK_ID_CURR
                ),

                features AS (
                    SELECT
                        *,

                        CASE
                            WHEN INST_RECORD_COUNT > 0
                            THEN
                                INST_LATE_PAYMENT_COUNT
                                * 1.0
                                / INST_RECORD_COUNT
                        END AS INST_LATE_PAYMENT_RATE,

                        CASE
                            WHEN INST_RECORD_COUNT > 0
                            THEN
                                INST_UNDERPAYMENT_COUNT
                                * 1.0
                                / INST_RECORD_COUNT
                        END AS INST_UNDERPAYMENT_RATE,

                        CASE
                            WHEN INST_RECORD_COUNT > 0
                            THEN
                                INST_OVERPAYMENT_COUNT
                                * 1.0
                                / INST_RECORD_COUNT
                        END AS INST_OVERPAYMENT_RATE,

                        CASE
                            WHEN INST_TOTAL_INSTALMENT_AMOUNT > 0
                            THEN
                                INST_TOTAL_PAYMENT_AMOUNT
                                / INST_TOTAL_INSTALMENT_AMOUNT
                            ELSE NULL
                        END AS INST_PAYMENT_RATIO

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
            "INST_RECORD_COUNT",
            "INST_CONTRACT_COUNT",
            "INST_AVG_VERSION",
            "INST_MAX_VERSION",
            "INST_AVG_NUMBER",
            "INST_MAX_NUMBER",
            "INST_AVG_INSTALMENT_AMOUNT",
            "INST_MAX_INSTALMENT_AMOUNT",
            "INST_TOTAL_INSTALMENT_AMOUNT",
            "INST_AVG_PAYMENT_AMOUNT",
            "INST_MAX_PAYMENT_AMOUNT",
            "INST_TOTAL_PAYMENT_AMOUNT",
            "INST_AVG_PAYMENT_DELAY",
            "INST_MAX_PAYMENT_DELAY",
            "INST_MIN_PAYMENT_DELAY",
            "INST_LATE_PAYMENT_COUNT",
            "INST_ON_TIME_OR_EARLY_COUNT",
            "INST_MISSING_PAYMENT_DATE_COUNT",
            "INST_AVG_PAYMENT_DIFFERENCE",
            "INST_MAX_PAYMENT_DIFFERENCE",
            "INST_UNDERPAYMENT_COUNT",
            "INST_OVERPAYMENT_COUNT",
            "INST_OLDEST_SCHEDULED_DAY",
            "INST_RECENT_SCHEDULED_DAY",
            "INST_OLDEST_PAYMENT_DAY",
            "INST_RECENT_PAYMENT_DAY",
            "INST_LATE_PAYMENT_RATE",
            "INST_UNDERPAYMENT_RATE",
            "INST_OVERPAYMENT_RATE",
            "INST_PAYMENT_RATIO",
        ]

        clients_with_history = int(dataframe["HAS_INSTALLMENTS_HISTORY"].sum())

        clients_without_history = int(len(dataframe) - clients_with_history)

        target_results = []

        target_values = sorted(dataframe[target_column].dropna().unique())

        for target_value in target_values:
            target_data = dataframe.loc[dataframe[target_column] == target_value]

            target_clients = len(target_data)

            target_with_history = int(target_data["HAS_INSTALLMENTS_HISTORY"].sum())

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
