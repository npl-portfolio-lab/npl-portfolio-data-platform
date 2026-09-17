from pathlib import Path

import pandas as pd

from npl_portfolio.core.duckdb_manager import DuckDBManager
from npl_portfolio.features.base_feature_builder import BaseDuckDBFeatureBuilder


class InstallmentsFeatureBuilder(BaseDuckDBFeatureBuilder):
    """
    Construye features de installments_payments a nivel cliente.

    La salida contiene exactamente una fila por SK_ID_CURR.

    TARGET no participa en la construcción de features.
    """

    def __init__(self, parquet_path: Path) -> None:
        self.parquet_path = parquet_path

        if not self.parquet_path.exists():
            raise FileNotFoundError(
                f"No existe el archivo: {self.parquet_path}"
            )

    def _get_query(self) -> str:
        return """
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

                        COUNT(DISTINCT SK_ID_PREV)
                            AS INST_CONTRACT_COUNT,

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
                )

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

                FROM installments_aggregated

                ORDER BY SK_ID_CURR
            """

    def _get_parameters(self) -> list[str]:
        return [str(self.parquet_path)]
