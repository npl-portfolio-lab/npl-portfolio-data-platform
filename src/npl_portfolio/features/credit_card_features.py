from pathlib import Path

import duckdb
import pandas as pd


class CreditCardFeatureBuilder:
    """
    Construye features de credit_card_balance a nivel cliente.

    La salida contiene exactamente una fila por SK_ID_CURR.

    TARGET no participa en la construcción de features.
    """

    def __init__(self, parquet_path: Path) -> None:
        self.parquet_path = parquet_path

        if not self.parquet_path.exists():
            raise FileNotFoundError(
                f"No existe el archivo: {self.parquet_path}"
            )

    def build(self) -> pd.DataFrame:
        """
        Construye las features históricas de tarjetas de crédito.
        """
        connection = duckdb.connect()

        try:
            query = """
                WITH credit_card_aggregated AS (
                    SELECT
                        SK_ID_CURR,

                        COUNT(*)
                            AS CC_RECORD_COUNT,

                        COUNT(DISTINCT SK_ID_PREV)
                            AS CC_CONTRACT_COUNT,

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
                )

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

                FROM credit_card_aggregated

                ORDER BY SK_ID_CURR
            """

            dataframe = connection.execute(
                query,
                [str(self.parquet_path)],
            ).fetchdf()

        finally:
            connection.close()

        return dataframe
