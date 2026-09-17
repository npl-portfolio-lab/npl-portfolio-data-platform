from pathlib import Path

import pandas as pd

from npl_portfolio.core.duckdb_manager import DuckDBManager
from npl_portfolio.features.base_feature_builder import BaseDuckDBFeatureBuilder


class POSCashFeatureBuilder(BaseDuckDBFeatureBuilder):
    """
    Construye features de POS_CASH_balance a nivel cliente.

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
                WITH pos_aggregated AS (
                    SELECT
                        SK_ID_CURR,

                        COUNT(*)
                            AS POS_RECORD_COUNT,

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
                )

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

                FROM pos_aggregated

                ORDER BY SK_ID_CURR
            """

    def _get_parameters(self) -> list[str]:
        return [str(self.parquet_path)]
