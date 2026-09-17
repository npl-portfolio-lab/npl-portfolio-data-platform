from pathlib import Path

import duckdb
import pandas as pd


class BureauBalanceFeatureBuilder:
    """
    Construye features de bureau_balance a nivel cliente.

    bureau_balance no contiene SK_ID_CURR, por lo que primero
    se mapea SK_ID_BUREAU contra bureau para obtener el cliente.

    Los registros de bureau_balance sin correspondencia en bureau
    no pueden participar en la agregación por SK_ID_CURR.

    TARGET no participa en la construcción de features.
    """

    def __init__(
        self,
        parquet_path: Path,
        bureau_path: Path,
    ) -> None:
        self.parquet_path = parquet_path
        self.bureau_path = bureau_path

        if not self.parquet_path.exists():
            raise FileNotFoundError(
                f"No existe el archivo: {self.parquet_path}"
            )

        if not self.bureau_path.exists():
            raise FileNotFoundError(
                f"No existe el archivo: {self.bureau_path}"
            )

    def build(self) -> pd.DataFrame:
        """
        Mapea bureau_balance hacia SK_ID_CURR mediante bureau
        y construye una fila de features por cliente.
        """
        connection = duckdb.connect()

        try:
            query = """
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

                        COUNT(DISTINCT SK_ID_BUREAU)
                            AS BB_BUREAU_CREDIT_COUNT,

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
                )

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

                FROM bureau_balance_aggregated

                ORDER BY SK_ID_CURR
            """

            dataframe = connection.execute(
                query,
                [
                    str(self.parquet_path),
                    str(self.bureau_path),
                ],
            ).fetchdf()

        finally:
            connection.close()

        return dataframe
