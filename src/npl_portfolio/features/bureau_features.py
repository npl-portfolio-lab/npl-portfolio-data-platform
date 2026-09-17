from pathlib import Path

import pandas as pd

from npl_portfolio.core.duckdb_manager import DuckDBManager
from npl_portfolio.features.base_feature_builder import BaseDuckDBFeatureBuilder


class BureauFeatureBuilder(BaseDuckDBFeatureBuilder):
    """Construye features de bureau por SK_ID_CURR."""

    FEATURE_COLUMNS = [
        "BUREAU_CREDIT_COUNT",
        "BUREAU_ACTIVE_COUNT",
        "BUREAU_CLOSED_COUNT",
        "BUREAU_DAYS_OVERDUE_COUNT",
        "BUREAU_AMOUNT_OVERDUE_COUNT",
        "BUREAU_AVG_DAYS_CREDIT",
        "BUREAU_MIN_DAYS_CREDIT",
        "BUREAU_TOTAL_CREDIT",
        "BUREAU_TOTAL_DEBT",
        "BUREAU_TOTAL_OVERDUE",
        "BUREAU_MAX_OVERDUE_DAYS",
    ]

    def __init__(self, parquet_path: Path) -> None:
        self.parquet_path = parquet_path

        if not self.parquet_path.exists():
            raise FileNotFoundError(
                f"No existe el archivo: {self.parquet_path}"
            )

    def _get_query(self) -> str:
        return """
            SELECT
                SK_ID_CURR,
                COUNT(*) AS BUREAU_CREDIT_COUNT,

                SUM(
                    CASE WHEN CREDIT_ACTIVE = 'Active'
                    THEN 1 ELSE 0 END
                ) AS BUREAU_ACTIVE_COUNT,

                SUM(
                    CASE WHEN CREDIT_ACTIVE = 'Closed'
                    THEN 1 ELSE 0 END
                ) AS BUREAU_CLOSED_COUNT,

                SUM(
                    CASE WHEN CREDIT_DAY_OVERDUE > 0
                    THEN 1 ELSE 0 END
                ) AS BUREAU_DAYS_OVERDUE_COUNT,

                SUM(
                    CASE WHEN AMT_CREDIT_SUM_OVERDUE > 0
                    THEN 1 ELSE 0 END
                ) AS BUREAU_AMOUNT_OVERDUE_COUNT,

                AVG(DAYS_CREDIT) AS BUREAU_AVG_DAYS_CREDIT,
                MIN(DAYS_CREDIT) AS BUREAU_MIN_DAYS_CREDIT,

                SUM(AMT_CREDIT_SUM) AS BUREAU_TOTAL_CREDIT,
                SUM(AMT_CREDIT_SUM_DEBT) AS BUREAU_TOTAL_DEBT,
                SUM(AMT_CREDIT_SUM_OVERDUE) AS BUREAU_TOTAL_OVERDUE,

                MAX(CREDIT_DAY_OVERDUE)
                    AS BUREAU_MAX_OVERDUE_DAYS

            FROM read_parquet(?)

            GROUP BY SK_ID_CURR
            ORDER BY SK_ID_CURR
        """

    def _get_parameters(self) -> list[str]:
        return [str(self.parquet_path)]
