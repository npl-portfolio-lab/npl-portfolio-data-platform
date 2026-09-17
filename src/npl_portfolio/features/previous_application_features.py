from pathlib import Path

from npl_portfolio.features.base_feature_builder import BaseDuckDBFeatureBuilder


class PreviousApplicationFeatureBuilder(BaseDuckDBFeatureBuilder):
    """Construye features de previous_application por SK_ID_CURR."""

    FEATURE_COLUMNS = [
        "PREV_APPLICATION_COUNT",
        "PREV_APPROVED_COUNT",
        "PREV_REFUSED_COUNT",
        "PREV_CANCELED_COUNT",
        "PREV_UNUSED_COUNT",
        "PREV_AVG_APPLICATION_AMOUNT",
        "PREV_AVG_CREDIT_AMOUNT",
        "PREV_TOTAL_APPLICATION_AMOUNT",
        "PREV_TOTAL_CREDIT_AMOUNT",
        "PREV_AVG_ANNUITY",
        "PREV_AVG_PAYMENT_COUNT",
        "PREV_AVG_CREDIT_DIFFERENCE",
        "PREV_AVG_DAYS_DECISION",
        "PREV_LAST_DECISION_DAYS",
        "PREV_APPROVAL_RATE",
        "PREV_REFUSAL_RATE",
        "PREV_CANCELLATION_RATE",
    ]

    def __init__(self, parquet_path: Path) -> None:
        self.parquet_path = parquet_path

        if not self.parquet_path.exists():
            raise FileNotFoundError(
                f"No existe el archivo: {self.parquet_path}"
            )

    def _get_query(self) -> str:
        return """
            WITH aggregated AS (
                SELECT
                    SK_ID_CURR,

                    COUNT(*) AS PREV_APPLICATION_COUNT,

                    SUM(
                        CASE
                            WHEN NAME_CONTRACT_STATUS = 'Approved'
                            THEN 1 ELSE 0
                        END
                    ) AS PREV_APPROVED_COUNT,

                    SUM(
                        CASE
                            WHEN NAME_CONTRACT_STATUS = 'Refused'
                            THEN 1 ELSE 0
                        END
                    ) AS PREV_REFUSED_COUNT,

                    SUM(
                        CASE
                            WHEN NAME_CONTRACT_STATUS = 'Canceled'
                            THEN 1 ELSE 0
                        END
                    ) AS PREV_CANCELED_COUNT,

                    SUM(
                        CASE
                            WHEN NAME_CONTRACT_STATUS = 'Unused offer'
                            THEN 1 ELSE 0
                        END
                    ) AS PREV_UNUSED_COUNT,

                    AVG(AMT_APPLICATION)
                        AS PREV_AVG_APPLICATION_AMOUNT,

                    AVG(AMT_CREDIT)
                        AS PREV_AVG_CREDIT_AMOUNT,

                    SUM(AMT_APPLICATION)
                        AS PREV_TOTAL_APPLICATION_AMOUNT,

                    SUM(AMT_CREDIT)
                        AS PREV_TOTAL_CREDIT_AMOUNT,

                    AVG(AMT_ANNUITY)
                        AS PREV_AVG_ANNUITY,

                    AVG(CNT_PAYMENT)
                        AS PREV_AVG_PAYMENT_COUNT,

                    AVG(AMT_APPLICATION - AMT_CREDIT)
                        AS PREV_AVG_CREDIT_DIFFERENCE,

                    AVG(DAYS_DECISION)
                        AS PREV_AVG_DAYS_DECISION,

                    MAX(DAYS_DECISION)
                        AS PREV_LAST_DECISION_DAYS

                FROM read_parquet(?)

                GROUP BY SK_ID_CURR
            )

            SELECT
                *,

                PREV_APPROVED_COUNT * 1.0
                    / PREV_APPLICATION_COUNT
                    AS PREV_APPROVAL_RATE,

                PREV_REFUSED_COUNT * 1.0
                    / PREV_APPLICATION_COUNT
                    AS PREV_REFUSAL_RATE,

                PREV_CANCELED_COUNT * 1.0
                    / PREV_APPLICATION_COUNT
                    AS PREV_CANCELLATION_RATE

            FROM aggregated

            ORDER BY SK_ID_CURR
        """

    def _get_parameters(self) -> list[str]:
        return [str(self.parquet_path)]
