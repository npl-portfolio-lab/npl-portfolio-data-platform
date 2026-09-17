from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from npl_portfolio.features.credit_card_features import (
    CreditCardFeatureBuilder,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CREDIT_CARD_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "home_credit"
    / "credit_card_balance.parquet"
)


FEATURE_COLUMNS = [
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


def build_eda_reference(
    parquet_path: Path,
) -> pd.DataFrame:
    """
    Reproduce independientemente la agregación utilizada
    originalmente durante el EDA de Credit Card.
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

        return connection.execute(
            query,
            [str(parquet_path)],
        ).fetchdf()

    finally:
        connection.close()


def main() -> None:
    print("=" * 90)
    print("VALIDACIÓN CREDIT CARD FEATURE BUILDER")
    print("=" * 90)

    print("\nConstruyendo features con CreditCardFeatureBuilder...")

    builder = CreditCardFeatureBuilder(
        parquet_path=CREDIT_CARD_PATH,
    )

    builder_features = builder.build()

    print("Construyendo referencia basada en EDA...")

    reference_features = build_eda_reference(
        CREDIT_CARD_PATH
    )

    print("\n===== DIMENSIONES =====")
    print(
        f"Builder:    "
        f"{len(builder_features):,} filas x "
        f"{len(builder_features.columns)} columnas"
    )
    print(
        f"Referencia: "
        f"{len(reference_features):,} filas x "
        f"{len(reference_features.columns)} columnas"
    )

    success = True

    if len(builder_features) != len(reference_features):
        print("\n[FAIL] El número de filas es diferente.")
        success = False

    if not builder_features["SK_ID_CURR"].is_unique:
        print("\n[FAIL] Builder contiene SK_ID_CURR duplicados.")
        success = False

    if not reference_features["SK_ID_CURR"].is_unique:
        print("\n[FAIL] Referencia contiene SK_ID_CURR duplicados.")
        success = False

    builder_features = builder_features.sort_values(
        "SK_ID_CURR"
    ).reset_index(drop=True)

    reference_features = reference_features.sort_values(
        "SK_ID_CURR"
    ).reset_index(drop=True)

    same_clients = builder_features[
        "SK_ID_CURR"
    ].equals(
        reference_features["SK_ID_CURR"]
    )

    if same_clients:
        print("\n[PASS] SK_ID_CURR coincide exactamente.")
    else:
        print("\n[FAIL] SK_ID_CURR no coincide.")
        success = False

    print("\n===== COMPARACIÓN DE FEATURES =====")

    for column in FEATURE_COLUMNS:
        builder_values = pd.to_numeric(
            builder_features[column],
            errors="coerce",
        ).to_numpy(dtype=float)

        reference_values = pd.to_numeric(
            reference_features[column],
            errors="coerce",
        ).to_numpy(dtype=float)

        equal = np.allclose(
            builder_values,
            reference_values,
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        )

        if equal:
            print(f"[PASS] {column}")
        else:
            print(f"[FAIL] {column}")
            success = False

    print("\n" + "=" * 90)

    if success:
        print("RESULTADO: PASS")
        print(
            "CreditCardFeatureBuilder reproduce las "
            "features utilizadas en el EDA."
        )
    else:
        print("RESULTADO: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
