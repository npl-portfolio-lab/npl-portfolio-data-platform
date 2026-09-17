from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from npl_portfolio.features.installments_features import (
    InstallmentsFeatureBuilder,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INSTALLMENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "home_credit"
    / "installments_payments.parquet"
)


FEATURE_COLUMNS = [
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


def build_eda_reference(
    parquet_path: Path,
) -> pd.DataFrame:
    """
    Reproduce independientemente la agregación utilizada
    originalmente durante el EDA de Installments.
    """
    connection = duckdb.connect()

    try:
        query = """
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

        return connection.execute(
            query,
            [str(parquet_path)],
        ).fetchdf()

    finally:
        connection.close()


def main() -> None:
    print("=" * 90)
    print("VALIDACIÓN INSTALLMENTS FEATURE BUILDER")
    print("=" * 90)

    print(
        "\nConstruyendo features con "
        "InstallmentsFeatureBuilder..."
    )

    builder = InstallmentsFeatureBuilder(
        parquet_path=INSTALLMENTS_PATH,
    )

    builder_features = builder.build()

    print("Construyendo referencia basada en EDA...")

    reference_features = build_eda_reference(
        INSTALLMENTS_PATH
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
        print(
            "\n[FAIL] El número de filas es diferente."
        )
        success = False

    if not builder_features["SK_ID_CURR"].is_unique:
        print(
            "\n[FAIL] Builder contiene "
            "SK_ID_CURR duplicados."
        )
        success = False

    if not reference_features["SK_ID_CURR"].is_unique:
        print(
            "\n[FAIL] Referencia contiene "
            "SK_ID_CURR duplicados."
        )
        success = False

    builder_features = (
        builder_features
        .sort_values("SK_ID_CURR")
        .reset_index(drop=True)
    )

    reference_features = (
        reference_features
        .sort_values("SK_ID_CURR")
        .reset_index(drop=True)
    )

    same_clients = builder_features[
        "SK_ID_CURR"
    ].equals(
        reference_features["SK_ID_CURR"]
    )

    if same_clients:
        print(
            "\n[PASS] SK_ID_CURR coincide exactamente."
        )
    else:
        print(
            "\n[FAIL] SK_ID_CURR no coincide."
        )
        success = False

    print(
        "\n===== COMPARACIÓN DE FEATURES ====="
    )

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
            "InstallmentsFeatureBuilder reproduce "
            "las features utilizadas en el EDA."
        )
    else:
        print("RESULTADO: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
