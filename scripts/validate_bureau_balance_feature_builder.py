from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from npl_portfolio.features.bureau_balance_features import (
    BureauBalanceFeatureBuilder,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BUREAU_BALANCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "home_credit"
    / "bureau_balance.parquet"
)

BUREAU_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "home_credit"
    / "bureau.parquet"
)


FEATURE_COLUMNS = [
    "BB_RECORD_COUNT",
    "BB_BUREAU_CREDIT_COUNT",
    "BB_OLDEST_MONTH",
    "BB_RECENT_MONTH",
    "BB_STATUS_0_COUNT",
    "BB_STATUS_1_COUNT",
    "BB_STATUS_2_COUNT",
    "BB_STATUS_3_COUNT",
    "BB_STATUS_4_COUNT",
    "BB_STATUS_5_COUNT",
    "BB_STATUS_C_COUNT",
    "BB_STATUS_X_COUNT",
    "BB_DPD_RECORD_COUNT",
    "BB_SEVERE_DPD_RECORD_COUNT",
    "BB_MAX_STATUS_LEVEL",
    "BB_DPD_RATE",
    "BB_SEVERE_DPD_RATE",
    "BB_CLOSED_STATUS_RATE",
    "BB_UNKNOWN_STATUS_RATE",
]


def build_eda_reference(
    bureau_balance_path: Path,
    bureau_path: Path,
) -> pd.DataFrame:
    """
    Reproduce independientemente la agregación utilizada
    originalmente durante el EDA de bureau_balance.
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

        return connection.execute(
            query,
            [
                str(bureau_balance_path),
                str(bureau_path),
            ],
        ).fetchdf()

    finally:
        connection.close()


def main() -> None:
    print("=" * 90)
    print("VALIDACIÓN BUREAU BALANCE FEATURE BUILDER")
    print("=" * 90)

    print(
        "\nConstruyendo features con "
        "BureauBalanceFeatureBuilder..."
    )

    builder = BureauBalanceFeatureBuilder(
        parquet_path=BUREAU_BALANCE_PATH,
        bureau_path=BUREAU_PATH,
    )

    builder_features = builder.build()

    print("Construyendo referencia basada en EDA...")

    reference_features = build_eda_reference(
        bureau_balance_path=BUREAU_BALANCE_PATH,
        bureau_path=BUREAU_PATH,
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
            "BureauBalanceFeatureBuilder reproduce "
            "las features utilizadas en el EDA."
        )
    else:
        print("RESULTADO: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
