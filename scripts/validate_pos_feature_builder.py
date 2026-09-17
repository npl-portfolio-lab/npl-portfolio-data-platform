from pathlib import Path

import duckdb
import numpy as np

from npl_portfolio.features.pos_cash_features import (
    POSCashFeatureBuilder,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

POS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "home_credit"
    / "POS_CASH_balance.parquet"
)


FEATURE_COLUMNS = [
    "POS_RECORD_COUNT",
    "POS_CONTRACT_COUNT",
    "POS_ACTIVE_RECORD_COUNT",
    "POS_COMPLETED_RECORD_COUNT",
    "POS_DPD_RECORD_COUNT",
    "POS_DPD_DEF_RECORD_COUNT",
    "POS_MAX_DPD",
    "POS_AVG_DPD",
    "POS_MAX_DPD_DEF",
    "POS_AVG_DPD_DEF",
    "POS_AVG_INSTALMENT",
    "POS_AVG_INSTALMENT_FUTURE",
    "POS_OLDEST_MONTH",
    "POS_RECENT_MONTH",
    "POS_DPD_RATE",
    "POS_DPD_DEF_RATE",
]


def build_reference_features():
    """
    Reproduce las fórmulas utilizadas originalmente durante el EDA.

    Esta consulta es únicamente de validación.
    No forma parte del pipeline productivo de Feature Engineering.
    """
    connection = duckdb.connect()

    try:
        query = """
            WITH pos_aggregated AS (
                SELECT
                    SK_ID_CURR,

                    COUNT(*) AS POS_RECORD_COUNT,

                    COUNT(DISTINCT SK_ID_PREV)
                        AS POS_CONTRACT_COUNT,

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

                    MAX(SK_DPD) AS POS_MAX_DPD,
                    AVG(SK_DPD) AS POS_AVG_DPD,

                    MAX(SK_DPD_DEF) AS POS_MAX_DPD_DEF,
                    AVG(SK_DPD_DEF) AS POS_AVG_DPD_DEF,

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

        return connection.execute(
            query,
            [str(POS_PATH)],
        ).fetchdf()

    finally:
        connection.close()


def main() -> None:
    print("===== VALIDACIÓN POS FEATURE BUILDER =====")
    print()

    builder = POSCashFeatureBuilder(
        parquet_path=POS_PATH,
    )

    print("Construyendo features con POSCashFeatureBuilder...")
    builder_features = builder.build()

    print("Construyendo referencia basada en EDA...")
    reference_features = build_reference_features()

    print()
    print("===== DIMENSIONES =====")
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

    if len(builder_features) != len(reference_features):
        raise AssertionError(
            "La cantidad de clientes no coincide."
        )

    if not builder_features["SK_ID_CURR"].is_unique:
        raise AssertionError(
            "El builder contiene SK_ID_CURR duplicados."
        )

    if not reference_features["SK_ID_CURR"].is_unique:
        raise AssertionError(
            "La referencia contiene SK_ID_CURR duplicados."
        )

    if not builder_features[
        "SK_ID_CURR"
    ].equals(reference_features["SK_ID_CURR"]):
        raise AssertionError(
            "Los SK_ID_CURR no coinciden."
        )

    print()
    print("===== COMPARACIÓN DE FEATURES =====")

    differences = []

    for column in FEATURE_COLUMNS:
        builder_values = builder_features[column].to_numpy()
        reference_values = reference_features[column].to_numpy()

        equal = np.allclose(
            builder_values,
            reference_values,
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        )

        status = "PASS" if equal else "FAIL"

        print(
            f"{column:<35} {status}"
        )

        if not equal:
            differences.append(column)

    print()

    if differences:
        print("RESULTADO: FAIL")
        print(
            "Features diferentes: "
            + ", ".join(differences)
        )
        raise AssertionError(
            "El nuevo builder no reproduce el EDA."
        )

    print("RESULTADO: PASS")
    print(
        "POSCashFeatureBuilder reproduce "
        "las features utilizadas en el EDA."
    )


if __name__ == "__main__":
    main()
