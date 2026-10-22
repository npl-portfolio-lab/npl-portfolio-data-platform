"""Fase 10.18: sensibilidad temporal de installments_payments V2.

Auditoría de solo lectura. No modifica datasets ni modelos.
Compara todas las características del constructor existente.
"""

import json
from pathlib import Path

import duckdb

from npl_portfolio.core.paths import PROJECT_ROOT
from npl_portfolio.features.installments_features import (
    InstallmentsFeatureBuilder,
)


SOURCE = (
    PROJECT_ROOT
    / "data/processed/home_credit/installments_payments.parquet"
)

CUTOFF_DAY = -2

REQUIRED_COLUMNS = {
    "SK_ID_CURR",
    "SK_ID_PREV",
    "NUM_INSTALMENT_VERSION",
    "NUM_INSTALMENT_NUMBER",
    "DAYS_INSTALMENT",
    "DAYS_ENTRY_PAYMENT",
    "AMT_INSTALMENT",
    "AMT_PAYMENT",
}


def audit(source: Path | None = None) -> dict:
    source = Path(source) if source is not None else SOURCE

    if not source.is_file():
        raise FileNotFoundError(source)

    con = duckdb.connect()
    con.execute("SET threads = 1")

    try:
        available = {
            row[0]
            for row in con.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)",
                [str(source)],
            ).fetchall()
        }

        missing = REQUIRED_COLUMNS - available

        if missing:
            raise ValueError(
                f"Columnas faltantes: {sorted(missing)}"
            )

        profile_result = con.execute(
            """
            SELECT
                COUNT(*) AS records,
                COUNT(DISTINCT SK_ID_CURR) AS clients,
                COUNT(DISTINCT SK_ID_PREV) AS contracts,
                MIN(DAYS_INSTALMENT) AS oldest_scheduled_day,
                MAX(DAYS_INSTALMENT) AS newest_scheduled_day,
                MIN(DAYS_ENTRY_PAYMENT) AS oldest_payment_day,
                MAX(DAYS_ENTRY_PAYMENT) AS newest_payment_day,
                COUNT(*) FILTER (
                    WHERE DAYS_INSTALMENT IS NULL
                ) AS null_scheduled_dates,
                COUNT(*) FILTER (
                    WHERE DAYS_ENTRY_PAYMENT IS NULL
                ) AS null_payment_dates,
                COUNT(*) FILTER (
                    WHERE DAYS_INSTALMENT > 0
                       OR DAYS_ENTRY_PAYMENT > 0
                ) AS future_records,
                COUNT(*) FILTER (
                    WHERE DAYS_INSTALMENT = -1
                ) AS recent_scheduled_records,
                COUNT(*) FILTER (
                    WHERE DAYS_ENTRY_PAYMENT = -1
                ) AS recent_payment_records,
                COUNT(*) FILTER (
                    WHERE DAYS_INSTALMENT <= ?
                      AND (
                          DAYS_ENTRY_PAYMENT <= ?
                          OR DAYS_ENTRY_PAYMENT IS NULL
                      )
                ) AS restricted_records
            FROM read_parquet(?)
            """,
            [CUTOFF_DAY, CUTOFF_DAY, str(source)],
        )

        profile_columns = [
            column[0] for column in profile_result.description
        ]

        temporal_profile = dict(
            zip(profile_columns, profile_result.fetchone())
        )

        temporal_profile["excluded_records"] = (
            temporal_profile["records"]
            - temporal_profile["restricted_records"]
        )

        builder = InstallmentsFeatureBuilder(source)
        original_query = builder._get_query()

        source_clause = "FROM read_parquet(?)"

        if original_query.count(source_clause) != 1:
            raise ValueError(
                "El constructor cambió: no se encontró "
                "una única lectura del Parquet."
            )

        restricted_query = original_query.replace(
            source_clause,
            """
            FROM read_parquet(?)
            WHERE DAYS_INSTALMENT <= -2
              AND (
                  DAYS_ENTRY_PAYMENT <= -2
                  OR DAYS_ENTRY_PAYMENT IS NULL
              )
            """,
            1,
        )

        con.execute(
            "CREATE TEMP TABLE original_features AS "
            + original_query,
            [str(source)],
        )

        con.execute(
            "CREATE TEMP TABLE restricted_features AS "
            + restricted_query,
            [str(source)],
        )

        columns = [
            row[1]
            for row in con.execute(
                "PRAGMA table_info('original_features')"
            ).fetchall()
            if row[1] != "SK_ID_CURR"
        ]

        changed_features = {}

        for column in columns:
            result = con.execute(
                f"""
                SELECT COUNT(*)
                FROM original_features o
                JOIN restricted_features r
                    USING (SK_ID_CURR)
                WHERE o."{column}"
                    IS DISTINCT FROM r."{column}"
                """
            ).fetchone()[0]

            if result:
                changed_features[column] = result

        comparison = con.execute(
            """
            SELECT
                COUNT(*) AS total_clients,
                COUNT(r.SK_ID_CURR) AS comparable_clients,
                COUNT(*) FILTER (
                    WHERE r.SK_ID_CURR IS NULL
                ) AS clients_without_history,
                COUNT(*) FILTER (
                    WHERE r.SK_ID_CURR IS NOT NULL
                      AND o.INST_RECORD_COUNT
                          IS DISTINCT FROM r.INST_RECORD_COUNT
                ) AS changed_record_count
            FROM original_features o
            LEFT JOIN restricted_features r
                ON o.SK_ID_CURR = r.SK_ID_CURR
            """
        )

        comparison_columns = [
            column[0] for column in comparison.description
        ]

        sensitivity = dict(
            zip(comparison_columns, comparison.fetchone())
        )

        sensitivity["evaluated_features"] = len(columns)
        sensitivity["changed_feature_count"] = len(
            changed_features
        )
        sensitivity["changed_features"] = changed_features

        return {
            "phase": "10.18",
            "source": (
                "data/processed/home_credit/"
                "installments_payments.parquet"
            ),
            "cutoff_day": CUTOFF_DAY,
            "temporal_profile": temporal_profile,
            "sensitivity": sensitivity,
            "risk_classification": (
                "Sensibilidad temporal identificada; "
                "Data Leakage no confirmado"
            ),
            "limitations": [
                "El corte -2 es una prueba de sensibilidad.",
                "No demuestra disponibilidad punto-en-tiempo.",
                "Los pagos sin fecha se conservan si la cuota "
                "cumple el corte temporal.",
                "Las diferencias se calculan solo para clientes "
                "con historial en ambos escenarios.",
                "Las comparaciones usan IS DISTINCT FROM "
                "sin tolerancia numérica.",
                "No modifica ni reentrena modelos ML V2.",
            ],
        }

    finally:
        con.close()


if __name__ == "__main__":
    print(
        json.dumps(
            audit(),
            indent=2,
            ensure_ascii=False,
        )
    )
