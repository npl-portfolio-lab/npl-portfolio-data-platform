"""Fase 10.15: auditoría temporal de previous_application.

Solo lectura. No modifica los datasets ni los modelos V2.
"""

import json

import duckdb

from npl_portfolio.core.paths import PROJECT_ROOT
from npl_portfolio.features.previous_application_features import (
    PreviousApplicationFeatureBuilder,
)


SOURCE = (
    PROJECT_ROOT
    / "data/processed/home_credit/previous_application.parquet"
)

TEMPORAL_COLUMNS = [
    "DAYS_DECISION",
    "DAYS_FIRST_DRAWING",
    "DAYS_FIRST_DUE",
    "DAYS_LAST_DUE_1ST_VERSION",
    "DAYS_LAST_DUE",
    "DAYS_TERMINATION",
]

SENTINEL = 365243


def audit():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)

    con = duckdb.connect()

    try:
        columns_available = {
            row[0]
            for row in con.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)",
                [str(SOURCE)],
            ).fetchall()
        }

        missing = set(TEMPORAL_COLUMNS) - columns_available

        if missing:
            raise ValueError(
                f"Columnas temporales faltantes: {sorted(missing)}"
            )

        total = con.execute(
            "SELECT COUNT(*) FROM read_parquet(?)",
            [str(SOURCE)],
        ).fetchone()[0]

        profiles = {}

        for column in TEMPORAL_COLUMNS:
            result = con.execute(
                f"""
                SELECT
                    COUNT(*) FILTER (
                        WHERE "{column}" IS NULL
                    ),
                    COUNT(*) FILTER (
                        WHERE "{column}" < 0
                    ),
                    COUNT(*) FILTER (
                        WHERE "{column}" = 0
                    ),
                    COUNT(*) FILTER (
                        WHERE "{column}" > 0
                          AND "{column}" <> ?
                    ),
                    COUNT(*) FILTER (
                        WHERE "{column}" = ?
                    ),
                    MIN("{column}"),
                    MAX("{column}")
                FROM read_parquet(?)
                """,
                [SENTINEL, SENTINEL, str(SOURCE)],
            ).fetchone()

            keys = [
                "nulls",
                "negatives",
                "zeros",
                "positive_non_sentinel",
                "sentinel_365243",
                "min",
                "max",
            ]

            profile = dict(zip(keys, result))

            categories_total = sum(
                profile[key]
                for key in keys[:5]
            )

            if categories_total != total:
                raise AssertionError(
                    f"Conteo inconsistente en {column}: "
                    f"{categories_total} != {total}"
                )

            profiles[column] = profile

        decision = profiles["DAYS_DECISION"]

        decision_before_reference = (
            decision["nulls"] == 0
            and decision["zeros"] == 0
            and decision["positive_non_sentinel"] == 0
            and decision["sentinel_365243"] == 0
        )

        return {
            "phase": "10.15",
            "source": str(SOURCE.relative_to(PROJECT_ROOT)),
            "total_records": total,
            "sentinel_value": SENTINEL,
            "temporal_profiles": profiles,
            "decision_strictly_before_reference": (
                decision_before_reference
            ),
            "feature_count": len(
                PreviousApplicationFeatureBuilder.FEATURE_COLUMNS
            ),
            "feature_columns": list(
                PreviousApplicationFeatureBuilder.FEATURE_COLUMNS
            ),
            "interpretation": (
                "Fechas futuras y valores especiales identificados; "
                "no constituyen por sí solos Data Leakage confirmado."
            ),
        }

    finally:
        con.close()


if __name__ == "__main__":
    result = audit()

    print("========== FASE 10.15 ==========")
    print(json.dumps(result, indent=2, ensure_ascii=False))
