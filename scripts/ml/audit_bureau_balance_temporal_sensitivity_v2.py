"""Fase 10.13: sensibilidad temporal de Bureau Balance.

Solo lectura. Reutiliza el SQL del constructor original.
No modifica datasets, modelos ni características V2.
"""

from pathlib import Path

import duckdb

from npl_portfolio.core.paths import PROJECT_ROOT
from npl_portfolio.features.bureau_balance_features import (
    BureauBalanceFeatureBuilder,
)

DATA_DIR = PROJECT_ROOT / "data"

BALANCE = (
    DATA_DIR / "processed/home_credit/bureau_balance.parquet"
)
BUREAU = DATA_DIR / "processed/home_credit/bureau.parquet"
REFERENCE = (
    DATA_DIR / "interim/features/bureau_balance_features.parquet"
)


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def build_queries(builder):
    original = builder._get_query()

    old = "FROM mapped_history"
    new = "FROM mapped_history WHERE MONTHS_BALANCE < 0"

    if original.count(old) != 1:
        raise ValueError(
            "No se encontró un único punto de agregación "
            "para aplicar el corte temporal."
        )

    conservative = original.replace(old, new, 1)

    return original, conservative


def run_audit():
    for path in (BALANCE, BUREAU, REFERENCE):
        if not path.is_file():
            raise FileNotFoundError(path)

    builder = BureauBalanceFeatureBuilder(
        parquet_path=BALANCE,
        bureau_path=BUREAU,
    )

    original, conservative = build_queries(builder)
    parameters = builder._get_parameters()

    con = duckdb.connect()

    try:
        print("========== FASE 10.13 ==========")
        print("AUDITORIA DE SENSIBILIDAD TEMPORAL")
        print()

        con.execute(
            "CREATE TEMP TABLE original AS " + original,
            parameters,
        )

        con.execute(
            "CREATE TEMP TABLE conservative AS " + conservative,
            parameters,
        )

        columns = [
            row[0]
            for row in con.execute(
                "DESCRIBE SELECT * FROM original"
            ).fetchall()
        ]

        features = [
            column
            for column in columns
            if column != "SK_ID_CURR"
        ]

        if len(features) != 19:
            raise AssertionError(
                f"Se esperaban 19 características: {len(features)}"
            )

        reference_columns = [
            row[0]
            for row in con.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)",
                [str(REFERENCE)],
            ).fetchall()
        ]

        if set(columns) != set(reference_columns):
            raise AssertionError(
                "El esquema original no coincide con el checkpoint."
            )

        comparisons = " OR ".join(
            f"o.{quote(c)} IS DISTINCT FROM r.{quote(c)}"
            for c in features
        )

        reference_check = con.execute(
            f"""
            SELECT
                COUNT(*) FILTER (
                    WHERE r.SK_ID_CURR IS NULL
                ),
                COUNT(*) FILTER (
                    WHERE r.SK_ID_CURR IS NOT NULL
                      AND ({comparisons})
                )
            FROM original o
            LEFT JOIN read_parquet(?) r
                ON o.SK_ID_CURR = r.SK_ID_CURR
            """,
            [str(REFERENCE)],
        ).fetchone()

        if any(reference_check):
            raise AssertionError(
                "La consulta original no reproduce el checkpoint: "
                f"{reference_check}"
            )

        original_count = con.execute(
            "SELECT COUNT(*) FROM original"
        ).fetchone()[0]

        reference_count = con.execute(
            "SELECT COUNT(*) FROM read_parquet(?)",
            [str(REFERENCE)],
        ).fetchone()[0]

        if original_count != reference_count:
            raise AssertionError(
                "Cantidad de clientes distinta del checkpoint."
            )

        print("[PASS] SQL original reproduce el checkpoint.")
        print(f"[PASS] Características comparadas: {len(features)}")
        print(f"[PASS] Clientes originales: {original_count:,}")

        changed_expressions = [
            (
                f"COUNT(*) FILTER (WHERE "
                f"o.{quote(c)} IS DISTINCT FROM c.{quote(c)}) "
                f"AS {quote(c)}"
            )
            for c in features
        ]

        results = con.execute(
            f"""
            SELECT
                COUNT(*) AS clientes,
                COUNT(*) FILTER (
                    WHERE c.SK_ID_CURR IS NULL
                ) AS clientes_sin_historial,
                {", ".join(changed_expressions)}
            FROM original o
            LEFT JOIN conservative c
                ON o.SK_ID_CURR = c.SK_ID_CURR
            """
        ).fetchone()

        print()
        print("========== RESULTADOS ==========")
        print(f"Clientes evaluados: {results[0]:,}")
        print(
            "Clientes sin historial tras corte: "
            f"{results[1]:,}"
        )

        for feature, changed in zip(features, results[2:]):
            print(f"{feature}: {changed:,} clientes con cambios")

        print()
        print("[PASS] Auditoría completada.")
        print(
            "[PENDIENTE] La comparación no certifica "
            "ausencia de Data Leakage."
        )

        return {
            "phase": "10.13",
            "scenario_original": "V2 sin corte temporal",
            "scenario_conservative": "MONTHS_BALANCE < 0",
            "clients_evaluated": int(results[0]),
            "clients_without_history": int(results[1]),
            "feature_count": len(features),
            "checkpoint_reproduced": True,
            "feature_changes": {
                feature: int(changed)
                for feature, changed in zip(features, results[2:])
            },
        }

    finally:
        con.close()


if __name__ == "__main__":
    run_audit()
