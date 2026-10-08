"""Auditoría de integridad de características históricas de V2.

Solo lectura. No modifica datasets ni artefactos ML.
"""

from dataclasses import dataclass
from pathlib import Path

import duckdb

from npl_portfolio.core.paths import PROJECT_ROOT


INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "features"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "features"

SOURCES = {
    "bureau": "HAS_BUREAU_HISTORY",
    "bureau_balance": "HAS_BUREAU_BALANCE_HISTORY",
    "previous_application": "HAS_PREVIOUS_APPLICATION_HISTORY",
    "pos_cash": "HAS_POS_HISTORY",
    "credit_card": "HAS_CREDIT_CARD_HISTORY",
    "installments": "HAS_INSTALLMENTS_HISTORY",
}


@dataclass(frozen=True)
class AuditResult:
    source: str
    feature_count: int
    matched_clients: int
    missing_history_clients: int
    value_mismatches: int
    flag_mismatches: int


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def get_columns(connection, path: Path) -> list[str]:
    return [
        row[0]
        for row in connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)",
            [str(path)],
        ).fetchall()
    ]


def check_unique_ids(connection, path: Path) -> tuple[int, int]:
    rows, unique_ids, null_ids = connection.execute(
        """
        SELECT
            COUNT(*),
            COUNT(DISTINCT SK_ID_CURR),
            COUNT(*) FILTER (WHERE SK_ID_CURR IS NULL)
        FROM read_parquet(?)
        """,
        [str(path)],
    ).fetchone()

    if rows != unique_ids or null_ids:
        raise AssertionError(
            f"Claves invalidas en {path}: "
            f"filas={rows}, unicos={unique_ids}, nulos={null_ids}"
        )

    return rows, unique_ids


def audit_source(
    connection,
    final_path: Path,
    source_path: Path,
    source_name: str,
    flag_name: str,
) -> AuditResult:
    final_columns = set(get_columns(connection, final_path))
    source_columns = get_columns(connection, source_path)

    if "SK_ID_CURR" not in source_columns:
        raise AssertionError(f"{source_name}: falta SK_ID_CURR")

    features = [
        column
        for column in source_columns
        if column != "SK_ID_CURR"
    ]

    missing = set(features) - final_columns

    if missing:
        raise AssertionError(
            f"{source_name}: columnas ausentes: {sorted(missing)}"
        )

    if flag_name not in final_columns:
        raise AssertionError(
            f"{source_name}: falta bandera {flag_name}"
        )

    comparisons = " OR ".join(
        (
            f"f.{quote_identifier(column)} "
            f"IS DISTINCT FROM s.{quote_identifier(column)}"
        )
        for column in features
    )

    if not comparisons:
        raise AssertionError(
            f"{source_name}: no hay caracteristicas para auditar"
        )

    query = f"""
        SELECT
            COUNT(*) FILTER (
                WHERE s.SK_ID_CURR IS NOT NULL
            ) AS matched_clients,

            COUNT(*) FILTER (
                WHERE s.SK_ID_CURR IS NULL
            ) AS missing_history_clients,

            COUNT(*) FILTER (
                WHERE s.SK_ID_CURR IS NOT NULL
                  AND ({comparisons})
            ) AS value_mismatches,

            COUNT(*) FILTER (
                WHERE f.{quote_identifier(flag_name)}
                    IS DISTINCT FROM
                    CASE
                        WHEN s.SK_ID_CURR IS NOT NULL THEN 1
                        ELSE 0
                    END
            ) AS flag_mismatches

        FROM read_parquet(?) AS f

        LEFT JOIN read_parquet(?) AS s
            ON f.SK_ID_CURR = s.SK_ID_CURR
    """

    matched, missing_history, values, flags = connection.execute(
        query,
        [str(final_path), str(source_path)],
    ).fetchone()

    return AuditResult(
        source=source_name,
        feature_count=len(features),
        matched_clients=matched,
        missing_history_clients=missing_history,
        value_mismatches=values,
        flag_mismatches=flags,
    )


def run_audit() -> list[AuditResult]:
    train = PROCESSED_DIR / "client_features_train.parquet"
    test = PROCESSED_DIR / "client_features_test.parquet"

    required = [train, test] + [
        INTERIM_DIR / f"{name}_features.parquet"
        for name in SOURCES
    ]

    missing = [str(path) for path in required if not path.is_file()]

    if missing:
        raise FileNotFoundError(
            "Archivos requeridos no encontrados:\n"
            + "\n".join(missing)
        )

    results = []
    connection = duckdb.connect()

    try:
        print("========== INTEGRIDAD DE DATASETS ==========")

        for name, path in (("TRAIN", train), ("TEST", test)):
            rows, _ = check_unique_ids(connection, path)
            print(f"[PASS] {name}: {rows:,} clientes unicos")

        train_rows, _ = check_unique_ids(connection, train)

        print("\n========== FUENTES HISTORICAS ==========")

        for source, flag in SOURCES.items():
            path = INTERIM_DIR / f"{source}_features.parquet"

            rows, _ = check_unique_ids(connection, path)

            result = audit_source(
                connection,
                train,
                path,
                source,
                flag,
            )

            results.append(result)

            total = (
                result.matched_clients
                + result.missing_history_clients
            )

            if total != train_rows:
                raise AssertionError(
                    f"{source}: cobertura inconsistente "
                    f"({total} != {train_rows})"
                )

            if result.value_mismatches or result.flag_mismatches:
                raise AssertionError(
                    f"{source}: diferencias de valores="
                    f"{result.value_mismatches}, "
                    f"banderas={result.flag_mismatches}"
                )

            print(
                f"[PASS] {source}: "
                f"filas={rows:,}, "
                f"features={result.feature_count}, "
                f"con_historial={result.matched_clients:,}, "
                f"sin_historial={result.missing_history_clients:,}, "
                "diferencias=0"
            )

        print("\n[PASS] Auditoria historica completada.")
        print(
            f"Columnas verificadas: "
            f"{sum(r.feature_count for r in results)}"
        )

        return results

    finally:
        connection.close()


if __name__ == "__main__":
    run_audit()
