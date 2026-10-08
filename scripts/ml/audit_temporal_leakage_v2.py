
"""Fase 10.12: auditoría temporal de fuentes históricas Home Credit."""

from pathlib import Path

import duckdb

from npl_portfolio.core.paths import PROJECT_ROOT


SOURCE_DIR = PROJECT_ROOT / "data" / "processed" / "home_credit"

TEMPORAL_COLUMNS = {
    "installments_payments": [
        "DAYS_INSTALMENT",
        "DAYS_ENTRY_PAYMENT",
    ],
    "credit_card_balance": ["MONTHS_BALANCE"],
    "POS_CASH_balance": ["MONTHS_BALANCE"],
    "bureau_balance": ["MONTHS_BALANCE"],
    "previous_application": ["DAYS_DECISION"],
    "bureau": ["DAYS_CREDIT"],
}


def audit_temporal_column(connection, path: Path, column: str) -> dict:
    result = connection.execute(
        f"""
        SELECT
            MIN("{column}"),
            MAX("{column}"),
            COUNT(*) FILTER (WHERE "{column}" > 0),
            COUNT(*) FILTER (WHERE "{column}" = 0),
            COUNT(*) FILTER (WHERE "{column}" IS NULL)
        FROM read_parquet(?)
        """,
        [str(path)],
    ).fetchone()

    return {
        "min": result[0],
        "max": result[1],
        "future_count": result[2],
        "zero_count": result[3],
        "null_count": result[4],
    }


def main() -> None:
    print("=" * 70)
    print("FASE 10.12 - AUDITORIA TEMPORAL V2")
    print("=" * 70)

    failures = []
    warnings = []

    connection = duckdb.connect()

    try:
        for source, columns in TEMPORAL_COLUMNS.items():
            path = SOURCE_DIR / f"{source}.parquet"

            if not path.is_file():
                failures.append(f"Fuente inexistente: {path}")
                continue

            schema = connection.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)",
                [str(path)],
            ).fetchall()

            available = {row[0] for row in schema}

            for column in columns:
                if column not in available:
                    failures.append(f"{source}: falta {column}")
                    continue

                result = audit_temporal_column(
                    connection,
                    path,
                    column,
                )

                label = f"{source}.{column}"

                if result["future_count"] > 0:
                    failures.append(
                        f"{label}: {result['future_count']} "
                        "fechas positivas"
                    )
                    print(f"[FAIL] {label}")
                else:
                    print(
                        f"[PASS] {label}: "
                        f"min={result['min']} "
                        f"max={result['max']} "
                        f"futuros=0"
                    )

                if result["zero_count"] > 0:
                    warnings.append(
                        f"{label}: {result['zero_count']:,} "
                        "registros en periodo cero"
                    )

                if result["null_count"] > 0:
                    warnings.append(
                        f"{label}: {result['null_count']:,} "
                        "fechas nulas"
                    )

    finally:
        connection.close()

    print("\n========== OBSERVACIONES ==========")

    for warning in warnings:
        print(f"[WARN] {warning}")

    if failures:
        print("\n========== ERRORES ==========")
        for failure in failures:
            print(f"[FAIL] {failure}")
        raise SystemExit(1)

    print("\n[OK] No se detectaron fechas positivas.")
    print("[PENDIENTE] Validar disponibilidad temporal y SQL.")
    print("[OK] Auditoria temporal inicial completada.")


if __name__ == "__main__":
    main()
