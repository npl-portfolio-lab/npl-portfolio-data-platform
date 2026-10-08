"""Fase 10.5: auditoría de particiones ML existentes."""

import json
from datetime import datetime, timezone

from npl_portfolio.core.paths import ML_DATA_DIR, ML_ARTIFACTS_DIR, DOCS_DIR
from npl_portfolio.core.duckdb_manager import DuckDBManager


TRAIN = ML_DATA_DIR / "ml_train.parquet"
VALIDATION = ML_DATA_DIR / "ml_validation.parquet"
TEST = ML_DATA_DIR / "ml_test.parquet"

REPORT = ML_ARTIFACTS_DIR / "independent_validation_audit.json"
DOCUMENT = DOCS_DIR / "ml" / "fase_10_5_validacion_independiente.md"


def main():
    print("=" * 70)
    print("FASE 10.5 - AUDITORIA DE INDEPENDENCIA")
    print("=" * 70)

    for path in (TRAIN, VALIDATION):
        if not path.is_file():
            raise FileNotFoundError(path)

    connection = DuckDBManager().connect()

    try:
        train_path = str(TRAIN.resolve())
        validation_path = str(VALIDATION.resolve())

        for name, path in (
            ("train", train_path),
            ("validation", validation_path),
        ):
            escaped_path = path.replace("'", "''")

            connection.execute(
                f"CREATE TEMP VIEW {name} AS "
                "SELECT SK_ID_CURR, TARGET "
                f"FROM read_parquet('{escaped_path}')"
            )

        results = {}

        for name in ("train", "validation"):
            row = connection.execute(
                f"""
                SELECT
                    COUNT(*) AS total,
                    COUNT(DISTINCT SK_ID_CURR) AS unique_ids,
                    COUNT(*) FILTER (WHERE TARGET = 1) AS positives,
                    COUNT(*) FILTER (WHERE TARGET = 0) AS negatives,
                    COUNT(*) FILTER (
                        WHERE SK_ID_CURR IS NULL OR TARGET IS NULL
                    ) AS null_rows,
                    COUNT(*) FILTER (
                        WHERE TARGET NOT IN (0, 1)
                    ) AS invalid_targets
                FROM {name}
                """
            ).fetchone()

            results[name] = {
                "rows": int(row[0]),
                "unique_ids": int(row[1]),
                "positives": int(row[2]),
                "negatives": int(row[3]),
                "null_rows": int(row[4]),
                "invalid_targets": int(row[5]),
            }

        overlap = connection.execute(
            """
            SELECT COUNT(*)
            FROM train t
            INNER JOIN validation v
              ON t.SK_ID_CURR = v.SK_ID_CURR
            """
        ).fetchone()[0]

    finally:
        connection.close()

    train = results["train"]
    validation = results["validation"]

    checks = {
        "train_unique_ids": train["rows"] == train["unique_ids"],
        "validation_unique_ids": (
            validation["rows"] == validation["unique_ids"]
        ),
        "no_overlap": overlap == 0,
        "no_nulls": (
            train["null_rows"] == 0
            and validation["null_rows"] == 0
        ),
        "binary_target": (
            train["invalid_targets"] == 0
            and validation["invalid_targets"] == 0
            and train["positives"] + train["negatives"] == train["rows"]
            and validation["positives"] + validation["negatives"]
            == validation["rows"]
        ),
    }

    report = {
        "phase": "10.5",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "train": train,
        "validation": validation,
        "overlapping_clients": int(overlap),
        "test_parquet_exists": TEST.is_file(),
        "checks": checks,
        "all_partition_checks_passed": all(checks.values()),
        "independent_test_available": False,
        "limitations": [
            "No se ha verificado un TEST etiquetado independiente.",
            "VALIDATION ya se utilizo para comparar modelos y thresholds.",
            "Pendiente auditar internamente PreprocessingContractBuilder.",
            "Pendiente auditar posibles fugas en feature engineering.",
        ],
    }

    # No declarar TEST validado solo porque exista un archivo.
    if TEST.is_file():
        report["limitations"].append(
            "Existe ml_test.parquet, pero su independencia "
            "y procedencia no han sido verificadas."
        )

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOCUMENT.parent.mkdir(parents=True, exist_ok=True)

    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Fase 10.5 — Auditoría de validación independiente",
        "",
        "## Objetivo",
        "",
        "Verificar la separación de TRAIN y VALIDATION y determinar "
        "si existe un conjunto TEST independiente.",
        "",
        "## Resultados",
        "",
        "| Indicador | TRAIN | VALIDATION |",
        "|---|---:|---:|",
        f"| Registros | {train['rows']:,} | {validation['rows']:,} |",
        f"| Clientes únicos | {train['unique_ids']:,} | "
        f"{validation['unique_ids']:,} |",
        f"| Positivos | {train['positives']:,} | "
        f"{validation['positives']:,} |",
        f"| Negativos | {train['negatives']:,} | "
        f"{validation['negatives']:,} |",
        "",
        f"- Clientes compartidos: **{overlap:,}**",
        f"- Comprobaciones de partición superadas: "
        f"**{all(checks.values())}**",
        f"- Archivo TEST etiquetado detectado: **{TEST.is_file()}**",
        "",
        "## Comprobaciones",
        "",
    ]

    for key, passed in checks.items():
        lines.append(f"- {key}: {'PASS' if passed else 'FAIL'}")

    lines.extend([
        "",
        "## Conclusión",
        "",
        "Las comprobaciones verifican la integridad y separación "
        "de las particiones existentes, pero no constituyen "
        "una validación independiente final.",
        "",
        "## Limitaciones",
        "",
    ])

    for limitation in report["limitations"]:
        lines.append(f"- {limitation}")

    lines.extend([
        "",
        "## Próximos pasos",
        "",
        "Diseñar un nuevo protocolo experimental con TEST "
        "etiquetado y aislado desde el inicio. Conservar los "
        "artefactos de las fases anteriores.",
        "",
        "## Artefactos",
        "",
        "- `data/artifacts/ml/independent_validation_audit.json`",
        "- `scripts/ml/audit_independent_validation.py`",
        "",
    ])

    DOCUMENT.write_text("\n".join(lines), encoding="utf-8")

    print(f"TRAIN: {train['rows']:,}")
    print(f"VALIDATION: {validation['rows']:,}")
    print(f"Clientes compartidos: {overlap:,}")

    for key, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {key}")

    print(f"[OK] Reporte: {REPORT}")
    print(f"[OK] Documentación: {DOCUMENT}")

    if not all(checks.values()):
        raise SystemExit(
            "[ERROR] La auditoría detectó problemas de integridad."
        )

    print("[OK] FASE 10.5 - AUDITORIA COMPLETADA")


if __name__ == "__main__":
    main()
