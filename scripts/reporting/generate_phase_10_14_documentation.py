"""Genera la documentación reproducible de la Fase 10.14."""

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from npl_portfolio.core.paths import PROJECT_ROOT
from npl_portfolio.features.bureau_features import (
    BureauFeatureBuilder,
)


AUDITOR_PATH = (
    PROJECT_ROOT
    / "scripts/ml/audit_bureau_temporal_availability_v2.py"
)

OUTPUT_DIR = PROJECT_ROOT / "docs/ml"
STEM = "fase_10_14_disponibilidad_temporal_bureau_v2"


def load_auditor():
    spec = importlib.util.spec_from_file_location(
        "bureau_temporal_availability_v2",
        AUDITOR_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("No se pudo cargar el auditor.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def validate_splits(module, expected_clients):
    """Comprueba cobertura y exclusividad de los clientes afectados."""
    con = duckdb.connect()

    try:
        con.execute(
            """
            CREATE TEMP TABLE affected AS
            SELECT DISTINCT SK_ID_CURR
            FROM read_parquet(?)
            WHERE DAYS_CREDIT_UPDATE > 0
            """,
            [str(module.BUREAU)],
        )

        con.execute(
            """
            CREATE TEMP TABLE split_membership (
                SK_ID_CURR BIGINT,
                split_name VARCHAR
            )
            """
        )

        for name, path in module.SPLITS.items():
            con.execute(
                """
                INSERT INTO split_membership
                SELECT DISTINCT a.SK_ID_CURR, ?
                FROM affected a
                INNER JOIN read_parquet(?) s
                    ON a.SK_ID_CURR = s.SK_ID_CURR
                """,
                [name, str(path)],
            )

        covered, duplicated = con.execute(
            """
            SELECT
                COUNT(*),
                COUNT(*) FILTER (WHERE split_count > 1)
            FROM (
                SELECT
                    SK_ID_CURR,
                    COUNT(*) AS split_count
                FROM split_membership
                GROUP BY SK_ID_CURR
            )
            """
        ).fetchone()

        if duplicated:
            raise AssertionError(
                f"{duplicated} clientes afectados aparecen "
                "en más de un split."
            )

        if covered != expected_clients:
            raise AssertionError(
                f"Cobertura incompleta: {covered} de "
                f"{expected_clients} clientes."
            )

        return {
            "clients_covered": int(covered),
            "clients_in_multiple_splits": int(duplicated),
            "exclusive_and_complete": True,
        }

    finally:
        con.close()


def generate():
    module = load_auditor()
    result = module.audit()

    features = list(BureauFeatureBuilder.FEATURE_COLUMNS)

    if len(features) != 11 or len(set(features)) != 11:
        raise AssertionError(
            "El constructor no contiene 11 características únicas."
        )

    if result["clients_in_bureau_features"] != result["clients"]:
        raise AssertionError(
            "No todos los clientes afectados aparecen "
            "en el checkpoint de características."
        )

    split_validation = validate_splits(
        module,
        result["clients"],
    )

    timestamp = datetime.now(timezone.utc).isoformat()

    evidence = {
        **result,
        "generated_at_utc": timestamp,
        "bureau_feature_columns": features,
        "split_validation": split_validation,
        "limitations": [
            "DAYS_CREDIT_UPDATE posterior indica riesgo potencial.",
            "No hay versiones históricas para reconstruir saldos "
            "y estados al momento de predicción.",
            "La presencia en ML V2 no demuestra fuga efectiva.",
        ],
    }

    feature_rows = "\n".join(
        f"| `{feature}` | Requiere evaluación point-in-time |"
        for feature in features
    )

    split_rows = "\n".join(
        f"| {name} | {count:,} |"
        for name, count in result["splits"].items()
    )

    report = f"""# Fase 10.14 — Disponibilidad temporal de Bureau V2

## Objetivo

Investigar actualizaciones posteriores al momento de referencia
en `bureau`, su presencia en ML V2 y los riesgos potenciales
para las características históricas.

## Criterio de auditoría

Se identifican registros con `DAYS_CREDIT_UPDATE > 0`.

Este criterio detecta actualizaciones posteriores al momento
de referencia, pero no permite reconstruir el contenido
histórico anterior a dichas actualizaciones.

## Resultados

| Indicador | Resultado |
|---|---:|
| Registros identificados | {result['records']:,} |
| Clientes afectados | {result['clients']:,} |
| Créditos afectados | {result['credits']:,} |
| Actualización mínima (días) | +{result['min_update']} |
| Actualización máxima (días) | +{result['max_update']} |
| Créditos activos | {result['active']:,} |
| Créditos cerrados | {result['closed']:,} |
| Créditos con deuda | {result['with_debt']:,} |
| Créditos con días de mora | {result['with_overdue_days']:,} |
| Créditos con saldo vencido | {result['with_overdue_amount']:,} |
| Créditos con vencimiento futuro | {result['future_enddate']:,} |
| Clientes presentes en features Bureau | {result['clients_in_bureau_features']:,} |

## Distribución en ML V2

| Conjunto | Clientes afectados |
|---|---:|
{split_rows}

La distribución se validó sin duplicación entre conjuntos.

- Clientes cubiertos: {split_validation['clients_covered']:,}.
- Clientes presentes en múltiples conjuntos:
  {split_validation['clients_in_multiple_splits']:,}.

La revisión de TEST es exclusivamente de integridad.
No se utiliza para optimizar el modelo ni seleccionar umbrales.

## Características del constructor Bureau

El constructor original define {len(features)} características.

| Característica | Evaluación |
|---|---|
{feature_rows}

La clasificación de estas características es preliminar.
No se ha demostrado individualmente que sus valores
contengan información posterior a la predicción.

## Interpretación

Los registros identificados presentan actualizaciones
posteriores al momento de referencia.

Aunque `DAYS_CREDIT_UPDATE` no participa directamente
en las agregaciones de Bureau, los estados y saldos
pueden depender de información actualizada.

No contamos con versiones históricas que permitan
reconstruir los valores exactos disponibles antes
de dichas actualizaciones.

Por tanto, la evidencia indica **riesgo temporal
potencial**, no Data Leakage confirmado.

## Decisión

**Mantener V2 intacta.**

No se modifican fuentes, características, splits,
checkpoints ni modelos.

Se requiere investigar la disponibilidad point-in-time
de las demás fuentes históricas antes de definir
una nueva política temporal.

## Reproducción

```bash
python scripts/ml/audit_bureau_temporal_availability_v2.py
python scripts/reporting/generate_phase_10_14_documentation.py
python -m pytest -q
```

## Trazabilidad

Generado automáticamente (UTC): `{timestamp}`.

La evidencia estructurada se almacena en el archivo
JSON correspondiente a esta fase.
"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / f"{STEM}.json"
    markdown_path = OUTPUT_DIR / f"{STEM}.md"

    json_path.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    markdown_path.write_text(
        report,
        encoding="utf-8",
    )

    print("[PASS] Auditor ejecutado.")
    print("[PASS] 11 características identificadas.")
    print("[PASS] Cobertura y exclusividad de splits verificadas.")
    print(f"[PASS] JSON: {json_path}")
    print(f"[PASS] Markdown: {markdown_path}")


if __name__ == "__main__":
    generate()
