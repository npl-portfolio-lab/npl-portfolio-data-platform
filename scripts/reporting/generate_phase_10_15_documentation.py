"""Documentación reproducible de la Fase 10.15."""

import importlib.util
import json
from pathlib import Path

from npl_portfolio.core.paths import PROJECT_ROOT


AUDITOR_PATH = (
    PROJECT_ROOT
    / "scripts/ml/audit_previous_application_temporal_v2.py"
)

OUTPUT_DIR = PROJECT_ROOT / "docs/ml"
STEM = "fase_10_15_disponibilidad_temporal_previous_application_v2"


def load_auditor():
    spec = importlib.util.spec_from_file_location(
        "previous_application_temporal_v2",
        AUDITOR_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("No se pudo cargar el auditor.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_evidence(evidence):
    """Verifica consistencia antes de publicar los documentos."""
    total = evidence["total_records"]
    profiles = evidence["temporal_profiles"]

    if evidence["feature_count"] != len(
        evidence["feature_columns"]
    ):
        raise AssertionError(
            "El número de características no coincide."
        )

    if len(set(evidence["feature_columns"])) != evidence["feature_count"]:
        raise AssertionError(
            "Hay características duplicadas."
        )

    categories = (
        "nulls",
        "negatives",
        "zeros",
        "positive_non_sentinel",
        "sentinel_365243",
    )

    for column, profile in profiles.items():
        count = sum(profile[key] for key in categories)

        if count != total:
            raise AssertionError(
                f"Conteo inconsistente en {column}: {count} != {total}"
            )

    decision = profiles["DAYS_DECISION"]

    expected_flag = (
        decision["nulls"] == 0
        and decision["zeros"] == 0
        and decision["positive_non_sentinel"] == 0
        and decision["sentinel_365243"] == 0
    )

    if evidence["decision_strictly_before_reference"] != expected_flag:
        raise AssertionError(
            "La clasificación de DAYS_DECISION es inconsistente."
        )

    return True


def generate():
    auditor = load_auditor()
    evidence = auditor.audit()

    validate_evidence(evidence)

    profiles = evidence["temporal_profiles"]

    temporal_rows = "\n".join(
        (
            f"| `{column}` | "
            f"{profile['nulls']:,} | "
            f"{profile['negatives']:,} | "
            f"{profile['zeros']:,} | "
            f"{profile['positive_non_sentinel']:,} | "
            f"{profile['sentinel_365243']:,} |"
        )
        for column, profile in profiles.items()
    )

    feature_rows = "\n".join(
        f"| `{feature}` |"
        for feature in evidence["feature_columns"]
    )

    decision_status = (
        "Sí"
        if evidence["decision_strictly_before_reference"]
        else "No"
    )

    report = f"""# Fase 10.15 — Disponibilidad temporal de previous_application V2

## Objetivo

Auditar la distribución temporal de `previous_application`
y revisar su relación con las características históricas
utilizadas por ML V2.

La auditoría es de solo lectura y no modifica modelos,
datasets, splits ni checkpoints.

## Fuente

`{evidence['source']}`

Total de registros: **{evidence['total_records']:,}**.

## Perfil temporal

El valor `{evidence['sentinel_value']}` se clasifica
separadamente de las fechas positivas reales.

| Columna | Nulos | Negativos | Ceros | Positivos sin sentinel | Sentinel 365243 |
|---|---:|---:|---:|---:|---:|
{temporal_rows}

## Validación de DAYS_DECISION

Todas las decisiones son estrictamente anteriores
a la referencia: **{decision_status}**.

- Mínimo: `{profiles['DAYS_DECISION']['min']}`.
- Máximo: `{profiles['DAYS_DECISION']['max']}`.
- Positivos reales: {profiles['DAYS_DECISION']['positive_non_sentinel']:,}.
- Valores especiales: {profiles['DAYS_DECISION']['sentinel_365243']:,}.

La ausencia de decisiones posteriores respalda
la consistencia temporal de esta columna,
pero no garantiza que todos los atributos asociados
a cada solicitud estuvieran disponibles históricamente.

## Características utilizadas

El constructor `PreviousApplicationFeatureBuilder`
define **{evidence['feature_count']}** características.

| Característica |
|---|
{feature_rows}

El constructor utiliza `DAYS_DECISION`, estados,
montos y condiciones de las solicitudes.

Las columnas `DAYS_FIRST_DRAWING`,
`DAYS_FIRST_DUE`, `DAYS_LAST_DUE_1ST_VERSION`,
`DAYS_LAST_DUE` y `DAYS_TERMINATION`
no se utilizan directamente en sus agregaciones.

## Hallazgos

1. `DAYS_DECISION` presenta únicamente valores negativos
   en el dataset auditado.
2. `DAYS_LAST_DUE_1ST_VERSION` contiene
   **{profiles['DAYS_LAST_DUE_1ST_VERSION']['positive_non_sentinel']:,}**
   valores positivos distintos de 365243.
3. Existen valores especiales `365243` en otras
   columnas temporales.
4. Las fechas futuras programadas no constituyen
   automáticamente Data Leakage.
5. No hay evidencia suficiente para confirmar
   la disponibilidad point-in-time de todos
   los estados y montos agregados.

## Clasificación del riesgo

**Riesgo temporal pendiente de evaluación;
Data Leakage no confirmado.**

No se recomienda modificar características
ni reentrenar ML V2 basándose únicamente
en estos resultados.

## Reproducción

```bash
python scripts/ml/audit_previous_application_temporal_v2.py
python scripts/reporting/generate_phase_10_15_documentation.py
python -m pytest -q
```

## Decisión

Conservar ML V2 intacta y continuar
la investigación de disponibilidad histórica
de las demás fuentes.
"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / f"{STEM}.json"
    markdown_path = OUTPUT_DIR / f"{STEM}.md"

    json_path.write_text(
        json.dumps(
            evidence,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    markdown_path.write_text(
        report,
        encoding="utf-8",
    )

    print("[PASS] Auditor ejecutado.")
    print("[PASS] Consistencia de evidencia validada.")
    print(
        "[PASS] Características documentadas:",
        evidence["feature_count"],
    )
    print(f"[PASS] JSON: {json_path}")
    print(f"[PASS] Markdown: {markdown_path}")


if __name__ == "__main__":
    generate()
