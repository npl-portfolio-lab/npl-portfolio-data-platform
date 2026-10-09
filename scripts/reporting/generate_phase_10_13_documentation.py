"""Genera evidencia reproducible de la Fase 10.13."""

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

from npl_portfolio.core.paths import PROJECT_ROOT


AUDITOR = (
    PROJECT_ROOT
    / "scripts/ml/audit_bureau_balance_temporal_sensitivity_v2.py"
)

DOCS_DIR = PROJECT_ROOT / "docs/ml"
REPORT = DOCS_DIR / "fase_10_13_sensibilidad_temporal_bureau_balance_v2.md"
JSON_REPORT = DOCS_DIR / "fase_10_13_sensibilidad_temporal_bureau_balance_v2.json"


def load_auditor():
    spec = importlib.util.spec_from_file_location(
        "bureau_balance_temporal_audit",
        AUDITOR,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("No se pudo cargar el auditor.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.run_audit


def generate():
    audit = load_auditor()
    result = audit()

    if not result["checkpoint_reproduced"]:
        raise AssertionError("El checkpoint no fue reproducido.")

    if result["feature_count"] != 19:
        raise AssertionError("Cantidad inesperada de características.")

    clients = result["clients_evaluated"]
    changes = result["feature_changes"]

    if len(changes) != 19:
        raise AssertionError("Resultados incompletos.")

    generated_at = datetime.now(timezone.utc).isoformat()

    evidence = {
        "generated_at_utc": generated_at,
        **result,
    }

    rows = []

    for feature, affected in changes.items():
        percentage = affected / clients * 100 if clients else 0

        rows.append(
            f"| `{feature}` | {affected:,} | {percentage:.2f}% |"
        )

    table = "\n".join(rows)

    document = f"""# Fase 10.13 — Sensibilidad temporal de Bureau Balance

## Objetivo

Evaluar el impacto de excluir los registros con
`MONTHS_BALANCE = 0` de las características históricas
de Bureau Balance, sin modificar los datos ni los modelos V2.

## Escenarios evaluados

- **Original:** {result['scenario_original']}.
- **Conservador:** `{result['scenario_conservative']}`.

Ambos escenarios utilizan la misma lógica de agregación
del constructor original.

## Validaciones

- Reproducción del checkpoint: **APROBADA**.
- Características comparadas: **{result['feature_count']}**.
- Clientes evaluados: **{clients:,}**.
- Clientes sin historial después del corte: **{result['clients_without_history']:,}**.

## Cambios por característica

Los conteos incluyen diferencias entre valores y valores nulos.

| Característica | Clientes con cambios | Porcentaje |
|---|---:|---:|
{table}

Los porcentajes utilizan como denominador los clientes
evaluados en la auditoría, no exclusivamente los clientes
del conjunto de entrenamiento del modelo.

## Interpretación

La exclusión del período cero modifica características
históricas de Bureau Balance, incluyendo conteos,
estados de morosidad y tasas derivadas.

Una variación en una tasa puede deberse a cambios en el
numerador, el denominador o ambos.

Estos resultados constituyen un análisis de sensibilidad.
No demuestran por sí solos que exista Data Leakage ni
que el corte conservador sea temporalmente correcto.

Para decidir una política point-in-time definitiva,
se necesita verificar la disponibilidad real de los
registros respecto al momento de predicción.

## Decisión de la fase

**Mantener V2 sin modificaciones.**

No se ejecuta reentrenamiento ni se alteran
checkpoints, datasets o modelos existentes.

## Reproducción

```bash
python scripts/ml/audit_bureau_balance_temporal_sensitivity_v2.py
python scripts/reporting/generate_phase_10_13_documentation.py
python -m pytest -q
```

## Trazabilidad

Generado automáticamente: `{generated_at}`.

Fuente: auditor de sensibilidad temporal de la Fase 10.13.
"""

    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    JSON_REPORT.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    REPORT.write_text(document, encoding="utf-8")

    print(f"[PASS] JSON generado: {JSON_REPORT}")
    print(f"[PASS] Markdown generado: {REPORT}")


if __name__ == "__main__":
    generate()
