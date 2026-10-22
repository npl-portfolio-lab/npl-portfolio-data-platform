"""Genera documentación Markdown de la Fase 10.18."""

from pathlib import Path

from npl_portfolio.core.paths import PROJECT_ROOT
from scripts.ml.audit_installments_temporal_sensitivity_v2 import audit


OUTPUT = (
    PROJECT_ROOT
    / "docs/ml/fase_10_18_sensibilidad_temporal_installments_v2.md"
)


def generate(output: Path = OUTPUT) -> Path:
    result = audit()
    profile = result["temporal_profile"]
    sensitivity = result["sensitivity"]

    feature_rows = "\n".join(
        f"| `{name}` | {count:,} |"
        for name, count in sensitivity["changed_features"].items()
    )

    content = f"""# Fase 10.18 — Sensibilidad temporal de installments_payments V2

## Objetivo

Evaluar el impacto de excluir registros próximos a la fecha
de referencia sobre las características construidas a nivel cliente.

## Metodología

Se compara el constructor original con una versión restringida
a `DAYS_INSTALMENT <= -2` y `DAYS_ENTRY_PAYMENT <= -2`,
permitiendo fechas de pago nulas.

La comparación reutiliza el SQL del constructor existente.
Se ejecuta con DuckDB y un único hilo.

## Perfil temporal

| Indicador | Resultado |
|---|---:|
| Registros | {profile["records"]:,} |
| Clientes | {profile["clients"]:,} |
| Contratos | {profile["contracts"]:,} |
| Registros excluidos | {profile["excluded_records"]:,} |
| Pagos sin fecha | {profile["null_payment_dates"]:,} |
| Registros con fechas futuras | {profile["future_records"]:,} |

## Sensibilidad

| Indicador | Resultado |
|---|---:|
| Clientes comparables | {sensitivity["comparable_clients"]:,} |
| Clientes sin historial restringido | {sensitivity["clients_without_history"]:,} |
| Clientes con cantidad de registros modificada | {sensitivity["changed_record_count"]:,} |
| Características evaluadas | {sensitivity["evaluated_features"]} |
| Características modificadas | {sensitivity["changed_feature_count"]} |

## Características modificadas

| Característica | Clientes afectados |
|---|---:|
{feature_rows}

## Interpretación

Los registros excluidos pueden modificar agregaciones,
promedios y tasas, incluso cuando representan pagos completos.

La sensibilidad temporal no demuestra por sí sola Data Leakage.

**Clasificación:** {result["risk_classification"]}.

## Limitaciones

""" + "\n".join(
        f"- {item}" for item in result["limitations"]
    ) + """

## Integridad del experimento

- No se modificaron los datasets.
- No se modificaron ni reentrenaron los modelos ML V2.
- No se publica evidencia JSON.
"""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    return output


if __name__ == "__main__":
    print(generate())
