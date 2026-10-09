# Fase 10.15 — Disponibilidad temporal de previous_application V2

## Objetivo

Auditar la distribución temporal de `previous_application`
y revisar su relación con las características históricas
utilizadas por ML V2.

La auditoría es de solo lectura y no modifica modelos,
datasets, splits ni checkpoints.

## Fuente

`data/processed/home_credit/previous_application.parquet`

Total de registros: **1,670,214**.

## Perfil temporal

El valor `365243` se clasifica
separadamente de las fechas positivas reales.

| Columna | Nulos | Negativos | Ceros | Positivos sin sentinel | Sentinel 365243 |
|---|---:|---:|---:|---:|---:|
| `DAYS_DECISION` | 0 | 1,670,214 | 0 | 0 | 0 |
| `DAYS_FIRST_DRAWING` | 673,065 | 62,705 | 0 | 0 | 934,444 |
| `DAYS_FIRST_DUE` | 673,065 | 956,504 | 0 | 0 | 40,645 |
| `DAYS_LAST_DUE_1ST_VERSION` | 673,065 | 678,188 | 705 | 224,392 | 93,864 |
| `DAYS_LAST_DUE` | 673,065 | 785,928 | 0 | 0 | 211,221 |
| `DAYS_TERMINATION` | 673,065 | 771,236 | 0 | 0 | 225,913 |

## Validación de DAYS_DECISION

Todas las decisiones son estrictamente anteriores
a la referencia: **Sí**.

- Mínimo: `-2922`.
- Máximo: `-1`.
- Positivos reales: 0.
- Valores especiales: 0.

La ausencia de decisiones posteriores respalda
la consistencia temporal de esta columna,
pero no garantiza que todos los atributos asociados
a cada solicitud estuvieran disponibles históricamente.

## Características utilizadas

El constructor `PreviousApplicationFeatureBuilder`
define **17** características.

| Característica |
|---|
| `PREV_APPLICATION_COUNT` |
| `PREV_APPROVED_COUNT` |
| `PREV_REFUSED_COUNT` |
| `PREV_CANCELED_COUNT` |
| `PREV_UNUSED_COUNT` |
| `PREV_AVG_APPLICATION_AMOUNT` |
| `PREV_AVG_CREDIT_AMOUNT` |
| `PREV_TOTAL_APPLICATION_AMOUNT` |
| `PREV_TOTAL_CREDIT_AMOUNT` |
| `PREV_AVG_ANNUITY` |
| `PREV_AVG_PAYMENT_COUNT` |
| `PREV_AVG_CREDIT_DIFFERENCE` |
| `PREV_AVG_DAYS_DECISION` |
| `PREV_LAST_DECISION_DAYS` |
| `PREV_APPROVAL_RATE` |
| `PREV_REFUSAL_RATE` |
| `PREV_CANCELLATION_RATE` |

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
   **224,392**
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
