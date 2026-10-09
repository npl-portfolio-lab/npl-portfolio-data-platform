# Fase 10.13 — Sensibilidad temporal de Bureau Balance

## Objetivo

Evaluar el impacto de excluir los registros con
`MONTHS_BALANCE = 0` de las características históricas
de Bureau Balance, sin modificar los datos ni los modelos V2.

## Escenarios evaluados

- **Original:** V2 sin corte temporal.
- **Conservador:** `MONTHS_BALANCE < 0`.

Ambos escenarios utilizan la misma lógica de agregación
del constructor original.

## Validaciones

- Reproducción del checkpoint: **APROBADA**.
- Características comparadas: **19**.
- Clientes evaluados: **134,542**.
- Clientes sin historial después del corte: **243**.

## Cambios por característica

Los conteos incluyen diferencias entre valores y valores nulos.

| Característica | Clientes con cambios | Porcentaje |
|---|---:|---:|
| `BB_RECORD_COUNT` | 129,829 | 96.50% |
| `BB_BUREAU_CREDIT_COUNT` | 4,486 | 3.33% |
| `BB_OLDEST_MONTH` | 243 | 0.18% |
| `BB_RECENT_MONTH` | 129,829 | 96.50% |
| `BB_STATUS_0_COUNT` | 73,671 | 54.76% |
| `BB_STATUS_1_COUNT` | 4,608 | 3.42% |
| `BB_STATUS_2_COUNT` | 497 | 0.37% |
| `BB_STATUS_3_COUNT` | 321 | 0.24% |
| `BB_STATUS_4_COUNT` | 302 | 0.22% |
| `BB_STATUS_5_COUNT` | 986 | 0.73% |
| `BB_STATUS_C_COUNT` | 102,328 | 76.06% |
| `BB_STATUS_X_COUNT` | 64,056 | 47.61% |
| `BB_DPD_RECORD_COUNT` | 5,609 | 4.17% |
| `BB_SEVERE_DPD_RECORD_COUNT` | 1,107 | 0.82% |
| `BB_MAX_STATUS_LEVEL` | 1,145 | 0.85% |
| `BB_DPD_RATE` | 47,833 | 35.55% |
| `BB_SEVERE_DPD_RATE` | 5,032 | 3.74% |
| `BB_CLOSED_STATUS_RATE` | 106,311 | 79.02% |
| `BB_UNKNOWN_STATUS_RATE` | 106,416 | 79.10% |

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

Generado automáticamente: `2026-10-09T00:07:14.030519+00:00`.

Fuente: auditor de sensibilidad temporal de la Fase 10.13.
