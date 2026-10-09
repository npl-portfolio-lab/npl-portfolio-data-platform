# Fase 10.14 — Disponibilidad temporal de Bureau V2

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
| Registros identificados | 17 |
| Clientes afectados | 17 |
| Créditos afectados | 17 |
| Actualización mínima (días) | +10 |
| Actualización máxima (días) | +372 |
| Créditos activos | 17 |
| Créditos cerrados | 0 |
| Créditos con deuda | 17 |
| Créditos con días de mora | 0 |
| Créditos con saldo vencido | 0 |
| Créditos con vencimiento futuro | 17 |
| Clientes presentes en features Bureau | 17 |

## Distribución en ML V2

| Conjunto | Clientes afectados |
|---|---:|
| TRAIN | 11 |
| VALIDATION | 4 |
| TEST | 2 |

La distribución se validó sin duplicación entre conjuntos.

- Clientes cubiertos: 17.
- Clientes presentes en múltiples conjuntos:
  0.

La revisión de TEST es exclusivamente de integridad.
No se utiliza para optimizar el modelo ni seleccionar umbrales.

## Características del constructor Bureau

El constructor original define 11 características.

| Característica | Evaluación |
|---|---|
| `BUREAU_CREDIT_COUNT` | Requiere evaluación point-in-time |
| `BUREAU_ACTIVE_COUNT` | Requiere evaluación point-in-time |
| `BUREAU_CLOSED_COUNT` | Requiere evaluación point-in-time |
| `BUREAU_DAYS_OVERDUE_COUNT` | Requiere evaluación point-in-time |
| `BUREAU_AMOUNT_OVERDUE_COUNT` | Requiere evaluación point-in-time |
| `BUREAU_AVG_DAYS_CREDIT` | Requiere evaluación point-in-time |
| `BUREAU_MIN_DAYS_CREDIT` | Requiere evaluación point-in-time |
| `BUREAU_TOTAL_CREDIT` | Requiere evaluación point-in-time |
| `BUREAU_TOTAL_DEBT` | Requiere evaluación point-in-time |
| `BUREAU_TOTAL_OVERDUE` | Requiere evaluación point-in-time |
| `BUREAU_MAX_OVERDUE_DAYS` | Requiere evaluación point-in-time |

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

Generado automáticamente (UTC): `2026-10-09T00:25:21.374353+00:00`.

La evidencia estructurada se almacena en el archivo
JSON correspondiente a esta fase.
