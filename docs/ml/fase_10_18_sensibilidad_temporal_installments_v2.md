# Fase 10.18 — Sensibilidad temporal de installments_payments V2

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
| Registros | 13,605,401 |
| Clientes | 339,587 |
| Contratos | 997,752 |
| Registros excluidos | 2 |
| Pagos sin fecha | 2,905 |
| Registros con fechas futuras | 0 |

## Sensibilidad

| Indicador | Resultado |
|---|---:|
| Clientes comparables | 339,587 |
| Clientes sin historial restringido | 0 |
| Clientes con cantidad de registros modificada | 2 |
| Características evaluadas | 30 |
| Características modificadas | 16 |

## Características modificadas

| Característica | Clientes afectados |
|---|---:|
| `INST_RECORD_COUNT` | 2 |
| `INST_AVG_VERSION` | 2 |
| `INST_AVG_NUMBER` | 2 |
| `INST_MAX_NUMBER` | 2 |
| `INST_AVG_INSTALMENT_AMOUNT` | 2 |
| `INST_TOTAL_INSTALMENT_AMOUNT` | 2 |
| `INST_AVG_PAYMENT_AMOUNT` | 2 |
| `INST_TOTAL_PAYMENT_AMOUNT` | 2 |
| `INST_AVG_PAYMENT_DELAY` | 2 |
| `INST_ON_TIME_OR_EARLY_COUNT` | 2 |
| `INST_AVG_PAYMENT_DIFFERENCE` | 2 |
| `INST_RECENT_SCHEDULED_DAY` | 2 |
| `INST_RECENT_PAYMENT_DAY` | 2 |
| `INST_UNDERPAYMENT_RATE` | 1 |
| `INST_OVERPAYMENT_RATE` | 1 |
| `INST_PAYMENT_RATIO` | 2 |

## Interpretación

Los registros excluidos pueden modificar agregaciones,
promedios y tasas, incluso cuando representan pagos completos.

La sensibilidad temporal no demuestra por sí sola Data Leakage.

**Clasificación:** Sensibilidad temporal identificada; Data Leakage no confirmado.

## Limitaciones

- El corte -2 es una prueba de sensibilidad.
- No demuestra disponibilidad punto-en-tiempo.
- Los pagos sin fecha se conservan si la cuota cumple el corte temporal.
- Las diferencias se calculan solo para clientes con historial en ambos escenarios.
- Las comparaciones usan IS DISTINCT FROM sin tolerancia numérica.
- No modifica ni reentrena modelos ML V2.

## Integridad del experimento

- No se modificaron los datasets.
- No se modificaron ni reentrenaron los modelos ML V2.
- No se publica evidencia JSON.
