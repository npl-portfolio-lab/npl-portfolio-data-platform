# Fase 10.19 — Matriz consolidada de riesgo temporal ML V2

## Objetivo

Consolidar la evidencia de las auditorías temporales 10.13 a 10.18 sin modificar el pipeline ML V2.

## Matriz de fuentes

| Fuente | Fase | Clasificación |
|---|---|---|
| Bureau Balance | 10.13 | Requiere evaluación punto-en-tiempo |
| Bureau | 10.14 | Riesgo temporal potencial; Data Leakage no confirmado |
| Previous Application | 10.15 | Requiere evaluación punto-en-tiempo |
| POS CASH | 10.16 | Sensibilidad temporal identificada; Data Leakage no confirmado |
| Credit Card | 10.17 | Sensibilidad temporal identificada; Data Leakage no confirmado |
| Installments | 10.18 | Sensibilidad temporal identificada; Data Leakage no confirmado |

## Distribución de variables del contrato ML V2

| Fuente | Variables numéricas |
|---|---:|
| Bureau | 12 |
| Bureau Balance | 20 |
| Previous Application | 18 |
| POS CASH | 17 |
| Credit Card | 29 |
| Installments | 31 |
| Application | 111 |
| **Total** | **238** |

Entradas categóricas: 16.

## Evidencia por fuente

### Bureau Balance — Fase 10.13

- **scenario_original:** V2 sin corte temporal
- **scenario_conservative:** MONTHS_BALANCE < 0
- **clients_evaluated:** 134,542
- **clients_without_history:** 243
- **feature_count:** 19
- **checkpoint_reproduced:** True
#### feature_changes

| Indicador | Resultado |
|---|---:|
| `BB_RECORD_COUNT` | 129,829 |
| `BB_BUREAU_CREDIT_COUNT` | 4,486 |
| `BB_OLDEST_MONTH` | 243 |
| `BB_RECENT_MONTH` | 129,829 |
| `BB_STATUS_0_COUNT` | 73,671 |
| `BB_STATUS_1_COUNT` | 4,608 |
| `BB_STATUS_2_COUNT` | 497 |
| `BB_STATUS_3_COUNT` | 321 |
| `BB_STATUS_4_COUNT` | 302 |
| `BB_STATUS_5_COUNT` | 986 |
| `BB_STATUS_C_COUNT` | 102,328 |
| `BB_STATUS_X_COUNT` | 64,056 |
| `BB_DPD_RECORD_COUNT` | 5,609 |
| `BB_SEVERE_DPD_RECORD_COUNT` | 1,107 |
| `BB_MAX_STATUS_LEVEL` | 1,145 |
| `BB_DPD_RATE` | 47,833 |
| `BB_SEVERE_DPD_RATE` | 5,032 |
| `BB_CLOSED_STATUS_RATE` | 106,311 |
| `BB_UNKNOWN_STATUS_RATE` | 106,416 |


### Bureau — Fase 10.14

- **records:** 17
- **clients:** 17
- **credits:** 17
- **min_update:** 10
- **max_update:** 372
- **active:** 17
- **closed:** 0
- **with_debt:** 17
- **with_overdue_days:** 0
- **with_overdue_amount:** 0
- **future_enddate:** 17
- **clients_in_bureau_features:** 17
#### splits

| Indicador | Resultado |
|---|---:|
| `TRAIN` | 11 |
| `VALIDATION` | 4 |
| `TEST` | 2 |


### Previous Application — Fase 10.15

- **total_records:** 1,670,214
- **sentinel_value:** 365,243
#### temporal_profiles

| Indicador | Resultado |
|---|---:|

##### DAYS_DECISION

| Característica | Resultado |
|---|---:|
| `nulls` | 0 |
| `negatives` | 1,670,214 |
| `zeros` | 0 |
| `positive_non_sentinel` | 0 |
| `sentinel_365243` | 0 |
| `min` | -2,922 |
| `max` | -1 |

##### DAYS_FIRST_DRAWING

| Característica | Resultado |
|---|---:|
| `nulls` | 673,065 |
| `negatives` | 62,705 |
| `zeros` | 0 |
| `positive_non_sentinel` | 0 |
| `sentinel_365243` | 934,444 |
| `min` | -2922.0 |
| `max` | 365243.0 |

##### DAYS_FIRST_DUE

| Característica | Resultado |
|---|---:|
| `nulls` | 673,065 |
| `negatives` | 956,504 |
| `zeros` | 0 |
| `positive_non_sentinel` | 0 |
| `sentinel_365243` | 40,645 |
| `min` | -2892.0 |
| `max` | 365243.0 |

##### DAYS_LAST_DUE_1ST_VERSION

| Característica | Resultado |
|---|---:|
| `nulls` | 673,065 |
| `negatives` | 678,188 |
| `zeros` | 705 |
| `positive_non_sentinel` | 224,392 |
| `sentinel_365243` | 93,864 |
| `min` | -2801.0 |
| `max` | 365243.0 |

##### DAYS_LAST_DUE

| Característica | Resultado |
|---|---:|
| `nulls` | 673,065 |
| `negatives` | 785,928 |
| `zeros` | 0 |
| `positive_non_sentinel` | 0 |
| `sentinel_365243` | 211,221 |
| `min` | -2889.0 |
| `max` | 365243.0 |

##### DAYS_TERMINATION

| Característica | Resultado |
|---|---:|
| `nulls` | 673,065 |
| `negatives` | 771,236 |
| `zeros` | 0 |
| `positive_non_sentinel` | 0 |
| `sentinel_365243` | 225,913 |
| `min` | -2874.0 |
| `max` | 365243.0 |

- **decision_strictly_before_reference:** True
- **feature_count:** 17
- **interpretation:** Fechas futuras y valores especiales identificados; no constituyen por sí solos Data Leakage confirmado.

### POS CASH — Fase 10.16

- **cutoff_month:** -2
#### temporal_profile

| Indicador | Resultado |
|---|---:|
| `records` | 10,001,358 |
| `clients` | 337,252 |
| `contracts` | 936,325 |
| `null_months` | 0 |
| `nonhistorical_months` | 0 |
| `recent_records` | 94,908 |
| `recent_clients` | 79,744 |
| `oldest_month` | -96 |
| `newest_month` | -1 |

#### sensitivity

| Indicador | Resultado |
|---|---:|
| `total_clients` | 337,252 |
| `clients_without_history` | 127 |
| `changed_record_count` | 79,744 |
| `changed_max_dpd` | 1,072 |
| `changed_avg_dpd` | 15,820 |
| `changed_dpd_rate` | 15,798 |


### Credit Card — Fase 10.17

- **cutoff_month:** -2
#### temporal_profile

| Indicador | Resultado |
|---|---:|
| `records` | 3,840,312 |
| `clients` | 103,558 |
| `contracts` | 104,307 |
| `null_months` | 0 |
| `nonhistorical_months` | 0 |
| `recent_records` | 62,356 |
| `recent_clients` | 61,885 |
| `oldest_month` | -96 |
| `newest_month` | -1 |

#### sensitivity

| Indicador | Resultado |
|---|---:|
| `total_clients` | 103,558 |
| `comparable_clients` | 103,136 |
| `clients_without_history` | 422 |
| `changed_record_count` | 61,463 |
| `changed_max_dpd` | 419 |
| `changed_avg_dpd` | 14,323 |
| `changed_dpd_rate` | 14,323 |
| `changed_avg_balance` | 47,923 |
| `changed_max_balance` | 5,926 |
| `changed_avg_credit_limit` | 35,926 |
| `changed_avg_utilization` | 37,843 |
| `changed_total_payment` | 30,147 |


### Installments — Fase 10.18

- **cutoff_day:** -2
#### temporal_profile

| Indicador | Resultado |
|---|---:|
| `records` | 13,605,401 |
| `clients` | 339,587 |
| `contracts` | 997,752 |
| `oldest_scheduled_day` | -2922.0 |
| `newest_scheduled_day` | -1.0 |
| `oldest_payment_day` | -4921.0 |
| `newest_payment_day` | -1.0 |
| `null_scheduled_dates` | 0 |
| `null_payment_dates` | 2,905 |
| `future_records` | 0 |
| `recent_scheduled_records` | 2 |
| `recent_payment_records` | 2 |
| `restricted_records` | 13,605,399 |
| `excluded_records` | 2 |

#### sensitivity

| Indicador | Resultado |
|---|---:|
| `total_clients` | 339,587 |
| `comparable_clients` | 339,587 |
| `clients_without_history` | 0 |
| `changed_record_count` | 2 |
| `evaluated_features` | 30 |
| `changed_feature_count` | 16 |

##### changed_features

| Característica | Resultado |
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


## Interpretación consolidada

La sensibilidad temporal no demuestra por sí sola Data Leakage. Se requiere validar la disponibilidad punto-en-tiempo de las características.

## Integridad del análisis

- Evidencia obtenida de los auditores existentes.
- No se modifican datasets ni modelos ML V2.
- No se reentrenan modelos.
- No se publican archivos JSON.
- La sensibilidad no equivale a leakage confirmado.

## Reproducción

```bash
python -m scripts.reporting.generate_phase_10_19_documentation
```
