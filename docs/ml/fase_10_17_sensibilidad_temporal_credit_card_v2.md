# Fase 10.17 — Sensibilidad temporal credit_card_balance V2

## Objetivo

Evaluar el historial temporal de tarjetas de crédito y medir la sensibilidad de características seleccionadas al excluir el mes -1.

## Metodología

- Fuente: `credit_card_balance.parquet`.
- Escenario original: historial completo.
- Escenario contrafactual: `MONTHS_BALANCE <= -2`.
- Comparación de características: solo clientes presentes en ambos escenarios.
- Clientes sin historial contrafactual: reportados por separado.
- DuckDB configurado con un hilo para reproducibilidad.
- Análisis de solo lectura; ML V2 permanece intacto.

## Perfil temporal

| Indicador | Valor |
|---|---:|
| records | 3,840,312 |
| clients | 103,558 |
| contracts | 104,307 |
| null_months | 0 |
| nonhistorical_months | 0 |
| recent_records | 62,356 |
| recent_clients | 61,885 |
| oldest_month | -96 |
| newest_month | -1 |

## Sensibilidad temporal

| Indicador | Clientes |
|---|---:|
| total_clients | 103,558 |
| comparable_clients | 103,136 |
| clients_without_history | 422 |
| changed_record_count | 61,463 |
| changed_max_dpd | 419 |
| changed_avg_dpd | 14,323 |
| changed_dpd_rate | 14,323 |
| changed_avg_balance | 47,923 |
| changed_max_balance | 5,926 |
| changed_avg_credit_limit | 35,926 |
| changed_avg_utilization | 37,843 |
| changed_total_payment | 30,147 |

## Interpretación

Los conteos `changed_*` representan clientes comparables con diferencias entre los dos escenarios.

La exclusión del mes -1 es un experimento de sensibilidad, no una corrección automática de leakage.

## Limitaciones

- No demuestra disponibilidad punto-en-tiempo.
- No evalúa todas las características del constructor.
- No modifica ni reentrena modelos ML V2.
- Las diferencias de características se calculan solo para clientes con historial en ambos escenarios.
- Las comparaciones de promedios usan IS DISTINCT FROM sin tolerancia numérica.

## Conclusión

Sensibilidad temporal identificada; Data Leakage no confirmado

La ausencia de meses posteriores a la referencia no demuestra por sí sola disponibilidad punto-en-tiempo. Los cambios observados no prueban Data Leakage.

## Reproducción

```bash
python scripts/ml/audit_credit_card_temporal_sensitivity_v2.py
python -m scripts.reporting.generate_phase_10_17_documentation
python -m pytest -q
```
