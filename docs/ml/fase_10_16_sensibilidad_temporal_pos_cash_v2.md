# Fase 10.16 — Sensibilidad temporal POS_CASH_balance V2

## Objetivo

Evaluar la distribución temporal de POS_CASH_balance y medir la sensibilidad de algunas características al excluir el mes -1.

## Fuente y metodología

- Fuente: `data/processed/home_credit/POS_CASH_balance.parquet`
- Referencia temporal: `MONTHS_BALANCE`.
- Escenario original: todos los registros disponibles.
- Escenario contrafactual: `MONTHS_BALANCE <= -2`.
- Unidad de comparación: cliente (`SK_ID_CURR`).
- Auditoría de solo lectura; no modifica ML V2.

## Perfil temporal

| Indicador | Valor |
|---|---:|
| Registros | 10,001,358 |
| Clientes | 337,252 |
| Contratos | 936,325 |
| Meses nulos | 0 |
| Meses >= 0 | 0 |
| Registros del mes -1 | 94,908 |
| Clientes del mes -1 | 79,744 |
| Mes más antiguo | -96 |
| Mes más reciente | -1 |

## Sensibilidad al excluir el mes -1

| Indicador | Clientes |
|---|---:|
| Clientes originales | 337,252 |
| Sin historial contrafactual | 127 |
| POS_RECORD_COUNT diferente | 79,744 |
| POS_MAX_DPD diferente | 1,072 |
| POS_AVG_DPD diferente | 15,820 |
| POS_DPD_RATE diferente | 15,798 |

## Alcance y limitaciones

- La comparación cubre cuatro métricas seleccionadas; no todas las características del constructor.
- Las diferencias incluyen clientes sin historial en el escenario contrafactual.
- `MONTHS_BALANCE < 0` indica meses anteriores a la referencia del dataset, pero no demuestra por sí solo disponibilidad operacional punto-en-tiempo.
- La exclusión del mes -1 es un análisis de sensibilidad, no una corrección automática de fuga de información.
- No se evalúa el impacto en predicciones ni métricas del modelo ML V2.

## Conclusión

Sensibilidad temporal identificada; Data Leakage no confirmado

No se confirma Data Leakage con esta evidencia. Se requiere una validación adicional de disponibilidad punto-en-tiempo para afirmar ausencia de fuga.

## Reproducción

```bash
python scripts/ml/audit_pos_cash_temporal_sensitivity_v2.py
python -m scripts.reporting.generate_phase_10_16_documentation
python -m pytest -q
```
