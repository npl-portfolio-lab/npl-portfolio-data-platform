# Fase 10.12 — Auditoría temporal e integridad histórica V2

> Documento generado automáticamente. No editar manualmente.

**Fecha de generación:** 2026-10-08 23:39 UTC

## 1. Objetivo

Verificar integridad histórica, correspondencia de características y controles temporales de los datasets V2.

La auditoría es de solo lectura y no reentrena modelos.

## 2. Integridad de datasets

| Dataset | Clientes únicos | Estado |
|---|---:|---|
| TRAIN | 307,511 | PASS |
| TEST | 48,744 | PASS |

## 3. Integridad de características históricas

| Fuente | Filas | Características | Con historial | Sin historial | Estado |
|---|---:|---:|---:|---:|---|
| bureau | 305,811 | 11 | 263,491 | 44,020 | PASS |
| bureau_balance | 134,542 | 19 | 92,231 | 215,280 | PASS |
| previous_application | 338,857 | 17 | 291,057 | 16,454 | PASS |
| pos_cash | 337,252 | 16 | 289,444 | 18,067 | PASS |
| credit_card | 103,558 | 28 | 86,905 | 220,606 | PASS |
| installments | 339,587 | 30 | 291,643 | 15,868 | PASS |

**Total de características verificadas:** 121

Las características se compararon para clientes presentes en ambas tablas. También se comprobaron las banderas de historial y la unicidad de las claves.

## 4. Auditoría temporal

| Fuente y columna | Mínimo | Máximo | Valores futuros |
|---|---:|---:|---:|
| installments_payments.DAYS_INSTALMENT | -2922.0 | -1.0 | 0 |
| installments_payments.DAYS_ENTRY_PAYMENT | -4921.0 | -1.0 | 0 |
| credit_card_balance.MONTHS_BALANCE | -96 | -1 | 0 |
| POS_CASH_balance.MONTHS_BALANCE | -96 | -1 | 0 |
| bureau_balance.MONTHS_BALANCE | -96 | 0 | 0 |
| previous_application.DAYS_DECISION | -2922 | -1 | 0 |
| bureau.DAYS_CREDIT | -2922 | 0 | 0 |

## 5. Advertencias

- installments_payments.DAYS_ENTRY_PAYMENT: 2,905 fechas nulas
- bureau_balance.MONTHS_BALANCE: 610,965 registros en periodo cero
- bureau.DAYS_CREDIT: 25 registros en periodo cero

## 6. Limitaciones

- La ausencia de fechas positivas no demuestra ausencia completa de fuga temporal.
- La disponibilidad de cada variable en el instante de predicción no está completamente demostrada.
- Los constructores históricos examinados no tienen filtros temporales explícitos.
- La integridad de las características no demuestra por sí sola reproducibilidad completa desde las fuentes.
- Esta auditoría no evalúa métricas predictivas sobre TEST.

## 7. Conclusión

Los controles ejecutados no detectaron inconsistencias en las características históricas verificadas ni fechas positivas en las siete columnas temporales auditadas.

La ausencia completa de *data leakage* permanece sin certificar debido a las limitaciones temporales documentadas.

## 8. Reproducción

```bash
python scripts/ml/audit_temporal_leakage_v2.py
python scripts/ml/audit_historical_feature_integrity_v2.py
python scripts/reporting/generate_phase_10_12_documentation.py --overwrite
python -m pytest -q
```
