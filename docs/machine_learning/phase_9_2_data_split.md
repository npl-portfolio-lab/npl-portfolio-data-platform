# Fase 9.2 — División estratificada TRAIN / VALIDATION

**Informe generado:** 2026-10-08 17:37 UTC

## 1. Objetivo

Separar los clientes en conjuntos independientes para entrenar
y evaluar los modelos, manteniendo una distribución comparable
de la variable objetivo `TARGET`.

## 2. Metodología

Se utiliza `StratifiedDatasetSplitter` con:

- TRAIN: 80 %.
- VALIDATION: 20 %.
- Semilla aleatoria: `42`.
- Identificador de cliente: `SK_ID_CURR`.

DuckDB permite materializar las particiones en Parquet sin
cargar todas las variables en memoria.

## 3. Resultados

| Indicador | SOURCE | TRAIN | VALIDATION |
|---|---:|---:|---:|
| Registros | 307,511 | 246,008 | 61,503 |
| Clientes únicos | 307,511 | 246,008 | 61,503 |
| Positivos TARGET=1 | 24,825 | 19,860 | 4,965 |
| Tasa positiva | 8.0729% | 8.0729% | 8.0728% |

Porcentaje real de VALIDATION: **20.00%**.

## 4. Validaciones

| Validación | Resultado |
|---|---|
| Cobertura completa | PASS |
| IDs únicos en TRAIN | PASS |
| IDs únicos en VALIDATION | PASS |
| Sin clientes compartidos | PASS |
| Schemas idénticos | PASS |
| Estratificación conservada | PASS |

Clientes compartidos entre TRAIN y VALIDATION: **0**.

## 5. Conclusión

Las particiones cumplen las verificaciones de cobertura,
unicidad, separación de clientes, consistencia de esquema
y conservación de la distribución de TARGET.

Esta validación no descarta por sí sola otras formas
de fuga de información en las variables predictoras.
