# Fase 10.5 — Auditoría de validación independiente

## Objetivo

Verificar la separación de TRAIN y VALIDATION y determinar si existe un conjunto TEST independiente.

## Resultados

| Indicador | TRAIN | VALIDATION |
|---|---:|---:|
| Registros | 246,008 | 61,503 |
| Clientes únicos | 246,008 | 61,503 |
| Positivos | 19,860 | 4,965 |
| Negativos | 226,148 | 56,538 |

- Clientes compartidos: **0**
- Comprobaciones de partición superadas: **True**
- Archivo TEST etiquetado detectado: **False**

## Comprobaciones

- train_unique_ids: PASS
- validation_unique_ids: PASS
- no_overlap: PASS
- no_nulls: PASS
- binary_target: PASS

## Conclusión

Las comprobaciones verifican la integridad y separación de las particiones existentes, pero no constituyen una validación independiente final.

## Limitaciones

- No se ha verificado un TEST etiquetado independiente.
- VALIDATION ya se utilizo para comparar modelos y thresholds.
- Pendiente auditar internamente PreprocessingContractBuilder.
- Pendiente auditar posibles fugas en feature engineering.

## Próximos pasos

Diseñar un nuevo protocolo experimental con TEST etiquetado y aislado desde el inicio. Conservar los artefactos de las fases anteriores.

## Artefactos

- `data/artifacts/ml/independent_validation_audit.json`
- `scripts/ml/audit_independent_validation.py`
