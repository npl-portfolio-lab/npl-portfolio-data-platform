# Fase 10.6 — Protocolo experimental V2

## 1. Objetivo

Construir un protocolo experimental reproducible con particiones
TRAIN, VALIDATION y TEST, manteniendo separados los artefactos
de experimentos anteriores.

## 2. Origen de datos

- Fuente: `data/processed/features/client_features_train.parquet`
- Total de registros: 307.511
- Identificador: `SK_ID_CURR`
- Variable objetivo: `TARGET`
- Semilla aleatoria: 42

En Home Credit, `TARGET=1` representa dificultad de pago;
no equivale directamente a recuperación de cartera castigada.

## 3. Distribución

| Partición | Registros | TARGET=1 | TARGET=0 | Lotes |
|---|---:|---:|---:|---:|
| TRAIN | 196.806 | 15.888 | 180.918 | 10 |
| VALIDATION | 49.202 | 3.972 | 45.230 | 3 |
| TEST | 61.503 | 4.965 | 56.538 | 4 |
| **TOTAL** | **307.511** | **24.825** | **282.686** | **17** |

Proporciones configuradas:

- TRAIN: 64%
- VALIDATION: 16%
- TEST: 20%

## 4. Preprocesamiento

Contrato construido exclusivamente desde TRAIN V2.

- Columnas numéricas: 238
- Columnas categóricas: 16
- Variables resultantes: 378
- Tamaño máximo de lote: 20000
- Contrato: `data/artifacts/ml/v2/preprocessing_contract.joblib`

Se utiliza imputación mediante estadísticas de TRAIN y
codificación OneHotEncoder con categorías aprendidas de TRAIN.

VALIDATION y TEST se transforman utilizando el mismo contrato.

## 5. Validaciones

Los scripts de auditoría comprueban:

- Integridad de identificadores y etiquetas.
- Ausencia de clientes compartidos entre particiones.
- Consistencia de esquemas Parquet.
- Coincidencia de conteos con los manifiestos.
- Dimensiones de matrices dispersas CSR.
- Valores finitos almacenados en las matrices.
- Consistencia de metadatos y conteos positivos.

La ejecución satisfactoria de las auditorías debe verificarse
por separado. Este documento se genera desde los manifiestos,
no desde un registro persistente de resultados de auditoría.

## 6. Independencia experimental

TRAIN se utiliza para aprender estadísticas y entrenar modelos.

VALIDATION se reserva para comparación y selección.

TEST se reserva para evaluación predictiva final y no debe
utilizarse para seleccionar modelos ni umbrales.

**Limitación:** TEST V2 contiene clientes procedentes del
conjunto etiquetado utilizado en experimentos anteriores.
Por ello no constituye una evaluación externa completamente
inédita respecto a todo el historial del proyecto.

## 7. Limitaciones pendientes

- Auditar disponibilidad temporal de las variables originales.
- Verificar posibles fugas de información en feature engineering.
- Incorporar una política para columnas completamente nulas.
- No se ha realizado comparación fila por fila de valores
  transformados contra el origen.
- No se han calculado métricas predictivas sobre TEST V2.

## 8. Artefactos

- `data/processed/ml/v2/`
- `data/artifacts/ml/v2/`
- `scripts/ml/build_ml_splits_v2.py`
- `scripts/ml/validate_ml_splits_v2.py`
- `scripts/ml/build_ml_preprocessed_v2.py`
- `scripts/ml/audit_preprocessed_v2.py`
- `src/npl_portfolio/ml/three_way_splitter.py`

## 9. Siguiente fase

Fase 10.7: entrenamiento y comparación de modelos V2,
utilizando TRAIN y VALIDATION.

---

Documento generado automáticamente: 2026-10-08 21:32 UTC.
