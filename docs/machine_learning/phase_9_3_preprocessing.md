# Fase 9.3 — Preprocesamiento y matrices sparse

**Informe generado:** 2026-10-08 17:37 UTC

## 1. Objetivo

Transformar las variables originales en matrices numéricas
adecuadas para entrenar modelos de Machine Learning,
controlando el consumo de memoria.

## 2. Metodología

El preprocesamiento utiliza:

- DuckDB para lectura de archivos Parquet.
- Procesamiento por lotes de 20.000 registros.
- Estadísticas de imputación aprendidas desde TRAIN.
- Codificación de variables categóricas.
- Matrices dispersas CSR almacenadas como archivos NPZ.
- Metadatos de cliente y TARGET almacenados en Parquet.

## 3. Resultados

| Indicador | TRAIN | VALIDATION |
|---|---:|---:|
| Registros | 246,008 | 61,503 |
| Features transformadas | 378 | 378 |
| Lotes | 13 | 4 |
| Elementos almacenados (nnz) | 41,378,157 | 10,343,166 |
| Formato | CSR | CSR |

El contrato registra:

- **238** medianas numéricas.
- **16** modas categóricas.
- **16** vocabularios categóricos.

## 4. Validaciones

| Comprobación | Resultado |
|---|---|
| Matrices y metadatos correspondientes | PASS |
| Formato CSR | PASS |
| Número consistente de features | PASS |
| Ausencia de NaN e infinitos | PASS |
| TARGET binario y sin nulos | PASS |
| IDs únicos por conjunto | PASS |
| Sin clientes compartidos | PASS |
| Cantidades esperadas de registros | PASS |

## 5. Conclusión

Los conjuntos transformados cumplen las comprobaciones
estructurales y de integridad realizadas.

El procesamiento por lotes y el formato sparse permiten
evitar materializar toda la matriz como un arreglo denso.

La evaluación de la calidad predictiva corresponde
a las fases posteriores de Machine Learning.
