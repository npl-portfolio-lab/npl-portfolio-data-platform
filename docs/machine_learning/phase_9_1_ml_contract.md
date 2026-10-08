# Fase 9.1 — Contrato de preprocesamiento

**Informe generado:** 2026-10-08 17:37 UTC

## 1. Objetivo

Definir y conservar las estadísticas necesarias para transformar
los datos de entrenamiento y validación de manera consistente,
evitando aprender información del conjunto VALIDATION.

## 2. Metodología

Se utiliza `PreprocessingContractBuilder` y DuckDB para calcular
las estadísticas a partir del archivo TRAIN.

- Variables numéricas: mediana para imputación.
- Variables categóricas: moda para imputación.
- Categorías: valores distintos registrados durante el ajuste.

El contrato se almacena con Joblib para reutilizarlo.

## 3. Resultados registrados

| Indicador | Resultado |
|---|---:|
| Variables con mediana | 238 |
| Variables con moda | 16 |
| Variables con vocabulario | 16 |
| Archivo del contrato | `preprocessing_contract.joblib` |

## 4. Validaciones

- El archivo del contrato existe y se puede cargar.
- Contiene estadísticas de preprocesamiento.
- Se verificaron las cantidades registradas en el contrato.

## 5. Conclusión

El contrato permite aplicar estadísticas consistentes a los
conjuntos de datos. Su implementación está diseñada para aprender
exclusivamente de TRAIN.

Esta revisión confirma la existencia y estructura del artefacto;
no demuestra por sí sola el origen histórico de cada estadística.
